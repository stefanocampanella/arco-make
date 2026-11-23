# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
import threading
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, override
from urllib.parse import urlparse
from zipfile import ZipFile

import cdsapi
import fsspec
import fsspec.core
import fsspec.utils
import numpy as np
import requests
import xarray as xr
from xarray.backends import AbstractDataStore, BackendEntrypoint
from xarray.core.types import ReadBuffer

from arcomake import chunk_plan, hdf5_lock
from arcomake.datetime_utils import DateInterval
from arcomake.temporary_utils import get_current_temp_registry, set_current_temp_registry

logger = logging.getLogger(__name__)

# Serializes the opening of GRIB files by concurrent EWDS requests (see `_process_request`).
_GRIB_OPEN_LOCK = threading.Lock()


class RunUnavailable(LookupError):
  """A forecast run its source does not have (not published yet, or not archived): a forecast
  download leaves the dataset out of that run (NaN, flagged), instead of failing it."""


def _steps_as_time(dataset: xr.Dataset, time_dim: str) -> xr.Dataset:
  """A forecast run as cfgrib opens it (`time` its initialization, `step` the lead times,
  `valid_time` their sum) along `time_dim`, the valid times: the steps of a recipe then apply as
  to any series. Scalar coordinates of the GRIB (`number`, `surface`, ...) are dropped, and so is
  cfgrib's `history` attribute: it names the temporary file read and the time of reading, so it
  differs between datasets (merged with identical attributes) and would publish a local path."""
  if "valid_time" not in dataset.coords:
    raise ValueError("a forecast run needs a valid_time coordinate (init + step)")
  if "step" not in dataset.dims:
    dataset = dataset.expand_dims("step")
  valid = dataset["valid_time"].values.reshape(-1)
  scalars = [name for name in dataset.coords if dataset[name].ndim == 0]
  dataset = dataset.drop_vars([*scalars, "valid_time"], errors="ignore")
  dataset = dataset.drop_vars("step").rename({"step": time_dim})
  attrs = {key: value for key, value in dataset.attrs.items() if key != "history"}
  return dataset.assign_coords({time_dim: valid}).drop_attrs(deep=False).assign_attrs(attrs)


def _drop_variables(dataset: xr.Dataset, names: str | Iterable[str] | None) -> xr.Dataset:
  """Apply `drop_variables` as xarray's own backends do: listed variables absent are ignored."""
  if names is None:
    return dataset
  return dataset.drop_vars([names] if isinstance(names, str) else list(names), errors="ignore")


POOL_VARIABLE = "COPERNICUSMARINE_MAX_POOL_CONNECTIONS"


def _copernicusmarine() -> Any:
  """The toolbox, imported at first use, with its log handlers removed.

  The toolbox reads some settings (COPERNICUSMARINE_MAX_POOL_CONNECTIONS) when it is imported.
  xarray imports this module whenever it lists its backends, possibly before the settings file
  is applied (library use, tests): importing the toolbox here would fix those settings early.

  Unless set, the pool is as large as Dask's thread count (at least the toolbox's 10): with 10
  connections for 32 threads, connections were discarded and opened again all the time (TLS
  handshakes were 29 % of the reading time), and an 8-day block took 37 % longer to compute,
  with 13 GiB more memory (logbook, 18 Nov 2026). The variable is set only for the import: the
  toolbox has read it then, and child processes or later code do not inherit it.
  """
  given = POOL_VARIABLE in os.environ
  if not given:
    os.environ[POOL_VARIABLE] = str(max(10, chunk_plan.dask_threads()))
  try:
    import copernicusmarine
  finally:
    if not given:
      del os.environ[POOL_VARIABLE]

  # Avoid annoying copernicusmarine log handling (its handlers are installed at import).
  cm_logger = logging.getLogger("copernicus_marine_root_logger")
  for handler in list(cm_logger.handlers):
    cm_logger.removeHandler(handler)
  cm_logger.setLevel(level=logging.getLevelName(logger.getEffectiveLevel()))
  return copernicusmarine


class CopernicusMarine(BackendEntrypoint):
  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    variables: str | Iterable[str] | None = None,
    start_datetime: str | datetime.datetime | None = None,
    end_datetime: str | datetime.datetime | None = None,
    dataset_version: str | None = None,
    dataset_part: str | None = None,
    service: str | None = None,
  ) -> xr.Dataset:

    url = urlparse(filename_or_obj)
    if url.path or url.query or url.fragment:
      warnings.warn(
        f"Possibly invalid Copernicus Marine URL: {filename_or_obj}. Ignoring path, query and fragment"
      )
    dataset_id = url.netloc
    if dataset_id == "":
      raise ValueError("Missing dataset ID.")

    if variables is None:
      variables_list: list[str] = []
    elif isinstance(variables, str):
      variables_list = [variables]
    else:
      variables_list = list(variables)

    dataset = _copernicusmarine().open_dataset(
      dataset_id=dataset_id,
      dataset_version=dataset_version,
      variables=variables_list,
      dataset_part=dataset_part,
      service=service,
      start_datetime=start_datetime,
      end_datetime=end_datetime,
      # IMPORTANT: the toolbox returns Dask arrays of its own chunking (e.g. time 2, all depths),
      # unless `chunk_size_limit` is falsy (undocumented): then it returns lazily indexed arrays,
      # and the planned chunks (`chunk_plan`) decide the reads. tests/live checks that this holds.
      chunk_size_limit=False,
    )
    assert isinstance(dataset, xr.Dataset)
    return _drop_variables(dataset, drop_variables)

  @override
  def guess_can_open(
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | ReadBuffer[Any]
    | bytes
    | memoryview
    | AbstractDataStore,
  ) -> bool:

    if not isinstance(filename_or_obj, str):
      return False
    else:
      url = urlparse(filename_or_obj)
      return url.scheme == "cm"


class NetCDFOverHTTP(BackendEntrypoint):
  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    timeout: float | None = None,
  ) -> xr.Dataset:
    """Download the file (requests wait at most `timeout` seconds for a connection or for data),
    then open it with netCDF4."""
    logger.info(f"Reading NetCDF from: {filename_or_obj}")

    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")

    def fetch(path) -> None:
      # Without a timeout, a stalled connection hangs until the job's time limit (6.2).
      request_timeout = None if timeout is None else (min(30.0, timeout), timeout)
      with requests.get(filename_or_obj, stream=True, timeout=request_timeout) as response:
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(path, "wb") as f:
          for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Fetched once per run (registry): every block of a download reads the same file, and a
    # server outage after the first block no longer fails the others (as on 16 Nov for GloFAS).
    temp_path = registry.cached_file(filename_or_obj, "downloaded.nc", fetch)

    # Under arco-make's HDF5 lock: netCDF4 is not thread-safe, and Dask's threads read other
    # netCDF files meanwhile.
    return hdf5_lock.open_dataset(temp_path, engine="netcdf4", drop_variables=drop_variables)

  @override
  def guess_can_open(
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | ReadBuffer[Any]
    | bytes
    | memoryview
    | AbstractDataStore,
  ) -> bool:

    if not isinstance(filename_or_obj, str):
      return False
    else:
      url = urlparse(filename_or_obj)
      return url.scheme == "http" or url.scheme == "https"


class ECMWFOpenData(BackendEntrypoint):
  """ECMWF's open data forecasts (IFS, AIFS), one run along valid time, from a copy of
  data.ecmwf.int's tree: AWS's `s3://ecmwf-forecasts` (anonymous, from early 2024 at 0.25 deg),
  Azure's or Google's.

  A run is a directory of GRIB2 files, one per step, each with a JSON-lines `.index` giving every
  field's byte range: only the fields asked are fetched (`fsspec`'s `cat_ranges`, concurrent),
  written to the run's temporary directory, read by cfgrib, and the files removed. Steps are those
  listed in the run's directory whose valid times are in [start, end], the end included: an
  accumulated field's last interval needs it (the instant fields' stamp at `end` is cut by the
  block's time selection, as is the last interval's NaN).
  """

  INDEX_SUFFIX = ".index"

  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    run_datetime: datetime.datetime | None = None,
    start_datetime: datetime.datetime | None = None,
    end_datetime: datetime.datetime | None = None,
    param: str | Iterable[str] | None = None,
    levtype: str = "sfc",
    model: str = "ifs",
    resolution: str = "0p25",
    stream: str = "oper",
    type: str = "fc",
    deaccumulate: bool = True,
    time_dim: str = "time",
    storage_options: dict[str, Any] | None = None,
  ) -> xr.Dataset:
    """The run of `run_datetime`, fields `param` (GRIB short names, e.g. "2t", "tp") of
    `levtype`, along `time_dim` (valid times). With `deaccumulate`, fields accumulated since the
    run's start (GRIB `stepType` "accum": precipitation, radiation) become the increments between
    consecutive steps, stamped at the start of theirs: a daily sum of day k is then exactly the
    accumulation at its end minus that at its start, whatever the steps (3-hourly, then 6-hourly
    for IFS). The accumulations are packed lossily in GRIB, so an increment can be a little
    negative (-1.2e-4 m of precipitation seen over a day): clip them in the recipe
    (`clip_negative`), as for ERA5. Variables keep cfgrib's names (`t2m` for "2t").

    Raises:
      RunUnavailable: if the source has no such run (not published yet, or not archived).
    """
    if run_datetime is None or start_datetime is None or end_datetime is None:
      raise ValueError("ecmwfopendata reads forecast runs: run, start and end are injected")
    if param is None:
      raise ValueError("ecmwfopendata: give `param`, the GRIB short names of the fields")
    params = [param] if isinstance(param, str) else list(param)
    root = str(filename_or_obj).rstrip("/")
    options = dict(storage_options or {})
    if fsspec.utils.get_protocol(root) in ("s3", "s3a"):
      options.setdefault("anon", True)
    fs, path = fsspec.core.url_to_fs(root, **options)
    run_dir = f"{path}/{run_datetime:%Y%m%d}/{run_datetime:%H}z/{model}/{resolution}/{stream}"
    prefix = f"{run_datetime:%Y%m%d%H}0000"
    try:
      listed = fs.ls(run_dir, detail=False)
    except FileNotFoundError:
      listed = []
    steps = self._steps(listed, prefix, stream, type)
    if not steps:
      # Not published (or not archived) only if the run's time has nothing at all: a run there
      # without this model, resolution, stream or type is a recipe's mistake, not a gap.
      time_dir = f"{path}/{run_datetime:%Y%m%d}/{run_datetime:%H}z"
      try:
        published = bool(fs.ls(time_dir, detail=False))
      except FileNotFoundError:
        published = False
      if published and listed:
        raise ValueError(
          f"{root}: the {model} {stream} run of {run_datetime:%Y-%m-%d %H:%M} has no indexed"
          f" files of type {type!r} (check `type` and `stream`)"
        )
      if published:
        raise ValueError(
          f"{root} has runs of {run_datetime:%Y-%m-%d %H:%M}, but not {model}/{resolution}/"
          f"{stream} (check `model`, `resolution` and `stream`)"
        )
      raise RunUnavailable(f"{root} has no {model} {stream} run of {run_datetime:%Y-%m-%d %H:%M}")
    first = start_datetime - run_datetime
    last = end_datetime - run_datetime
    wanted = [s for s in steps if first <= datetime.timedelta(hours=s) <= last]
    if not wanted or datetime.timedelta(hours=wanted[-1]) != last:
      raise ValueError(
        f"the {model} run of {run_datetime:%Y-%m-%d %H:%M} has no step at {end_datetime} (steps"
        f" to {steps[-1]} h): the block must end on a step (an accumulation's last interval"
        " needs it)"
      )
    self._no_gaps(wanted, run_datetime, model)
    names = [f"{run_dir}/{prefix}-{s}h-{stream}-{type}" for s in wanted]
    indexes = fs.cat([f"{name}{self.INDEX_SUFFIX}" for name in names])
    # Per field, per step, the byte ranges of its messages (several for levels or members).
    fields: dict[str, list[list[tuple[str, int, int]]]] = {p: [] for p in params}
    for name in names:
      found: dict[str, list[tuple[str, int, int]]] = {p: [] for p in params}
      for line in indexes[f"{name}{self.INDEX_SUFFIX}"].decode().splitlines():
        if not line.strip():
          continue
        entry = json.loads(line)
        if entry.get("levtype") == levtype and entry.get("param") in found:
          offset, length = int(entry["_offset"]), int(entry["_length"])
          found[entry["param"]].append((f"{name}.grib2", offset, offset + length))
      for param, ranges in found.items():
        fields[param].append(ranges)
    # Every step needs the same messages of a field (none missing at some steps).
    if missing := [p for p, per_step in fields.items() if len({len(r) for r in per_step}) != 1]:
      raise ValueError(
        f"the {model} run of {run_datetime:%Y-%m-%d %H:%M} lacks {missing} ({levtype}) at some"
        f" of the steps {wanted}"
      )
    if absent := [p for p, per_step in fields.items() if not per_step[0]]:
      raise ValueError(
        f"the {model} run of {run_datetime:%Y-%m-%d %H:%M} has no {absent} ({levtype})"
      )
    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")
    directory = registry.create_temp_dir()
    try:
      # One concurrent fetch of every message of every field.
      ranges = [r for per_step in fields.values() for step in per_step for r in step]
      paths, starts, ends = map(list, zip(*ranges, strict=True))
      blobs = iter(fs.cat_ranges(paths, starts, ends))
      datasets = []
      for name, per_step in fields.items():
        grib = directory / f"{name}.grib2"
        grib.write_bytes(b"".join(next(blobs) for step in per_step for _ in step))
        datasets.append(self._read(grib, time_dim, deaccumulate))
    finally:
      shutil.rmtree(directory, ignore_errors=True)
    return _drop_variables(xr.merge(datasets, join="exact"), drop_variables)

  @staticmethod
  def _no_gaps(steps: list[int], run: datetime.datetime, model: str) -> None:
    """Fail if a step is missing inside the steps read (a file not published yet, or lost): the
    spacing of a run's steps never shrinks (IFS: 3 hours to 144 h, then 6), so a spacing larger
    than the next one is a gap; and the last spacing may not be larger than the one before (a
    gap before the last step; a block ending one step after the spacing grows is refused too,
    which daily blocks of IFS never do). A daily mean would average fewer samples, silently. A
    step missing just where the spacing grows (141 h for IFS) looks like the change of spacing,
    and is not seen: ECMWF publishes whole runs, step by step, and a run still being published
    lacks its last steps (no step at the block's end)."""
    spacings = [b - a for a, b in itertools.pairwise(steps)]
    for i, (spacing, following) in enumerate(itertools.pairwise(spacings)):
      last = i + 2 == len(spacings)
      if spacing > following or (last and following > spacing):
        at = i if spacing > following else i + 1
        raise ValueError(
          f"the {model} run of {run:%Y-%m-%d %H:%M} lacks a step between {steps[at]} and"
          f" {steps[at + 1]} h (being published, or lost)"
        )

  @staticmethod
  def _steps(listed: Iterable[str], prefix: str, stream: str, type: str) -> list[int]:
    """The steps (hours) of a run's GRIB files with an index, sorted."""
    names = {pathlib.PurePosixPath(p).name for p in listed}
    pattern = re.compile(rf"{prefix}-(\d+)h-{re.escape(stream)}-{re.escape(type)}\.grib2")
    steps = []
    for name in names:
      match = pattern.fullmatch(name)
      if match and f"{name.removesuffix('.grib2')}{ECMWFOpenData.INDEX_SUFFIX}" in names:
        steps.append(int(match.group(1)))
    return sorted(steps)

  @staticmethod
  def _read(grib: pathlib.Path, time_dim: str, deaccumulate: bool) -> xr.Dataset:
    """One field's steps, loaded (its file is removed after), along valid time."""
    with _GRIB_OPEN_LOCK:
      dataset = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""}).load()
    dataset = _steps_as_time(dataset, time_dim)
    if not deaccumulate:
      return dataset
    for name, var in dataset.data_vars.items():
      if var.attrs.get("GRIB_stepType") == "accum":
        # The increment over [t_i, t_i+1], stamped t_i; the last stamp has none.
        increments = var.shift({time_dim: -1}) - var
        dataset[name] = increments.assign_attrs(var.attrs | {"GRIB_stepType": "increment"})
    return dataset

  @override
  def guess_can_open(self, filename_or_obj) -> bool:
    return False  # explicit: engine = "ecmwfopendata"


class EarlyWarningDataStore(BackendEntrypoint):
  """
  Support for downloading from the Early Warning Data Store (EWDS) using cdsapi.
  Minimal implementation: it monkey-patches the cdsapi logger to avoid unformatted output,
  it downloads the whole set of dates at once, loads and returns the dataset.
  For a more robust and feature-rich implementation, see: https://github.com/bopen/xarray-ecmwf/tree/main
  When available, ARCO alternatives should be preferred.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # No progress bars; cdsapi messages go to this module's logger.
    self.progress = False
    self.client_logger = logger

  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    start_datetime: datetime.datetime | None = None,
    end_datetime: datetime.datetime | None = None,
    system_version: str | None = None,
    hydrological_model: str | None = None,
    product_type: str | None = None,
    timespan: str | None = None,
    variable: str | Iterable[str] | None = None,
    time_dim: str = "time",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
    max_workers: int | None = None,
    timeout: float | None = None,
    run_datetime: datetime.datetime | None = None,
  ) -> xr.Dataset:
    url = urlparse(filename_or_obj)
    if url.path or url.query or url.fragment:
      warnings.warn(
        f"Possibly invalid EWDS URL: {filename_or_obj}. Ignoring path, query and fragment"
      )
    dataset_name = url.netloc
    # A forecast run (`run_datetime`) takes no timespan: its lead times are the request's.
    required = [system_version, hydrological_model, product_type, variable]
    if run_datetime is None:
      required.append(timespan)
    if any(arg is None for arg in required):
      raise ValueError("Missing required argument.")
    if start_datetime is None or end_datetime is None:
      raise ValueError("Missing required argument.")
    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")
    base = {
      "system_version": [system_version],
      "hydrological_model": [hydrological_model],
      "product_type": [product_type],
      **({"timespan": [timespan]} if timespan is not None else {}),
      "variable": [variable] if isinstance(variable, str) else variable,
      "data_format": "grib2",
      "download_format": "zip",
    }
    client_options = {} if timeout is None else {"timeout": timeout}
    if run_datetime is not None:
      run = self._forecast_run(
        dataset_name, base, run_datetime, start_datetime, end_datetime, time_dim, client_options
      )
      return _drop_variables(run, drop_variables)
    groups = self._request_groups(dataset_name, base, start_datetime, end_datetime, client_options)
    requests = [base | self._year_request(dates) for dates in groups]
    raw_datasets = self._process_requests(
      dataset_name, requests, max_workers=max_workers, **client_options
    )

    datasets: list[xr.Dataset] = []
    for dates, dataset in zip(groups, raw_datasets, strict=True):
      # TODO:
      #  Unfortunately, it was impossible to get documentation on `valid_time` for historical GLOFAS data, hence
      #  the provider documentation should clarify the difference between the two, e.g., if renaming the time
      #  coordinate is needed (after appropriately resizing the download time interval)
      #    ds = ds.rename(valid_time='time')
      if extra_coords := [
        name for name in dataset.coords if name not in [latitude_dim, longitude_dim, time_dim]
      ]:
        dataset = dataset.drop_vars(extra_coords)
      # A one-day request (one month, one day) gives no time dimension.
      if len(dates) == 1:
        dataset = dataset.expand_dims(dim=time_dim, axis=0)
        dataset = dataset.assign_coords({time_dim: dates})
      else:  # the request's months x days: keep only the interval's days
        given = dataset[time_dim].values.astype("datetime64[D]")
        wanted = np.array([date.date() for date in dates], dtype="datetime64[D]")
        if missing := sorted(set(wanted.tolist()) - set(given.tolist())):
          # The service leaves out dates not published yet, if others of the request are.
          raise ValueError(
            f"EWDS gave no data for {len(missing)} day(s) of {dataset_name}, from {missing[0]}"
            f" to {missing[-1]}: not published yet?"
          )
        dataset = dataset.isel({time_dim: np.isin(given, wanted)})
      datasets.append(dataset)

    return _drop_variables(xr.concat(datasets, dim=time_dim), drop_variables)

  def _forecast_run(
    self,
    dataset_name: str,
    base: dict[str, Any],
    run: datetime.datetime,
    start: datetime.datetime,
    end: datetime.datetime,
    time_dim: str,
    client_options: dict[str, Any],
  ) -> xr.Dataset:
    """The forecast run of `run` whose valid times are in [start, end), along `time_dim`: one
    request, for the run's day and its lead hours (multiples of `LEAD_HOURS`, as the service
    gives them).

    The service's cost estimate is asked first: a run it refuses (not published yet; it answers
    as for any combination it lacks) raises `RunUnavailable`.
    """
    leads = self._lead_hours(run, start, end)
    if not leads:
      raise ValueError(f"no lead time of the run {run} is in [{start}, {end})")
    request = base | {
      "year": [f"{run.year}"],
      "month": [f"{run.month:02}"],
      "day": [f"{run.day:02}"],
      "leadtime_hour": [str(hours) for hours in leads],
    }
    client = self.get_cdsapi_client(
      progress=False, client_logger=self.client_logger, **client_options
    )
    # Without an estimate, the run is requested anyway: a run the service lacks then fails.
    if (costs := self._estimate(client, dataset_name, request)) is not None:
      if not self._valid(costs):
        # The service answers so for any combination it lacks: a run not published yet, and a
        # recipe's mistake (a variable, a product type). The same request for a run a week
        # older tells which.
        older = run - datetime.timedelta(days=7)
        again = self._estimate(
          client,
          dataset_name,
          request
          | {"year": [f"{older.year}"], "month": [f"{older.month:02}"]}
          | {"day": [f"{older.day:02}"]},
        )
        if again is not None and not self._valid(again):
          reason = str(costs.get("invalid_reason") or "").splitlines()[:1]
          raise ValueError(
            f"{dataset_name} refuses the request for the run of {run:%Y-%m-%d} and for a run a"
            f" week older (check the recipe's variable, product type and versions): {reason}"
          )
        raise RunUnavailable(f"{dataset_name} has no run of {run:%Y-%m-%d %H:%M}")
      if costs["cost"] > costs["limit"]:
        raise ValueError(
          f"{dataset_name}: the run of {run:%Y-%m-%d} costs {costs['cost']}, above the service's"
          f" limit of {costs['limit']} (fields: lead times x variables)"
        )
    (dataset,) = self._process_requests(dataset_name, [request], **client_options)
    dataset = _steps_as_time(dataset, time_dim)
    wanted = np.array([run + datetime.timedelta(hours=h) for h in leads], dtype="datetime64[ns]")
    if not np.array_equal(dataset[time_dim].values, wanted):
      raise ValueError(
        f"{dataset_name} gave the run {run:%Y-%m-%d} with other lead times than asked: {leads}"
      )
    return dataset

  # The lead times of the EWDS forecasts are whole days (GloFAS: 24 to 720 hours).
  LEAD_HOURS = 24

  @classmethod
  def _lead_hours(
    cls, run: datetime.datetime, start: datetime.datetime, end: datetime.datetime
  ) -> list[int]:
    """The lead hours of the run whose valid times are in [start, end)."""
    first = max(1, -(-(start - run) // datetime.timedelta(hours=cls.LEAD_HOURS)))
    hours = []
    lead = first * cls.LEAD_HOURS
    while run + datetime.timedelta(hours=lead) < end:
      hours.append(lead)
      lead += cls.LEAD_HOURS
    return hours

  def _request_groups(
    self,
    dataset_name: str,
    base: dict[str, Any],
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    client_options: dict[str, Any],
  ) -> list[list[datetime.datetime]]:
    """The interval's days, one group per request: per year (see `_dates_by_year`) where the
    service accepts the request, else per month, as exact as before.

    A request over several months asks for every date of its months x days: the service refuses
    it whole if some are not published yet (near the end of the record), and refuses requests
    above its limit (500 fields: days x variables, so a year of two variables). Its cost
    estimate tells, before any wait in the queue.
    """
    groups: list[list[datetime.datetime]] = []
    client = None
    for dates in self._dates_by_year(start_datetime, end_datetime):
      months = [list(g) for _, g in itertools.groupby(dates, key=lambda day: day.month)]
      if len(months) > 1:
        client = client or self.get_cdsapi_client(
          progress=False, client_logger=self.client_logger, **client_options
        )
        if not self._accepted(client, dataset_name, base | self._year_request(dates)):
          groups.extend(months)
          continue
      groups.append(dates)
    return groups

  @staticmethod
  def _estimate(client: Any, dataset_name: str, request: dict[str, Any]) -> dict[str, Any] | None:
    """The service's cost estimate of `request`; None if unavailable (an older client, an
    error)."""
    datastores = getattr(client, "client", None)  # the legacy cdsapi client's datastores client
    if datastores is None:
      return None
    try:
      return datastores.estimate_costs(dataset_name, request)
    except Exception as error:
      logger.warning(f"EWDS cost estimate failed ({error})")
      return None

  @staticmethod
  def _valid(costs: dict[str, Any]) -> bool:
    """Whether a cost estimate says its request is a valid combination of values. A valid
    request comes back with `request_is_valid` None (measured 23 Nov 2026), not True: only False
    refuses it."""
    return costs.get("request_is_valid") is not False

  @classmethod
  def _accepted(cls, client: Any, dataset_name: str, request: dict[str, Any]) -> bool:
    """Whether the service's cost estimate accepts `request` (valid, within the limit). False if
    the estimate is unavailable: requests per month then, as before."""
    costs = cls._estimate(client, dataset_name, request)
    if costs is None:
      logger.info("No EWDS cost estimate: one request per month")
      return False
    accepted = cls._valid(costs) and costs["cost"] <= costs["limit"]
    if not accepted:
      reason = str(costs.get("invalid_reason") or f"cost {costs['cost']} > {costs['limit']}")
      logger.info(
        f"EWDS refuses {request['year']} in one request ({reason.splitlines()[0]}):"
        " one request per month"
      )
    return bool(accepted)

  @staticmethod
  def _dates_by_year(
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
  ) -> list[list[datetime.datetime]]:
    """The interval's days, grouped by year: one request each.

    The EWDS runs one request of a user at a time, about two minutes apart (measured 18 Nov
    2026): a request per month cost a 32-day block, usually spanning two months, an extra wait.
    """
    days = DateInterval(start=start_datetime, end=end_datetime).to_list(
      freq=datetime.timedelta(days=1)
    )
    return [list(group) for _, group in itertools.groupby(days, key=lambda day: day.year)]

  @staticmethod
  def _year_request(dates: list[datetime.datetime]) -> dict[str, Any]:
    """The `year`, `month` and `day` of a request for the given days of one year.

    The service gives every valid date of months x days, so days spanning two months partly also
    bring days outside them (at most about a month's worth for a 32-day block); `open_dataset`
    drops them.
    """
    return {
      "year": f"{dates[0].year}",
      "month": sorted({f"{date.month:02}" for date in dates}),
      "day": sorted({f"{date.day:02}" for date in dates}),
    }

  def _process_requests(
    self,
    dataset_name: str,
    requests: Iterable[dict[str, Any]],
    max_workers: int | None = None,
    **kwargs,
  ) -> list[xr.Dataset]:
    """Submit requests to the Climate Data Store, download temporary files, and return datasets.
    Requests are processed asynchronously on a thread pool.
    """
    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")

    def _worker(request: dict[str, Any]) -> xr.Dataset:
      return self._process_request(dataset_name, request, **kwargs)

    def initializer():
      set_current_temp_registry(registry)

    with ThreadPoolExecutor(initializer=initializer, max_workers=max_workers) as executor:
      logger.info(f"Downloading {dataset_name} from EWDS using {executor._max_workers} threads")
      return list(executor.map(_worker, requests))

  def _process_request(self, dataset_name: str, request: dict[str, Any], **kwargs) -> xr.Dataset:
    """Submit a request to the Climate Data Store, download its GRIB files, and return a dataset.

    Every file (the zip and its content) lives in a directory of the current temporary registry,
    so it is removed when the registry closes, including on SIGTERM (intent 9).
    """
    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")
    path = registry.create_temp_dir()
    archive = path / "request.zip"

    client = self.get_cdsapi_client(
      progress=self.progress, client_logger=self.client_logger, **kwargs
    )
    client.retrieve(dataset_name, request, str(archive))
    with ZipFile(archive) as zipfile:
      zipfile.extractall(path=path)
    archive.unlink()  # The extracted files are all that is needed.
    # FIXME: when reading grib files eccodes emits the following warning:
    #    ECCODES WARNING :  g2date:unpack_long: Date is not valid! year=0 month=0 day=0
    #  see: https://git.ecmwf.int/users/erds/repos/eccodes/browse/src/accessor/grib_accessor_class_g2date.cc?at=f10d3f3c1be3737a231d4a2c19f45535de4d4424#71-75
    #  Also, cdsapi download one NetCDF per variable, ugly. :(
    # Requests run on a thread pool, but ecCodes' definition parser is not thread-safe, and
    # cfgrib only locks reads, not opens: concurrent opens crashed the process (segfault).
    with _GRIB_OPEN_LOCK:
      return xr.open_mfdataset(list(path.glob("*.grib")), engine="cfgrib")

  # Credits to the amazing Stefano Piani from OGS
  @staticmethod
  def get_cdsapi_client(url: str | None = None, client_logger=None, **kwargs):
    """Returns a cdsapi.Client instance.

    It also configures the returned client to use a specific logger (if
    submitted)

    Args:
      url (str): the url of the endpoint of the cdsapi. If it is None, the client reads it
        from its configuration (the CDSAPI_URL environment variable or ~/.cdsapirc).
      client_logger (logging.Logger): Logger that the returned client
        will use to print its messages
      **kwargs: Other arguments of `cdsapi.Client` (e.g. `key`, read from CDSAPI_KEY or
        ~/.cdsapirc when not given).

    Returns:
      cdsapi.Client instance
    """
    if client_logger is not None:

      def debug_callback(*args, **kwargs):
        return client_logger.debug(*args, **kwargs)

      def info_callback(*args, **kwargs):
        return client_logger.info(*args, **kwargs)

      def warning_callback(*args, **kwargs):
        return client_logger.warning(*args, **kwargs)

      def error_callback(*args, **kwargs):
        return client_logger.error(*args, **kwargs)
    else:
      debug_callback = None
      info_callback = None
      warning_callback = None
      error_callback = None

    client_kwargs = {
      "debug_callback": debug_callback,
      "info_callback": info_callback,
      "warning_callback": warning_callback,
      "error_callback": error_callback,
      **kwargs,
    }

    if url is not None:
      client_kwargs["url"] = url

    cdsapi_client = cdsapi.Client(**client_kwargs)

    # This is a horrible hack that probably will become not necessary in the
    # next version of cdsapi. It removes the "logging decorator", which is a
    # context manager that changes the configuration of the logger
    if cdsapi_client.__class__.__name__.startswith("Legacy"):  # noqa: SIM102
      if hasattr(cdsapi_client, "logging_decorator"):
        cdsapi_client.logging_decorator = lambda x: x  # type: ignore

    return cdsapi_client

  @override
  def guess_can_open(
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | ReadBuffer[Any]
    | bytes
    | memoryview
    | AbstractDataStore,
  ) -> bool:

    if not isinstance(filename_or_obj, str):
      return False
    else:
      url = urlparse(filename_or_obj)
      return url.scheme == "ewds"


class Icechunk(BackendEntrypoint):
  """Icechunk repositories (the outputs of arco-make): `engine = "icechunk"`.

  The path is a local directory, an `s3://` or a `gs://` URL; `storage_options` as in
  `icechunk_utils`. Reads the tip of branch `main`, or the version named by `branch`, `tag` or
  `snapshot_id`. Decoding options are those of xarray's zarr backend; `consolidated` is accepted
  and ignored (Icechunk lists metadata without it).
  """

  description = "Open an Icechunk repository (a version of it) with xarray's zarr backend"
  url = "https://icechunk.io"

  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    branch: str | None = None,
    tag: str | None = None,
    snapshot_id: str | None = None,
    group: str | None = None,
    consolidated: bool | None = None,
    mask_and_scale: bool = True,
    decode_times: Any = True,
    concat_characters: bool = True,
    decode_coords: Any = True,
    use_cftime: bool | None = None,
    decode_timedelta: Any = None,
  ) -> xr.Dataset:
    from xarray.backends.zarr import ZarrBackendEntrypoint

    from arcomake import icechunk_utils

    store = icechunk_utils.readonly_store(
      filename_or_obj, storage_options, branch=branch, tag=tag, snapshot_id=snapshot_id
    )
    # xarray reads zarr stores; its type hints do not list them.
    return ZarrBackendEntrypoint().open_dataset(
      store,  # ty: ignore[invalid-argument-type]
      drop_variables=drop_variables,
      group=group,
      consolidated=False,
      zarr_format=3,
      mask_and_scale=mask_and_scale,
      decode_times=decode_times,
      concat_characters=concat_characters,
      decode_coords=decode_coords,
      use_cftime=use_cftime,
      decode_timedelta=decode_timedelta,
    )

  @override
  def guess_can_open(self, filename_or_obj) -> bool:
    return False  # explicit `engine = "icechunk"` only: no request to find out
