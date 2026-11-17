# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import logging
import os
import threading
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, override
from urllib.parse import urlparse
from zipfile import ZipFile

import cdsapi
import requests
import xarray as xr
from xarray.backends import AbstractDataStore, BackendEntrypoint
from xarray.core.types import ReadBuffer

from arcomake.datetime_utils import DateInterval
from arcomake.temporary_utils import get_current_temp_registry, set_current_temp_registry

logger = logging.getLogger(__name__)

# Serializes the opening of GRIB files by concurrent EWDS requests (see `_process_request`).
_GRIB_OPEN_LOCK = threading.Lock()


def _drop_variables(dataset: xr.Dataset, names: str | Iterable[str] | None) -> xr.Dataset:
  """Apply `drop_variables` as xarray's own backends do: listed variables absent are ignored."""
  if names is None:
    return dataset
  return dataset.drop_vars([names] if isinstance(names, str) else list(names), errors="ignore")


def _copernicusmarine() -> Any:
  """The toolbox, imported at first use, with its log handlers removed.

  The toolbox reads some settings (COPERNICUSMARINE_MAX_POOL_CONNECTIONS) when it is imported.
  xarray imports this module whenever it lists its backends, possibly before the settings file
  is applied (library use, tests): importing the toolbox here would fix those settings early.
  """
  import copernicusmarine

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
      # and the part's `chunks` decide the reads. tests/live checks that this still holds.
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
  ) -> xr.Dataset:
    logger.info(f"Downloading NetCDF from: {filename_or_obj}")

    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")
    temp_dir = registry.create_temp_dir()

    temp_path = temp_dir / "downloaded.nc"
    with requests.get(filename_or_obj, stream=True) as response:
      response.raise_for_status()  # Raise an exception for HTTP errors

      with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

    # Open the downloaded file as xarray Dataset
    dataset = xr.open_dataset(temp_path, engine="netcdf4", drop_variables=drop_variables)
    return dataset

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
  ) -> xr.Dataset:
    url = urlparse(filename_or_obj)
    if url.path or url.query or url.fragment:
      warnings.warn(
        f"Possibly invalid EWDS URL: {filename_or_obj}. Ignoring path, query and fragment"
      )
    dataset_name = url.netloc
    if any(
      arg is None for arg in [system_version, hydrological_model, product_type, timespan, variable]
    ):
      raise ValueError("Missing required argument.")
    if start_datetime is None or end_datetime is None:
      raise ValueError("Missing required argument.")
    registry = get_current_temp_registry()
    if registry is None:
      raise ValueError("No temporary directory registry available.")
    consecutive_dates = self._consecutive_dates_by_month(start_datetime, end_datetime)

    requests = [
      {
        "system_version": [system_version],
        "hydrological_model": [hydrological_model],
        "product_type": [product_type],
        "timespan": [timespan],
        "variable": [variable] if isinstance(variable, str) else variable,
        "year": f"{dates[0].year}",
        "month": [f"{dates[0].month:02}"],
        "day": [f"{date.day:02}" for date in dates],
        "data_format": "grib2",
        "download_format": "zip",
      }
      for dates in consecutive_dates
    ]

    raw_datasets = self._process_requests(dataset_name, requests, max_workers=max_workers)

    datasets: list[xr.Dataset] = []
    for dates, dataset in zip(consecutive_dates, raw_datasets, strict=True):
      # TODO:
      #  Unfortunately, it was impossible to get documentation on `valid_time` for historical GLOFAS data, hence
      #  the provider documentation should clarify the difference between the two, e.g., if renaming the time
      #  coordinate is needed (after appropriately resizing the download time interval)
      #    ds = ds.rename(valid_time='time')
      if extra_coords := [
        name for name in dataset.coords if name not in [latitude_dim, longitude_dim, time_dim]
      ]:
        dataset = dataset.drop_vars(extra_coords)
      # _process_request drops the time dimension if of length one
      if len(dates) == 1:
        dataset = dataset.expand_dims(dim=time_dim, axis=0)
        dataset = dataset.assign_coords({time_dim: dates})
      datasets.append(dataset)

    return _drop_variables(xr.concat(datasets, dim=time_dim), drop_variables)

  @staticmethod
  def _consecutive_dates_by_month(
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
  ) -> list[list[datetime.datetime]]:
    date_interval = DateInterval(start=start_datetime, end=end_datetime)
    days = date_interval.to_list(freq=datetime.timedelta(days=1))

    def partition[T](f: Callable[[T, T], bool], sequence: Iterable[T]) -> list[list[T]]:
      part: list[list[T]] = []
      subseq: list[T] = []
      last: T | None = None
      for current in sequence:
        if (last is None) or (not subseq) or f(last, current):
          subseq.append(current)
        else:
          part.append(subseq)
          subseq = [current]
        last = current
      part.append(subseq)
      return part

    def same_month_and_year(x: datetime.datetime, y: datetime.datetime) -> bool:
      return x.month == y.month and x.year == y.year

    return partition(same_month_and_year, days)

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
