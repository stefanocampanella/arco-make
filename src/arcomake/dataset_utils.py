# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import logging
import pathlib
import tempfile
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
from zarr.storage import DirectoryStore, TempStore, ZipStore

from arcomake.datetime_utils import (
  DateInterval,
  IterableDateInterval,
  may_parse_datetime,
  may_parse_timedelta,
)
from arcomake.processing import Process
from arcomake.providers import Provider

logger = logging.getLogger(__name__)


def bar(progress):
  if progress:
    return ProgressBar()
  else:
    return nullcontext()


def get_dataset(
  provider: Provider,
  configs: dict[str, Any],
  date_intervals: IterableDateInterval | None,
  mask: xr.DataArray | None = None,
  progress: bool = False,
) -> xr.Dataset:
  """
  Downloads and pre-processes a dataset based on provided configurations.
  """
  postprocess = Process(steps=configs.get("postprocess"), mask=mask)
  with TempStore() as temporary_store:
    def _download_step(date_interval: DateInterval | None, **kwargs):
      if date_interval is not None:
        logger.info(f"Downloading and postprocessing {date_interval}")
      # When downloading from Copernicus Marine Data Store or Climate Data Store, the typical case is a large dataset,
      # spanning a long time period, with several sets of variables in different datasets (bio, phys, etc.),
      # which needs to be downloaded one piece at a time. Hence, `parts` list in the TOML configuration file
      # represent different pieces of the same dataset.
      # When downloading from CDS, a temporary directory is needed to store partial netCDF files.
      with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        fragment_datasets = []
        for ds_conf in configs.get("parts", []):
          ds = provider.open_dataset(backend_kwargs=ds_conf, date_interval=date_interval, tmpdir=tempdir)
          fragment_datasets.append(ds)
        fragment = xr.merge(fragment_datasets, join="exact")
        fragment = postprocess(fragment)
        with bar(progress):
          # Requires that fragment fits into memory
          fragment = fragment.compute()
      # Here we save the fragment to a temporary Zarr store using default parameters.
      # Being the fragment underlying data numpy arrays, the chunk size will be determined by zarr,
      # which tends to produce small chunks (1MB without compression).
      # This might be optimized, but as we are saving it fast local storage (SSD), it is probably fine.
      for var in fragment.data_vars:
        fragment[var].encoding["compressor"] = None
      logger.info(f"Saving temporary dataset to {temporary_store.path}")
      fragment.to_zarr(store=temporary_store, **kwargs)

    # TODO: in terms of memory consumption, it would make more sense to later open lazily the datasets written/appended
    #  to disk here. However, this would require the caller to pass the temporary directory directory, so that the file
    #  are not cancelled and the datasets can be opened later on.
    # For each date_interval: download, postprocess, and append the dataset to a temporary Zarr
    if date_intervals is None:
      _download_step(date_interval=None, mode="w")
    else:
      is_first_fragment = True
      for date_interval in date_intervals:
        if is_first_fragment:
          _download_step(date_interval=date_interval, mode="w")
          is_first_fragment = False
        else:
          _download_step(date_interval=date_interval, mode="a-", append_dim="time")

    # Load the temporary Zarr into memory and return it
    dataset = xr.load_dataset(temporary_store, engine='zarr',
                              backend_kwargs={"overwrite_encoded_chunks": True})
    return dataset


# The date interval returned by `parse_timeseries_arguments` will be used during download to take into account which
# job of a SLURM job array is currently being processed.
# Each element of the SLURM job array will consume its interval of dates using sub-intervals
# (of size equal to `tmp_step`).
def parse_timeseries_arguments(
  configs: dict[str, Any],
  output_path: pathlib.Path,
  start: datetime.datetime | str | None = None,
  end: datetime.datetime | str | None = None,
  array_id: int | None = None,
) -> tuple[DateInterval, pathlib.Path]:

  # The start and end could be overridden using command line arguments, so we need to check them first.
  start = start or configs.get("start")
  end = end or configs.get("end")
  if start is None or end is None:
    raise ValueError(
      "Start and end dates must be specified in the top table of the config file "
      "or as command line arguments. See --help for more information."
    )
  start = may_parse_datetime(start)
  end = may_parse_datetime(end)
  date_interval = DateInterval(start=start, end=end)

  # If we are in a SLURM job array, recompute corresponding start and end dates.
  # The array ID is the index of the job in the SLURM job array.
  # In this case, we append the start and end dates to the output path.
  if array_id is not None:
    array_step = configs.get("array_step")
    if array_step is None:
      raise ValueError(
        "Configs top table should contain the 'array_step' key when "
        "downloading using SLURM arrays. See --help for more information."
      )
    array_step = may_parse_timedelta(array_step)
    array_total_time_interval = DateInterval(start=start, end=end)
    array_time_intervals = IterableDateInterval(interval=array_total_time_interval, step=array_step)
    date_interval = array_time_intervals[array_id]
    output_path = output_path / f"{date_interval.start.strftime('%Y%m%d')}-{date_interval.end.strftime('%Y%m%d')}"
  logger.info(f"Downloading {date_interval} (array id {array_id}) to {output_path}")

  return date_interval, output_path


# FIXME: the code should handle both Zarr (using a DirectoryStore or a ZipStore) and NetCDF files.
def open_dataset_wo_static(path: pathlib.Path, time_dim: str = "time", chunks=None) -> xr.Dataset:
  """
  Open a dataset from a single Zarr file/store or a directory containing multiple Zarr zip files.

  - If `path` is a directory with one or more .zip files, open all of them via xarray.open_mfdataset(engine='zarr').
  - In all other cases, open it via xarray.open_dataset(engine='zarr').

  Returns a xarray.Dataset filtered to only data variables that include the provided time dimension.
  """
  path = pathlib.Path(path)
  if not path.exists():
    raise ValueError(f"Input path {path} does not exist")

  def _drop_static_vars(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    ds = ds.drop_vars([name for (name, var) in ds.data_vars.items() if time_dim not in var.dims])
    return ds

  # As the Dask graph tends to be huge it's important to avoid inline_array=True,
  # see: https://docs.dask.org/en/latest/generated/dask.array.from_array.html#dask.array.from_array
  if path.is_dir():
    zip_files = sorted(p for p in path.glob("*.zip"))
    if zip_files:
      # noinspection PyTypeChecker
      ds = xr.open_mfdataset(
        [str(p) for p in zip_files],
        preprocess=lambda ds: _drop_static_vars(ds, time_dim),
        engine="zarr",
        combine="by_coords",
        inline_array=False,
        chunks=chunks,
      )
      return ds

  ds = xr.open_dataset(str(path), engine="zarr", inline_array=False, chunks=chunks)
  ds = _drop_static_vars(ds, time_dim)

  return ds


def open_mfdataset(
  paths: Sequence[str | pathlib.Path], time_dim: str = "time", chunks=None
) -> xr.Dataset:
  """
  Open multiple zipped Zarr datasets and combine them as xarray.open_mfdataset would, with a
  specific behavior for static variables (those without the provided time dimension):

  - Time-varying variables (containing `time_dim` among their dimensions) are merged along
    coordinates (typically along the time dimension) using xarray.open_mfdataset(combine='by_coords').
  - Static variables (that do not contain `time_dim`) are expected to be identical across the
    input datasets if duplicated; they are validated and included once, as-is, in the output.

  Parameters
  ---------
  paths: Sequence[str | pathlib.Path]
      List of paths to zipped Zarr stores (.zip). They must exist. The function does not support
      directories; pass individual .zip paths instead.
  time_dim: str
      Name of the time dimension. Variables that do not include this dimension are considered static.
  chunks: Any
      Chunking specification forwarded to xarray open calls. Use None to keep existing chunking.

  Returns
  -------
  xr.Dataset
      Dataset obtained by combining the time-varying variables by coordinates and adding the static
      variables (validated to be equal across inputs) unchanged.
  """
  if not isinstance(paths, (list, tuple)):
    raise TypeError("paths must be a sequence of path-like strings pointing to zipped Zarr stores")
  if len(paths) == 0:
    raise ValueError("paths cannot be empty")

  str_paths = [str(pathlib.Path(p)) for p in paths]
  for p in str_paths:
    if not pathlib.Path(p).exists():
      raise ValueError(f"Input path {p} does not exist")

  # Phase 1: scan inputs to collect and validate static variables (no `time_dim`).
  static_vars: dict[str, xr.DataArray] = {}

  def _collect_and_validate_static(ds: xr.Dataset):
    nonlocal static_vars
    for name, var in ds.data_vars.items():
      if time_dim not in var.dims:
        if name in static_vars:
          # Ensure equality (values and coordinates). Attributes are ignored.
          if not var.equals(static_vars[name]):  # ty: ignore
            raise ValueError(
              f"Static variable '{name}' differs across inputs. All static variables must be identical."
            )
        else:
          static_vars[name] = var  # ty: ignore

  # Open each dataset quickly to inspect static variables. Keep inline_array=False to avoid huge graphs.
  for p in str_paths:
    ds = xr.open_dataset(p, engine="zarr", inline_array=False, chunks=chunks)
    try:
      _collect_and_validate_static(ds)
    finally:
      ds.close()

  # Phase 2: combine time-varying variables by coordinates using open_mfdataset
  def _drop_static(ds: xr.Dataset) -> xr.Dataset:
    to_drop = [name for name, var in ds.data_vars.items() if time_dim not in var.dims]
    if to_drop:
      # Drop only those present to avoid errors if some files lack certain static vars
      ds = ds.drop_vars(to_drop, errors="ignore")
    return ds

  ds_dynamic = xr.open_mfdataset(
    str_paths,
    engine="zarr",
    combine="by_coords",
    preprocess=_drop_static,
    inline_array=False,
    chunks=chunks,
  )

  # Merge back the validated static variables (if any)
  if static_vars:
    static_ds = xr.Dataset({k: v for k, v in static_vars.items()})
    # xr.merge will align coordinates as needed; prefer dynamic attrs
    ds_dynamic = xr.merge([ds_dynamic, static_ds], combine_attrs="override")

  return ds_dynamic


def save_to_zarr(
  dataset: xr.Dataset,
  path: pathlib.Path,
  configs: dict[str, Any],
  progress: bool = False,
) -> None:
  logger.info(f"Saving dataset to {path} with {configs}")
  # No need to copy configs here before popping elements out of it, as it will not be reused
  if rechunk_conf := configs.pop("chunk", {}):
    dataset = dataset.chunk(**rechunk_conf)
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      if dataset[var].encoding and dataset[var].encoding.get("chunks"):
        del dataset[var].encoding["chunks"]
  if compressor_conf := configs.pop("compressor", {}):
    for var in dataset.data_vars:
      dataset[var].encoding["compressor"] = Blosc(**compressor_conf)
  else:
    for var in dataset.data_vars:
      dataset[var].encoding["compressor"] = None
  if path.suffix == ".zip":
    # Notice that parallel writes to Zarr using zip store are (apparently) not supported.
    store = ZipStore(path=str(path), mode="w", compression=0, allowZip64=True)
  else:
    store = DirectoryStore(path=str(path))
  if postprocess_conf := configs.pop("postprocess", None):
    postprocess = Process(steps=postprocess_conf)
    dataset = postprocess(dataset)
  with bar(progress):
    dataset.to_zarr(store=store, compute=True, mode="w", **configs)


def valid_datetime_index(idx: pd.DatetimeIndex) -> bool:
  # Check that the following implementation works for both DateTimeIndex and CFTimeIndex
  if idx.has_duplicates:
    dups = idx[idx.duplicated()]
    dup_values = pd.DatetimeIndex(dups.unique())
    preview = ", ".join(str(ts) for ts in dup_values[:5])
    more = "" if len(dup_values) <= 5 else f" and {len(dup_values) - 5} more"
    warnings.warn(f"Duplicate timestamps found in time coordinate: {preview}{more}")
    return False

  if len(idx) > 0:
    # noinspection PyTypeChecker
    expected = pd.date_range(start=idx[0], periods=len(idx), freq="D", tz=getattr(idx, "tz", None))
    if not idx.equals(expected):
      # Report missing or irregular timestamps for easier debugging
      # Compute missing by comparing against the sorted unique expected sequence
      sorted_idx = idx.sort_values()
      expected_full = pd.date_range(
        start=sorted_idx[0],
        end=sorted_idx[-1],
        freq="D",
        tz=getattr(sorted_idx, "tz", None),
      )
      missing = expected_full.difference(sorted_idx)
      preview = ", ".join(str(ts) for ts in missing[:5])
      more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
      warnings.warn(f"Missing or irregular dates detected: {preview}{more}")
      return False

  return True


def valid_cftime_index(idx: xr.CFTimeIndex) -> bool:
  # Equivalent checks for xarray.CFTimeIndex (cftime-based calendars)
  # Duplicates check
  if idx.has_duplicates:
    dups = idx[idx.duplicated()]
    dup_values = dups.unique()  # CFTimeIndex of unique duplicate timestamps
    preview = ", ".join(str(ts) for ts in dup_values[:5])
    more = "" if len(dup_values) <= 5 else f" and {len(dup_values) - 5} more"
    warnings.warn(f"Duplicate timestamps found in time coordinate: {preview}{more}")
    return False

  # Regularity check (daily frequency across CF calendars)
  if len(idx) > 0:
    calendar = getattr(idx, "calendar", None)
    # Build expected daily sequence with same calendar
    expected = xr.cftime_range(start=idx[0], periods=len(idx), freq="D", calendar=calendar)
    if not idx.equals(expected):
      # Compute missing days between min and max dates for helpful diagnostics
      sorted_idx = idx.sort_values()
      expected_full = xr.cftime_range(
        start=sorted_idx[0], end=sorted_idx[-1], freq="D", calendar=calendar
      )
      missing = expected_full.difference(sorted_idx)
      preview = ", ".join(str(ts) for ts in missing[:5])
      more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
      warnings.warn(f"Missing or irregular dates detected: {preview}{more}")
      return False

  return True


def valid_time_coordinate(dataset: xr.Dataset, time_dim: str = "time") -> bool:
  idx = dataset[time_dim].to_index()
  if isinstance(idx, pd.DatetimeIndex):
    passed = valid_datetime_index(idx)
  elif isinstance(idx, xr.CFTimeIndex):
    passed = valid_cftime_index(idx)
  else:
    raise ValueError(f"Unexpected index type: {type(idx).__name__}")
  return passed
