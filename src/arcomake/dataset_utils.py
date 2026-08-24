# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import logging
import pathlib
import tempfile
from typing import Any

import xarray as xr
from dask.delayed import Delayed
from numcodecs import Blosc
from zarr.storage import DirectoryStore, ZipStore

from arcomake.datetime_utils import (
  IterableDateInterval,
  may_parse_timedelta,
)
from arcomake.processing import process

logger = logging.getLogger(__name__)


def open_dataset(
  configs: dict[str, Any],
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
) -> xr.Dataset:
  """
  Downloads and pre-processes a dataset based on provided configurations.
  """
  # When downloading from Copernicus Marine Data Store or Climate Data Store, the typical case is a large dataset,
  # spanning a long time period, with several sets of variables in different datasets (bio, phys, etc.),
  # which needs to be downloaded one piece at a time. Hence, `parts` list in the TOML configuration file
  # represents different pieces of the same dataset.
  parts = []
  for part_conf in configs.get("parts", []):
    engine = part_conf.get("engine")
    if engine == "earlywarningdatastore" or engine == "copernicusmarine":
      part_conf.update(start_datetime=start_datetime, end_datetime=end_datetime)
    part = xr.open_dataset(**part_conf)
    part = part.arcomake.time_sel(start_datetime, end_datetime)
    parts.append(part)
  dataset: xr.Dataset = xr.merge(parts, join="exact")
  dataset = process(dataset=dataset, steps=configs.get("postprocess", []))
  return dataset


def maybe_checkpointing_open_dataset(
  configs: dict[str, Any],
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
  time_dim: str = "time",
) -> xr.Dataset:
  checkpointing_conf = configs.get("checkpointing", {})
  checkpointing_step = checkpointing_conf.get("step")
  if checkpointing_step is None:
    return open_dataset(configs, start_datetime, end_datetime)
  checkpointing_step = may_parse_timedelta(checkpointing_step)
  if checkpointing_step >= end_datetime - start_datetime:
    return open_dataset(configs, start_datetime, end_datetime)

  compressor_conf = checkpointing_conf.get("compressor")
  compressor = None if compressor_conf is None else Blosc(**compressor_conf)

  checkpoint = tempfile.TemporaryDirectory(suffix=".zarr", delete=False)
  logger.info(f"Checkpointing to {checkpoint.name} every {checkpointing_step}")
  checkpoint_store = DirectoryStore(checkpoint.name)
  date_intervals = IterableDateInterval(start_datetime, end_datetime, checkpointing_step)
  is_first_checkpoint = True
  for date_interval in date_intervals:
    with open_dataset(configs, date_interval.start, date_interval.end) as dataset:
      for var in dataset.data_vars:
        dataset[var].encoding["compressor"] = compressor
      logger.info(f"Saving checkpoint {date_interval}")
      if is_first_checkpoint:
        dataset.to_zarr(store=checkpoint_store, mode="w", compute=True)
        is_first_checkpoint = False
      else:
        dataset.to_zarr(store=checkpoint_store, mode="a-", append_dim=time_dim, compute=True)
    del dataset

  # Open the checkpointed dataset, set the close function to remove the temporary directory when done
  logger.info(f"Opening checkpointed dataset from {checkpoint.name}")
  dataset = xr.open_zarr(
    store=checkpoint_store, overwrite_encoded_chunks=True, chunks=configs.get("chunks")
  )
  dataset.set_close(checkpoint.cleanup)
  return dataset


# FIXME: the code should handle both Zarr (using a DirectoryStore or a ZipStore) and NetCDF files.
def open_dataset_wo_static(
  path: str | pathlib.Path, time_dim: str = "time", chunks=None
) -> xr.Dataset:
  """
  Open a dataset from a single Zarr file/store or a directory containing multiple Zarr zip files.

  - If `path` is a directory with one or more .zip files, open all of them via xarray.open_mfdataset(engine='zarr').
  - In all other cases, open it via xarray.open_dataset(engine='zarr').

  Returns a xarray.Dataset filtered to only data variables that include the provided time dimension.
  """
  path = pathlib.Path(path)

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


def open_archive(path: str | pathlib.Path, time_dim: str = "time", **kwargs) -> xr.Dataset:
  """
  Open multiple zipped Zarr datasets and combine them as xarray.open_mfdataset would, with a
  specific behavior for static variables (those without the provided time dimension):

  - Time-varying variables (containing `time_dim` among their dimensions) are merged along
    coordinates (typically along the time dimension) using xarray.open_mfdataset(combine='by_coords').
  - Static variables (that do not contain `time_dim`) are expected to be identical across the
    input datasets if duplicated; they are validated and included once, as-is, in the output.

  Parameters
  ---------
  path: str | pathlib.Path
      Directory of .zip Zarr datasets.
  time_dim: str
      Name of the time dimension. Variables that do not include this dimension are considered static.

  Returns
  -------
  xr.Dataset
      Dataset obtained by combining the time-varying variables by coordinates and adding the static
      variables (validated to be equal across inputs) unchanged.
  """
  path = pathlib.Path(path)
  zip_file_paths = [str(file_path) for file_path in sorted(path.glob("*.zip"))]
  if len(zip_file_paths) == 0:
    raise ValueError("Provided path does not contain any .zip files.")
  logger.info(f"Reading {len(zip_file_paths)} .zip datasets from {path}")

  # Open each dataset quickly to inspect static variables. Keep inline_array=False to avoid huge graphs.
  static_vars: dict[str, xr.DataArray] = {}
  for file_path in zip_file_paths:
    with xr.open_dataset(file_path, engine="zarr", inline_array=False, **kwargs) as ds:
      for name, var in ds.data_vars.items():
        # 'last_updated' attribute can differ across inputs, so remove it
        var.attrs.pop('last_updated', None)
        if time_dim not in var.dims:
          if name in static_vars:
            try:
              xr.testing.assert_identical(var, static_vars[name])  # type: ignore
            except AssertionError as exc:
              raise ValueError(
                f"Static variable '{name}' differs across inputs. All static variables must be identical."
              ) from exc
          else:
            static_vars[name] = var  # type: ignore

  # Combine time-varying variables by coordinates using open_mfdataset
  def _drop_static(ds: xr.Dataset) -> xr.Dataset:
    to_drop = [name for name, var in ds.data_vars.items() if time_dim not in var.dims]
    if to_drop:
      # Drop only those present to avoid errors if some files lack certain static vars
      ds = ds.drop_vars(to_drop)
    # Drop the possibly conflicting 'last_updated' attribute so that combining with
    # combine_attrs="no_conflicts" does not fail when it differs across inputs.
    ds.attrs.pop("last_updated", None)
    return ds

  ds_dynamic = xr.open_mfdataset(
    zip_file_paths,
    engine="zarr",
    combine="by_coords",
    combine_attrs="no_conflicts",
    preprocess=_drop_static,
    inline_array=False,
    **kwargs,
  )

  # Merge back the validated static variables (if any)
  if static_vars:
    ds_static = xr.Dataset({k: v for k, v in static_vars.items()})
    # xr.merge will align coordinates as needed; prefer dynamic attrs
    ds_dynamic: xr.Dataset = xr.merge(
      [ds_dynamic, ds_static], compat="no_conflicts", combine_attrs="no_conflicts"
    )

  return ds_dynamic


def save_to_zarr(
  dataset: xr.Dataset,
  path: pathlib.Path,
  configs: dict[str, Any],
  compute=True,
) -> xr.backends.ZarrStore | Delayed:
  logger.info(f"Saving dataset to {path} with {configs}")
  # Copy configs before popping elements out of it as, for example, so that save_to_zarr can be called multiple times.
  configs = configs.copy()
  if rechunk_conf := configs.pop("chunk", {}):
    # Set the on-disk Zarr chunk layout via encoding, without altering the
    # underlying Dask chunking. This requires the existing Dask chunks to be
    # an integer multiple of (and evenly divide into) the requested chunks
    # along each dimension; otherwise to_zarr will raise a ValueError.
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      dims = dataset[var].dims
      chunk_sizes = tuple(
        rechunk_conf[dim] if dim in rechunk_conf else dataset[var].sizes[dim] for dim in dims
      )
      dataset[var].encoding["chunks"] = chunk_sizes
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
  xarray_zarr_store = dataset.to_zarr(store=store, compute=compute, mode="w", **configs)
  xarray_zarr_store._close_store_on_close = True
  return xarray_zarr_store
