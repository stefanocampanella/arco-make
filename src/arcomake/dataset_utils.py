# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import contextlib
import contextvars
import datetime
import logging
import pathlib
import tempfile
import threading
from collections.abc import Hashable, Iterable, Mapping
from contextlib import ExitStack, nullcontext
from typing import Any, Literal, Self

import fsspec
import xarray as xr
import zarr.storage
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
from pydantic import BaseModel, ConfigDict, Field
from xarray.backends.common import BackendEntrypoint
from xarray.coders import CFDatetimeCoder, CFTimedeltaCoder
from zarr.storage import DirectoryStore, ZipStore

from arcomake.datetime_utils import (
  IterableDateInterval,
  may_parse_timedelta,
)
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


_CURRENT_TEMP_REGISTRY: contextvars.ContextVar["TempDirectoryRegistry | None"] = (
  contextvars.ContextVar("current_temp_registry", default=None)
)


def get_current_temp_registry() -> "TempDirectoryRegistry | None":
  """Get the active temporary directory registry in the current context, if any."""
  return _CURRENT_TEMP_REGISTRY.get()


class TempDirectoryRegistry:
  """Thread-safe and context-scoped manager for temporary directories and resources.

  Ensures temporary directories remain intact while lazy computations (like Dask)
  stream data from disk, and are deterministically deleted upon exiting the context.
  """

  def __init__(self):
    self._stack = contextlib.ExitStack()
    self._lock = threading.Lock()
    self._token: contextvars.Token[TempDirectoryRegistry | None] | None = None

  def create_temp_dir(
    self,
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | pathlib.Path | None = None,
    **kwargs,
  ) -> pathlib.Path:
    """Create a temporary directory tracked by this registry."""
    with self._lock:
      tmpdir = self._stack.enter_context(
        tempfile.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=dir, **kwargs)
      )
      return pathlib.Path(tmpdir)

  def register[T: contextlib.AbstractContextManager[Any]](self, context_or_cleanup: T) -> T:
    """Register an existing context manager (e.g. TemporaryDirectory) for cleanup."""
    with self._lock:
      return self._stack.enter_context(context_or_cleanup)

  def close(self):
    """Clean up all registered temporary directories."""
    with self._lock:
      self._stack.close()

  def __enter__(self) -> Self:
    self._token = _CURRENT_TEMP_REGISTRY.set(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    try:
      self.close()
    finally:
      if self._token is not None:
        _CURRENT_TEMP_REGISTRY.reset(self._token)
        self._token = None


EngineType = (
  Literal["netcdf4", "scipy", "pydap", "h5netcdf", "zarr"] | type[BackendEntrypoint] | str
)


ChunksConfig = (
  int | str | dict[Hashable, int | Literal["auto"] | tuple[int, ...] | None] | tuple[int, ...]
)


class ReadConfig(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

  engine: EngineType | None = None
  chunks: ChunksConfig | None = None
  cache: bool | None = None
  decode_cf: bool | None = None
  mask_and_scale: bool | Mapping[str, bool] | None = None
  decode_times: bool | CFDatetimeCoder | Mapping[str, bool | CFDatetimeCoder] | None = None
  decode_timedelta: bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None = None
  use_cftime: bool | Mapping[str, bool] | None = None
  concat_characters: bool | Mapping[str, bool] | None = None
  decode_coords: Literal["coordinates", "all"] | bool | None = None
  drop_variables: str | Iterable[str] | None = None
  create_default_indexes: bool = True
  inline_array: bool = False
  chunked_array_type: str | None = None
  from_array_kwargs: dict[str, Any] | None = None
  backend_kwargs: dict[str, Any] | None = None


class CompressorConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  cname: Literal["lz4", "lz4hc", "zstd", "zlib", "snappy"]
  clevel: int = Field(ge=0, le=9)
  shuffle: Literal[0, 1, 2] | None = None
  blocksize: int | None = Field(default=None, ge=0)


class CopernicusMarinePartConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["copernicusmarine"] | None = "copernicusmarine"
  variables: str | list[str] | None = None
  drop_variables: str | list[str] | None = None
  dataset_version: str | None = None
  dataset_part: str | None = None
  service: str | None = None
  start_datetime: datetime.datetime | str | None = None
  end_datetime: datetime.datetime | str | None = None
  chunks: ChunksConfig | None = None
  storage_options: dict[str, Any] | None = None


class NetCDFOverHTTPPartConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["netcdfoverhttp"] | None = "netcdfoverhttp"
  drop_variables: str | list[str] | None = None
  chunks: ChunksConfig | None = None


class EarlyWarningDataStorePartConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["earlywarningdatastore"] | None = "earlywarningdatastore"
  system_version: str | None = None
  hydrological_model: str | None = None
  product_type: str | None = None
  timespan: str | None = None
  variable: str | list[str] | None = None
  variables: str | list[str] | None = None
  drop_variables: str | list[str] | None = None
  start_datetime: datetime.datetime | str | None = None
  end_datetime: datetime.datetime | str | None = None
  time_dim: str | None = None
  latitude_dim: str | None = None
  longitude_dim: str | None = None
  chunks: ChunksConfig | None = None


class ZarrPartConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["zarr"] | None = "zarr"
  consolidated: bool | None = None
  storage_options: dict[str, Any] | None = None
  chunks: ChunksConfig | None = None
  drop_variables: str | list[str] | None = None


class GenericPartConfig(BaseModel):
  model_config = ConfigDict(extra="allow")

  filename_or_obj: str
  engine: str | None = None
  chunks: ChunksConfig | None = None
  drop_variables: str | list[str] | None = None


DatasetPartConfig = (
  CopernicusMarinePartConfig
  | NetCDFOverHTTPPartConfig
  | EarlyWarningDataStorePartConfig
  | ZarrPartConfig
  | GenericPartConfig
)


class CheckpointingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  step: str | None = None
  compressor: CompressorConfig | None = None


class DatasetConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  skip: bool = False
  checkpointing: CheckpointingConfig | None = None
  parts: list[DatasetPartConfig] = Field(default_factory=list)
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)


class VariableEncodingConfig(BaseModel):
  model_config = ConfigDict(extra="allow")

  units: str | None = None
  calendar: str | None = None
  dtype: str | None = None
  chunks: list[int] | None = None
  compressor: CompressorConfig | None = None
  fill_value: float | int | str | bool | None = None


class SaveConfig(BaseModel):
  model_config = ConfigDict(extra="allow")

  consolidated: bool | None = None
  chunks: ChunksConfig | None = None
  compressor: CompressorConfig | None = None
  encoding: dict[str, VariableEncodingConfig] | None = None
  storage_options: dict[str, Any] | None = None


def download_and_process(
  configs: DatasetConfig,
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
  for raw_part in configs.parts:
    part_conf = raw_part.model_dump(exclude_unset=True)
    engine = getattr(raw_part, "engine", None) or part_conf.get("engine")
    if engine == "earlywarningdatastore" or engine == "copernicusmarine":
      part_conf.update(start_datetime=start_datetime, end_datetime=end_datetime)
    part = xr.open_dataset(**part_conf)
    part = part.arcomake.time_sel(start_datetime, end_datetime)
    parts.append(part)
  if not parts:
    raise ValueError("No parts found in the dataset configuration")
  dataset = xr.merge(parts, join="exact")
  assert isinstance(dataset, xr.Dataset)
  if configs.postprocess:
    dataset = dataset.arcomake.process(steps=configs.postprocess)
    assert isinstance(dataset, xr.Dataset)
  return dataset


def maybe_checkpointing_download_and_process(
  configs: DatasetConfig,
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
  time_dim: str = "time",
  progress: bool = True,
) -> xr.Dataset:
  checkpointing_conf = configs.checkpointing
  checkpointing_step = checkpointing_conf.step if checkpointing_conf else None
  compressor_conf = checkpointing_conf.compressor if checkpointing_conf else None
  compressor = (
    Blosc(**compressor_conf.model_dump(exclude_none=True)) if compressor_conf is not None else None
  )

  if checkpointing_step is None:
    return download_and_process(configs, start_datetime, end_datetime)
  parsed_step = may_parse_timedelta(checkpointing_step)
  if parsed_step >= end_datetime - start_datetime:
    return download_and_process(configs, start_datetime, end_datetime)

  registry = get_current_temp_registry()
  if registry is not None:
    checkpoint_path = registry.create_temp_dir(suffix=".zarr")
    checkpoint = None
  else:
    checkpoint = tempfile.TemporaryDirectory(suffix=".zarr")
    checkpoint_path = pathlib.Path(checkpoint.name)

  logger.info(f"Checkpointing to {checkpoint_path} every {parsed_step}")
  checkpoint_store = DirectoryStore(str(checkpoint_path))
  date_intervals = IterableDateInterval(start_datetime, end_datetime, parsed_step)
  is_first_checkpoint = True
  for date_interval in date_intervals:
    with download_and_process(configs, date_interval.start, date_interval.end) as dataset:
      for var in dataset.data_vars:
        dataset[var].encoding["compressor"] = compressor
      logger.info(f"Saving checkpoint {date_interval}")
      progress_bar = nullcontext if not progress else ProgressBar
      with progress_bar():
        if is_first_checkpoint:
          dataset.to_zarr(store=checkpoint_store, mode="w", compute=True)
          is_first_checkpoint = False
        else:
          dataset.to_zarr(store=checkpoint_store, mode="a-", append_dim=time_dim, compute=True)
    del dataset

  # Open the checkpointed dataset
  logger.info(f"Opening checkpointed dataset from {checkpoint_path}")
  dataset = xr.open_zarr(store=checkpoint_store, overwrite_encoded_chunks=True)
  assert isinstance(dataset, xr.Dataset)
  if checkpoint is not None:
    dataset.set_close(checkpoint.cleanup)
  return dataset


def open_archive(
  path: str, time_dim: str = "time", attrs_to_drop: list[str] | None = None, **kwargs
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
  fs, fs_path = fsspec.url_to_fs(path)
  zip_file_paths = [str(file_path) for file_path in sorted(fs.glob(fs_path + "/*.zip"))]
  if len(zip_file_paths) == 0:
    raise ValueError("Provided path does not contain any .zip files.")
  logger.info(f"Reading {len(zip_file_paths)} .zip datasets from {path}")

  # Drop the possibly conflicting attributes so that combining with
  # combine_attrs="no_conflicts" does not fail when it differs across inputs.
  attrs_to_drop: list[str] = [] if attrs_to_drop is None else attrs_to_drop

  # Open each dataset quickly to inspect static variables. Keep inline_array=False to avoid huge graphs.
  static_vars: dict[str, xr.DataArray] = {}
  for file_path in zip_file_paths:
    with xr.open_dataset(file_path, **kwargs) as ds:
      for name, var in ds.data_vars.items():
        for key in attrs_to_drop:
          var.attrs.pop(key, None)
        if time_dim not in var.dims:
          if name in static_vars:
            try:
              xr.testing.assert_identical(var, static_vars[name])  # type: ignore
            except AssertionError as exc:
              raise ValueError(
                f"Static variable '{name}' differs across inputs. All static variables must be identical."
              ) from exc
          else:
            static_vars[name] = var.copy(deep=True)  # type: ignore

  # Combine time-varying variables by coordinates using open_mfdataset
  def _drop_static(ds: xr.Dataset) -> xr.Dataset:
    to_drop = [name for name, var in ds.data_vars.items() if time_dim not in var.dims]
    if to_drop:
      # Drop only those present to avoid errors if some files lack certain static vars
      ds = ds.drop_vars(to_drop)
    # Drop the possibly conflicting 'last_updated' attribute so that combining with
    # combine_attrs="no_conflicts" does not fail when it differs across inputs.
    for key in attrs_to_drop:
      ds.attrs.pop(key, None)
    return ds

  ds_dynamic = xr.open_mfdataset(
    zip_file_paths,
    combine="by_coords",
    combine_attrs="no_conflicts",
    preprocess=_drop_static,
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


def safe_to_zarr(
  dataset: xr.Dataset,
  destination: str | pathlib.Path | zarr.storage.BaseStore | fsspec.mapping.FSMap,
  configs: SaveConfig | None = None,
  compute: bool = True,
  *,
  store_stack: ExitStack | None = None,
) -> xr.backends.ZarrStore | Delayed:
  """Write a dataset, explicitly managing the underlying store's lifetime.

  Eager writes close their store before returning, including on failure. The
  returned Xarray backend is already closed and needs no further cleanup.
  For ``compute=False``, the caller must supply an active ``ExitStack`` and
  finish computing the returned task before leaving it. The stack closes the
  original store even if graph construction or computation fails, or the task
  is never computed. Calling ``close`` on a Delayed object is not cleanup.

  Zip stores must not be written from multiple processes; use directory stores
  for distributed writes and archive the completed output separately.
  """
  if not compute and store_stack is None:
    raise ValueError("Delayed writes require a store_stack kept open through computation")
  if configs is None:
    configs = SaveConfig()

  logger.info(f"Saving dataset to {destination} with {configs}")
  chunk_conf = configs.chunks
  compressor = (
    Blosc(**configs.compressor.model_dump(exclude_none=True))
    if configs.compressor is not None
    else None
  )
  to_zarr_kwargs = configs.model_dump(
    exclude_none=True, exclude={"chunks", "compressor", "storage_options"}
  )

  if chunk_conf:
    # Set the on-disk Zarr chunk layout via encoding, without altering the
    # underlying Dask chunking. This requires the existing Dask chunks to be
    # an integer multiple of (and evenly divide into) the requested chunks
    # along each dimension; otherwise to_zarr will raise a ValueError.
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      dims = dataset[var].dims
      if isinstance(chunk_conf, dict):
        chunk_sizes = tuple(
          chunk_conf[dim] if dim in chunk_conf else dataset[var].sizes[dim] for dim in dims
        )
      elif isinstance(chunk_conf, tuple):
        chunk_sizes = chunk_conf
      else:
        chunk_sizes = tuple(chunk_conf for _ in dims)
      dataset[var].encoding["chunks"] = chunk_sizes
  for var in dataset.data_vars:
    dataset[var].encoding["compressor"] = compressor

  if "encoding" in to_zarr_kwargs and isinstance(to_zarr_kwargs["encoding"], dict):
    for var_enc in to_zarr_kwargs["encoding"].values():
      if isinstance(var_enc, dict) and isinstance(var_enc.get("compressor"), dict):
        var_enc["compressor"] = Blosc(**var_enc["compressor"])

  if isinstance(destination, (zarr.storage.BaseStore, fsspec.mapping.FSMap)):
    store = destination
  elif isinstance(destination, str) and "://" in destination:
    storage_options = getattr(configs, "storage_options", None) or {}
    store = fsspec.get_mapper(destination, **storage_options)
  else:
    path_obj = pathlib.Path(destination)
    if path_obj.suffix == ".zip":
      # Notice that parallel writes to Zarr using zip store are (apparently) not supported.
      store = ZipStore(path=str(path_obj), mode="w", compression=0, allowZip64=True)
    else:
      store = DirectoryStore(path=str(path_obj))

  close_fn = getattr(store, "close", None)
  if not compute:
    assert store_stack is not None
    if close_fn is not None:
      store_stack.callback(close_fn)
    return dataset.to_zarr(store=store, compute=False, mode="w", **to_zarr_kwargs)
  try:
    return dataset.to_zarr(store=store, compute=True, mode="w", **to_zarr_kwargs)
  finally:
    if close_fn is not None:
      close_fn()
