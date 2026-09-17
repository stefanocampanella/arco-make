# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import logging
import pathlib
import tempfile
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal

import xarray as xr
from dask.delayed import Delayed
from numcodecs import Blosc
from pydantic import BaseModel, ConfigDict, Field
from xarray.backends.common import BackendEntrypoint
from xarray.coders import CFDatetimeCoder, CFTimedeltaCoder
from zarr.storage import DirectoryStore, ZipStore

from arcomake.datetime_utils import (
  IterableDateInterval,
  may_parse_timedelta,
)
from arcomake.processing import ProcessingStepConfig, process

logger = logging.getLogger(__name__)


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

  # @model_validator(mode="after")
  # def validate_chunks(self) -> Self:
  #   if (
  #     self.chunks is not None
  #     and isinstance(self.chunks, dict)
  #     and not all(isinstance(value, int) or value == "auto" for value in self.chunks.values())
  #   ):
  #     raise ValueError("Chunk option value must be a dictionary with integer or 'auto' values")
  #   return self


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


def open_dataset(
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
  dataset: xr.Dataset = xr.merge(parts, join="exact")
  if configs.postprocess:
    dataset = process(dataset=dataset, steps=configs.postprocess)
  return dataset


def maybe_checkpointing_open_dataset(
  configs: DatasetConfig,
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
  time_dim: str = "time",
) -> xr.Dataset:
  checkpointing_conf = configs.checkpointing
  checkpointing_step = checkpointing_conf.step if checkpointing_conf else None
  compressor_conf = checkpointing_conf.compressor if checkpointing_conf else None
  compressor = (
    Blosc(**compressor_conf.model_dump(exclude_none=True)) if compressor_conf is not None else None
  )

  if checkpointing_step is None:
    return open_dataset(configs, start_datetime, end_datetime)
  parsed_step = may_parse_timedelta(checkpointing_step)
  if parsed_step >= end_datetime - start_datetime:
    return open_dataset(configs, start_datetime, end_datetime)

  checkpoint = tempfile.TemporaryDirectory(suffix=".zarr", delete=False)
  logger.info(f"Checkpointing to {checkpoint.name} every {parsed_step}")
  checkpoint_store = DirectoryStore(checkpoint.name)
  date_intervals = IterableDateInterval(start_datetime, end_datetime, parsed_step)
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
  dataset = xr.open_zarr(store=checkpoint_store, overwrite_encoded_chunks=True)
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


def open_archive(
  path: str | pathlib.Path, time_dim: str = "time", attrs_to_drop: list[str] | None = None, **kwargs
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
  path = pathlib.Path(path)
  zip_file_paths = [str(file_path) for file_path in sorted(path.glob("*.zip"))]
  if len(zip_file_paths) == 0:
    raise ValueError("Provided path does not contain any .zip files.")
  logger.info(f"Reading {len(zip_file_paths)} .zip datasets from {path}")

  # Drop the possibly conflicting attributes so that combining with
  # combine_attrs="no_conflicts" does not fail when it differs across inputs.
  attrs_to_drop: list[str] = [] if attrs_to_drop is None else attrs_to_drop

  # Open each dataset quickly to inspect static variables. Keep inline_array=False to avoid huge graphs.
  static_vars: dict[str, xr.DataArray] = {}
  for file_path in zip_file_paths:
    with xr.open_dataset(file_path, engine="zarr", inline_array=False, **kwargs) as ds:
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
            static_vars[name] = var  # type: ignore

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
  configs: SaveConfig | None = None,
  compute: bool = True,
) -> xr.backends.ZarrStore | Delayed:
  if configs is None:
    configs = SaveConfig()

  logger.info(f"Saving dataset to {path} with {configs}")
  rechunk_conf = configs.chunks
  compressor = (
    Blosc(**configs.compressor.model_dump(exclude_none=True))
    if configs.compressor is not None
    else None
  )
  to_zarr_kwargs = configs.model_dump(exclude_none=True, exclude={"chunk", "compressor"})

  if rechunk_conf:
    # Set the on-disk Zarr chunk layout via encoding, without altering the
    # underlying Dask chunking. This requires the existing Dask chunks to be
    # an integer multiple of (and evenly divide into) the requested chunks
    # along each dimension; otherwise to_zarr will raise a ValueError.
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      dims = dataset[var].dims
      if isinstance(rechunk_conf, dict):
        chunk_sizes = tuple(
          rechunk_conf[dim] if dim in rechunk_conf else dataset[var].sizes[dim] for dim in dims
        )
      elif isinstance(rechunk_conf, tuple):
        chunk_sizes = rechunk_conf
      else:
        chunk_sizes = tuple(rechunk_conf for _ in dims)
      dataset[var].encoding["chunks"] = chunk_sizes
  for var in dataset.data_vars:
    dataset[var].encoding["compressor"] = compressor

  if "encoding" in to_zarr_kwargs and isinstance(to_zarr_kwargs["encoding"], dict):
    for var_enc in to_zarr_kwargs["encoding"].values():
      if isinstance(var_enc, dict) and isinstance(var_enc.get("compressor"), dict):
        var_enc["compressor"] = Blosc(**var_enc["compressor"])

  if path.suffix == ".zip":
    # Notice that parallel writes to Zarr using zip store are (apparently) not supported.
    store = ZipStore(path=str(path), mode="w", compression=0, allowZip64=True)
  else:
    store = DirectoryStore(path=str(path))
  xarray_zarr_store = dataset.to_zarr(store=store, compute=compute, mode="w", **to_zarr_kwargs)
  xarray_zarr_store._close_store_on_close = True
  return xarray_zarr_store
