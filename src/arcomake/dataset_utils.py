# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import logging
import pathlib
import tempfile
from collections.abc import Callable, Hashable, Mapping
from contextlib import nullcontext
from typing import Annotated, Any, Literal

import fsspec
import xarray as xr
import zarr.errors
import zarr.storage
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from xarray.backends.common import BackendEntrypoint
from xarray.backends.plugins import guess_engine
from xarray.coders import CFDatetimeCoder, CFTimedeltaCoder
from zarr.storage import DirectoryStore

from arcomake.datetime_utils import (
  IterableDateInterval,
  may_parse_timedelta,
)
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.temporary_utils import get_current_temp_registry

logger = logging.getLogger(__name__)


EngineType = (
  Literal["netcdf4", "scipy", "pydap", "h5netcdf", "zarr"] | type[BackendEntrypoint] | str
)


ChunksConfig = (
  int | str | dict[Hashable, int | Literal["auto"] | tuple[int, ...] | None] | tuple[int, ...]
)


class OpenDatasetArguments(BaseModel):
  """Arguments of `xarray.open_dataset`, except `engine`."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

  chunks: ChunksConfig | None = None
  cache: bool | None = None
  decode_cf: bool | None = None
  mask_and_scale: bool | Mapping[str, bool] | None = None
  decode_times: bool | CFDatetimeCoder | Mapping[str, bool | CFDatetimeCoder] | None = None
  decode_timedelta: bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None = None
  use_cftime: bool | Mapping[str, bool] | None = None
  concat_characters: bool | Mapping[str, bool] | None = None
  decode_coords: Literal["coordinates", "all"] | bool | None = None
  drop_variables: str | list[str] | None = None
  create_default_indexes: bool = True
  inline_array: bool = False
  chunked_array_type: str | None = None
  from_array_kwargs: dict[str, Any] | None = None
  backend_kwargs: dict[str, Any] | None = None
  # Options of the fsspec filesystem of a remote input (credentials, `requests_timeout`...).
  storage_options: dict[str, Any] | None = None
  # Zarr inputs: whether to read consolidated metadata.
  consolidated: bool | None = None


class ReadConfig(OpenDatasetArguments):
  """Arguments of `xarray.open_dataset` (the `[read]` table of a recipe)."""

  engine: EngineType | None = None


class CompressorConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  cname: Literal["lz4", "lz4hc", "zstd", "zlib", "snappy"]
  clevel: int = Field(ge=0, le=9)
  shuffle: Literal[0, 1, 2] | None = None
  blocksize: int | None = Field(default=None, ge=0)


def _engine_given_or_implied(data: Any, implied: Callable[[str], bool]) -> Any:
  """A part without `engine` matches a dedicated model only if its URL implies that engine.

  Otherwise it falls through to `GenericPartConfig`, and xarray guesses the engine.
  """
  if isinstance(data, Mapping) and data.get("engine") is None:
    url = data.get("filename_or_obj")
    if not (isinstance(url, str) and implied(url)):
      raise ValueError("no `engine` given, and the URL does not imply this one")
  return data


def _has_scheme(url: str, *schemes: str) -> bool:
  return url.split("://", 1)[0].lower() in schemes if "://" in url else False


class CopernicusMarinePartConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["copernicusmarine"] | None = "copernicusmarine"

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(data, lambda url: _has_scheme(url, "cm"))

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

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(
      data, lambda url: _has_scheme(url, "http", "https") and not url.endswith(".zarr")
    )


class EarlyWarningDataStorePartConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["earlywarningdatastore"] | None = "earlywarningdatastore"
  system_version: str | None = None
  hydrological_model: str | None = None
  product_type: str | None = None
  timespan: str | None = None
  variable: str | list[str] | None = None
  drop_variables: str | list[str] | None = None
  start_datetime: datetime.datetime | str | None = None
  end_datetime: datetime.datetime | str | None = None
  time_dim: str | None = None
  latitude_dim: str | None = None
  longitude_dim: str | None = None
  chunks: ChunksConfig | None = None

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(data, lambda url: _has_scheme(url, "ewds"))


class ZarrPartConfig(OpenDatasetArguments):
  """A Zarr part: the arguments of `xarray.open_dataset`, read through a strict store."""

  filename_or_obj: str
  engine: Literal["zarr"] | None = "zarr"

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(data, lambda url: url.rstrip("/").endswith(".zarr"))


class GenericPartConfig(ReadConfig):
  """A part read by any other xarray engine, with the arguments of `xarray.open_dataset`."""

  filename_or_obj: str

  @field_validator("engine")
  @classmethod
  def _not_a_dedicated_engine(cls, value: EngineType | None) -> EngineType | None:
    # Engines with a dedicated model must match it: a typo there is not a generic part.
    if value in DEDICATED_ENGINES:
      raise ValueError(f"invalid options for engine {value!r}")
    return value


DEDICATED_ENGINES = frozenset(
  {"copernicusmarine", "netcdfoverhttp", "earlywarningdatastore", "zarr"}
)


DatasetPartConfig = (
  CopernicusMarinePartConfig
  | NetCDFOverHTTPPartConfig
  | EarlyWarningDataStorePartConfig
  | ZarrPartConfig
  | GenericPartConfig
)


class VariableEncodingConfig(BaseModel):
  """One variable's encoding in `[save.encoding]`, as `xarray.Dataset.to_zarr` takes it.

  The fill value is `_FillValue` (`fill_value` is accepted too); `scale_factor` and `add_offset`
  pack floats into integers. `chunks` are explicit sizes, one per dimension.
  """

  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  units: str | None = None
  calendar: str | None = None
  dtype: str | None = None
  chunks: list[Annotated[int, Field(ge=1)]] | None = None
  compressor: CompressorConfig | None = None
  fill_value: float | int | str | bool | None = Field(
    default=None,
    validation_alias=AliasChoices("_FillValue", "fill_value"),
    serialization_alias="_FillValue",
  )
  scale_factor: float | None = None
  add_offset: float | None = None


StoredChunk = Annotated[int, Field(ge=-1)]
# The layout on disk: one size per dimension (-1: the whole dimension), or one size for all.
StoredChunksConfig = StoredChunk | dict[str, StoredChunk] | tuple[StoredChunk, ...]


class SaveConfig(BaseModel):
  """The `[save]` table: layout, compression and encodings of the stored arrays."""

  model_config = ConfigDict(extra="forbid")

  consolidated: bool | None = None
  chunks: StoredChunksConfig | None = None
  compressor: CompressorConfig | None = None
  encoding: dict[str, VariableEncodingConfig] | None = None
  storage_options: dict[str, Any] | None = None
  group: str | None = None

  @field_validator("chunks")
  @classmethod
  def _no_zero_chunks(cls, value: StoredChunksConfig | None) -> StoredChunksConfig | None:
    sizes = (
      value.values() if isinstance(value, dict) else value if isinstance(value, tuple) else [value]
    )
    if any(size == 0 for size in sizes or []):
      raise ValueError("chunk sizes are positive, or -1 for a whole dimension")
    return value


class CheckpointingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  step: datetime.timedelta
  compressor: CompressorConfig | None = None

  @field_validator("step", mode="before")
  @classmethod
  def parse_step(cls, value: str | datetime.timedelta) -> datetime.timedelta:
    return may_parse_timedelta(value)


def strict_zarr_store(
  path: str,
  *,
  fs: fsspec.AbstractFileSystem | None = None,
  storage_options: Mapping[str, Any] | None = None,
) -> zarr.storage.FSStore:
  """A read-only fsspec Zarr store on which only a missing object is a missing chunk.

  zarr 2 reads a chunk whose object does not exist as the fill value (that is how sparse stores
  work). Its `FSStore` does the same, by default, for `PermissionError` and any `OSError`, which
  include timeouts and connection errors: a failed read would become NaN with no message, and a
  later mean would skip it. This store raises those errors instead.

  Args:
    path: URL of the store, or its path on `fs`.
    fs: The filesystem of `path`; by default, the one its URL names.
    storage_options: Options of that filesystem (when `fs` is not given).
  """
  return zarr.storage.FSStore(
    path, mode="r", fs=fs, exceptions=(KeyError,), **dict(storage_options or {})
  )


def reads_as_zarr(filename_or_obj: str, engine: str | None) -> bool:
  """Whether `xarray.open_dataset(filename_or_obj, engine=engine)` uses the zarr engine.

  Without an engine, xarray guesses one from the name (e.g. a `.zarr` suffix).
  """
  if engine is not None:
    return engine == "zarr"
  try:
    return guess_engine(filename_or_obj) == "zarr"
  except ValueError:
    return False


def open_input(filename_or_obj: Any, **kwargs: Any) -> xr.Dataset:
  """`xarray.open_dataset`, raising read errors of Zarr stores given as fsspec URLs.

  A URL (`gs://`, `s3://`, `https://`, chained `::` URLs...) read with the zarr engine, given or
  guessed by xarray, is opened as a `strict_zarr_store`, with the `storage_options` given. Local
  paths keep zarr's directory stores, which already raise read errors. Other engines are
  not affected.
  """
  is_url = isinstance(filename_or_obj, str) and (
    "://" in filename_or_obj or "::" in filename_or_obj
  )
  if is_url and reads_as_zarr(filename_or_obj, kwargs.get("engine")):
    kwargs["engine"] = "zarr"
    storage_options = kwargs.pop("storage_options", None)
    filename_or_obj = strict_zarr_store(filename_or_obj, storage_options=storage_options)
  # xarray accepts zarr stores, but its type hints do not list them.
  return xr.open_dataset(filename_or_obj, **kwargs)  # ty: ignore[invalid-argument-type]


class DatasetConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  skip: bool = False
  checkpointing: CheckpointingConfig | None = None
  parts: list[DatasetPartConfig] = Field(default_factory=list)
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)


def download_and_process(
  configs: DatasetConfig,
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
  time_dim: str = "time",
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
    if engine is not None:
      part_conf["engine"] = engine
    if engine == "earlywarningdatastore" or engine == "copernicusmarine":
      part_conf.update(start_datetime=start_datetime, end_datetime=end_datetime)
    part = open_input(**part_conf)
    part = part.arcomake.time_sel(start_datetime, end_datetime, time_dim=time_dim)
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
    return download_and_process(configs, start_datetime, end_datetime, time_dim=time_dim)
  parsed_step = may_parse_timedelta(checkpointing_step)
  if parsed_step >= end_datetime - start_datetime:
    return download_and_process(configs, start_datetime, end_datetime, time_dim=time_dim)

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
    with download_and_process(
      configs, date_interval.start, date_interval.end, time_dim=time_dim
    ) as downloaded:
      # A shallow copy has its own encodings: the downloaded dataset is left unchanged.
      dataset = downloaded.copy(deep=False)
      if compressor is not None:
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


def safe_to_zarr(
  dataset: xr.Dataset,
  destination: str | pathlib.Path,
  configs: SaveConfig | None = None,
  compute: bool = True,
  *,
  mode: Literal["w", "w-", "a", "a-", "r+", "r"] | None = None,
  append_dim: str | None = None,
) -> xr.backends.ZarrStore | Delayed:
  """Write a dataset to a Zarr store (a local directory or an fsspec URL) with the layout,
  compression and encodings of `[save]`.

  With `compute=False`, returns the task that writes the data (the metadata are written now).
  """
  if configs is None:
    configs = SaveConfig()
  mode = mode or "w"
  # The stored layout is set through the variables' encodings. A shallow copy has its own
  # encodings (data are shared), so the caller's dataset is left unchanged.
  dataset = dataset.copy(deep=False)

  logger.info(f"Saving dataset to {destination} with {configs}")
  chunk_conf = configs.chunks
  compressor = (
    Blosc(**configs.compressor.model_dump(exclude_none=True))
    if configs.compressor is not None
    else None
  )
  # by_alias: encodings use xarray's names (`_FillValue`).
  to_zarr_kwargs = configs.model_dump(
    exclude_none=True, by_alias=True, exclude={"chunks", "compressor", "storage_options"}
  )

  if chunk_conf is not None:
    # Set the on-disk Zarr chunk layout via encoding, without altering the
    # underlying Dask chunking. This requires the existing Dask chunks to be
    # an integer multiple of (and evenly divide into) the requested chunks
    # along each dimension; otherwise to_zarr will raise a ValueError.
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      dims = dataset[var].dims
      if isinstance(chunk_conf, dict):
        requested = [chunk_conf.get(str(dim), -1) for dim in dims]
      elif isinstance(chunk_conf, tuple):
        requested = list(chunk_conf)
      else:
        requested = [chunk_conf for _ in dims]
      # -1 means the whole dimension. Resolve it here: xarray's check that Dask chunks do not
      # straddle Zarr chunks passes for -1 (`n % -1 == 0`), and overlapping Dask chunks would
      # then write into the same Zarr chunk concurrently, losing data.
      chunk_sizes = tuple(
        dataset[var].sizes[dim] if size == -1 else size
        for dim, size in zip(dims, requested, strict=True)
      )
      dataset[var].encoding["chunks"] = chunk_sizes
  if compressor is not None:
    # Without `[save] compressor`, the variables keep their own (or zarr's default) compressor.
    for var in dataset.data_vars:
      dataset[var].encoding["compressor"] = compressor

  if "encoding" in to_zarr_kwargs and isinstance(to_zarr_kwargs["encoding"], dict):
    for var_enc in to_zarr_kwargs["encoding"].values():
      if isinstance(var_enc, dict) and isinstance(var_enc.get("compressor"), dict):
        var_enc["compressor"] = Blosc(**var_enc["compressor"])

  if isinstance(destination, str) and "://" in destination:
    storage_options = configs.storage_options or {}
    store = fsspec.get_mapper(destination, **storage_options)
  else:
    store = DirectoryStore(path=str(destination))

  if mode in ("a", "a-", "r+") and "encoding" in to_zarr_kwargs and to_zarr_kwargs["encoding"]:
    try:
      group = to_zarr_kwargs.get("group")
      zg = (
        zarr.open_group(store, mode="r", path=group) if group else zarr.open_group(store, mode="r")
      )
      existing_keys = set(zg.keys())
      to_zarr_kwargs["encoding"] = {
        k: v for k, v in to_zarr_kwargs["encoding"].items() if k not in existing_keys
      }
      if not to_zarr_kwargs["encoding"]:
        to_zarr_kwargs.pop("encoding", None)
    except zarr.errors.GroupNotFoundError:
      # Nothing written yet (e.g. mode "a" on a new store): every encoding applies.
      pass

  return dataset.to_zarr(
    store=store, compute=compute, mode=mode, append_dim=append_dim, **to_zarr_kwargs
  )
