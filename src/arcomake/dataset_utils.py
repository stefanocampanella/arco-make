# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import logging
import pathlib
import tempfile
import warnings
from collections.abc import Callable, Generator, Hashable, Mapping
from contextlib import contextmanager, nullcontext
from typing import Annotated, Any, Literal

import fsspec
import fsspec.utils
import numpy as np
import xarray as xr
import zarr
import zarr.errors
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from xarray.backends.common import BackendEntrypoint
from xarray.coders import CFDatetimeCoder, CFTimedeltaCoder
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.errors import ZarrUserWarning
from zarr.registry import get_codec_class

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


class CodecConfig(BaseModel):
  """A Zarr v3 compressor (a bytes-to-bytes codec), as Zarr array metadata write it.

  For example `{name = "blosc", configuration = {cname = "lz4", clevel = 1, shuffle = "shuffle"}}`
  or `{name = "zstd", configuration = {level = 3}}`. The codec is built when the recipe is loaded,
  so unknown codecs and invalid configurations fail before any work.
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  configuration: dict[str, Any] = Field(default_factory=dict)

  @model_validator(mode="after")
  def _known_compressor(self) -> "CodecConfig":
    try:
      codec_class = get_codec_class(self.name)
      codec_class.from_dict(self.model_dump())
    except Exception as error:  # unknown codec, unknown or invalid configuration
      raise ValueError(f"invalid codec {self.name!r}: {error}") from None
    if not issubclass(codec_class, BytesBytesCodec):
      raise ValueError(f"codec {self.name!r} is not a compressor (bytes-to-bytes codec)")
    return self


# Compressors of an array, applied in order; an empty list stores the data uncompressed.
CompressorsConfig = list[CodecConfig]


class PartTimeOffset(BaseModel):
  """`time_offset` of a part: added to its time coordinate before each block's time selection.

  For values stamped at the end of the period they describe, e.g. ERA5 hourly accumulations
  (`time_offset = "-1h"`: the value stamped 00:00 covers 23:00-00:00 of the previous day), so
  that each value is stamped with the start of its period and a day holds its own 24 hours. The
  part is read over the block moved by the opposite amount (backends that take a time range get
  it shifted).
  """

  time_offset: datetime.timedelta | None = None

  @field_validator("time_offset", mode="before")
  @classmethod
  def _parse_time_offset(cls, value: Any) -> Any:
    return may_parse_timedelta(value) if isinstance(value, str) else value


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


class CopernicusMarinePartConfig(PartTimeOffset):
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


class NetCDFOverHTTPPartConfig(PartTimeOffset):
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


class EarlyWarningDataStorePartConfig(PartTimeOffset):
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


class ZarrPartConfig(OpenDatasetArguments, PartTimeOffset):
  """A Zarr part: the arguments of `xarray.open_dataset`, read through a strict store."""

  filename_or_obj: str
  engine: Literal["zarr"] | None = "zarr"

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(data, lambda url: url.rstrip("/").endswith(".zarr"))


class GenericPartConfig(ReadConfig, PartTimeOffset):
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
  pack floats into integers. `chunks` and `shards` are explicit sizes, one per dimension.
  `compressors` replace those of `[save]` for this variable.
  """

  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  units: str | None = None
  calendar: str | None = None
  dtype: str | None = None
  chunks: list[Annotated[int, Field(ge=1)]] | None = None
  shards: list[Annotated[int, Field(ge=1)]] | None = None
  compressors: CompressorsConfig | None = None
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


def _sizes(value: StoredChunksConfig) -> list[int]:
  if isinstance(value, dict):
    return list(value.values())
  return list(value) if isinstance(value, tuple) else [value]


class SaveConfig(BaseModel):
  """The `[save]` table: layout, compression and encodings of the stored arrays.

  `chunks` are the units of reading. `shards`, if given, group whole chunks into one stored
  object each (Zarr v3 sharding), in the same form as `chunks`: per dimension (-1, or a dimension
  not listed: the whole dimension), one size per dimension, or one size for all. Without
  `shards`, every chunk is an object. Dask chunks may not straddle shards when writing.
  """

  model_config = ConfigDict(extra="forbid")

  consolidated: bool | None = None
  chunks: StoredChunksConfig | None = None
  shards: StoredChunksConfig | None = None
  # Compressors of the data variables; without them, zarr's default (zstd).
  compressors: CompressorsConfig | None = None
  encoding: dict[str, VariableEncodingConfig] | None = None
  storage_options: dict[str, Any] | None = None
  group: str | None = None

  @field_validator("chunks", "shards")
  @classmethod
  def _no_zero_chunks(cls, value: StoredChunksConfig | None) -> StoredChunksConfig | None:
    if value is not None and 0 in _sizes(value):
      raise ValueError("sizes are positive, or -1 for a whole dimension")
    return value

  @model_validator(mode="after")
  def _shards_hold_whole_chunks(self) -> "SaveConfig":
    """Shards hold whole chunks: checked here wherever both sizes are explicit. A whole
    dimension (-1, or not listed) is only known once the data are open: `safe_to_zarr` checks
    those before writing (before any data are computed, unless a source is checkpointed)."""
    if self.shards is None:
      return self
    if self.chunks is None:
      raise ValueError("shards need chunks: each shard holds a whole number of chunks")
    for where, shard, chunk in _explicit_size_pairs(self.shards, self.chunks):
      if shard > 0 and chunk > 0 and shard % chunk:
        raise ValueError(f"shards {where}({shard}) must be a multiple of chunks ({chunk})")
    return self


def _explicit_size_pairs(
  shards: StoredChunksConfig, chunks: StoredChunksConfig
) -> list[tuple[str, int, int]]:
  """(where, shard, chunk) for every dimension whose sizes both forms state, without the data."""
  if isinstance(shards, dict) or isinstance(chunks, dict):
    if isinstance(shards, tuple) or isinstance(chunks, tuple):
      return []  # positions against names: unknown without the data
    names = set(shards) if isinstance(shards, dict) else set()
    names |= set(chunks) if isinstance(chunks, dict) else set()
    pairs = []
    for name in sorted(names):
      shard = shards.get(name, -1) if isinstance(shards, dict) else shards
      chunk = chunks.get(name, -1) if isinstance(chunks, dict) else chunks
      pairs.append((f"along {name!r} ", shard, chunk))
    return pairs
  if isinstance(shards, tuple) and isinstance(chunks, tuple):
    if len(shards) != len(chunks):
      raise ValueError("shards and chunks give sizes for different numbers of dimensions")
    return [
      (f"at position {i} ", s, c) for i, (s, c) in enumerate(zip(shards, chunks, strict=True))
    ]
  if isinstance(shards, tuple):
    assert isinstance(chunks, int)
    return [(f"at position {i} ", s, chunks) for i, s in enumerate(shards)]
  assert isinstance(shards, int)
  if isinstance(chunks, tuple):
    return [(f"at position {i} ", shards, c) for i, c in enumerate(chunks)]
  return [("", shards, chunks)]


class CheckpointingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  step: datetime.timedelta
  compressors: CompressorsConfig | None = None

  @field_validator("step", mode="before")
  @classmethod
  def parse_step(cls, value: str | datetime.timedelta) -> datetime.timedelta:
    return may_parse_timedelta(value)


def is_remote(path: Any) -> bool:
  """Whether `path` is a URL of a remote filesystem (`file://` and plain paths are local)."""
  return isinstance(path, str | pathlib.Path) and fsspec.utils.get_protocol(str(path)) not in (
    "file",
    "local",
  )


def storage_options_for(path: Any, options: Mapping[str, Any] | None) -> dict[str, Any] | None:
  """`options` for a remote `path`, `None` for a local one: zarr 3 refuses storage options for
  local stores (zarr 2 accepted and ignored them), and recipes may share them across both."""
  if options is None or not len(options) or not is_remote(path):
    return None
  return dict(options)


@contextmanager
def quiet_consolidated_metadata() -> Generator[None]:
  """Consolidated metadata are written by choice: hide zarr's warning that they are not part of
  the format 3 specification."""
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore", message="Consolidated metadata is currently not part", category=ZarrUserWarning
    )
    yield


def open_input(filename_or_obj: Any, **kwargs: Any) -> xr.Dataset:
  """`xarray.open_dataset`, the one way arco-make opens its inputs.

  Read errors of Zarr stores raise (N8): zarr 3 reads only a missing object as the fill value
  (sparse stores); timeouts and connection errors raise (`tests/test_read_errors.py`).
  `storage_options` are dropped for local paths.
  """
  if "storage_options" in kwargs:
    options = storage_options_for(filename_or_obj, kwargs.pop("storage_options"))
    if options is not None:
      kwargs["storage_options"] = options
  return xr.open_dataset(filename_or_obj, **kwargs)


class DatasetConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  skip: bool = False
  checkpointing: CheckpointingConfig | None = None
  parts: list[DatasetPartConfig] = Field(default_factory=list)
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)


def _shift_time(part: xr.Dataset, offset: datetime.timedelta, time_dim: str) -> xr.Dataset:
  """`part` with `offset` added to its time coordinate (attributes and encoding kept)."""
  if time_dim not in part.coords:
    raise ValueError(f"time_offset needs a {time_dim!r} coordinate, which the part lacks")
  time = part[time_dim].variable
  return part.assign_coords({time_dim: time.copy(data=time.values + np.timedelta64(offset))})


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
    offset = raw_part.time_offset or datetime.timedelta(0)
    part_conf = raw_part.model_dump(exclude_unset=True, exclude={"time_offset"})
    engine = getattr(raw_part, "engine", None) or part_conf.get("engine")
    if engine is not None:
      part_conf["engine"] = engine
    if engine == "earlywarningdatastore" or engine == "copernicusmarine":
      part_conf.update(start_datetime=start_datetime - offset, end_datetime=end_datetime - offset)
    part = open_input(**part_conf)
    if offset:
      part = _shift_time(part, offset, time_dim)
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


def _codecs(compressors: CompressorsConfig) -> list[dict[str, Any]]:
  """The compressors of a recipe as zarr takes them (JSON metadata)."""
  return [codec.model_dump() for codec in compressors]


_ZARR3_CODECS = {
  "compressors": BytesBytesCodec,
  "filters": ArrayArrayCodec,
  "serializer": ArrayBytesCodec,
}


def _without_zarr2_codecs(dataset: xr.Dataset) -> xr.Dataset:
  """A shallow copy (own encodings, shared data) without the codecs read from Zarr v2 sources.

  Variables read from a format 2 store carry numcodecs codecs in their encoding, which zarr 3
  cannot write in format 3: they are dropped, so the recipe's codecs or zarr's defaults apply.
  Codecs read from format 3 stores are kept. An empty list of compressors read from a source
  (uncompressed, in either format) is dropped too: outputs are compressed unless the recipe says
  `compressors = []` (N1).
  """
  dataset = dataset.copy(deep=False)
  for variable in dataset.variables.values():
    encoding = variable.encoding
    encoding.pop("compressor", None)  # zarr 2's single compressor
    for key, kind in _ZARR3_CODECS.items():
      codecs = encoding.get(key)
      if codecs is None:
        continue
      codecs = codecs if isinstance(codecs, list | tuple) else [codecs]
      if not all(isinstance(c, kind) for c in codecs) or (key == "compressors" and not codecs):
        del encoding[key]
  return dataset


def _write_zarr(dataset: xr.Dataset, path: str, **kwargs: Any) -> xr.backends.ZarrStore | Delayed:
  """`Dataset.to_zarr` in Zarr format 3 (consolidated metadata as asked, by default)."""
  with quiet_consolidated_metadata():
    written = dataset.to_zarr(path, zarr_format=3, **kwargs)
  assert isinstance(written, xr.backends.ZarrStore | Delayed)
  return written


def maybe_checkpointing_download_and_process(
  configs: DatasetConfig,
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
  time_dim: str = "time",
  progress: bool = True,
) -> xr.Dataset:
  checkpointing_conf = configs.checkpointing
  checkpointing_step = checkpointing_conf.step if checkpointing_conf else None
  compressors = (
    _codecs(checkpointing_conf.compressors)
    if checkpointing_conf is not None and checkpointing_conf.compressors is not None
    else None
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
  date_intervals = IterableDateInterval(start_datetime, end_datetime, parsed_step)
  is_first_checkpoint = True
  for date_interval in date_intervals:
    with download_and_process(
      configs, date_interval.start, date_interval.end, time_dim=time_dim
    ) as downloaded:
      # A copy with its own encodings: the downloaded dataset is left unchanged.
      dataset = _without_zarr2_codecs(downloaded)
      if compressors is not None:
        for var in dataset.data_vars:
          dataset[var].encoding["compressors"] = compressors
      logger.info(f"Saving checkpoint {date_interval}")
      progress_bar = nullcontext if not progress else ProgressBar
      with progress_bar():
        if is_first_checkpoint:
          _write_zarr(dataset, str(checkpoint_path), mode="w", consolidated=False)
          is_first_checkpoint = False
        else:
          _write_zarr(
            dataset, str(checkpoint_path), mode="a-", append_dim=time_dim, consolidated=False
          )
    del dataset

  # Open the checkpointed dataset
  logger.info(f"Opening checkpointed dataset from {checkpoint_path}")
  dataset = xr.open_zarr(str(checkpoint_path), consolidated=False, overwrite_encoded_chunks=True)
  assert isinstance(dataset, xr.Dataset)
  if checkpoint is not None:
    dataset.set_close(checkpoint.cleanup)
  return dataset


def _resolve_sizes(conf: StoredChunksConfig, var: xr.DataArray) -> tuple[int, ...]:
  """Sizes per dimension of `var` from `[save] chunks` or `shards`.

  -1 (or a dimension not listed) is the whole dimension. Resolved here: xarray's check that Dask
  chunks do not straddle Zarr chunks passes for -1 (`n % -1 == 0`), and overlapping Dask chunks
  would then write into the same Zarr chunk concurrently, losing data.
  """
  dims = var.dims
  if isinstance(conf, dict):
    requested = [conf.get(str(dim), -1) for dim in dims]
  elif isinstance(conf, tuple):
    requested = list(conf)
  else:
    requested = [conf for _ in dims]
  return tuple(
    var.sizes[dim] if size == -1 else size for dim, size in zip(dims, requested, strict=True)
  )


def safe_to_zarr(
  dataset: xr.Dataset,
  destination: str | pathlib.Path,
  configs: SaveConfig | None = None,
  compute: bool = True,
  *,
  mode: Literal["w", "w-", "a", "a-", "r+", "r"] | None = None,
  append_dim: str | None = None,
) -> xr.backends.ZarrStore | Delayed:
  """Write a dataset to a Zarr store in format 3 (a local directory or an fsspec URL) with the
  layout, compression and encodings of `[save]`.

  `[save] chunks`, `shards` and `compressors` apply to the data variables; `[save.encoding.VAR]`
  is merged over them, key by key. Codecs read from Zarr v2 sources are dropped (see
  `_without_zarr2_codecs`). With `compute=False`, returns the task that writes the data (the
  metadata are written now).

  Raises:
    ValueError: for a `.zip` destination (zip outputs were removed with Zarr v3), shards that do
      not hold whole chunks, or `[save.encoding]` naming a variable the dataset lacks.
  """
  if configs is None:
    configs = SaveConfig()
  mode = mode or "w"
  path = str(destination)
  if path.rstrip("/").lower().endswith(".zip"):
    raise ValueError(f"{path}: zip outputs are not supported; write a directory or a URL")
  # The stored layout is set through the variables' encodings, on a copy: the caller's dataset is
  # left unchanged.
  dataset = _without_zarr2_codecs(dataset)

  logger.info(f"Saving dataset to {destination} with {configs}")
  chunk_conf = configs.chunks
  if chunk_conf is not None:
    # Set the on-disk Zarr chunk layout via encoding, without altering the
    # underlying Dask chunking. This requires the existing Dask chunks to be
    # an integer multiple of (and evenly divide into) the requested chunks
    # (or shards) along each dimension; otherwise to_zarr will raise a ValueError.
    # see: https://github.com/pydata/xarray/issues/4380
    for var in dataset.data_vars:
      encoding = dataset[var].encoding
      chunk_sizes = _resolve_sizes(chunk_conf, dataset[var])
      encoding["chunks"] = chunk_sizes
      if configs.shards is None:
        # Shards read from a format 3 source were sized for its chunks, not these.
        encoding.pop("shards", None)
        continue
      shard_sizes = _resolve_sizes(configs.shards, dataset[var])
      for dim, shard, chunk in zip(dataset[var].dims, shard_sizes, chunk_sizes, strict=True):
        if shard % chunk:
          raise ValueError(
            f"{var}: shards along {dim!r} ({shard}) must be a multiple of chunks ({chunk})"
          )
      encoding["shards"] = shard_sizes
  if configs.compressors is not None:
    # Without `[save] compressors`, the variables keep their own (zarr 3) codecs or zarr's default.
    for var in dataset.data_vars:
      dataset[var].encoding["compressors"] = _codecs(configs.compressors)

  # by_alias: encodings use xarray's names (`_FillValue`).
  for name, conf in (configs.encoding or {}).items():
    if name not in dataset.variables:
      raise ValueError(f"[save.encoding] names {name!r}, which the dataset does not have")
    encoding = dataset.variables[name].encoding
    overrides = conf.model_dump(exclude_none=True, by_alias=True)
    if "chunks" in overrides:
      overrides["chunks"] = tuple(overrides["chunks"])
      if "shards" not in overrides and configs.shards is None:
        encoding.pop("shards", None)
    if "shards" in overrides:
      overrides["shards"] = tuple(overrides["shards"])
    encoding.update(overrides)

  to_zarr_kwargs = configs.model_dump(exclude_none=True, include={"consolidated", "group"})
  return _write_zarr(
    dataset,
    path,
    compute=compute,
    mode=mode,
    append_dim=append_dim,
    storage_options=storage_options_for(path, configs.storage_options),
    **to_zarr_kwargs,
  )


def consolidate_metadata(path: str, storage_options: Mapping[str, Any] | None = None) -> None:
  """Write the consolidated metadata of a Zarr store, as `safe_to_zarr` does."""
  options = storage_options_for(path, storage_options)
  with quiet_consolidated_metadata():
    zarr.consolidate_metadata(zarr.open_group(str(path), mode="r+", storage_options=options).store)
