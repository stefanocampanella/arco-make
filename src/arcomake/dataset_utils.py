# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import datetime
import functools
import logging
import math
import pathlib
import re
import shutil
import tempfile
import warnings
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Annotated, Any, Literal, Self

import aiohttp
import fsspec
import fsspec.utils
import numpy as np
import xarray as xr
from dask.base import tokenize
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from xarray.backends.common import BackendEntrypoint
from xarray.coders import CFDatetimeCoder, CFTimedeltaCoder
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.errors import ZarrUserWarning
from zarr.registry import get_codec_class

from arcomake import chunk_plan, hdf5_lock, icechunk_utils, sharding
from arcomake.datetime_utils import (
  IterableDateInterval,
  may_parse_timedelta,
  naive_utc,
)
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.steps import needs_for
from arcomake.temporary_utils import get_current_temp_registry

logger = logging.getLogger(__name__)


EngineType = (
  Literal["netcdf4", "scipy", "pydap", "h5netcdf", "zarr"] | type[BackendEntrypoint] | str
)


ChunksConfig = (
  int | str | dict[Hashable, int | Literal["auto"] | tuple[int, ...] | None] | tuple[int, ...]
)


class OpenDatasetArguments(BaseModel):
  """Arguments of `xarray.open_dataset`, except `engine` and `chunks` (download parts are
  chunked as planned: `chunk_plan`)."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
  # Options of the fsspec filesystem of a remote input (credentials...). Download parts take their
  # request timeouts from `timeout` (`PartTimeout`); `[read]` tables get none from arco-make.
  storage_options: dict[str, Any] | None = None
  # Zarr inputs: whether to read consolidated metadata.
  consolidated: bool | None = None


class ReadConfig(OpenDatasetArguments):
  """Arguments of `xarray.open_dataset` (the `[read]` table of a recipe)."""

  engine: EngineType | None = None
  chunks: ChunksConfig | None = None
  # A group of the input (Zarr, Icechunk, netCDF), e.g. "climatology" of a compute-climatology
  # output (its groups: `climatology.GROUPS`).
  group: str | None = None


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


class PartStaticTime(BaseModel):
  """`static_time` of a part: its variables are read at this instant, without the time
  dimension, instead of over each block.

  For fields constant in time that a source stores as a time series, e.g. ERA5's geopotential at
  the surface and land-sea mask: one field instead of every step of every block, the same in
  every block, and none of the glitches some steps have. Give such variables a part
  of their own (the time-varying ones of the same source in another part). The instant must be a
  stamp of the source (after `time_offset`, if any).
  """

  static_time: datetime.datetime | None = None

  @field_validator("static_time")
  @classmethod
  def _without_zone(cls, value: datetime.datetime | None) -> datetime.datetime | None:
    return naive_utc(value, "static_time is a stamp of the source")


class PartAlignment(BaseModel):
  """`rename` and `coordinate_tolerance` of a part: to merge a part whose names or coordinates
  differ slightly from the other parts' (a static file of the same grid from another publisher).

  `rename` renames the part's variables, coordinates or dimensions right after it is opened, e.g.
  `{lat = "latitude", lon = "longitude"}`. With `coordinate_tolerance`, each of the part's index
  coordinates that an earlier part of the dataset also has, with the same size, takes the earlier
  part's values if they differ by at most this much (absolute; else the dataset fails): parts are
  merged only on equal coordinates. Not on the first part; a part none of whose coordinates match
  fails, and so do missing values at other positions than the earlier part's.
  """

  rename: dict[str, str] | None = None
  coordinate_tolerance: float | None = Field(default=None, gt=0)


def _snap_coordinates(
  part: xr.Dataset, earlier: Sequence[xr.Dataset], tolerance: float
) -> xr.Dataset:
  """`part` with the index coordinates it shares with an earlier part (same name and size) taken
  from that part, if no value differs by more than `tolerance` (`PartAlignment`)."""
  matched = False
  for name, index in part.indexes.items():
    reference = next(
      (p[name] for p in earlier if name in p.indexes and p.sizes[name] == part.sizes[name]), None
    )
    if reference is None or not (
      np.issubdtype(index.dtype, np.number) and np.issubdtype(reference.dtype, np.number)
    ):
      continue
    values = np.asarray(index, dtype="float64")
    expected = np.asarray(reference.values, dtype="float64")
    missing = np.isnan(values)
    if (missing != np.isnan(expected)).any():
      raise ValueError(
        f"coordinate {name!r} has missing values where an earlier part's has values, or the"
        " reverse: it cannot be matched"
      )
    difference = float(np.abs(values[~missing] - expected[~missing]).max(initial=0.0))
    if difference > tolerance:
      raise ValueError(
        f"coordinate {name!r} differs from an earlier part's by {difference:g}, more than the"
        f" part's coordinate_tolerance {tolerance:g}"
      )
    part = part.assign_coords({name: (name, reference.values, part[name].attrs)})
    matched = True
  if not matched:
    raise ValueError(
      "coordinate_tolerance: no index coordinate of the part matches an earlier part's (same"
      " name and size; `rename` may be missing)"
    )
  return part


# Seconds a source may stay silent (no connection, no data) before the request fails. gcsfs
# waits forever by default (N7): two downloads hung on a read from Google Cloud Storage.
DEFAULT_TIMEOUT = 120.0


class PartTimeout(BaseModel):
  """`timeout` of a part: seconds a source may stay silent before a request to it fails
  (default `DEFAULT_TIMEOUT`).

  Applied to Zarr stores on Google Cloud Storage (gcsfs's `requests_timeout`, which has none by
  default), to NetCDF files over HTTP (connection at most 30 s, then each wait for data) and to
  EWDS requests; on S3, only when given (botocore's connect and read timeouts, 60 s by default
  otherwise). Copernicus Marine parts do not take it: the toolbox sets its own timeouts and
  retries. Neither do Icechunk inputs, whose storage has its own
  (`network_stream_timeout_seconds` on S3).

  On GCS (`gcs_timeout`) it bounds the connection (at most 30 s) and each wait for data, not
  the whole request: a stalled answer fails with aiohttp's `SocketTimeoutError`, which gcsfs
  requests again (6 attempts in all, waits of about 1, 2, 4, 8 and 16 s between them), so one
  stalled chunk costs about the timeout, not the block. GCS answers stall now and then (one ERA5
  chunk took 212 s on 17 Nov, four at once past 600 s, each block waiting for them alone). When
  GCS does not answer at all, a chunk fails after 6 attempts (about 3.5 minutes if no connection
  can be made, 12.6 if connections are made but nothing comes, at the default), then its block
  is retried. gcsfs's own `requests_timeout` is refused in a part's storage options: as a
  number it bounds a whole request, and gcsfs does not retry the plain `TimeoutError` it raises.

  No timeout bounds a whole request: large chunks over slow links are not cut, and waits for a
  pooled connection end as the requests holding them do. An answer that trickles in more often
  than once per `timeout` would not be cut either; none was seen.
  """

  timeout: float | None = Field(default=None, gt=0)

  @model_validator(mode="after")
  def _no_gcsfs_timeout(self) -> Self:
    options = getattr(self, "storage_options", None) or {}
    if _is_gcs(getattr(self, "filename_or_obj", None)) and "requests_timeout" in options:
      raise ValueError(
        "storage_options.requests_timeout: set the part's `timeout` instead (seconds of silence,"
        " retried by gcsfs; requests_timeout is a total, which gcsfs does not retry)"
      )
    return self


def _is_gcs(url: Any) -> bool:
  """Whether `url` names a Google Cloud Storage object (chained URLs such as `simplecache::gs://`
  are not: their storage options are per protocol)."""
  return isinstance(url, str) and fsspec.utils.get_protocol(url) in ("gs", "gcs")


def gcs_timeout(seconds: float) -> aiohttp.ClientTimeout:
  """gcsfs's `requests_timeout` for a part's `timeout`: the connection and each wait for data,
  no total (see `PartTimeout`)."""
  return aiohttp.ClientTimeout(total=None, sock_connect=min(30.0, seconds), sock_read=seconds)


def _with_timeout(part_conf: dict[str, Any], engine: str | None, timeout: float | None) -> None:
  """Give the part's source a request timeout (`PartTimeout`), unless its options set one."""
  seconds = timeout if timeout is not None else DEFAULT_TIMEOUT
  if engine == "icechunk":  # not fsspec: its storage refuses gcsfs and botocore options
    return
  if engine in ("netcdfoverhttp", "earlywarningdatastore"):
    part_conf.setdefault("timeout", seconds)
    return
  url = part_conf.get("filename_or_obj")
  if not isinstance(url, str):
    return
  protocol = fsspec.utils.get_protocol(url)
  options = dict(part_conf.get("storage_options") or {})
  if _is_gcs(url):  # a recipe cannot set requests_timeout here (`PartTimeout`)
    options.setdefault("requests_timeout", gcs_timeout(seconds))
  elif protocol in ("s3", "s3a") and timeout is not None:
    config = dict(options.get("config_kwargs") or {})
    config.setdefault("connect_timeout", timeout)
    config.setdefault("read_timeout", timeout)
    options["config_kwargs"] = config
  else:
    return
  part_conf["storage_options"] = options


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


class CopernicusMarinePartConfig(PartTimeOffset, PartAlignment):
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
  storage_options: dict[str, Any] | None = None


class NetCDFOverHTTPPartConfig(PartTimeOffset, PartTimeout, PartStaticTime, PartAlignment):
  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["netcdfoverhttp"] | None = "netcdfoverhttp"
  drop_variables: str | list[str] | None = None

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(
      data, lambda url: _has_scheme(url, "http", "https") and not url.endswith(".zarr")
    )


class EarlyWarningDataStorePartConfig(PartTimeOffset, PartTimeout, PartAlignment):
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
  # The forecast run to read (a forecast download injects it); without, the historical product.
  run_datetime: datetime.datetime | None = None
  time_dim: str | None = None
  latitude_dim: str | None = None
  longitude_dim: str | None = None

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(data, lambda url: _has_scheme(url, "ewds"))


class ECMWFOpenDataPartConfig(PartTimeOffset, PartTimeout, PartAlignment):
  """A forecast run of ECMWF's open data (`backends.ECMWFOpenData`): `filename_or_obj` is the
  root of a copy of data.ecmwf.int's tree (e.g. "s3://ecmwf-forecasts", read anonymously),
  `param` the fields' GRIB short names. Read only in forecast downloads, which inject the run."""

  model_config = ConfigDict(extra="forbid")

  filename_or_obj: str
  engine: Literal["ecmwfopendata"]
  param: str | list[str]
  levtype: str | None = None
  model: str | None = None
  resolution: str | None = None
  stream: str | None = None
  type: str | None = None
  deaccumulate: bool | None = None
  drop_variables: str | list[str] | None = None
  storage_options: dict[str, Any] | None = None
  time_dim: str | None = None
  run_datetime: datetime.datetime | None = None
  start_datetime: datetime.datetime | None = None
  end_datetime: datetime.datetime | None = None


class ZarrPartConfig(
  OpenDatasetArguments, PartTimeOffset, PartTimeout, PartStaticTime, PartAlignment
):
  """A Zarr part: the arguments of `xarray.open_dataset`, read through a strict store."""

  filename_or_obj: str
  engine: Literal["zarr"] | None = "zarr"

  @model_validator(mode="before")
  @classmethod
  def _engine(cls, data: Any) -> Any:
    return _engine_given_or_implied(data, lambda url: url.rstrip("/").endswith(".zarr"))


class GenericPartConfig(
  OpenDatasetArguments, PartTimeOffset, PartTimeout, PartStaticTime, PartAlignment
):
  """A part read by any other xarray engine, with the arguments of `xarray.open_dataset`."""

  filename_or_obj: str
  engine: EngineType | None = None

  @field_validator("engine")
  @classmethod
  def _not_a_dedicated_engine(cls, value: EngineType | None) -> EngineType | None:
    # Engines with a dedicated model must match it: a typo there is not a generic part.
    if value in DEDICATED_ENGINES:
      raise ValueError(f"invalid options for engine {value!r}")
    return value


DEDICATED_ENGINES = frozenset(
  {"copernicusmarine", "netcdfoverhttp", "earlywarningdatastore", "ecmwfopendata", "zarr"}
)


DatasetPartConfig = (
  CopernicusMarinePartConfig
  | NetCDFOverHTTPPartConfig
  | EarlyWarningDataStorePartConfig
  | ECMWFOpenDataPartConfig
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
  `shard_order` is the physical order of the chunks inside a shard (see `sharding`): zarr's
  "morton" by default; "lexicographic" puts the chunks of one step of the first dimension next to
  each other (with time first, a day's whole depth profile), "colexicographic" those of one step
  of the last.
  """

  model_config = ConfigDict(extra="forbid")

  chunks: StoredChunksConfig | None = None
  shards: StoredChunksConfig | None = None
  shard_order: sharding.ShardOrder = sharding.DEFAULT
  # Compressors of the data variables; without them, zarr's default (zstd).
  compressors: CompressorsConfig | None = None
  encoding: dict[str, VariableEncodingConfig] | None = None
  storage_options: dict[str, Any] | None = None
  group: str | None = None
  # How long versions of the output replaced by later runs are kept (one of the two; without
  # them, all are kept): by age (e.g. "30D"), or as the number of versions kept, the current
  # one included. Applied after each successful run (see `icechunk_utils.Retention`).
  keep_history: datetime.timedelta | None = None
  keep_versions: Annotated[int, Field(ge=1)] | None = None

  @field_validator("keep_history", mode="before")
  @classmethod
  def _parse_keep_history(cls, value: Any) -> Any:
    # A duration with its unit ("30D", "12h", "0s"): a bare number would be read as nanoseconds
    # (pandas) or seconds (pydantic), and expire nearly the whole history.
    if not isinstance(value, str) or not re.search(r"[A-Za-z]", value):
      raise ValueError('keep_history is a duration with its unit, e.g. "30D"')
    return may_parse_timedelta(value)

  @model_validator(mode="after")
  def _order_needs_shards(self) -> "SaveConfig":
    if self.shard_order != sharding.DEFAULT and self.shards is None:
      raise ValueError("shard_order orders the chunks inside shards: give [save] shards")
    return self

  @model_validator(mode="after")
  def _one_retention(self) -> "SaveConfig":
    if self.keep_history is not None and self.keep_versions is not None:
      raise ValueError("give keep_history or keep_versions, not both")
    if self.keep_history is not None and self.keep_history < datetime.timedelta(0):
      raise ValueError("keep_history is a duration, not negative")
    return self

  def retention(self) -> "icechunk_utils.Retention | None":
    """The retention of the output's versions, if the recipe sets one."""
    if self.keep_history is None and self.keep_versions is None:
      return None
    return icechunk_utils.Retention(self.keep_history, self.keep_versions)

  @field_validator("chunks", "shards")
  @classmethod
  def _no_zero_chunks(cls, value: StoredChunksConfig | None) -> StoredChunksConfig | None:
    if value is not None and 0 in _sizes(value):
      raise ValueError("sizes are positive, or -1 for a whole dimension")
    return value

  @model_validator(mode="after")
  def _shards_hold_whole_chunks(self) -> "SaveConfig":
    """Shards hold whole chunks: checked here wherever both sizes are explicit. A whole
    dimension (-1, or not listed) is only known once the data are open: `prepare_for_writing` checks
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
  `storage_options` are dropped for local paths. netCDF4 and HDF5 files are opened and read
  under one lock (`hdf5_lock`): those libraries are not thread-safe, and blocks run in threads.
  """
  if "storage_options" in kwargs:
    options = storage_options_for(filename_or_obj, kwargs.pop("storage_options"))
    if options is not None:
      kwargs["storage_options"] = options
  return hdf5_lock.open_dataset(filename_or_obj, **kwargs)


# Download recipes give no chunk sizes: they are planned for each block (`chunk_plan`).
PLANNED_CHUNKS = "chunks are planned for each block (download --memory-limit; --plan shows them)"


def refuse_chunk_steps(steps: list[ProcessingStepConfig]) -> list[ProcessingStepConfig]:
  """`steps` of a download recipe, which may not set chunk sizes: no `chunk` step, no regrid
  `output_chunks` (they are planned).

  Raises:
    ValueError: naming the step.
  """
  for number, config in enumerate(steps, start=1):
    if config.name == "chunk":
      raise ValueError(f"step {number} (chunk): {PLANNED_CHUNKS}: remove the step")
    kwargs = (config.model_extra or {}).get("kwargs") or {}
    if config.name == "regrid" and isinstance(kwargs, Mapping) and "output_chunks" in kwargs:
      raise ValueError(f"step {number} (regrid): {PLANNED_CHUNKS}: remove `output_chunks`")
  return steps


class DatasetConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  skip: bool = False
  checkpointing: CheckpointingConfig | None = None
  parts: list[DatasetPartConfig] = Field(default_factory=list)
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  # In a forecast recipe: how the source gives runs ("runs": the parts read each run; "series":
  # read by valid time), and the flag variable telling where the values are the run's own
  # forecast (see `forecast`).
  forecast: Literal["runs", "series"] = "series"
  flag: str | None = None

  @model_validator(mode="after")
  def _runs_from_run_engines(self) -> Self:
    # Engines that read only runs belong to "runs" datasets (of forecast recipes).
    for number, part in enumerate(self.parts, start=1):
      if getattr(part, "engine", None) in RUN_ONLY_ENGINES and self.forecast != "runs":
        raise ValueError(
          f'part {number} reads forecast runs only: give its dataset forecast = "runs" (in a'
          " recipe with a [forecast] table)"
        )
    # A "runs" dataset reads its runs from parts of engines that read runs; its other parts (a
    # static file, say: GloFAS's drainage network) are read as in any block.
    engines = {getattr(part, "engine", None) for part in self.parts}
    if self.forecast == "runs" and self.parts and not engines & RUN_ENGINES:
      raise ValueError(
        'forecast = "runs": no part reads forecast runs (engines that do:'
        f" {', '.join(sorted(RUN_ENGINES))})"
      )
    return self

  @model_validator(mode="before")
  @classmethod
  def _no_part_chunks(cls, data: Any) -> Any:
    # Before the parts are parsed: one message, not one per model of the union.
    if isinstance(data, Mapping):
      for number, part in enumerate(data.get("parts") or [], start=1):
        if isinstance(part, Mapping) and "chunks" in part:
          raise ValueError(f"part {number}: {PLANNED_CHUNKS}: remove `chunks`")
    return data

  @field_validator("parts")
  @classmethod
  def _tolerance_after_the_first_part(
    cls, parts: list[DatasetPartConfig]
  ) -> list[DatasetPartConfig]:
    # The tolerance takes coordinates from earlier parts: the first one has none.
    if parts and parts[0].coordinate_tolerance is not None:
      raise ValueError(
        "the first part has no earlier part to take coordinates from: give"
        " coordinate_tolerance to a later part"
      )
    return parts

  @field_validator("postprocess")
  @classmethod
  def _no_chunk_steps(cls, steps: list[ProcessingStepConfig]) -> list[ProcessingStepConfig]:
    return refuse_chunk_steps(steps)

  @field_validator("postprocess")
  @classmethod
  def _no_output_attrs(cls, steps: list[ProcessingStepConfig]) -> list[ProcessingStepConfig]:
    # Datasets are merged with identical attributes: the output's extent belongs after merging.
    for number, config in enumerate(steps, start=1):
      if config.name == "coverage_attrs":
        raise ValueError(
          f"step {number}: coverage_attrs describes the output: give it in the recipe's"
          " [[postprocess]], after merging"
        )
    return steps


# The step lists of a recipe: its own steps (download, process) and each dataset's.
_STEP_LISTS = ("postprocess", "process")


def inline_grids(data: Any) -> Any:
  """A recipe whose `regrid` steps name a grid of its `[grids]` table (`grid = "quarter"`), with
  the table's arguments in their place: a grid is written once per recipe.

  Raises:
    ValueError: for a name the table lacks.
  """
  if not isinstance(data, Mapping):
    return data
  grids = data.get("grids") or {}
  if not isinstance(grids, Mapping):
    raise ValueError("[grids] is a table of grids, each a table of xarray_regrid.Grid arguments")

  def inlined(steps: Any, where: str) -> Any:
    if not isinstance(steps, list):
      return steps
    result = []
    for number, step in enumerate(steps, start=1):
      named = isinstance(step, Mapping) and step.get("name") == "regrid"
      if named and isinstance(step.get("grid"), str):
        if step["grid"] not in grids:
          known = ", ".join(sorted(grids)) or "none"
          raise ValueError(f"{where} step {number}: no grid {step['grid']!r} in [grids] ({known})")
        if not isinstance(grid := grids[step["grid"]], Mapping):
          raise ValueError(f"grids.{step['grid']} is not a table of xarray_regrid.Grid arguments")
        step = {**step, "grid": dict(grid)}
      result.append(step)
    return result

  data = {key: inlined(value, key) if key in _STEP_LISTS else value for key, value in data.items()}
  datasets = data.get("datasets")
  if isinstance(datasets, Mapping):
    data["datasets"] = {
      name: {**dataset, "postprocess": inlined(dataset["postprocess"], f"{name}.postprocess")}
      if isinstance(dataset, Mapping) and "postprocess" in dataset
      else dataset
      for name, dataset in datasets.items()
    }
  return data


def _shift_time(part: xr.Dataset, offset: datetime.timedelta, time_dim: str) -> xr.Dataset:
  """`part` with `offset` added to its time coordinate (attributes and encoding kept)."""
  if time_dim not in part.coords:
    raise ValueError(f"time_offset needs a {time_dim!r} coordinate, which the part lacks")
  time = part[time_dim].variable
  return part.assign_coords({time_dim: time.copy(data=time.values + np.timedelta64(offset))})


def _static_at(part: xr.Dataset, instant: datetime.datetime, time_dim: str) -> xr.Dataset:
  """`part` at `instant`, without the time dimension (`static_time`)."""
  if time_dim not in part.coords:
    raise ValueError(f"static_time needs a {time_dim!r} coordinate, which the part lacks")
  try:
    static = part.sel({time_dim: np.datetime64(instant, "ns")})
  except KeyError:
    raise ValueError(f"static_time {instant} is not a stamp of the part") from None
  return static.drop_vars(time_dim)


def _source_chunks(dataset: xr.Dataset) -> dict[Hashable, int]:
  """The chunks of the source's storage along each dimension (the least common multiple over the
  variables), as the backend reports them (`preferred_chunks`)."""
  chunks: dict[Hashable, int] = {}
  for var in dataset.data_vars.values():
    for dim, size in (var.encoding.get("preferred_chunks") or {}).items():
      if dim in var.dims and isinstance(size, int) and size > 0:
        chunks[dim] = math.lcm(chunks.get(dim, 1), size)
  return chunks


def _kept_variables(steps: Sequence[ProcessingStepConfig], dataset: xr.Dataset) -> set[Hashable]:
  """The data variables of `dataset` that its leading selection steps (`select_variables`,
  `drop_vars`) keep: the others are dropped before any computation, so they are not planned
  for (an ERA5 store's pressure-level variables, say)."""
  kept = set(dataset.data_vars)
  for config in steps:
    arguments = config.model_dump(exclude_unset=True)
    name = arguments.pop("name")
    if name == "select_variables":
      kept &= set(arguments["variables"])
    elif name == "drop_vars":
      names = arguments.get("names", ())
      kept -= {names} if isinstance(names, str) else set(names)
    else:
      break
  return kept


def _planned(
  parts: list[tuple[str, xr.Dataset]],
  steps: Sequence[ProcessingStepConfig],
  budget: chunk_plan.Budget,
  *,
  time_dim: str,
  label: str,
) -> list[xr.Dataset]:
  """The parts of a dataset (their block's slices, not yet chunked), each with a token naming
  its Dask arrays (lazily indexed source arrays are not tokenized), in the chunks
  `chunk_plan.plan` gives them together, so that merged variables share their chunks: for
  `budget`, the needs of the `steps` that will process them, and the sources' chunks, over the
  variables the leading selection steps keep. The plan is logged."""
  needs = chunk_plan.Needs()
  variables: dict[Hashable, chunk_plan.Shape] = {}
  preferred: dict[Hashable, int] = {}
  for number, (_, part) in enumerate(parts, start=1):
    needs = needs.merge(needs_for(steps, part))
    used = part[sorted(_kept_variables(steps, part), key=str)]
    for name, shape in chunk_plan.shapes(used).items():
      variables[name if name not in variables else f"{name} (part {number})"] = shape
    for dim, size in _source_chunks(used).items():
      preferred[dim] = math.lcm(preferred.get(dim, 1), size)
  planned = chunk_plan.plan(variables, budget, needs, grow_dim=time_dim, preferred=preferred)
  logger.info(f"{label}: {planned.describe()}")
  for warning in planned.warnings:
    logger.warning(f"{label}: {warning}")
  return [
    part.chunk({dim: size for dim, size in planned.chunks.items() if dim in part.dims}, token=token)
    for token, part in parts
  ]


# The engines that read forecast runs (`run_datetime`), along valid time.
RUN_ENGINES = frozenset({"earlywarningdatastore", "ecmwfopendata"})
# Those that read nothing else (the EWDS has historical products too).
RUN_ONLY_ENGINES = frozenset({"ecmwfopendata"})


def download_and_process(
  configs: DatasetConfig,
  start_datetime: datetime.datetime,
  end_datetime: datetime.datetime,
  time_dim: str = "time",
  budget: chunk_plan.Budget | None = None,
  name: str = "dataset",
  later_steps: Sequence[ProcessingStepConfig] = (),
  run_datetime: datetime.datetime | None = None,
  part_ends: list[np.datetime64] | None = None,
) -> xr.Dataset:
  """The dataset of `configs` over [start, end): its parts read, time-sliced and merged, then its
  steps applied (lazily: computed when written).

  With `part_ends`, the last time stamp of each part as opened (moved by its offset, before the
  block's time selection) is appended to it: a forecast tells from it whether a series is a run's
  own forecast (`forecast.series_is_the_run`).

  With `run_datetime`, the parts of engines that read runs (`RUN_ENGINES`) read that forecast
  run (their backends return it along valid time, so [start, end) are valid times); at least one
  must. The others are read as in any block (a static file, say).

  With a `budget`, the parts are chunked by `chunk_plan.plan` for their slices of the block and
  the needs of the dataset's steps and of the `later_steps` (the recipe's steps after merging);
  without, they are read as their backends return them.
  """
  # When downloading from Copernicus Marine Data Store or Climate Data Store, the typical case is a large dataset,
  # spanning a long time period, with several sets of variables in different datasets (bio, phys, etc.),
  # which needs to be downloaded one piece at a time. Hence, `parts` list in the TOML configuration file
  # represents different pieces of the same dataset.
  opened: list[tuple[str, xr.Dataset]] = []
  for raw_part in configs.parts:
    offset = raw_part.time_offset or datetime.timedelta(0)
    static_time = getattr(raw_part, "static_time", None)
    part_conf = raw_part.model_dump(
      exclude_unset=True,
      exclude={"time_offset", "timeout", "static_time", "rename", "coordinate_tolerance"},
    )
    engine = getattr(raw_part, "engine", None) or part_conf.get("engine")
    if engine is not None:
      part_conf["engine"] = engine
    if engine != "copernicusmarine":
      _with_timeout(part_conf, engine, getattr(raw_part, "timeout", None))
    if engine in ("earlywarningdatastore", "copernicusmarine", "ecmwfopendata"):
      part_conf.update(start_datetime=start_datetime - offset, end_datetime=end_datetime - offset)
    if run_datetime is not None and engine in RUN_ENGINES:
      part_conf["run_datetime"] = run_datetime
    part = open_input(**part_conf)
    if raw_part.rename is not None:
      part = part.rename(raw_part.rename)
    if raw_part.coordinate_tolerance is not None:
      part = _snap_coordinates(part, [p for _, p in opened], raw_part.coordinate_tolerance)
    if offset:
      part = _shift_time(part, offset, time_dim)
    # A part read at one instant (`static_time`) says nothing of where a series ends.
    if (
      part_ends is not None
      and static_time is None
      and time_dim in part.coords
      and part[time_dim].size
    ):
      part_ends.append(np.datetime64(part[time_dim].values.max(), "ns"))
    if static_time is not None:
      part = _static_at(part, static_time, time_dim)
    else:
      part = part.arcomake.time_sel(start_datetime, end_datetime, time_dim=time_dim)
    opened.append((tokenize(repr(raw_part), start_datetime, end_datetime, run_datetime), part))
  if not opened:
    raise ValueError("No parts found in the dataset configuration")
  if budget is not None:
    steps = [*configs.postprocess, *later_steps]
    parts = _planned(opened, steps, budget, time_dim=time_dim, label=name)
  else:
    parts = [part for _, part in opened]
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
  budget: chunk_plan.Budget | None = None,
  name: str = "dataset",
  later_steps: Sequence[ProcessingStepConfig] = (),
  run_datetime: datetime.datetime | None = None,
  part_ends: list[np.datetime64] | None = None,
) -> xr.Dataset:
  checkpointing_conf = configs.checkpointing
  checkpointing_step = checkpointing_conf.step if checkpointing_conf else None
  compressors = (
    _codecs(checkpointing_conf.compressors)
    if checkpointing_conf is not None and checkpointing_conf.compressors is not None
    else None
  )

  if checkpointing_step is None:
    return download_and_process(
      configs,
      start_datetime,
      end_datetime,
      time_dim=time_dim,
      budget=budget,
      name=name,
      later_steps=later_steps,
      run_datetime=run_datetime,
      part_ends=part_ends,
    )
  parsed_step = may_parse_timedelta(checkpointing_step)
  if parsed_step >= end_datetime - start_datetime:
    return download_and_process(
      configs,
      start_datetime,
      end_datetime,
      time_dim=time_dim,
      budget=budget,
      name=name,
      later_steps=later_steps,
      run_datetime=run_datetime,
      part_ends=part_ends,
    )

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
      configs,
      date_interval.start,
      date_interval.end,
      time_dim=time_dim,
      budget=budget,
      name=name,
      later_steps=later_steps,
      run_datetime=run_datetime,
      part_ends=part_ends,
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
          # Planned chunks may straddle the checkpoint's (those of its first write).
          _write_zarr(
            dataset, str(checkpoint_path), mode="w", consolidated=False, align_chunks=True
          )
          is_first_checkpoint = False
        else:
          _write_zarr(
            dataset,
            str(checkpoint_path),
            mode="a-",
            append_dim=time_dim,
            consolidated=False,
            align_chunks=True,
          )
    del dataset

  # Open the checkpointed dataset
  logger.info(f"Opening checkpointed dataset from {checkpoint_path}")
  dataset = xr.open_zarr(str(checkpoint_path), consolidated=False, overwrite_encoded_chunks=True)
  assert isinstance(dataset, xr.Dataset)
  # Removed when the dataset is closed (at the end of its block): a run of many blocks would
  # otherwise keep every block's checkpoints on disk until it ends.
  if checkpoint is not None:
    dataset.set_close(checkpoint.cleanup)
  else:
    dataset.set_close(functools.partial(shutil.rmtree, checkpoint_path, ignore_errors=True))
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


def prepare_for_writing(dataset: xr.Dataset, configs: SaveConfig | None = None) -> xr.Dataset:
  """A copy of `dataset` whose encodings carry the layout, compression and encodings of `[save]`.

  `[save] chunks`, `shards` and `compressors` apply to the data variables; `[save.encoding.VAR]`
  is merged over them, key by key. Codecs read from Zarr v2 sources are dropped (see
  `_without_zarr2_codecs`). The caller's dataset is left unchanged.

  Raises:
    ValueError: shards that do not hold whole chunks, or `[save.encoding]` naming a variable the
      dataset lacks.
  """
  configs = configs or SaveConfig()
  dataset = _without_zarr2_codecs(dataset)
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
  return dataset


def _check_output_path(destination: str | pathlib.Path) -> str:
  path = str(destination)
  if path.rstrip("/").lower().endswith(".zip"):
    raise ValueError(f"{path}: zip outputs are not supported; write a directory or a URL")
  return path


def write_output(
  dataset: xr.Dataset,
  destination: "str | pathlib.Path | icechunk_utils.Run",
  configs: SaveConfig | None = None,
  *,
  message: str,
  mode: Literal["w", "w-", "a", "a-", "r+"] = "w",
  append_dim: str | None = None,
  command: str = "write",
  metadata: Mapping[str, Any] | None = None,
  align_chunks: bool = False,
  recipe: str | None = None,
  keep: int | None = None,
) -> str:
  """Write a dataset with the layout of `[save]` (see `prepare_for_writing`) in one commit.

  With `keep`, the dataset (one step along `append_dim`) is written after the last `keep` steps
  stored, the others dropped: a rolling window of forecast runs
  (`icechunk_utils.Run.write_rolled`). Runs only.

  With `align_chunks`, Dask chunks that straddle the stored chunks (or shards) are rechunked to
  fit them (see `icechunk_utils.Run.write`); else such a layout is refused.

  `destination` is a run (`icechunk_utils.Run`, which the caller finishes) or the path of a
  repository: then the write is a run of its own (labelled with `command`), finished at once
  (`main` moves to it, and the recipe's retention applies). Returns the commit.

  Raises:
    ValueError: for a `.zip` destination (zip outputs were removed with Zarr v3), or a layout
      `prepare_for_writing` refuses.
  """
  if isinstance(destination, icechunk_utils.Run) and recipe is not None:
    raise ValueError("write_output: a run carries its recipe's fingerprint (`Run(recipe=...)`)")
  configs = configs or SaveConfig()
  prepared = prepare_for_writing(dataset, configs)
  if isinstance(destination, icechunk_utils.Run):
    logger.info(f"Writing to {destination.path} ({message}) with {configs}")
    if keep is not None:
      if append_dim is None:
        raise ValueError("write_output: a rolling write (keep) goes along append_dim")
      return destination.write_rolled(
        prepared,
        dim=append_dim,
        keep=keep,
        message=message,
        group=configs.group,
        metadata=metadata,
        align_chunks=align_chunks,
        shard_order=configs.shard_order,
      )
    return destination.write(
      prepared,
      message=message,
      mode=mode,
      append_dim=append_dim,
      group=configs.group,
      metadata=metadata,
      align_chunks=align_chunks,
      shard_order=configs.shard_order,
    )
  path = _check_output_path(destination)
  logger.info(f"Writing to {path} ({message}) with {configs}")
  return icechunk_utils.write(
    prepared,
    path,
    storage_options=configs.storage_options,
    message=message,
    mode=mode,
    append_dim=append_dim,
    group=configs.group,
    retention=configs.retention(),
    command=command,
    align_chunks=align_chunks,
    shard_order=configs.shard_order,
    recipe=recipe,
  )


def write_tree(
  groups: Mapping[str, xr.Dataset],
  destination: "str | pathlib.Path | icechunk_utils.Run",
  configs: SaveConfig,
  *,
  label: str,
  recipe: str | None = None,
  metadata: Mapping[str, Any] | None = None,
) -> str:
  """Write datasets into the groups of one repository (a DataTree; under `[save] group`, if
  set), each with the layout of `[save]`, in one commit of a run named `label`
  (`icechunk_utils.Run`; with `metadata`), which replaces what the repository held; `main` moves
  to it once all are written. Returns the new tip of `main`. `destination` is a repository's
  path, or a run opened before the data were computed (it carries the recipe's fingerprint), so
  that a version promoted meanwhile is not replaced (`PromotionError`).

  Each group is computed by its own write: work they share is done again for each (Dask
  optimizes each write on its own and fuses a shared task into each), so compute it first (the
  climatology writes its groups from a temporary store).

  Raises:
    icechunk_utils.PromotionError: if `main` changed meanwhile (the run stays on its branch).
  """
  if isinstance(destination, icechunk_utils.Run):
    if recipe is not None:
      raise ValueError("write_tree: a run carries its recipe's fingerprint (`Run(recipe=...)`)")
    run = destination
  else:
    path = _check_output_path(destination)
    run = icechunk_utils.Run(path, configs.storage_options, label, recipe=recipe)
  logger.info(f"Writing groups {list(groups)} to {run.path} ({label}) with {configs}")
  parent = "" if configs.group is None else f"{configs.group}/"
  run.write_groups(
    {f"{parent}{name}": prepare_for_writing(dataset, configs) for name, dataset in groups.items()},
    message=label,
    align_chunks=True,
    shard_order=configs.shard_order,
    metadata=metadata,
  )
  return run.finish(configs.retention())
