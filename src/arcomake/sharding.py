# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The physical order of the chunks inside a shard (`[save] shard_order`).

Zarr's sharding codec writes the chunks of a shard in Morton order unless told otherwise
(`ShardingCodec(subchunk_write_order=...)`: "morton", "lexicographic", "colexicographic"). The
order is not part of the array's metadata: a reader finds every chunk through the shard's index,
whatever the order, but reads chunks next to each other in fewer requests (zarr merges ranges
less than 1 MiB apart). Lexicographic order (C order over the chunk grid) puts the chunks of one
step of the first dimension next to each other: with time first, a day's whole depth profile.

Because the order is not stored, every write would use the default: zarr builds the codec of an
existing array from its metadata (appends, regions), and builds it itself when xarray creates an
array. `write_order` selects, through zarr's codec registry (`codecs.sharding_indexed` in
`zarr.config`), a subclass whose default is the order asked, for every array opened meanwhile;
arrays created in a write need their metadata written first (`create_arrays`) and the values
written into them afterwards (mode "r+", by the writers in `icechunk_utils` and `dataset_utils`).
The setting is process-wide while active, and reaches Dask's threads, not the workers of a
distributed client.
"""

import contextlib
from collections.abc import Generator
from typing import Any, Literal

import xarray as xr
import zarr
from zarr.codecs import ShardingCodec
from zarr.registry import register_codec

ShardOrder = Literal["morton", "lexicographic", "colexicographic"]
DEFAULT: ShardOrder = "morton"  # zarr's


class LexicographicShardingCodec(ShardingCodec):
  """Zarr's sharding codec writing chunks in lexicographic (C) order by default."""

  def __init__(self, *args: Any, subchunk_write_order: Any = "lexicographic", **kwargs: Any):
    super().__init__(*args, subchunk_write_order=subchunk_write_order, **kwargs)


class ColexicographicShardingCodec(ShardingCodec):
  """Zarr's sharding codec writing chunks in colexicographic (Fortran) order by default."""

  def __init__(self, *args: Any, subchunk_write_order: Any = "colexicographic", **kwargs: Any):
    super().__init__(*args, subchunk_write_order=subchunk_write_order, **kwargs)


_CODECS: dict[ShardOrder, type[ShardingCodec]] = {
  "lexicographic": LexicographicShardingCodec,
  "colexicographic": ColexicographicShardingCodec,
}
for _codec in _CODECS.values():
  register_codec("sharding_indexed", _codec)


def creates_arrays(order: ShardOrder, mode: str) -> bool:
  """Whether a write in `mode` with this order must write the metadata first (see the module)."""
  return order != DEFAULT and mode in ("w", "w-")


def create_arrays(
  dataset: xr.Dataset, store: Any, *, mode: Literal["w", "w-", "a", "a-", "r+"], group: str | None
) -> None:
  """Write the metadata of `dataset`'s arrays (and the values of those not in Dask) into
  `store`, for values written afterwards in mode "r+" into the arrays zarr opens.

  No Dask values are written here, so xarray's check that Dask chunks do not straddle stored
  ones is left to the write of the values, which may align them (`align_chunks`).
  """
  dataset.to_zarr(
    store,
    mode=mode,
    group=group,
    compute=False,
    safe_chunks=False,
    consolidated=False,
    zarr_format=3,
  )


@contextlib.contextmanager
def write_order(order: ShardOrder) -> Generator[None]:
  """Within the block, shards of the arrays zarr opens are written in `order`."""
  if order == DEFAULT:
    yield
    return
  codec = _CODECS[order]
  with zarr.config.set({"codecs.sharding_indexed": f"{codec.__module__}.{codec.__qualname__}"}):
    yield
