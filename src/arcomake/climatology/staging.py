# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The temporary store both paths compute into, and its copy into the output with the
layout of `[save]`."""

import dataclasses
import logging
import math
import pathlib
import shutil
from collections.abc import Hashable, Mapping
from typing import Any

import dask
import dask.array
import xarray as xr

from arcomake import icechunk_utils
from arcomake.chunk_plan import COPIES, Budget, detect_budget, stored_warnings
from arcomake.dask_utils import (
  check_workers_reach,
)
from arcomake.dataset_utils import (
  SaveConfig,
  prepare_for_writing,
  write_tree,
)
from arcomake.temporary_utils import get_current_temp_registry

logger = logging.getLogger(__name__)


def _staging_chunks(
  dims: tuple[Hashable, ...],
  shape: tuple[int, ...],
  dask_chunks: tuple[int, ...],
  unit: tuple[int, ...] | None,
  itemsize: int,
  time_dim: str,
  target: int,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
  """The chunks of a variable in the temporary store, and the Dask chunks of its copy (None:
  those of the store).

  The copy's chunks are the stored units (`unit`: shards, else chunks), with along `time_dim` as
  many whole units as `target` bytes hold (at least one). The store's chunks are Dask's (the
  first of each dimension), with the copy's along `time_dim` and the units' where they divide
  Dask's: copying reads each chunk of the store once, and one-bin units do not make a store of
  tiny files.
  """
  if unit is None or time_dim not in dims:
    return dask_chunks, None
  copy = [min(u, n) for u, n in zip(unit, shape, strict=True)]
  axis = dims.index(time_dim)
  units = max(1, target // (math.prod(copy) * itemsize))
  copy[axis] = min(shape[axis], copy[axis] * units)
  stored = [
    c if axis_ == axis or d % c == 0 else d
    for axis_, (d, c) in enumerate(zip(dask_chunks, copy, strict=True))
  ]
  return tuple(stored), tuple(copy)


@dataclasses.dataclass
class Stage:
  """A temporary plain Zarr v3 store for the groups of the output (a DataTree), unsharded, and how
  to copy them into the repository with the layout of `[save]` (`stage`, `copy`)."""

  path: pathlib.Path
  staged: dict[str, xr.Dataset]  # the groups with the store's chunks as encodings, for writing
  copies: dict[str, dict[Hashable, tuple[int, ...] | None]]  # the copy's chunks per variable
  # Per group, the attributes, and every variable's attributes and encoding, as computed: the
  # copy takes them back (the computed datasets themselves are not kept).
  restored: dict[
    str, tuple[dict[Hashable, Any], dict[Hashable, tuple[dict[Hashable, Any], dict[str, Any]]]]
  ]
  threads: int | None  # threads of the copy (None: Dask's)

  def write(self, compute: bool = True) -> None:
    """Write the groups into the store (with `compute=False`, their metadata only)."""
    xr.DataTree.from_dict(self.staged).to_zarr(
      self.path, mode="w", compute=compute, consolidated=False, zarr_format=3, align_chunks=True
    )


def stage(
  groups: Mapping[str, xr.Dataset], save: SaveConfig, time_dim: str, *, workers: bool = True
) -> Stage:
  """A temporary store for `groups` (in the current registry's directory, `TMPDIR`; with
  `workers`, checked to be seen by every worker of a distributed client, which write into it).

  Its chunks are `_staging_chunks`: Dask's, and along `time_dim` whole stored units. Writing into
  it needs the memory of the series being computed; written directly, shards spanning all levels
  held every level's statistics until the last window (logbook, 21 Nov). The copy reads one
  group of stored units per task: tasks holding more than their share of memory are warned
  about, and the copy runs with as many threads as hold their units (`_copy_threads`).
  """
  registry = get_current_temp_registry()
  if registry is None:
    raise RuntimeError("the climatology's temporary store needs a TempDirectoryRegistry")
  path = registry.create_temp_dir(suffix=".zarr")
  if workers:
    check_workers_reach(path, "the temporary store")
  budget = detect_budget()
  staged, copies, units, size = {}, {}, [], 0
  for group, dataset in groups.items():
    prepared = prepare_for_writing(dataset, save)
    for warning in stored_warnings(prepared, budget):
      logger.warning(warning)
    staged[group] = dataset.drop_encoding()
    copies[group] = {}
    for name, var in staged[group].data_vars.items():
      stored, copy = _staging_chunks(
        var.dims,
        var.shape,
        tuple(sizes[0] for sizes in var.chunks or [(n,) for n in var.shape]),
        prepared[name].encoding.get("shards") or prepared[name].encoding.get("chunks"),
        var.dtype.itemsize,
        time_dim,
        budget.target,
      )
      var.encoding["chunks"] = stored
      copies[group][name] = copy
      # A copy task reads whole chunks of the store: it holds the larger of each.
      held = stored if copy is None else tuple(map(max, stored, copy))
      units.append(math.prod(held) * var.dtype.itemsize)
      size += var.nbytes
  free = shutil.disk_usage(path).free
  if free < size:
    logger.warning(
      f"The temporary store may not fit in {path}: {size / 2**30:.1f} GiB before compression,"
      f" {free / 2**30:.1f} GiB free"
    )
  logger.info(f"Temporary store {path} ({size / 2**30:.1f} GiB before compression)")
  restored = {
    group: (
      dict(dataset.attrs),
      {name: (dict(var.attrs), dict(var.encoding)) for name, var in dataset.variables.items()},
    )
    for group, dataset in groups.items()
  }
  return Stage(path, staged, copies, restored, _copy_threads(units, budget))


def _copy_threads(units: list[int], budget: Budget) -> int | None:
  """Threads for the copy: as many as hold `COPIES` times the largest unit each within the
  budget's memory, at least one (None: all of Dask's). Each copy task holds its unit, the pieces
  read for it and its encoded form: 3.1 GiB for a shard of 8 bins, 49 levels and the 0.25 deg
  grid (1.6 GB); 32 of them did not fit in 110 GB, and 8 copied as fast as 16 (logbook, 21 Nov)."""
  largest = max(units, default=0)
  fits = budget.usable // (COPIES * largest) if largest else budget.threads
  return None if fits >= budget.threads else max(1, int(fits))


def copy(
  stage: Stage,
  run: icechunk_utils.Run,
  save: SaveConfig,
  *,
  label: str,
  metadata: Mapping[str, Any],
) -> None:
  """The groups of `stage`, opened lazily in Dask chunks of their copies, written into `run`'s
  repository in one commit (`write_tree`), with their encodings and attributes as computed.

  The copy runs in this process, on local threads, at most `stage.threads`: Dask's distributed
  schedulers have no such setting, and every copy task holds a whole stored unit."""
  groups = {}
  for group, (attrs, variables) in stage.restored.items():
    copy = xr.open_zarr(stage.path, group=group, consolidated=False, chunks=None)
    copy = copy.assign(
      {
        name: var.chunk(
          dict(zip(var.dims, stage.copies[group][name] or var.encoding["chunks"], strict=True))
        )
        for name, var in copy.data_vars.items()
      }
    )
    for name, (var_attrs, encoding) in variables.items():
      copy[name].attrs = dict(var_attrs)
      copy[name].encoding = dict(encoding)
    copy.attrs = dict(attrs)
    groups[group] = copy
  local: dict[str, Any] = {"scheduler": "threads"}
  if stage.threads is not None:
    logger.info(f"Copying with {stage.threads} threads (each holds a whole stored unit)")
    local["num_workers"] = stage.threads
  with dask.config.set(local):
    write_tree(groups, run, save, label=label, metadata=metadata)
