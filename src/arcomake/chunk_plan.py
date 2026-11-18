# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Chunk sizes planned from shapes, the needs of later steps and a memory budget.

Download recipes give no chunk sizes: each part of a block is chunked by `plan`, from the shapes
and dtypes of its variables in the block, what the dataset's steps need along each dimension
(`Needs`, declared by the steps) and the memory each task may use (`Budget`). Before writing,
`align_to_stored` rechunks the dimensions whose Dask chunks straddle the stored chunks (or
shards). Everything here is a pure function of its arguments.
"""

import dataclasses
import math
from collections.abc import Hashable, Iterable, Mapping

import dask
import xarray as xr
from dask.utils import format_bytes, parse_bytes

# Share of the memory limit the planned tasks may use (the rest: the interpreter, libraries,
# connection pools, the written block's buffers).
USABLE = 0.8
# Memory a running task holds, in chunks of its input: the input, the output, temporaries, and
# finished results waiting for the tasks that use them. A heuristic, checked against measured
# peaks (logbook, group 8).
COPIES = 4


class PlanError(ValueError):
  """No chunking fits the memory budget."""


@dataclasses.dataclass(frozen=True)
class Needs:
  """What the steps of a dataset need along its dimensions.

  Attributes:
    whole: dimensions in one chunk (e.g. the horizontal dimensions of `regrid`).
    gathered: dimensions a step joins into one chunk itself (e.g. depth for `get_bottom`): its
      tasks hold them whole, so they count as whole when sizing, but the dataset keeps them split.
    multiple: chunk sizes along a dimension are multiples of this (e.g. the steps of one bin of
      `resample`, so that no bin straddles chunks).
  """

  whole: frozenset[Hashable] = frozenset()
  gathered: frozenset[Hashable] = frozenset()
  multiple: Mapping[Hashable, int] = dataclasses.field(default_factory=dict)

  def merge(self, other: "Needs") -> "Needs":
    multiple = dict(self.multiple)
    for dim, factor in other.multiple.items():
      multiple[dim] = math.lcm(multiple.get(dim, 1), factor)
    return Needs(self.whole | other.whole, self.gathered | other.gathered, multiple)

  def restricted_to(self, dims: Iterable[Hashable]) -> "Needs":
    """These needs on `dims` only (needs on dimensions a dataset lacks do not apply to it)."""
    dims = set(dims)
    return Needs(
      self.whole & dims,
      self.gathered & dims,
      {dim: factor for dim, factor in self.multiple.items() if dim in dims},
    )


@dataclasses.dataclass(frozen=True)
class Budget:
  """The memory planned tasks may use.

  Attributes:
    memory_limit: bytes of memory of the process (see `detect_budget`).
    threads: threads computing at once for one block.
    blocks: blocks computed at once (`download --blocks`): each has its own threads.
    source: where `memory_limit` comes from, for the log.
  """

  memory_limit: int
  threads: int
  blocks: int = 1
  source: str = "given"

  @property
  def per_block(self) -> int:
    """Bytes for the tasks of one block."""
    return int(self.memory_limit * USABLE / self.blocks)

  @property
  def per_task(self) -> int:
    """Bytes of input chunk a task may hold (`COPIES` of it are held while it runs)."""
    return self.per_block // (self.threads * COPIES)

  @property
  def target(self) -> int:
    """Bytes of the chunks planned: `per_task`, or Dask's `array.chunk-size` if smaller (larger
    chunks only make fewer, longer tasks)."""
    return min(self.per_task, parse_bytes(dask.config.get("array.chunk-size")))

  def describe(self) -> str:
    return (
      f"target {format_bytes(self.target)} per chunk, at most {format_bytes(self.per_task)} per"
      f" task: {format_bytes(self.memory_limit)} ({self.source}) x {USABLE} /"
      f" ({self.threads} threads x {self.blocks} block(s) x {COPIES})"
    )


@dataclasses.dataclass(frozen=True)
class Shape:
  """Dimensions, sizes and bytes per element of one variable."""

  dims: tuple[Hashable, ...]
  shape: tuple[int, ...]
  itemsize: int


def shapes(dataset: xr.Dataset) -> dict[Hashable, Shape]:
  """The `Shape` of every data variable of `dataset`."""
  return {
    name: Shape(tuple(var.dims), tuple(var.shape), var.dtype.itemsize)
    for name, var in dataset.data_vars.items()
  }


@dataclasses.dataclass(frozen=True)
class Plan:
  """Chunk sizes per dimension, with the largest chunk and the largest task they make.

  Attributes:
    chunks: size per dimension (whole dimensions included).
    largest: bytes of the largest chunk, and its variable.
    gathered: bytes of the largest task of a step that gathers dimensions, and its variable
      (None without such steps).
    warnings: what does not fit the budget.
  """

  chunks: dict[Hashable, int]
  largest: tuple[int, Hashable | None]
  gathered: tuple[int, Hashable | None] | None
  warnings: tuple[str, ...]

  def describe(self) -> str:
    chunks = ", ".join(f"{dim}: {size}" for dim, size in self.chunks.items())
    text = f"chunks {{{chunks}}}, largest {format_bytes(self.largest[0])} ({self.largest[1]})"
    if self.gathered is not None:
      text += f", gathered tasks {format_bytes(self.gathered[0])} ({self.gathered[1]})"
    return text


def _bytes(
  chunks: Mapping[Hashable, int],
  variables: Mapping[Hashable, Shape],
  whole: Iterable[Hashable] = (),
) -> tuple[int, Hashable | None]:
  """Bytes of the largest chunk among `variables`, with `whole` dimensions whole, and its name."""
  whole = set(whole)
  largest: tuple[int, Hashable | None] = (0, None)
  for name, var in variables.items():
    size = var.itemsize
    for dim, length in zip(var.dims, var.shape, strict=True):
      size *= length if dim in whole else min(chunks[dim], length)
    if size > largest[0]:
      largest = (size, name)
  return largest


def _fit(
  dim: Hashable,
  limit: int,
  chunks: Mapping[Hashable, int],
  variables: Mapping[Hashable, Shape],
  unit: int,
  size: int,
) -> int:
  """The largest multiple of `unit` along `dim` (at least `unit`, at most `size`) whose chunks,
  the other dimensions as in `chunks`, hold at most `limit` bytes."""
  per_step = _bytes({**chunks, dim: 1}, {n: v for n, v in variables.items() if dim in v.dims})
  steps = size if per_step[0] == 0 else limit // per_step[0]
  return max(unit, min(size, steps // unit * unit))


def plan(
  variables: Mapping[Hashable, Shape],
  budget: Budget,
  needs: Needs | None = None,
  grow_dim: Hashable | None = "time",
) -> Plan:
  """Chunk sizes for `variables`, about `budget.target` bytes each.

  Every dimension starts whole, except `grow_dim`, which starts at its unit (the multiple its
  steps need, 1 without). If a chunk is still above the target, the dimensions that need not be
  whole are split, outermost first, to multiples of their unit. Then `grow_dim` grows, in
  multiples of its unit, up to the target or its length. Chunks, and the tasks of steps that
  gather dimensions (those dimensions whole), above `budget.per_task` are reported as warnings.

  Raises:
    PlanError: if even chunks of one unit along every dimension that may be split leave no room
      for one task of a block (`COPIES` chunks within the block's memory).
  """
  sizes: dict[Hashable, int] = {}
  order: list[Hashable] = []  # dimensions, outermost first, from the largest variables
  for var in sorted(variables.values(), key=lambda v: -math.prod(v.shape) * v.itemsize):
    for dim, length in zip(var.dims, var.shape, strict=True):
      sizes[dim] = max(sizes.get(dim, 0), length)
      if dim not in order:
        order.append(dim)
  needs = (needs or Needs()).restricted_to(sizes)
  units = {dim: min(max(sizes[dim], 1), needs.multiple.get(dim, 1)) for dim in sizes}
  chunks = dict(sizes)
  grow = grow_dim if grow_dim in sizes and grow_dim not in needs.whole else None
  if grow is not None:
    chunks[grow] = units[grow]
  target = budget.target
  for dim in order:
    if _bytes(chunks, variables)[0] <= target:
      break
    if dim == grow or dim in needs.whole:
      continue
    chunks[dim] = _fit(dim, target, chunks, variables, units[dim], sizes[dim])
  if grow is not None and _bytes(chunks, variables)[0] <= target:
    chunks[grow] = _fit(grow, target, chunks, variables, units[grow], sizes[grow])
  gathered = _bytes(chunks, variables, needs.gathered) if needs.gathered else None
  largest = _bytes(chunks, variables)
  warnings = []
  checks = [(largest, "chunk")]
  if gathered is not None:
    gathered_dims = ", ".join(sorted(map(str, needs.gathered)))
    checks.append((gathered, f"task gathering {gathered_dims}"))
  for (size, name), what in checks:
    if size > budget.per_task:
      warnings.append(
        f"{name}: {what} of {format_bytes(size)}, above {format_bytes(budget.per_task)} per task"
      )
    if size * COPIES > budget.per_block:
      raise PlanError(
        f"{name}: a {what} of {format_bytes(size)} needs about {format_bytes(size * COPIES)}"
        f" ({COPIES} copies), more than the {format_bytes(budget.per_block)} a block may use"
        f" ({budget.describe()}): give more memory (--memory-limit), or fewer blocks at once"
      )
  return Plan(chunks, largest, gathered, tuple(warnings))


def _straddles(chunks: tuple[int, ...], unit: int) -> bool:
  """Whether Dask `chunks` along a dimension split stored chunks of `unit` (all but the last
  must be multiples of it)."""
  return any(size % unit for size in chunks[:-1])


def align_to_stored(dataset: xr.Dataset, budget: Budget) -> xr.Dataset:
  """`dataset` with its variables rechunked where their Dask chunks straddle the stored chunks
  (or shards) their encodings give, as Zarr writes require; other dimensions keep their chunks.

  Along such a dimension a variable gets the largest multiple of its stored size within
  `budget.target`, its other dimensions as they are. Variables are aligned one by one: merged
  datasets may chunk a dimension differently.
  """
  updates = {}
  for name, var in dataset.data_vars.items():
    stored = var.encoding.get("shards") or var.encoding.get("chunks")
    if var.chunks is None or stored is None:
      continue
    chunks = {dim: max(sizes) for dim, sizes in zip(var.dims, var.chunks, strict=True)}
    shape = {name: Shape(tuple(var.dims), tuple(var.shape), var.dtype.itemsize)}
    rechunk: dict[Hashable, int] = {}
    for dim, unit, sizes in zip(var.dims, stored, var.chunks, strict=True):
      unit = min(int(unit), var.sizes[dim])
      if _straddles(tuple(sizes), unit):
        rechunk[dim] = _fit(dim, budget.target, chunks, shape, unit, var.sizes[dim])
        chunks[dim] = rechunk[dim]
    if rechunk:
      updates[name] = var.chunk(rechunk)
  return dataset.assign(updates) if updates else dataset
