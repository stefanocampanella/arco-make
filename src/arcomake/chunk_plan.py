# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Chunk sizes planned from shapes, the needs of later steps and a memory budget.

Download recipes give no chunk sizes: each part of a block is chunked by `plan`, from the shapes
and dtypes of its variables in the block, what the dataset's steps need along each dimension
(`Needs`, declared by the steps) and the memory each task may use (`Budget`). Writes align the
planned chunks to the stored layout (xarray's `align_chunks`). Everything here is a pure function
of its arguments, except `detect_budget`.
"""

import dataclasses
import math
import os
from collections.abc import Hashable, Iterable, Mapping

import dask
import dask.system
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
    threads: threads computing at once.
    source: where `memory_limit` comes from, for the log.
  """

  memory_limit: int
  threads: int
  source: str = "given"

  @property
  def usable(self) -> int:
    """Bytes the tasks may use, all threads together (`USABLE` of the limit)."""
    return int(self.memory_limit * USABLE)

  @property
  def per_task(self) -> int:
    """Bytes of input chunk a task may hold (`COPIES` of it are held while it runs)."""
    return self.usable // (self.threads * COPIES)

  @property
  def target(self) -> int:
    """Bytes of the chunks planned: `per_task`, or Dask's `array.chunk-size` if smaller (larger
    chunks only make fewer, longer tasks)."""
    return min(self.per_task, parse_bytes(dask.config.get("array.chunk-size")))

  def describe(self) -> str:
    return (
      f"target {format_bytes(self.target)} per chunk, at most {format_bytes(self.per_task)} per"
      f" task: {format_bytes(self.memory_limit)} ({self.source}) x {USABLE} /"
      f" ({self.threads} threads x {COPIES})"
    )


def dask_threads() -> int:
  """Threads of Dask's local scheduler: one for the synchronous one, else `num_workers` or the
  CPUs this process may use."""
  if dask.config.get("scheduler", None) in ("synchronous", "sync", "single-threaded"):
    return 1
  return int(dask.config.get("num_workers", None) or dask.system.CPU_COUNT)


def detect_budget(memory_limit: int | None = None) -> Budget:
  """The budget of this process: `memory_limit` if given, else `ARCOMAKE_MEMORY_LIMIT` if set
  (the settings file exports it), else the memory it may use (the least of physical memory, its
  cgroup's limit, so a Slurm job's, and its RLIMIT_RSS), shared among the threads of Dask's local
  scheduler. Only `download` passed its `--memory-limit`: the climatology, sizing its series and
  copy threads, ignored the variable."""
  if memory_limit is not None:
    return Budget(memory_limit, dask_threads(), "given")
  from arcomake.settings import parse_memory

  if variable := os.environ.get("ARCOMAKE_MEMORY_LIMIT", "").strip():
    return Budget(parse_memory(variable), dask_threads(), "ARCOMAKE_MEMORY_LIMIT")
  from distributed.system import MEMORY_LIMIT

  return Budget(MEMORY_LIMIT, dask_threads(), "detected")


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
  whole: Iterable[Hashable] = (),
  preferred: int | None = None,
) -> int:
  """The largest multiple of `unit` along `dim` (at least `unit`, at most `size`) whose chunks,
  the other dimensions as in `chunks` (`whole` ones whole), hold at most `limit` bytes; a
  multiple of `preferred` too (the source's chunk, read once by one task) if one fits."""
  per_step = _bytes(
    {**chunks, dim: 1}, {n: v for n, v in variables.items() if dim in v.dims}, whole
  )[0]
  steps = size if per_step == 0 else limit // per_step
  if preferred is not None:
    aligned = min(size, math.lcm(unit, preferred))
    if aligned <= steps:
      return min(size, steps // aligned * aligned)
  return max(unit, min(size, steps // unit * unit))


def plan(
  variables: Mapping[Hashable, Shape],
  budget: Budget,
  needs: Needs | None = None,
  grow_dim: Hashable | None = "time",
  preferred: Mapping[Hashable, int] | None = None,
) -> Plan:
  """Chunk sizes for `variables`, about `budget.target` bytes each.

  Every dimension starts whole, except `grow_dim`, which starts at its unit (the multiple its
  steps need, 1 without). Steps that gather dimensions hold them whole in each task: while such
  tasks are above `budget.per_task`, the other dimensions that need not be whole are split,
  outermost first. Then, while a chunk is above the target, so are the gathered ones. Then
  `grow_dim` grows, in multiples of its unit, up to the target (gathered tasks up to
  `budget.per_task`) or its length. Splits prefer multiples of the source's chunks
  (`preferred`), when they fit. Chunks, and gathering tasks, above `budget.per_task` are
  reported as warnings.

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
  source: dict[Hashable, int] = {
    dim: int(size) for dim, size in (preferred or {}).items() if dim in sizes and size > 0
  }
  units = {dim: min(max(sizes[dim], 1), needs.multiple.get(dim, 1)) for dim in sizes}
  chunks = dict(sizes)
  grow = grow_dim if grow_dim in sizes and grow_dim not in needs.whole else None
  if grow is not None:
    chunks[grow] = units[grow]
  target = budget.target

  def split(dims: Iterable[Hashable], limit: int, whole: frozenset[Hashable]) -> None:
    for dim in dims:
      if _bytes(chunks, variables, whole)[0] <= limit:
        return
      chunks[dim] = _fit(
        dim, limit, chunks, variables, units[dim], sizes[dim], whole, source.get(dim)
      )

  free = [dim for dim in order if dim != grow and dim not in needs.whole]
  if needs.gathered:
    split([dim for dim in free if dim not in needs.gathered], budget.per_task, needs.gathered)
  split(free, target, frozenset())
  if grow is not None and _bytes(chunks, variables)[0] <= target:
    size = _fit(grow, target, chunks, variables, units[grow], sizes[grow], (), source.get(grow))
    if needs.gathered:
      size = min(
        size,
        _fit(grow, budget.per_task, chunks, variables, units[grow], sizes[grow], needs.gathered),
      )
    chunks[grow] = size
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
    if size * COPIES > budget.usable:
      raise PlanError(
        f"{name}: a {what} of {format_bytes(size)} needs about {format_bytes(size * COPIES)}"
        f" ({COPIES} copies), more than the {format_bytes(budget.usable)} the tasks may use"
        f" ({budget.describe()}): give more memory (--memory-limit)"
      )
  return Plan(chunks, largest, gathered, tuple(warnings))


def stored_warnings(prepared: xr.Dataset, budget: Budget) -> list[str]:
  """What the stored layout of `prepared` (encodings set for writing) costs beyond the budget:
  writing a stored chunk (or shard) needs it whole in one task, whatever the planned chunks."""
  warnings = []
  for name, var in prepared.data_vars.items():
    stored = var.encoding.get("shards") or var.encoding.get("chunks")
    if stored is None:
      continue
    size = math.prod(min(int(s), n) for s, n in zip(stored, var.shape, strict=True))
    size *= var.dtype.itemsize
    if size > budget.per_task:
      kind = "shards" if var.encoding.get("shards") else "chunks"
      warnings.append(
        f"{name}: stored {kind} of {format_bytes(size)}, above {format_bytes(budget.per_task)} per"
        " task: each is written whole by one task"
      )
  return warnings
