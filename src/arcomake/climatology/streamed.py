# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The streaming path (`--streaming`): every series (variable, index of `split_dims`)
computed apart in memory, as many at a time as fit, written into its region of a
temporary store, then copied into the output.

`_stream` writes the metadata and coordinates of both groups into the temporary store first,
from a lazy template of the outputs (never computed). A pool of threads then computes the series
(`_series`), as many at once as fit in the memory budget (`_series_at_once`, `_series_bytes`).
Each series (`_compute_series`):

1. reads its values in blocks of about 1 GB along time, the next block while the current one is
   added, and adds them to its per-bin count, mean and m2 by Welford's update in float64
   (`_stream_bin_statistics`, `kernels.welford`);
2. applies the window, in bands of points in parallel (`_window_statistics_in_memory`,
   `kernels.window_bins_last`, the graph path's kernel);
3. writes both outputs into its region of the temporary store, where it is alone: each chunk of
   the store holds one index of the split dimensions.

Everything runs in this process, on local threads, whatever `--dask-setup` says: the results
live here. Then the store is copied into the output (`staging.copy`).
"""

import itertools
import logging
import math
import os
import time
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import dask
import dask.array
import numpy as np
import numpy.typing as npt
import xarray as xr

from arcomake import icechunk_utils, provenance
from arcomake.chunk_plan import Budget, detect_budget
from arcomake.climatology import binning, kernels, staging
from arcomake.climatology.config import (
  GROUPS,
  METHOD_KEY,
  BinningType,
  ClimatologyConfig,
  WeightingType,
)
from arcomake.temporary_utils import TempDirectoryRegistry

logger = logging.getLogger(__name__)


def compute(
  dataset: xr.Dataset, configs: ClimatologyConfig, output_path: str, *, label: str
) -> None:
  """`_stream` in a temporary directory registry (its store is removed at the end)."""
  with TempDirectoryRegistry():
    _stream(dataset, configs, output_path, label=label)


def _stream(
  dataset: xr.Dataset, configs: ClimatologyConfig, output_path: str, *, label: str
) -> None:
  """Every series (variable, index of `split_dims`) apart, written into its region of a
  temporary store, then copied into the output with the layout of `[save]` (`staging.stage`,
  `staging.copy`).

  The groups' metadata and coordinates are written first, from a template with the postprocess
  steps applied; then every series is computed in memory (`_compute_series`), postprocessed and
  written into its region, as many at a time as fit in memory (`_series_at_once`). Postprocess
  steps therefore have to work on one series (attributes, renames, chunking); a series gets every
  coordinate of the template at its index, so steps reading coordinates see what they see on the
  graph path.
  """
  steps_of = dict(
    zip(GROUPS, (configs.postprocess_climatology, configs.postprocess_anomaly_std), strict=True)
  )
  split_dims = configs.split_dims or []
  unknown = [d for d in split_dims if d not in dataset.dims]
  if unknown:
    raise ValueError(f"split_dims {unknown} are not dimensions of the input: {list(dataset.dims)}")
  # Opened first: a version of `main` promoted while computing is not replaced.
  run = icechunk_utils.Run(
    output_path,
    configs.save.storage_options,
    label,
    recipe=provenance.fingerprint(configs, exclude={"read"}),
  )
  template = _output_template(dataset, configs.period, configs.time_dim)
  groups = {
    group: template.arcomake.process(steps=steps) if steps else template
    for group, steps in steps_of.items()
  }
  # One index of the split dimensions per chunk of the store (after the postprocess steps, which
  # may chunk): each series writes its region alone. Coordinates are written with the metadata
  # (in memory, not chunked), so that series computed at once never write the same chunk.
  # Everything runs in this process.
  groups = {
    group: dataset.chunk({d: 1 for d in split_dims if d in dataset.dims}).assign_coords(
      {name: coord.compute() for name, coord in dataset.coords.items()}
    )
    for group, dataset in groups.items()
  }
  stage = staging.stage(groups, configs.save, configs.time_dim, workers=False)
  stage.write(compute=False)  # metadata only: the template's data are never computed
  # Results are chunked as the store, so that Dask writes the chunks of a region in parallel.
  stored_chunks = {
    group: {name: var.encoding["chunks"] for name, var in staged.data_vars.items()}
    for group, staged in stage.staged.items()
  }
  # The template's coordinates (time aside: a series' bins are its own), for the steps.
  coordinates = template.coords.to_dataset().drop_vars(configs.time_dim)

  def write(results: tuple[xr.Dataset, xr.Dataset], index: dict[str, int], label: str) -> None:
    writing = time.monotonic()
    # Explicit regions: "auto" maps a dimension without coordinate to index 0.
    region = {configs.time_dim: slice(0, configs.period)} | {
      d: slice(i, i + 1) for d, i in index.items()
    }
    # Every coordinate of the template at the series' index, as the steps see on the graph path:
    # a series carries only its variable's own (`coverage_attrs` found no `depth` on a 2D
    # variable's series). Dropped again before writing.
    at = {d: slice(i, i + 1) for d, i in index.items() if d in coordinates.dims}
    around = coordinates.isel(at).coords
    for (group, steps), result in zip(steps_of.items(), results, strict=True):
      if steps:
        missing = {name: coord for name, coord in around.items() if name not in result.coords}
        result = result.assign_coords(missing).arcomake.process(steps=steps)
      chunks = stored_chunks[group]
      result = (
        result.drop_attrs()
        .drop_encoding()
        .chunk(
          {
            dim: size
            for name in result.data_vars
            for dim, size in zip(result[name].dims, chunks[str(name)], strict=True)
          }
        )
      )
      result = result.drop_vars(list(result.coords))  # written with the metadata
      result.to_zarr(
        stage.path, group=group, region=region, mode="r+", consolidated=False, zarr_format=3
      )
    logger.info(f"{label} written in {time.monotonic() - writing:.0f} s")

  def compute_and_write(name: str, index: dict[str, int]) -> None:
    piece = dataset[[name]].isel({d: slice(i, i + 1) for d, i in index.items()})
    climatology, variance = _compute_series(
      piece,
      period=configs.period,
      window=configs.window,
      weighting=configs.weighting,
      weighting_scale=configs.weighting_scale,
      time_dim=configs.time_dim,
      bins=configs.bins,
    )
    anomaly_std = np.sqrt(variance.clip(min=0.0))
    assert isinstance(anomaly_std, xr.Dataset)
    write((climatology, anomaly_std), index, f"{name} {index}")

  series = _series(dataset, split_dims)
  at_once = _series_at_once(dataset, series, configs.period, configs.time_dim, detect_budget())
  started = time.monotonic()
  # Reads and writes use local threads whatever --dask-setup says: results live in this process.
  # Each worker computes a series and writes it; series run apart, so a series' accumulators are
  # freed before the next one of its worker starts.
  # Dask's threads mostly wait for reads (S3 latency): each series keeps its own pool of them.
  pool = ThreadPoolExecutor(max_workers=at_once)
  try:
    with dask.config.set(scheduler="threads"):
      futures = {
        pool.submit(compute_and_write, name, index): (name, index) for name, index in series
      }
      for done, future in enumerate(as_completed(futures), start=1):
        future.result()  # the first error ends the run
        name, index = futures[future]
        elapsed = time.monotonic() - started
        logger.info(
          f"Series {done}/{len(series)} ({name} {index}) written; {elapsed / done:.0f} s per"
          f" series, {elapsed / done * (len(series) - done) / 60:.0f} min left"
        )
  finally:
    # On an error or an interrupt, series not started are cancelled; running ones finish.
    pool.shutdown(wait=True, cancel_futures=True)
  logger.info(f"{len(series)} series computed; copying into {output_path}")
  staging.copy(stage, run, configs.save, label=label, metadata={METHOD_KEY: "streaming"})
  logger.info("Climatology writes completed, series by series.")


def _series(dataset: xr.Dataset, split_dims: list[str]) -> list[tuple[str, dict[str, int]]]:
  """(variable, {dim: index}) for every series: each variable, each index of its split dims."""
  series = []
  for name, var in dataset.data_vars.items():
    dims = [d for d in split_dims if d in var.dims]
    for index in itertools.product(*(range(var.sizes[d]) for d in dims)):
      series.append((str(name), dict(zip(dims, index, strict=True))))
  return series


def _series_at_once(
  dataset: xr.Dataset,
  series: Sequence[tuple[str, dict[str, int]]],
  period: int,
  time_dim: str,
  budget: Budget,
) -> int:
  """Series computed at a time: as many of the largest as fit in the budget's memory
  (`_series_bytes`), at least one, at most the series and the budget's threads.

  Each series reads on its own, so more series keep more reads in flight; their Welford updates
  take turns (`kernels.welford`)."""
  largest = max(
    (_series_bytes(dataset[name], index, period, time_dim) for name, index in series), default=1
  )
  at_once = max(1, min(len(series), budget.threads, budget.usable // largest))
  logger.info(
    f"{len(series)} series, {at_once} at a time (up to {largest / 2**30:.1f} GiB each;"
    f" {budget.usable / 2**30:.1f} GiB usable, {budget.source})"
  )
  return at_once


def _series_bytes(var: xr.DataArray, index: Mapping[str, int], period: int, time_dim: str) -> int:
  """Memory one series of `var` (at `index` of its split dimensions) holds at most while it is
  computed and written, the largest of its three phases:

  - reading: the per-bin count, mean and m2 (20 bytes per bin and point) and three blocks read
    (the one added, the next one being read, and Dask's pieces of it);
  - the window: the per-bin statistics, the climatology and the variance, and each thread's band
    (contiguous copies of its count, mean and m2, and its two results);
  - writing: four arrays of results (climatology, variance, its clip and square root).
  """
  piece = var.isel({d: slice(i, i + 1) for d, i in index.items()})
  points = math.prod(n for d, n in piece.sizes.items() if d != time_dim)
  out = binning.output_dtype(var.dtype).itemsize
  steps = min(_read_block(piece, time_dim), piece.sizes[time_dim])
  block = steps * points * var.dtype.itemsize
  statistics, result = period * points * 20, period * points * out
  bands = min(_window_threads(), -(-points // _WINDOW_BAND))
  window = bands * min(_WINDOW_BAND, points) * period * (20 + 2 * out)
  return max(statistics + 3 * block, statistics + 2 * result + window, 4 * result)


def _read_block(da: xr.DataArray, time_dim: str, target_bytes: float = 1e9) -> int:
  """Steps to read at once: about `target_bytes`, a whole number of the input's time chunks.

  Large reads keep many requests in flight (one 4 MB step at a time was latency-bound on S3);
  following the input's chunks (or shards) means no chunk is fetched twice.
  """
  step_bytes = max(
    1, da.dtype.itemsize * int(np.prod([da.sizes[d] for d in da.dims if d != time_dim]))
  )
  steps = max(1, int(target_bytes // step_bytes))
  axis = da.get_axis_num(time_dim)
  if da.chunks is not None:
    chunk = int(da.chunks[axis][0])
  else:  # stored chunks (or shards), as the zarr backend reports them
    preferred = da.encoding.get("preferred_chunks", {}).get(time_dim)
    stored = da.encoding.get("chunks")
    chunk = int(preferred or (stored[axis] if stored else 1))
  return max(chunk, steps // chunk * chunk)


def _output_template(dataset: xr.Dataset, period: int, time_dim: str) -> xr.Dataset:
  """The outputs' layout, lazily (never computed): `period` bins along time, other dims kept."""
  data_vars = {}
  for name, var in dataset.data_vars.items():
    shape = tuple(period if d == time_dim else var.sizes[d] for d in var.dims)
    dtype = binning.output_dtype(var.dtype)
    data_vars[name] = (var.dims, dask.array.zeros(shape, dtype=dtype, chunks=-1))
  coords = {
    name: coord.load() for name, coord in dataset.coords.items() if time_dim not in coord.dims
  }
  coords[time_dim] = np.arange(period)
  return xr.Dataset(data_vars, coords=coords)


def _compute_series(
  piece: xr.Dataset,
  period: int,
  window: int,
  weighting: WeightingType,
  weighting_scale: float | None,
  time_dim: str,
  bins: BinningType,
) -> tuple[xr.Dataset, xr.Dataset]:
  """Climatology and variance of one series, in memory: one read pass, then the window in NumPy."""
  offsets, weights = binning.window(window, weighting, weighting_scale)
  results: tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] = ({}, {})
  for name, var in piece.data_vars.items():
    order = [time_dim, *(d for d in var.dims if d != time_dim)]
    started = time.monotonic()
    count, mean, m2 = _stream_bin_statistics(var, period, bins, time_dim)
    read = time.monotonic()
    dtype = binning.output_dtype(var.dtype)
    statistics = _window_statistics_in_memory(count, mean, m2, offsets, weights, dtype=dtype)
    logger.info(
      f"{name}: read and binned in {read - started:.0f} s, window in {time.monotonic() - read:.0f} s"
    )
    coords: dict[Hashable, Any] = {d: var[d] for d in order[1:] if d in var.coords}
    coords[time_dim] = np.arange(period)
    for out, values in zip(results, statistics, strict=True):
      out[str(name)] = xr.DataArray(values, dims=order, coords=coords).transpose(*var.dims)
  climatology, variance = xr.Dataset(results[0]), xr.Dataset(results[1])
  assert isinstance(climatology, xr.Dataset) and isinstance(variance, xr.Dataset)
  return climatology, variance


def _stream_bin_statistics(
  da: xr.DataArray, period: int, bins: BinningType, time_dim: str, block: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Per-bin count, mean and sum of squared deviations of one series, streaming over time.

  Reads `block` steps at a time (by default about 1 GB, in whole chunks along time: see
  `_read_block`; the next block is read while this one is added) and updates
  the bins with Welford's algorithm, in float64. Returns arrays with the bins on axis 0 and the
  other dimensions of `da` (time removed) after it. Memory: about 20 bytes per bin and point.
  """
  da = da.transpose(time_dim, ...)
  size = da.sizes[time_dim]
  labels = binning.time_bins(da[time_dim], period, bins).labels
  shape = (period, *da.shape[1:])
  count = np.zeros(shape, dtype=np.int32)
  mean = np.zeros(shape, dtype=np.float64)
  m2 = np.zeros(shape, dtype=np.float64)

  def read(start: int) -> np.ndarray:
    return np.asarray(da.isel({time_dim: slice(start, start + steps)}).values)

  steps = _read_block(da, time_dim) if block is None else block
  starts = list(range(0, size, steps))
  with ThreadPoolExecutor(max_workers=1) as prefetch:
    pending = prefetch.submit(read, starts[0]) if starts else None
    for k, start in enumerate(starts):
      assert pending is not None
      values = pending.result()
      if k + 1 < len(starts):
        pending = prefetch.submit(read, starts[k + 1])
      kernels.welford(count, mean, m2, values, labels[start : start + len(values)])
  mean[count == 0] = np.nan
  return count, mean, m2


# Points per band of the window over a series' statistics, and the threads computing bands.
_WINDOW_BAND = 1 << 15


def _window_threads() -> int:
  return min(16, os.cpu_count() or 1)


def _window_statistics_in_memory(
  count: np.ndarray,
  mean: np.ndarray,
  m2: np.ndarray,
  offsets: list[int],
  weights: np.ndarray,
  workers: int | None = None,
  band: int = _WINDOW_BAND,
  dtype: npt.DTypeLike = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
  """The window step of `graph._compute_climatology` on in-memory per-bin statistics (bins on
  axis 0), in `dtype`.

  Same kernel as the lazy path, over bands of `band` points (all axes after the bins, flattened),
  in parallel, so that memory stays a few times the size of one band per thread.
  """
  shape = mean.shape
  count, mean, m2 = (a.reshape(shape[0], -1) for a in (count, mean, m2))
  climatology = np.empty(mean.shape, dtype=dtype)
  variance = np.empty(mean.shape, dtype=dtype)

  def window_band(start: int) -> None:
    part = (slice(None), slice(start, start + band))
    c, v = kernels.window_bins_last(
      count[part].T, mean[part].T, m2[part].T, offsets, weights, dtype
    )
    climatology[part] = c.T
    variance[part] = v.T

  # Bands are independent, and the kernel releases the GIL: compute them in parallel.
  with ThreadPoolExecutor(max_workers=workers or _window_threads()) as pool:
    list(pool.map(window_band, range(0, mean.shape[1], band)))
  return climatology.reshape(shape), variance.reshape(shape)
