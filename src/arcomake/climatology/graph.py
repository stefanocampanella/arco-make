# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
#
#  The following implementation is the result of a number of experiments, mostly to overcome performance issues.
#  In most cases, (injudicious) re-chunking was the source of these issues.
#  Indeed, in that case Dask ended up creating huge task graphs (22GB), and jobs died, even with small timeseries.
#  In particular, the `groupby` in the computation of the climatology is sensitive to the chunking of the dataset.
#  Early on optimizations reusing intermediate results[1] made things worse, making the task graph even bigger and more
#  complex, and (un)surprisingly turned out to be detrimental.
#
#  I tried the following:
#    1. Compute one stat (climatology, mean, std, diff) at a time; which is the current approach.
#    2. Adjust chunking when reading from disk, and avoid ZipStore.
#    3. Investigate the use of flox, and in general optimizations related to groupby operations.
#
# Regarding the latter, see, for example:
#    1. https://discourse.pangeo.io/t/optimizing-climatology-calculation-with-xarray-and-dask/2453
#    2. https://flox.readthedocs.io/en/latest/user-stories/climatology.html
#    3. https://xarray.dev/blog/flox
#
#  With these and other optimizations, now we're able to get the job to the end without errors and compute all the
#  stats, including the climatology. The climatology uses vectorized reductions over periods (a streaming Welford
#  implementation was removed in 9dcb661); flox was 4x slower than folding the time axis (logbook, 22 Nov).
#
#  Even without rechunking, the communication is still a bottleneck. Hence, further optimization might be:
#   1. Try to set up a Dask cluster using UCX (which appears to be experimental) to reduce communication time.
#
# [1]: mean could be computed, using a reasonable approximation, as the mean of the climatology, and the std could reuse
# the value of the mean.
#
# Say you have a collection of values {x_i} and labels {l_i} so that each label corresponds to multiple values.
# Then you can compute the mean of the whole collection x_mean, or the mean of the averages for each label x_clim_mean.
# If the number of values for each label is the same, then the two quantities are strictly equal.
# But it's not true in general.
# Indeed, in the case of a daily climatology, the two would differ because of leap years, which would introduce a
# relative error of the order of 1/365.
# Finally, the dataset might not start on the 1st of January or end before the 31 of December,
# which would further distort the results. However, the quantities computed here are aimed at standardizing the input
# features in a deep-learning model, and therefore such approximations are reasonably acceptable.
"""The graph path (`--graph`, the default): the statistics of every variable in one Dask
graph, computed into a temporary store, then copied into the output."""

import functools
import logging
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from arcomake import icechunk_utils, provenance
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
  """Everything in one Dask graph (`_compute_climatology`), computed into a temporary store and
  copied into the output's groups (`staging.stage`, `staging.copy`)."""
  # Opened first: a version of `main` promoted while computing is not replaced.
  run = icechunk_utils.Run(
    output_path,
    configs.save.storage_options,
    label,
    # `split_dims` is ignored here.
    recipe=provenance.fingerprint(configs, exclude={"read", "split_dims"}),
  )
  climatology, anomaly_var = _compute_climatology(
    dataset,
    time_dim=configs.time_dim,
    period=configs.period,
    window=configs.window,
    weighting=configs.weighting,
    weighting_scale=configs.weighting_scale,
    bins=configs.bins,
  )
  # Every coordinate without time, as streaming's template has them: the reductions drop those
  # on dimensions of no output variable (cell bounds, along their `bnds` dimension).
  static = {
    name: coord for name, coord in dataset.coords.items() if configs.time_dim not in coord.dims
  }
  climatology, anomaly_var = (
    result.assign_coords({name: c for name, c in static.items() if name not in result.coords})
    for result in (climatology, anomaly_var)
  )
  if configs.postprocess_climatology:
    climatology = climatology.arcomake.process(steps=configs.postprocess_climatology)
  anomaly_std = np.sqrt(anomaly_var.clip(min=0.0))
  assert isinstance(anomaly_std, xr.Dataset)
  if configs.postprocess_anomaly_std:
    anomaly_std = anomaly_std.arcomake.process(steps=configs.postprocess_anomaly_std)
  # Computed once into a temporary store (removed with the registry; both groups at once, so
  # what they share is computed once), then copied.
  with TempDirectoryRegistry():
    stage = staging.stage(
      dict(zip(GROUPS, (climatology, anomaly_std), strict=True)), configs.save, configs.time_dim
    )
    stage.write()
    staging.copy(stage, run, configs.save, label=label, metadata={METHOD_KEY: "graph"})
  logger.info("Climatology and anomaly standard deviation committed.")


def balanced_tree_reduce[T](
  func: Callable[[T, T], T],
  seq: Sequence[T],
  start: int = 0,
  end: int | None = None,
) -> T:
  """Divide-and-conquer reduction using index bounds to avoid intermediate slices."""
  if end is None:
    end = len(seq)
  count = end - start
  if count == 0:
    raise ValueError("Cannot reduce empty range")
  if count == 1:
    return seq[start]

  mid = start + count // 2
  return func(
    balanced_tree_reduce(func, seq, start, mid),
    balanced_tree_reduce(func, seq, mid, end),
  )


BinStatistics = tuple[xr.Dataset, xr.Dataset, xr.Dataset]


def _fold_statistics(folded: xr.Dataset) -> BinStatistics:
  """Count, mean and sum of squared deviations over `_period`, skipping NaN (0 without values).

  Two passes, each level from its own chunks. They were tied with `bind` to a checkpoint of
  every chunk's mean, which kept Dask from reusing the folded data (read again from the input)
  but made every level wait for all levels' means: the statistics of all levels were computed
  first and held until their windows. Dask's one-pass variance was slower and needed more memory
  over several years (its per-chunk intermediates; logbook, 21 Nov)."""
  count = folded.count("_period")
  mean = folded.mean("_period", skipna=True)
  m2 = ((folded - mean) ** 2).sum("_period", skipna=True)
  return count, mean, m2


def _combine(a: BinStatistics, b: BinStatistics) -> BinStatistics:
  """Statistics of the union of two samples (Chan et al.); an empty sample has mean NaN."""
  n_a, mean_a, m2_a = a
  n_b, mean_b, m2_b = b
  n = n_a + n_b
  mean_a0, mean_b0 = mean_a.fillna(0.0), mean_b.fillna(0.0)
  mean = ((n_a * mean_a0 + n_b * mean_b0) / n).where(n > 0)
  m2 = m2_a + m2_b + ((mean_b0 - mean_a0) ** 2 * n_a * n_b / n).where(n > 0, 0.0)
  return n, mean, m2


def _bin_statistics(
  dataset: xr.Dataset, period: int, bins: BinningType, time_dim: str
) -> BinStatistics:
  """Per bin: the number of values, their mean and their sum of squared deviations (NaN skipped).

  The bins are along `time_dim`, numbered 0 to `period - 1`.
  """
  size = dataset.sizes[time_dim]
  if bins == "day_of_year":
    _, is_feb_29, position = binning.day_of_year_positions(dataset[time_dim], period)
  else:
    if size % period != 0:
      raise ValueError(
        f"position bins need a whole number of periods ({size} steps, period {period}): leap"
        ' days and partial years shift every later bin; use bins = "day_of_year" for daily data'
      )
    is_feb_29, position = np.zeros(size, dtype=bool), np.arange(size)
  bin_coords = {time_dim: np.arange(period)}

  # Fold the time axis into (_period, bin): step k of a folded period is bin k. Missing steps (an
  # incomplete first or last year, gaps) become NaN, which the statistics skip.
  regular = ~is_feb_29
  n_periods = -(-int(position.max(initial=0) + 1) // period)
  folded = (
    dataset.isel({time_dim: regular})
    .assign_coords({time_dim: position[regular]})
    .reindex({time_dim: np.arange(n_periods * period)})
    .coarsen({time_dim: period}, boundary="exact")
    .construct({time_dim: ("_period", time_dim)})
    .assign_coords(bin_coords)
  )
  folded = _chunked_for_the_window(folded, time_dim)
  count, mean, m2 = _fold_statistics(folded)
  if is_feb_29.any():
    leap = dataset.isel({time_dim: is_feb_29}).rename({time_dim: "_period"})
    if folded.chunks:
      # In the tiles of the other days, or combining them would cut every bin's statistics into
      # the intersections of both tilings.
      leap = leap.chunk(
        {dim: folded.chunksizes[dim] for dim in leap.dims if dim not in ("_period", time_dim)}
      )
    feb_28 = (
      count.isel({time_dim: binning.FEB_29_BIN}),
      mean.isel({time_dim: binning.FEB_29_BIN}),
      m2.isel({time_dim: binning.FEB_29_BIN}),
    )
    merged = _combine(feb_28, _fold_statistics(leap))
    is_feb_28 = xr.DataArray(
      np.arange(period) == binning.FEB_29_BIN, dims=[time_dim], coords=bin_coords
    )
    count, mean, m2 = (
      xr.where(is_feb_28, new, old) for old, new in zip((count, mean, m2), merged, strict=True)
    )
  assert isinstance(count, xr.Dataset) and isinstance(mean, xr.Dataset)
  assert isinstance(m2, xr.Dataset)
  return count, mean, m2


def _chunked_for_the_window(folded: xr.Dataset, time_dim: str) -> xr.Dataset:
  """`folded` (periods x bins x ...) chunked once, before the statistics: one period per chunk and
  every bin, so that the reduction over periods and the window over bins need no rechunk; the
  other dimensions in "auto" tiles under Dask's chunk size, those split one index per chunk kept
  so. Left to Dask, the window's rechunk to whole bins split the grid into small tiles and built
  an all-to-all graph (747,727 tasks for one 3D variable at 1 deg; 26 GiB for three variables of
  8 days)."""
  # Per variable: variables may be chunked differently (`Dataset.chunks` would raise).
  sizes = [var.chunksizes for var in folded.data_vars.values() if var.chunks is not None]
  if not sizes:
    return folded
  chunks: dict[Hashable, Any] = {"_period": 1, time_dim: -1}
  for dim in {dim for var_sizes in sizes for dim in var_sizes}:
    if dim not in chunks:
      split = all(set(var_sizes[dim]) == {1} for var_sizes in sizes if dim in var_sizes)
      chunks[dim] = 1 if split else "auto"
  # "auto" tiles follow each variable's data type: unified, all variables share them.
  return folded.chunk(chunks).unify_chunks()


def _compute_climatology(
  dataset: xr.Dataset,
  period: int,
  window: int = 1,
  weighting: WeightingType = "constant",
  weighting_scale: float | None = None,
  time_dim: str = "time",
  bins: BinningType = "position",
) -> tuple[xr.Dataset, xr.Dataset]:
  """
  Compute the climatological mean and unbiased sample variance of every data variable.

  Bin b pools the values of bins b + k for k in a centred `window` (wrapping around the
  period), each weighted by the weight of its offset k. Missing values (NaN, missing steps,
  incomplete periods) are skipped: each bin counts only the values it has (B6).

  Parameters
  ----------
  dataset : xr.Dataset
      Input Dataset, sorted along the time dimension.
  period : int
      Number of bins (e.g. 365 daily bins).
  window : int, default 1
      Number of bins of the (centred) window; 1 gives the plain per-bin statistics.
  weighting : {"constant", "gaussian", "geometric"}, default "constant"
      Weighting of the window's bins (see ``binning.window_weights``).
  weighting_scale : float or None, default None
      Scale of the weighting kernel, in bins (see ``binning.window_weights``).
  time_dim : str, default "time"
      Name of the time dimension.
  bins : {"position", "day_of_year"}, default "position"
      ``"position"``: step k of each run of `period` steps is in bin k. ``"day_of_year"``: daily
      data; the bin is the day of a 365-day year, and Feb 29 is in Feb 28's bin.

  Returns
  -------
  tuple[xr.Dataset, xr.Dataset]
      ``(mean, variance)``, each with `period` bins along ``time_dim`` (numbered from 0) and the
      input's data types. ``variance`` is the unbiased (reliability weights) variance, NaN where
      the window holds fewer than two values.
  """
  if bins == "day_of_year" and period != 365:
    raise ValueError(f"day_of_year bins need period = 365, got {period}")
  count, mean, m2 = _bin_statistics(dataset, period, bins, time_dim)
  half = window // 2
  offsets = list(range(-half, window - half))
  weights = binning.window_weights(offsets, weighting, scale=weighting_scale)

  results: tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] = ({}, {})
  for name, var in mean.data_vars.items():
    dtype = dataset[name].dtype if np.issubdtype(dataset[name].dtype, np.floating) else np.float64
    # One implementation of the window for both paths, chunk by chunk (bins last, as apply_ufunc
    # puts the core dimension), in the output's type.
    statistics = xr.apply_ufunc(
      functools.partial(kernels.window_bins_last, offsets=offsets, weights=weights, dtype=dtype),
      count[name],
      var,
      m2[name],
      input_core_dims=[[time_dim]] * 3,
      output_core_dims=[[time_dim]] * 2,
      dask="parallelized",
      output_dtypes=[dtype, dtype],
      dask_gufunc_kwargs={"allow_rechunk": True},
    )
    for out, value in zip(results, statistics, strict=True):
      out[str(name)] = value.transpose(*var.dims)
  return xr.Dataset(results[0]), xr.Dataset(results[1])
