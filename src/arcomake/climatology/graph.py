# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The graph path (`--graph`, the default): the statistics of every variable in one Dask
graph, computed into a temporary store, then copied into the output.

1. Per-bin statistics (`_bin_statistics`): the time axis is folded into periods x bins (each
   step at its position on a grid of whole periods, `binning.time_bins`; missing steps become
   NaN), then reduced over the periods in two passes, the count and the mean, then the sum of
   squared deviations from the mean (`_fold_statistics`). Feb 29 is not on the grid: its
   statistics are combined into Feb 28's bin (`_combine`).
2. The window, chunk by chunk (`kernels.window_bins_last`, through `apply_ufunc`): the folded
   data are chunked once, one period and every bin per chunk (`_chunked_for_the_window`), so that
   neither step rechunks.
3. Both outputs are computed together into the temporary store, and copied (`staging`).

Dask holds the folded data of several years and levels at once: at 0.25 deg its memory grows
with the years (logbook, 21 Nov), where `streamed` stays flat.
"""

import functools
import logging
from collections.abc import Hashable
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
  count, mean, m2 = _bin_statistics(dataset, period, bins, time_dim)
  offsets, weights = binning.window(window, weighting, weighting_scale)

  results: tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] = ({}, {})
  for name, var in mean.data_vars.items():
    dtype = binning.output_dtype(dataset[name].dtype)
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


# Per bin: the number of values, their mean and their sum of squared deviations.
BinStatistics = tuple[xr.Dataset, xr.Dataset, xr.Dataset]


def _bin_statistics(
  dataset: xr.Dataset, period: int, bins: BinningType, time_dim: str
) -> BinStatistics:
  """Per bin: the number of values, their mean and their sum of squared deviations (NaN skipped).

  The bins are along `time_dim`, numbered 0 to `period - 1`.
  """
  _, is_feb_29, position = binning.time_bins(dataset[time_dim], period, bins)
  bin_coords = {time_dim: np.arange(period)}

  # Fold the time axis into (_period, bin): step k of a folded period is bin k. Missing steps (an
  # incomplete first or last year, gaps) become NaN, which the statistics skip. In turn: the
  # regular days, their positions as the time coordinate, every position of whole periods (NaN
  # where no step), and the time axis cut into periods (`construct` reshapes, no copy).
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
  """Statistics of the union of two samples (Chan et al.); an empty sample has mean NaN.

  n = n_a + n_b; mean = (n_a mean_a + n_b mean_b) / n; m2 = m2_a + m2_b + (mean_b - mean_a)^2
  n_a n_b / n. Empty samples are taken with mean 0, so that they leave the other unchanged."""
  n_a, mean_a, m2_a = a
  n_b, mean_b, m2_b = b
  n = n_a + n_b
  mean_a0, mean_b0 = mean_a.fillna(0.0), mean_b.fillna(0.0)
  mean = ((n_a * mean_a0 + n_b * mean_b0) / n).where(n > 0)
  m2 = m2_a + m2_b + ((mean_b0 - mean_a0) ** 2 * n_a * n_b / n).where(n > 0, 0.0)
  return n, mean, m2
