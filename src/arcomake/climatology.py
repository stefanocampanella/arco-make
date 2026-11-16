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
#  implementation was removed in 9dcb661); flox can be used for the grouped reductions (`use_flox`).
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
import datetime
import logging
import operator
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import Literal, get_args

import click
import dask
import numpy as np
import xarray as xr
from dask.graph_manipulation import bind
from pydantic import BaseModel, ConfigDict, Field

from arcomake.cli_utils import (
  check_if_overwriting,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_utils import DaskSetupType, get_client
from arcomake.dataset_utils import ReadConfig, SaveConfig, open_input, safe_to_zarr
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)

# Weighting scheme applied to the steps of the (centered) window used to
# compute the climatology. The weights depend only on the distance from the
# center of the window and are therefore constant from one year to the next.
WeightingType = Literal["constant", "gaussian", "geometric"]
# How time steps fall into bins: by position in each period, or by day of a 365-day year.
BinningType = Literal["position", "day_of_year"]


class ReferencePeriod(BaseModel):
  """The right-open interval `[start, end)` of the input the statistics are computed over."""

  model_config = ConfigDict(extra="forbid")

  start: datetime.datetime
  end: datetime.datetime


class ClimatologyConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  time_dim: str = "time"
  period: int = Field(ge=1, description="Number of climatology bins")
  bins: BinningType = Field(
    default="position",
    description=(
      'How steps fall into bins: "position" (step k of each period of `period` steps is in bin k)'
      ' or "day_of_year" (daily data, 365 bins, Feb 29 in Feb 28\'s bin)'
    ),
  )
  reference_period: ReferencePeriod | None = Field(
    default=None, description="Compute the statistics over [start, end) only (default: all)"
  )
  window: int = Field(ge=1, description="Number of bins to average over")
  weighting: WeightingType
  weighting_scale: float | None = Field(
    default=None, ge=0.0, description="Scale of the weighting function in bins"
  )
  use_flox: bool = Field(
    default=False, description="Whether to use flox for grouped periodic reductions"
  )
  read: ReadConfig
  preprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_climatology: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_anomaly_std: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig


@click.command()
@click.argument(
  "configs_path",
  required=True,
  type=str,
  callback=check_path(dir_okay=False),
)
@click.argument(
  "input_path",
  required=True,
  type=str,
  callback=check_path(),
)
@click.argument(
  "climatology_output_path",
  required=True,
  type=str,
)
@click.argument(
  "anomaly_std_output_path",
  required=True,
  type=str,
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--dask-setup",
  default="mpi",
  type=click.Choice(
    get_args(DaskSetupType),
    case_sensitive=False,
  ),
  help="Type of Dask scheduler to use.",
)
@click.option(
  "--frisky",
  "use_frisky",
  default=False,
  is_flag=True,
  help="Hijack Dask client with frisky, this is an experimental feature.",
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
)
def compute_climatology(
  configs_path: str,
  input_path: str,
  climatology_output_path: str,
  anomaly_std_output_path: str,
  overwrite: bool = False,
  dask_setup: DaskSetupType = "mpi",
  use_frisky: bool = False,
  log_level: str = "info",
):
  """Compute a periodic climatology and the standard deviation of the anomalies.

  After the [[preprocess]] steps, and within `reference_period` if set, each time step falls in
  one of `period` bins: by position (`bins = "position"`: step k of every run of `period` steps
  is in bin k) or by day of the year (`bins = "day_of_year"`: daily data, 365 bins, Feb 29 in
  Feb 28's bin, later days of a leap year in the bin of their date). For each bin, the
  climatology is the weighted mean of the values of a centred `window` of bins (wrapping around
  the period), and the anomaly standard deviation is the unbiased weighted standard deviation of
  the same values. Missing values are skipped: each bin counts the values it has.

  Assumed, not checked: the time axis is sorted and every data variable has the time dimension.
  The two outputs have `period` steps along time, the bin index (0 is Jan 1 for day-of-year
  bins).

  Example::

    period = 365
    bins = "day_of_year"
    reference_period = {start = 1993-01-01, end = 2019-01-01}
    window = 29
    weighting = "gaussian"
    weighting_scale = 7.0
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  with get_client(dask_setup=dask_setup, use_frisky=use_frisky):
    try:
      # Read configs, then check the output paths (with the recipe's storage options).
      configs = read_configs(configs_path, schema=ClimatologyConfig)
      for path in (climatology_output_path, anomaly_std_output_path):
        check_if_overwriting(path, overwrite=overwrite, save_configs=configs.save)
      # Open input dataset
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with open_input(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Preprocess input dataset
        if configs.preprocess:
          dataset = dataset.arcomake.process(steps=configs.preprocess)
        # Statistics over the reference period only, if one is set.
        if configs.reference_period is not None:
          dataset = dataset.arcomake.time_sel(
            configs.reference_period.start, configs.reference_period.end, configs.time_dim
          )
          if dataset.sizes[configs.time_dim] == 0:
            raise ValueError(f"No data in the reference period {configs.reference_period}")
        # Compute the climatological mean and the unbiased sample variance
        # of the anomalies using vectorized reductions across periods and window smoothing.
        if not dataset.data_vars:
          raise ValueError("Input dataset contains no data variables.")
        climatology, anomaly_var = _compute_climatology(
          dataset,
          time_dim=configs.time_dim,
          period=configs.period,
          window=configs.window,
          weighting=configs.weighting,
          weighting_scale=configs.weighting_scale,
          use_flox=configs.use_flox,
          bins=configs.bins,
        )
        # Postprocess climatology and anomaly standard deviation.
        if configs.postprocess_climatology:
          climatology = climatology.arcomake.process(steps=configs.postprocess_climatology)
        anomaly_std = np.sqrt(anomaly_var.clip(min=0.0))
        assert isinstance(anomaly_std, xr.Dataset)
        if configs.postprocess_anomaly_std:
          anomaly_std = anomaly_std.arcomake.process(steps=configs.postprocess_anomaly_std)
        # Compute and save the climatology and anomaly std in parallel.
        logger.info("Processing climatology and anomaly standard deviation.")
        with ExitStack() as store_stack:
          _climatology_delayed_save = safe_to_zarr(
            climatology,
            climatology_output_path,
            configs=configs.save,
            compute=False,
            store_stack=store_stack,
          )
          _anomaly_std_delayed_save = safe_to_zarr(
            anomaly_std,
            anomaly_std_output_path,
            configs=configs.save,
            compute=False,
            store_stack=store_stack,
          )
          dask.compute(_climatology_delayed_save, _anomaly_std_delayed_save)
        logger.info("Climatology writes completed; output stores closed.")
      logger.info("Input dataset closed; closing Dask client.")
    except Exception as exc:
      logger.exception("An error occurred while processing the climatology and anomaly std.")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
  logger.info("Climatology complete; Dask client closed.")


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


def _window_weights(
  offsets: list[int], weighting: WeightingType, scale: float | None = None
) -> np.ndarray:
  """
  Compute the weights of the steps of the (centered) window.

  The weight of a step depends only on its (signed) distance in steps from the
  center of the window, and is therefore constant from one year to the next.

  Parameters
  ----------
  offsets : list[int]
      Signed offsets of the window steps relative to its center.
  weighting : {"constant", "gaussian", "geometric"}
      Weighting scheme. ``"constant"`` gives equal weights (plain average),
      ``"gaussian"`` uses a Gaussian smoothing kernel, and ``"geometric"`` uses
      weights decaying geometrically with the distance from the center.
  scale : float or None, default None
      Scale of the kernel, in step units. For ``"gaussian"`` it is the standard
      deviation of the kernel, and for ``"geometric"`` it is the per-step decay
      ratio (in ``(0, 1]``). If ``None``, a default derived from the window size
      is used: half the window half-width for ``"gaussian"``, and the ratio
      reaching one tenth at the window edge for ``"geometric"``. Ignored for
      ``"constant"``.

  Returns
  -------
  np.ndarray
      Array of weights aligned with ``offsets``. Only the relative magnitude of
      the weights matters, hence they are left unnormalised.
  """
  positions = np.asarray(offsets, dtype=float)
  half = int(np.abs(positions).max()) if positions.size else 0
  if weighting == "constant" or half == 0:
    return np.ones_like(positions)
  if weighting == "gaussian":
    sigma = scale if scale is not None else half / 2.0
    if sigma <= 0:
      raise ValueError(f"Gaussian scale (standard deviation) must be positive, got {sigma}")
    return np.exp(-0.5 * (positions / sigma) ** 2)
  if weighting == "geometric":
    ratio = scale if scale is not None else 0.1 ** (1.0 / half)
    if not 0.0 < ratio <= 1.0:
      raise ValueError(f"Geometric scale (decay ratio) must be in (0, 1], got {ratio}")
    return ratio ** np.abs(positions)
  raise ValueError(f"Unsupported weighting scheme: {weighting}")


# Days before each month in a 365-day year: the day-of-year bins of `bins = "day_of_year"`.
_DAYS_BEFORE_MONTH = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
# Feb 29 counts in Feb 28's bin (0-based day 58 of a 365-day year).
FEB_29_BIN = 58

BinStatistics = tuple[xr.Dataset, xr.Dataset, xr.Dataset]


def day_of_year_bins(time: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
  """The 0-based day of a 365-day year of each time stamp, and which stamps are Feb 29.

  Days after Feb 29 in a leap year take the bin of the same date in other years; Feb 29 itself
  takes Feb 28's bin (`FEB_29_BIN`). Works for numpy and cftime dates with standard months.
  """
  month = np.asarray(time.dt.month.values)
  day = np.asarray(time.dt.day.values)
  is_feb_29 = (month == 2) & (day == 29)
  if np.any((day > _DAYS_IN_MONTH[month - 1]) & ~is_feb_29):
    raise ValueError("day_of_year bins need a calendar with standard month lengths")
  bins = _DAYS_BEFORE_MONTH[month - 1] + day - 1
  bins[is_feb_29] = FEB_29_BIN
  return bins, is_feb_29


def _fold_statistics(folded: xr.Dataset) -> BinStatistics:
  """Count, mean and sum of squared deviations over `_period`, skipping NaN."""
  count = folded.count("_period")
  mean = folded.mean("_period", skipna=True)
  m2 = ((bind(folded, mean) - mean) ** 2).sum("_period", skipna=True)
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
  dataset: xr.Dataset, period: int, bins: BinningType, time_dim: str, use_flox: bool
) -> BinStatistics:
  """Per bin: the number of values, their mean and their sum of squared deviations (NaN skipped).

  The bins are along `time_dim`, numbered 0 to `period - 1`.
  """
  size = dataset.sizes[time_dim]
  if bins == "day_of_year":
    labels, is_feb_29 = day_of_year_bins(dataset[time_dim])
  else:
    labels, is_feb_29 = np.arange(size) % period, np.zeros(size, dtype=bool)
  bin_coords = {time_dim: np.arange(period)}

  flox_xarray = None
  if use_flox:
    try:
      import flox.xarray as flox_xarray
    except ImportError:
      logger.warning("flox is not installed; falling back to native xarray implementation.")
  if flox_xarray is not None:
    group = xr.DataArray(labels, dims=[time_dim], coords={time_dim: dataset[time_dim]}, name="_bin")
    expected = np.arange(period)
    count = flox_xarray.xarray_reduce(
      dataset, group, func="count", expected_groups=expected, method="cohorts"
    )
    mean = flox_xarray.xarray_reduce(
      dataset, group, func="nanmean", expected_groups=expected, method="cohorts"
    )
    var = flox_xarray.xarray_reduce(
      dataset, group, func="nanvar", ddof=0, expected_groups=expected, method="cohorts"
    )
    assert isinstance(count, xr.Dataset) and isinstance(mean, xr.Dataset)
    assert isinstance(var, xr.Dataset)
    as_bins = [s.rename({"_bin": time_dim}).assign_coords(bin_coords) for s in (count, mean, var)]
    m2 = as_bins[2] * as_bins[0]
    assert isinstance(m2, xr.Dataset)
    return as_bins[0], as_bins[1], m2

  # Fold the time axis into (_period, bin): step k of a folded period is bin k. Missing steps (an
  # incomplete first or last year, gaps) become NaN, which the statistics skip.
  if bins == "day_of_year":
    years = np.asarray(dataset[time_dim].dt.year.values)
    position = (years - years.min()) * period + labels
  else:
    position = np.arange(size)
  regular = ~is_feb_29
  if np.unique(position[regular]).size != np.count_nonzero(regular):
    raise ValueError("day_of_year bins need at most one time step per day")
  n_periods = -(-int(position.max(initial=0) + 1) // period)
  folded = (
    dataset.isel({time_dim: regular})
    .assign_coords({time_dim: position[regular]})
    .reindex({time_dim: np.arange(n_periods * period)})
    .coarsen({time_dim: period}, boundary="exact")
    .construct({time_dim: ("_period", time_dim)})
    .assign_coords(bin_coords)
  )
  count, mean, m2 = _fold_statistics(folded)
  if is_feb_29.any():
    leap = dataset.isel({time_dim: is_feb_29}).rename({time_dim: "_period"})
    feb_28 = (
      count.isel({time_dim: FEB_29_BIN}),
      mean.isel({time_dim: FEB_29_BIN}),
      m2.isel({time_dim: FEB_29_BIN}),
    )
    merged = _combine(feb_28, _fold_statistics(leap))
    is_feb_28 = xr.DataArray(np.arange(period) == FEB_29_BIN, dims=[time_dim], coords=bin_coords)
    count, mean, m2 = (
      xr.where(is_feb_28, new, old) for old, new in zip((count, mean, m2), merged, strict=True)
    )
  assert isinstance(count, xr.Dataset) and isinstance(mean, xr.Dataset)
  assert isinstance(m2, xr.Dataset)
  return count, mean, m2


def _compute_climatology(
  dataset: xr.Dataset,
  period: int,
  window: int = 1,
  weighting: WeightingType = "constant",
  weighting_scale: float | None = None,
  time_dim: str = "time",
  use_flox: bool = False,
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
      Weighting of the window's bins (see ``_window_weights``).
  weighting_scale : float or None, default None
      Scale of the weighting kernel, in bins (see ``_window_weights``).
  time_dim : str, default "time"
      Name of the time dimension.
  use_flox : bool, default False
      Use ``flox`` for the grouped reductions (falls back to xarray without it).
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
  count, mean, m2 = _bin_statistics(dataset, period, bins, time_dim, use_flox)

  half = window // 2
  offsets = list(range(-half, window - half))
  terms = list(
    zip(offsets, _window_weights(offsets, weighting, scale=weighting_scale), strict=True)
  )

  def shifted(stat: xr.Dataset, offset: int) -> xr.Dataset:
    return stat.roll({time_dim: -offset}, roll_coords=False)

  def window_sum(parts: list[xr.Dataset]) -> xr.Dataset:
    return balanced_tree_reduce(operator.add, parts)

  # Reliability weights: each value of bin b + k has the weight w_k of its offset.
  mean0 = mean.fillna(0.0)
  v1 = window_sum([float(w) * shifted(count, o) for o, w in terms])
  v2 = window_sum([float(w) ** 2 * shifted(count, o) for o, w in terms])
  climatology = (window_sum([float(w) * shifted(count * mean0, o) for o, w in terms]) / v1).where(
    v1 > 0
  )
  clim0 = climatology.fillna(0.0)
  total_m2 = window_sum(
    [
      float(w) * (shifted(m2, o) + shifted(count, o) * (shifted(mean0, o) - clim0) ** 2)
      for o, w in terms
    ]
  )
  denominator = v1 - v2 / v1.where(v1 > 0)
  variance = (total_m2 / denominator).where(denominator > 0)

  def as_input_types(result: xr.Dataset) -> xr.Dataset:
    return result.astype(
      {
        name: dataset[name].dtype
        for name in result.data_vars
        if np.issubdtype(dataset[name].dtype, np.floating)
      }
    )

  return as_input_types(climatology), as_input_types(variance)
