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
#  With these and other optimizations, now we're able to get the job to the end without errors and compute the all the
#  stats, including the climatology. Optionally, the climatology can now be computed using Welford's algorithm.
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
import logging
import pathlib
from itertools import cycle
from typing import Literal, get_args

import click
import dask
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from arcomake.cli_utils import (
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client, maybe_wait
from arcomake.dataset_utils import ReadConfig, SaveConfig, save_to_zarr
from arcomake.processing import ProcessingStepConfig, process

logger = logging.getLogger(__name__)

# Weighting scheme applied to the steps of the (centered) window used to
# compute the climatology. The weights depend only on the distance from the
# center of the window and are therefore constant from one year to the next.
WeightingType = Literal["constant", "gaussian", "geometric"]


class PostprocessConfig(BaseModel):
  model_config = ConfigDict(extra="allow")

  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)


class ClimatologyConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  time_dim: str = "time"
  period: int = Field(description="Period of the climatology bins")
  window: int = Field(ge=1, description="Number of bins to average over")
  weighting: WeightingType
  weighting_scale: float | None = Field(
    default=None, ge=0.0, description="Scale of the weighting function in bins"
  )
  read: ReadConfig
  preprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_climatology: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_anomaly_std: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, exists=True, dir_okay=False),
)
@click.argument(
  "input_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, exists=True),
)
@click.argument(
  "climatology_output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, writable=True),
)
@click.argument(
  "anomaly_std_output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, writable=True),
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--scheduler-type",
  default="mpi",
  type=click.Choice(
    get_args(SchedulerOptionType),
    case_sensitive=False,
  ),
  help="Type of Dask scheduler to use.",
)
@click.option(
  "--sync-step/--no-sync-step",
  "sync_step",
  default=False,
  help="Whether to persist the result in memory during climatology computation.",
  show_default=True,
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
)
def compute_climatology(
  config_path: pathlib.Path,
  input_path: pathlib.Path,
  climatology_output_path: pathlib.Path,
  anomaly_std_output_path: pathlib.Path,
  overwrite: bool = False,
  scheduler_type: SchedulerOptionType = "mpi",
  sync_step: bool = False,
  log_level: str = "info",
):
  """
  Compute the climatology and anomaly std of a Zarr dataset and save them to Zarr.

  The time coordinate is validated (sorted, no missing dates or duplicates,
  uniformly spaced with a daily-multiple frequency dividing the calendar year)
  and the dataset is converted to the given fixed-day calendar to handle leap
  years. Data variables without the time dimension (e.g., static masks) are
  dropped. Two outputs are written: the climatological average over
  calendar-year bins, and the standard deviation of the anomalies with
  respect to that climatology.

  Parameters
  ----------
  config_path : pathlib.Path
      Path to TOML configuration file containing configuration options.
  input_path : pathlib.Path
      Path to the input Zarr dataset.
  climatology_output_path : pathlib.Path
      Path to the climatology output.
  anomaly_std_output_path : pathlib.Path
      Path to the anomaly std output.
  overwrite : bool, default False
      Whether to overwrite the outputs if they already exist.
  scheduler_type : SchedulerOptionType, default "mpi"
      Type of Dask scheduler to use.
  sync_step : bool, default False
      Whether to persist the result in memory during climatology computation.
  log_level : {"debug", "info", "warning", "error", "critical"}, default "info"
      Logging verbosity.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Check output paths.
  check_output_path(climatology_output_path, overwrite=overwrite)
  check_output_path(anomaly_std_output_path, overwrite=overwrite)

  # Read configs
  configs = read_configs(config_path, schema=ClimatologyConfig)

  # Set up Dask client.
  with get_client(scheduler_type=scheduler_type):
    climatology = xr.Dataset()
    anomaly_std = xr.Dataset()
    try:
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with xr.open_dataset(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Preprocess input dataset
        if configs.preprocess:
          dataset = process(dataset=dataset, steps=configs.preprocess)
        # We assume that after preprocessing the time coordinate of the dataset is:
        #   1. sorted,
        #   2. without missing dates or duplicates,
        #   3. uniformly spaced,
        #   4. that all variables have a time dimension,
        #   5. that period times the size of a climatology bin divides the calendar year,
        #   6. that the dataset contains an integer number of periods.
        # We just check for the latter.
        assert dataset[configs.time_dim].size % configs.period == 0, (
          "Dataset must contain an integer number of full periods."
        )
        # Welford's online algorithm yields both the climatological mean and the
        # unbiased sample variance of the anomalies in a single pass over the data.
        climatology, anomaly_var = _online_climatology(
          dataset,
          time_dim=configs.time_dim,
          period=configs.period,
          window=configs.window,
          weighting=configs.weighting,
          weighting_scale=configs.weighting_scale,
          sync_step=sync_step,
        )
        # Postprocess climatology and anomaly standard deviation.
        if configs.postprocess_climatology:
          climatology = process(dataset=climatology, steps=configs.postprocess_climatology)
        anomaly_std = np.sqrt(anomaly_var.clip(min=0.0))
        if configs.postprocess_anomaly_std:
          anomaly_std = process(dataset=anomaly_std, steps=configs.postprocess_anomaly_std)
        # Compute and save the climatology and anomaly std in parallel.
        _climatology_delayed_save = save_to_zarr(
          climatology, climatology_output_path, configs=configs.save, compute=False
        )
        _anomaly_std_delayed_save = save_to_zarr(
          anomaly_std, anomaly_std_output_path, configs=configs.save, compute=False
        )
        dask.compute(_climatology_delayed_save, _anomaly_std_delayed_save)
        # Manually close the store, see: https://github.com/pydata/xarray/issues/4076
        _climatology_delayed_save.close()
        _anomaly_std_delayed_save.close()
    except Exception as exc:
      logger.exception("An error occurred while processing the climatology and anomaly std.")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
    finally:
      # Close datasets.
      climatology.close()
      anomaly_std.close()


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


def _online_climatology(
  dataset: xr.Dataset,
  period: int,
  window: int = 1,
  weighting: WeightingType = "constant",
  weighting_scale: float | None = None,
  time_dim: str = "time",
  sync_step=False,
) -> tuple[xr.Dataset, xr.Dataset]:
  """
  Average a dataset over consecutive windows of ``step_size`` steps along time.

  The dataset is split along the time dimension into consecutive windows of
  ``period`` steps, and the windows are combined elementwise using a weighted
  Welford online algorithm, so that only one window is combined at a time.
  This yields both the running (streaming) weighted mean and the unbiased weighted
  sample variance across windows.

  Optionally, statistics can be computed over a ``window`` of steps centered on
  each bin, rather than over a single bin. In that case, for each yearly window
  and each bin, the ``window`` neighbouring steps (wrapped around the calendar
  year) are folded into the running statistics, so that each bin aggregates the
  values of a ``window``-step window centered on it, across all years. For
  example, with a centered window of 15 days on daily data, the statistics of
  the 8th of January are computed from the 1st to the 15th of January of every
  year.

  Each step of the window is weighted according to ``weighting`` by a factor
  that depends only on its distance from the center of the window (see
  ``_window_weights``). Since these weights are the same for every yearly
  window, they are constant from one year to the next. With ``"constant"``
  weighting the algorithm reduces to the plain (unweighted) Welford algorithm.

  Parameters
  ----------
  dataset : xr.Dataset
      Input dataset, assumed to be sorted and uniformly spaced along the time
      dimension.
  period : int
      Number of time steps in each window (i.e., per calendar year).
  window : int, default 1
      Number of steps of the (centered) window over which statistics are
      computed for each bin. ``window=1`` reduces to the plain per-bin
      climatology.
  weighting : {"constant", "gaussian", "geometric"}, default "constant"
      Weighting scheme applied to the steps of the window. See
      ``_window_weights`` for details.
  weighting_scale : float or None, default None
      Scale of the weighting kernel, in step units (Gaussian standard deviation
      or geometric decay ratio). If ``None``, a default derived from the window
      size is used. See ``_window_weights`` for details.
  time_dim : str, default "time"
      Name of the time dimension to average over.

  Returns
  -------
  tuple[xr.Dataset, xr.Dataset]
      A pair ``(mean, variance)``, each of size ``step_size`` along
      ``time_dim``. ``mean`` is the weighted climatological average
      across windows and ``variance`` is the unbiased weighted sample variance
      across windows (with reliability weights, i.e. normalised by
      ``W - (sum of squared weights) / W``, which reduces to ``n - 1`` for
      constant weights).
  """
  size = dataset.sizes[time_dim]
  bins = np.arange(period, dtype=int)
  dataset = dataset.assign_coords({time_dim: np.fromiter(cycle(bins), dtype=int, count=size)})
  # Offsets of a `window`-step window centered on each bin.
  half = window // 2
  offsets = list(range(-half, window - half))
  # Per-step weights, constant from one year to the next.
  weights = _window_weights(offsets, weighting, scale=weighting_scale)

  avg = xr.Dataset()
  m2 = xr.Dataset()
  # Running sums of the weights and of the squared weights (West's algorithm).
  weight_total = 0.0
  weight_sq_total = 0.0
  start = 0
  first_iteration = True

  while start < size:
    end = min(start + period, size)
    value = dataset.isel({time_dim: slice(start, end)})
    # Inner loop over the steps of the window centered on each bin. Each offset
    # contributes the value `offset` steps away (wrapped around the year),
    # weighted by `weight`, so that every bin aggregates a weighted
    # `window`-step window centered on it.
    for offset, weight in zip(offsets, weights, strict=True):
      weight = float(weight)
      weight_total += weight
      weight_sq_total += weight * weight
      shifted = value.roll({time_dim: -offset}, roll_coords=False)
      if first_iteration:
        avg: xr.Dataset = shifted
        # Welford's aggregated squared distance from the running mean (M2 accumulator).
        m2 = xr.zeros_like(avg)
        first_iteration = False
        continue
      # Notice: it is assumed that the dataset contains an integer number of periods.
      # However, in the case of missing values, it would be enough to set them to avg
      # shifted = shifted.where(shifted.notnull(), avg)
      # We don't do that to not hurt the performance.
      # Weighted Welford (West) online update for mean and squared-distance accumulator.
      delta = shifted - avg
      avg = avg + (weight / weight_total) * delta
      delta2 = shifted - avg
      m2 = m2 + delta * delta2 * weight
    if sync_step:
      avg = avg.persist()
      m2 = m2.persist()
      maybe_wait([avg, m2], rebalance=True)
    start = end

  # Unbiased weighted sample variance with reliability weights (normalised by
  # W - (sum of squared weights) / W, which reduces to n - 1 for constant weights).
  # noinspection PyTypeChecker
  denom = weight_total - (weight_sq_total / weight_total)
  variance = m2 / denom if denom > 0.0 else xr.zeros_like(m2)

  return avg, variance
