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
from arcomake.dask_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import ReadConfig, SaveConfig, safe_to_zarr
from arcomake.processing_utils import ProcessingStepConfig

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
  "--scheduler-type",
  default="mpi",
  type=click.Choice(
    get_args(SchedulerOptionType),
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
  scheduler_type: SchedulerOptionType = "mpi",
  use_frisky: bool = False,
  use_flox: bool = False,
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
  configs_path : pathlib.Path
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
  log_level : {"debug", "info", "warning", "error", "critical"}, default "info"
      Logging verbosity.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Check output paths.
  check_if_overwriting(climatology_output_path, overwrite=overwrite)
  check_if_overwriting(anomaly_std_output_path, overwrite=overwrite)

  # Read configs
  configs = read_configs(configs_path, schema=ClimatologyConfig)

  # Set up Dask client.
  with get_client(scheduler_type=scheduler_type, use_frisky=use_frisky):
    try:
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with xr.open_dataset(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Preprocess input dataset
        if configs.preprocess:
          dataset = dataset.arcomake.process(steps=configs.preprocess)
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
          use_flox=use_flox,
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


def _compute_climatology(
  dataset: xr.Dataset,
  period: int,
  window: int = 1,
  weighting: WeightingType = "constant",
  weighting_scale: float | None = None,
  time_dim: str = "time",
  use_flox: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
  """
  Compute the climatological mean and unbiased sample variance for a single DataArray.

  Parameters
  ----------
  dataset : xr.Dataset
      Input Dataset, assumed to be sorted and uniformly spaced along the time
      dimension.
  period : int
      Number of time steps in each period (i.e., per calendar year).
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
  use_flox : bool, default False
      Whether to use ``flox`` for grouped periodic reductions. If ``True``
      and ``flox`` is not installed, falls back to the native xarray
      implementation.

  Returns
  -------
  tuple[xr.DataArray, xr.DataArray]
      A pair ``(mean, variance)``, each of size ``period`` along
      ``time_dim``. ``mean`` is the weighted climatological average
      and ``variance`` is the unbiased weighted sample variance of the anomalies.
  """
  size = dataset.sizes[time_dim]
  n_periods = size // period
  bins = np.arange(period, dtype=int)

  flox_xarray = None
  if use_flox:
    try:
      import flox.xarray as flox_xarray
    except ImportError:
      logger.warning("flox is not installed; falling back to native xarray implementation.")

  if flox_xarray is not None:
    # Choose the optimal reduction strategy based on chunk alignment:
    # 'blockwise' provides maximal speed and zero inter-worker communication
    # when time chunks align with period boundaries; 'cohorts' prevents memory
    # spikes when chunks span arbitrary periodic boundaries.
    time_chunks = dataset.chunksizes.get(time_dim, (size,))
    is_aligned = all(c % period == 0 for c in time_chunks)
    reduction_method = "blockwise" if is_aligned else "cohorts"
    group = xr.DataArray(
      np.tile(bins, n_periods),
      dims=[time_dim],
      coords={time_dim: dataset[time_dim]},
      name=time_dim,
    )
    # Parallel computation of periodic mean in a single pass
    raw_mean = flox_xarray.xarray_reduce(
      dataset,
      group,
      func="mean",
      expected_groups=bins,
      method=reduction_method,
    )
    # Direct variance calculation without intermediate array materialization
    raw_var = flox_xarray.xarray_reduce(
      dataset,
      group,
      func="var",
      ddof=0,
      expected_groups=bins,
      method=reduction_method,
    )
    assert isinstance(raw_mean, xr.Dataset) and isinstance(raw_var, xr.Dataset)
  else:
    # Reshape time dimension into (_period, time_dim)
    folded = (
      dataset.assign_coords({time_dim: np.arange(size)})
      .chunk({time_dim: period})
      .coarsen({time_dim: period}, boundary="exact")
      .construct({time_dim: ("_period", time_dim)})
      .assign_coords({time_dim: bins})
    )
    raw_mean = folded.mean(dim="_period")
    raw_var = ((bind(folded, raw_mean) - raw_mean) ** 2).mean(dim="_period")

  # Centered window smoothing
  half = window // 2
  offsets = list(range(-half, window - half))
  weights = _window_weights(offsets, weighting, scale=weighting_scale)

  w_total = float(np.sum(weights))
  w_sq_total = float(np.sum(weights**2))

  if window == 1:
    weighted_mean = raw_mean
    m2_per_period = raw_var
  else:
    mean_terms = [
      float(w) * raw_mean.roll({time_dim: -offset}, roll_coords=False)
      for offset, w in zip(offsets, weights, strict=True)
    ]
    weighted_mean = balanced_tree_reduce(operator.add, mean_terms) / w_total

    var_smooth_terms = [
      float(w) * raw_var.roll({time_dim: -offset}, roll_coords=False)
      for offset, w in zip(offsets, weights, strict=True)
    ]
    smoothed_raw_var = balanced_tree_reduce(operator.add, var_smooth_terms)

    mean_dispersion_terms = [
      float(w) * (raw_mean.roll({time_dim: -offset}, roll_coords=False) - weighted_mean) ** 2
      for offset, w in zip(offsets, weights, strict=True)
    ]
    between_bin_var = balanced_tree_reduce(operator.add, mean_dispersion_terms)

    m2_per_period = smoothed_raw_var + between_bin_var

  total_m2 = n_periods * m2_per_period
  total_w = n_periods * w_total
  total_w_sq = n_periods * w_sq_total
  denom = total_w - (total_w_sq / total_w)
  variance = total_m2 / denom if denom > 0.0 else xr.zeros_like(total_m2)
  return weighted_mean, variance
