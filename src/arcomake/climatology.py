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
import functools
import itertools
import logging
import math
import os
import pathlib
import shutil
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, get_args

import click
import dask
import dask.array
import numpy as np
import numpy.typing as npt
import xarray as xr
from icechunk.xarray import to_icechunk
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arcomake import icechunk_utils, provenance, sharding
from arcomake.chunk_plan import detect_budget, stored_warnings
from arcomake.cli_utils import (
  check_aws_profile,
  check_destination,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_utils import DaskSetupType, get_client
from arcomake.dataset_utils import (
  ReadConfig,
  SaveConfig,
  open_input,
  prepare_for_writing,
  write_into_session,
  write_tree,
)
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.temporary_utils import TempDirectoryRegistry, get_current_temp_registry

logger = logging.getLogger(__name__)

# Weighting scheme applied to the steps of the (centered) window used to
# compute the climatology. The weights depend only on the distance from the
# center of the window and are therefore constant from one year to the next.
WeightingType = Literal["constant", "gaussian", "geometric"]
# How time steps fall into bins: by position in each period, or by day of a 365-day year.
BinningType = Literal["position", "day_of_year"]
# The groups of the output repository: the climatology and the anomaly standard deviation.
GROUPS = ("climatology", "anomaly_std")
# Commit metadata: how the statistics were computed ("graph" or "streaming").
METHOD_KEY = "arcomake_method"


class ReferencePeriod(BaseModel):
  """The right-open interval `[start, end)` of the input the statistics are computed over."""

  model_config = ConfigDict(extra="forbid")

  start: datetime.datetime
  end: datetime.datetime

  @model_validator(mode="after")
  def _ordered_and_naive(self) -> "ReferencePeriod":
    if self.start.tzinfo is not None or self.end.tzinfo is not None:
      raise ValueError("reference_period dates have no time zone (time axes are naive)")
    if self.start >= self.end:
      raise ValueError(f"reference_period start ({self.start}) must be before end ({self.end})")
    return self


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
  split_dims: list[str] | None = Field(
    default=None,
    description=(
      "With --streaming (one series at a time): each data variable, and each index along these"
      ' dimensions, is a series (e.g. ["depth"]; default: one series per variable). Warned'
      " about without --streaming"
    ),
  )
  read: ReadConfig
  preprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_climatology: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_anomaly_std: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig

  @model_validator(mode="after")
  def _no_save_group(self) -> "ClimatologyConfig":
    if self.save.group is not None:
      raise ValueError(f"[save] group is not supported: the output's groups are {GROUPS}")
    return self

  @model_validator(mode="after")
  def _split_dims_not_time(self) -> "ClimatologyConfig":
    if self.split_dims is not None and self.time_dim in self.split_dims:
      raise ValueError(f"split_dims cannot include the time dimension {self.time_dim!r}")
    return self

  @model_validator(mode="after")
  def _day_of_year_bins(self) -> "ClimatologyConfig":
    if self.bins == "day_of_year" and self.period != 365:
      raise ValueError(f'bins = "day_of_year" needs period = 365, got {self.period}')
    return self


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
  "output_path",
  required=True,
  type=str,
)
@click.option(
  "--replace/--no-replace",
  help=(
    "Write a new version of an existing output (an Icechunk repository): it replaces what `main`"
    " shows once the run succeeds; earlier versions stay in its history until the recipe's"
    " [save] keep_history or keep_versions expires them"
  ),
  default=False,
  is_flag=True,
)
@click.option(
  "--streaming/--graph",
  default=False,
  help=(
    "Compute one series at a time (each variable, and each index of the recipe's `split_dims`),"
    " streaming over time, instead of everything in one Dask graph (the default)"
  ),
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
  output_path: str,
  replace: bool = False,
  streaming: bool = False,
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

  The output is one Icechunk repository with two groups, `climatology` and `anomaly_std`
  (xarray: `open_datatree`, or `open_zarr` with `group=`), written in one commit. Assumed, not
  checked: the time axis is sorted and every data variable has the time dimension. Both groups
  have `period` steps along time, the bin index (0 is Jan 1 for day-of-year bins).

  Example::

    period = 365
    bins = "day_of_year"
    reference_period = {start = 1991-01-01, end = 2021-01-01}
    window = 15
    weighting = "gaussian"
    weighting_scale = 3.0
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  with get_client(dask_setup=dask_setup, use_frisky=use_frisky):
    try:
      # Read configs, then check the output paths (with the recipe's storage options).
      configs = read_configs(configs_path, schema=ClimatologyConfig)
      _warn_ignored_split_dims(configs, streaming)
      check_aws_profile(input_path, output_path)
      check_destination(output_path, replace=replace, save_configs=configs.save)
      # The run's label (its branch, commit metadata and message).
      label = icechunk_utils.run_label("compute-climatology")
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
        if not dataset.data_vars:
          raise ValueError("Input dataset contains no data variables.")
        _compute_and_write(dataset, configs, output_path, label=label, streaming=streaming)
      logger.info("Input dataset closed; closing Dask client.")
    except click.ClickException:
      raise
    except icechunk_utils.PromotionError as exc:
      raise click.ClickException(str(exc)) from exc
    except Exception as exc:
      logger.exception("An error occurred while processing the climatology and anomaly std.")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
  logger.info("Climatology complete; Dask client closed.")


def _compute_and_write(
  dataset: xr.Dataset,
  configs: ClimatologyConfig,
  output_path: str,
  *,
  label: str,
  streaming: bool,
) -> None:
  """Compute the climatology and anomaly std of `dataset` and write them into the groups of
  `output_path` (`GROUPS`), one series at a time (`_compute_streaming`) or in one Dask graph
  (`_compute_graph`)."""
  if streaming:
    _compute_streaming(dataset, configs, output_path, label=label)
  else:
    _compute_graph(dataset, configs, output_path, label=label)


def _warn_ignored_split_dims(configs: ClimatologyConfig, streaming: bool) -> None:
  """`split_dims` without `--streaming` (`[]` too: it selected streaming before the option)."""
  if configs.split_dims is not None and not streaming:
    logger.warning(
      f"split_dims {configs.split_dims} ignored: they apply to --streaming only (without it,"
      " everything is one Dask graph)"
    )


def _compute_graph(
  dataset: xr.Dataset, configs: ClimatologyConfig, output_path: str, *, label: str
) -> None:
  """Everything in one Dask graph (`_compute_climatology`), computed into a temporary store and
  copied into the output's groups (`_checkpointed`)."""
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
  if configs.postprocess_climatology:
    climatology = climatology.arcomake.process(steps=configs.postprocess_climatology)
  anomaly_std = np.sqrt(anomaly_var.clip(min=0.0))
  assert isinstance(anomaly_std, xr.Dataset)
  if configs.postprocess_anomaly_std:
    anomaly_std = anomaly_std.arcomake.process(steps=configs.postprocess_anomaly_std)
  # Computed once into a temporary store (removed with the registry), then copied.
  with TempDirectoryRegistry():
    climatology, anomaly_std = _checkpointed(
      [climatology, anomaly_std], configs.save, configs.time_dim
    )
    write_tree(
      dict(zip(GROUPS, (climatology, anomaly_std), strict=True)),
      output_path,
      configs.save,
      label=label,
      # `split_dims` is ignored here.
      recipe=provenance.fingerprint(configs, exclude={"read", "split_dims"}),
      metadata={METHOD_KEY: "graph"},
    )
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


def _day_of_year_positions(time: xr.DataArray, period: int) -> tuple[np.ndarray, ...]:
  """Day-of-year bins, Feb 29 flags and positions on a (year, bin) grid of daily steps.

  Raises:
    ValueError: if two steps fall on the same day (sub-daily data).
  """
  labels, is_feb_29 = day_of_year_bins(time)
  years = np.asarray(time.dt.year.values)
  position = (years - years.min()) * period + labels
  regular = ~is_feb_29
  if np.unique(position[regular]).size != np.count_nonzero(regular):
    raise ValueError("day_of_year bins need at most one time step per day")
  return labels, is_feb_29, position


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
  dataset: xr.Dataset, period: int, bins: BinningType, time_dim: str, use_flox: bool
) -> BinStatistics:
  """Per bin: the number of values, their mean and their sum of squared deviations (NaN skipped).

  The bins are along `time_dim`, numbered 0 to `period - 1`.
  """
  size = dataset.sizes[time_dim]
  if bins == "day_of_year":
    labels, is_feb_29, position = _day_of_year_positions(dataset[time_dim], period)
  else:
    if size % period != 0:
      raise ValueError(
        f"position bins need a whole number of periods ({size} steps, period {period}): leap"
        ' days and partial years shift every later bin; use bins = "day_of_year" for daily data'
      )
    labels, is_feb_29 = np.arange(size) % period, np.zeros(size, dtype=bool)
    position = np.arange(size)
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
    # "blockwise" needs no communication when time chunks hold whole periods of position bins;
    # "cohorts" bounds memory otherwise.
    time_chunks = dataset.chunksizes.get(time_dim, (size,))
    aligned = bins == "position" and all(c % period == 0 for c in time_chunks)
    method = "blockwise" if aligned else "cohorts"
    count = flox_xarray.xarray_reduce(
      dataset, group, func="count", expected_groups=expected, method=method
    )
    mean = flox_xarray.xarray_reduce(
      dataset, group, func="nanmean", expected_groups=expected, method=method
    )
    var = flox_xarray.xarray_reduce(
      dataset, group, func="nanvar", ddof=0, expected_groups=expected, method=method
    )
    assert isinstance(count, xr.Dataset) and isinstance(mean, xr.Dataset)
    assert isinstance(var, xr.Dataset)
    as_bins = [s.rename({"_bin": time_dim}).assign_coords(bin_coords) for s in (count, mean, var)]
    m2 = as_bins[2] * as_bins[0]
    assert isinstance(m2, xr.Dataset)
    return as_bins[0], as_bins[1], m2

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


def _check_workers_reach(path: pathlib.Path) -> None:
  """On a distributed client, fail unless every worker sees `path`: chunks a worker writes
  where the reader does not look would be read as missing, the fill value (NaN), silently."""
  try:
    from distributed import default_client

    client = default_client()
  except (ImportError, ValueError):
    return
  blind = sorted(w for w, seen in client.run(os.path.isdir, str(path)).items() if not seen)
  if blind:
    raise RuntimeError(
      f"Dask workers {', '.join(blind)} do not see the temporary store {path}: set TMPDIR to a"
      " filesystem every worker reaches"
    )


def _checkpointed(outputs: list[xr.Dataset], save: SaveConfig, time_dim: str) -> list[xr.Dataset]:
  """`outputs` computed into a temporary plain Zarr v3 store, and opened again in Dask chunks of
  their stored units (the shards of `save`, else its chunks).

  All outputs are written at once (a DataTree, one group each), so what they share, the window
  and everything before it, is computed once. The store has no shards (its chunks:
  `_staging_chunks`). Writing it needs the memory of a few levels' statistics; written
  directly, shards spanning all levels held every level's statistics until the last window
  (logbook, 21 Nov). The copies compute nothing: they read, one task per group of stored units
  (a shard of 8 bins, 49 levels and the 0.25 deg grid, 1.6 GB, is alone; units above a task's
  share of memory are warned about).

  The store is in the temporary directory of the current registry (`TMPDIR`), checked to be
  seen by every worker of a distributed client.
  """
  registry = get_current_temp_registry()
  if registry is None:
    raise RuntimeError("the climatology's temporary store needs a TempDirectoryRegistry")
  path = registry.create_temp_dir(suffix=".zarr")
  _check_workers_reach(path)
  budget = detect_budget()
  copies: list[dict[Hashable, tuple[int, ...] | None]] = []
  groups: dict[str, xr.Dataset] = {}
  size = 0
  for index, dataset in enumerate(outputs):
    prepared = prepare_for_writing(dataset, save)
    for warning in stored_warnings(prepared, budget):
      logger.warning(warning)
    staged = dataset.drop_encoding()
    copies.append({})
    for name, var in staged.data_vars.items():
      stored, copies[-1][name] = _staging_chunks(
        var.dims,
        var.shape,
        tuple(sizes[0] for sizes in var.chunks or [(n,) for n in var.shape]),
        prepared[name].encoding.get("shards") or prepared[name].encoding.get("chunks"),
        var.dtype.itemsize,
        time_dim,
        budget.target,
      )
      var.encoding["chunks"] = stored
      size += var.nbytes
    groups[str(index)] = staged
  free = shutil.disk_usage(path).free
  if free < size:
    logger.warning(
      f"The temporary store may not fit in {path}: {size / 2**30:.1f} GiB before compression,"
      f" {free / 2**30:.1f} GiB free"
    )
  logger.info(f"Computing the outputs into {path} ({size / 2**30:.1f} GiB before compression)")
  xr.DataTree.from_dict(groups).to_zarr(
    path, mode="w", consolidated=False, zarr_format=3, align_chunks=True
  )
  opened: list[xr.Dataset] = []
  for index, dataset in enumerate(outputs):
    copy = xr.open_zarr(path, group=str(index), consolidated=False, chunks=None)
    copy = copy.assign(
      {
        name: var.chunk(
          dict(zip(var.dims, copies[index][name] or var.encoding["chunks"], strict=True))
        )
        for name, var in copy.data_vars.items()
      }
    )
    # As computed: the copy is written as the outputs would have been.
    for name, var in dataset.variables.items():
      copy[name].attrs = dict(var.attrs)
      copy[name].encoding = dict(var.encoding)
    copy.attrs = dict(dataset.attrs)
    opened.append(copy)
  return opened


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
  weights = _window_weights(offsets, weighting, scale=weighting_scale)

  results: tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] = ({}, {})
  for name, var in mean.data_vars.items():
    dtype = dataset[name].dtype if np.issubdtype(dataset[name].dtype, np.floating) else np.float64
    # One implementation of the window for both paths, chunk by chunk (bins last, as apply_ufunc
    # puts the core dimension), in the output's type.
    statistics = xr.apply_ufunc(
      functools.partial(_window_bins_last, offsets=offsets, weights=weights, dtype=dtype),
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


def _window_bins_last(
  count: np.ndarray,
  mean: np.ndarray,
  m2: np.ndarray,
  offsets: list[int],
  weights: np.ndarray,
  dtype: npt.DTypeLike = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
  """The window step on per-bin statistics with the bins on the last axis, by the numba kernel
  (`_window_kernel`): climatology and variance in `dtype`, computed in float64."""
  shape = mean.shape
  bins = shape[-1]
  climatology = np.empty((mean.size // max(bins, 1), bins), dtype=dtype)
  variance = np.empty_like(climatology)
  _window_kernel()(
    *(np.ascontiguousarray(a.reshape(-1, bins)) for a in (count, mean, m2)),
    np.asarray(offsets, dtype=np.int64),
    np.asarray(weights, dtype=np.float64),
    climatology,
    variance,
  )
  return climatology.reshape(shape), variance.reshape(shape)


def _window_statistics_in_memory(
  count: np.ndarray,
  mean: np.ndarray,
  m2: np.ndarray,
  offsets: list[int],
  weights: np.ndarray,
  workers: int | None = None,
  band: int = 1 << 15,
  dtype: npt.DTypeLike = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
  """The window step of `_compute_climatology` on in-memory per-bin statistics (bins on axis 0),
  in `dtype`.

  Same kernel as the lazy path, over bands of `band` points (all axes after the bins, flattened),
  in parallel, so that memory stays a few times the size of one band per thread.
  """
  shape = mean.shape
  count, mean, m2 = (a.reshape(shape[0], -1) for a in (count, mean, m2))
  climatology = np.empty(mean.shape, dtype=dtype)
  variance = np.empty(mean.shape, dtype=dtype)

  def window_band(start: int) -> None:
    part = (slice(None), slice(start, start + band))
    c, v = _window_bins_last(count[part].T, mean[part].T, m2[part].T, offsets, weights, dtype)
    climatology[part] = c.T
    variance[part] = v.T

  # Bands are independent, and the kernel releases the GIL: compute them in parallel.
  with ThreadPoolExecutor(max_workers=workers or min(16, os.cpu_count() or 1)) as pool:
    list(pool.map(window_band, range(0, mean.shape[1], band)))
  return climatology.reshape(shape), variance.reshape(shape)


_WINDOW_KERNEL_LOCK = threading.Lock()


def _window_kernel():
  """The numba window kernel (one for all threads: numba compiles it once, at its first call).

  Dask's threads ask for it at once on their first chunks: `functools.cache` alone let each of
  them make (and compile) its own."""
  with _WINDOW_KERNEL_LOCK:
    return _new_window_kernel()


@functools.cache
def _new_window_kernel():
  """The numba window kernel, made on first use (importing numba is slow)."""
  import numba

  @numba.njit(nogil=True)
  def window_rows(count, mean, m2, offsets, weights, climatology, variance):  # pragma: no cover
    """Row by row (a point's bins): bin i pools bins i + offsets[j] (wrapping around), weighted
    by weights[j], skipping empty bins; reliability-weights variance. In float64, in the order
    of the former NumPy implementation (same bits)."""
    points, bins = mean.shape
    for p in range(points):
      for i in range(bins):
        v1 = 0.0
        v2 = 0.0
        total = 0.0
        for j in range(offsets.shape[0]):
          b = (i + offsets[j]) % bins
          n = np.float64(count[p, b])
          if n > 0:
            w = weights[j]
            v1 += w * n
            v2 += w * w * n
            total += w * (n * np.float64(mean[p, b]))
        if v1 > 0:
          c = total / v1
          total = 0.0
          for j in range(offsets.shape[0]):
            b = (i + offsets[j]) % bins
            n = np.float64(count[p, b])
            if n > 0:
              d = np.float64(mean[p, b]) - c
              total += weights[j] * (d * d * n + np.float64(m2[p, b]))
          denominator = v1 - v2 / v1
          climatology[p, i] = c
          variance[p, i] = total / denominator if denominator > 0 else np.nan
        else:
          climatology[p, i] = np.nan
          variance[p, i] = np.nan

  return window_rows


def _series(dataset: xr.Dataset, split_dims: list[str]) -> list[tuple[str, dict[str, int]]]:
  """(variable, {dim: index}) for every series: each variable, each index of its split dims."""
  series = []
  for name, var in dataset.data_vars.items():
    dims = [d for d in split_dims if d in var.dims]
    for index in itertools.product(*(range(var.sizes[d]) for d in dims)):
      series.append((str(name), dict(zip(dims, index, strict=True))))
  return series


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
  if bins == "day_of_year" and period != 365:
    raise ValueError(f"day_of_year bins need period = 365, got {period}")
  half = window // 2
  offsets = list(range(-half, window - half))
  weights = _window_weights(offsets, weighting, scale=weighting_scale)
  results: tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] = ({}, {})
  for name, var in piece.data_vars.items():
    order = [time_dim, *(d for d in var.dims if d != time_dim)]
    started = time.monotonic()
    count, mean, m2 = _stream_bin_statistics(var, period, bins, time_dim)
    read = time.monotonic()
    dtype = var.dtype if np.issubdtype(var.dtype, np.floating) else np.float64
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


def _one_index_per_chunk(save: SaveConfig, split_dims: list[str]) -> SaveConfig:
  """`save` with stored chunks (and shards, if any) of 1 along the split dimensions, which each
  series writes alone.

  Raises:
    ValueError: if `[save] chunks` or `shards` ask for another size along a split dimension.
  """
  update = {}
  for key in ("chunks", "shards"):
    sizes = getattr(save, key)
    if key == "shards" and sizes is None:
      continue
    if sizes is not None and not isinstance(sizes, dict):
      raise ValueError(f"with split_dims, [save] {key} must be a table of sizes per dimension")
    sizes = dict(sizes or {})
    for dim in split_dims:
      if sizes.get(dim, 1) != 1:
        raise ValueError(f"with split_dims, [save] {key} along {dim!r} must be 1 (one series each)")
      sizes[dim] = 1
    update[key] = sizes
  return save.model_copy(update=update)


def _output_template(dataset: xr.Dataset, period: int, time_dim: str) -> xr.Dataset:
  """The outputs' layout, lazily (never computed): `period` bins along time, other dims kept."""
  data_vars = {}
  for name, var in dataset.data_vars.items():
    shape = tuple(period if d == time_dim else var.sizes[d] for d in var.dims)
    dtype = var.dtype if np.issubdtype(var.dtype, np.floating) else np.float64
    data_vars[name] = (var.dims, dask.array.zeros(shape, dtype=dtype, chunks=-1))
  coords = {
    name: coord.load() for name, coord in dataset.coords.items() if time_dim not in coord.dims
  }
  coords[time_dim] = np.arange(period)
  return xr.Dataset(data_vars, coords=coords)


def _compute_streaming(
  dataset: xr.Dataset, configs: ClimatologyConfig, output_path: str, *, label: str
) -> None:
  """One series (variable, index of `split_dims`) at a time, written into its region.

  The groups' metadata are written first, from a template with the postprocess steps applied;
  then every series is computed in memory (`_compute_series`), postprocessed and written into its
  region. Postprocess steps therefore have to work on one series (attributes, renames, chunking).
  """
  outputs = tuple(
    zip(GROUPS, (configs.postprocess_climatology, configs.postprocess_anomaly_std), strict=True)
  )
  split_dims = configs.split_dims or []
  unknown = [d for d in split_dims if d not in dataset.dims]
  if unknown:
    raise ValueError(f"split_dims {unknown} are not dimensions of the input: {list(dataset.dims)}")
  save = _one_index_per_chunk(configs.save, split_dims)
  template = _output_template(dataset, configs.period, configs.time_dim)
  # One run (its own branch) and one session on it, committed once every series is written:
  # an interrupted run commits nothing, and `main` is untouched.
  run = icechunk_utils.Run(
    output_path,
    save.storage_options,
    label,
    recipe=provenance.fingerprint(configs, exclude={"read"}),
  )
  session = run.session()
  icechunk_utils.empty(session)  # what an existing repository held (--replace) goes
  for group, steps in outputs:
    # Metadata only: the data of the template are never computed.
    write_into_session(
      template.arcomake.process(steps=steps) if steps else template,
      session,
      save.model_copy(update={"group": group}),
      compute=False,
    )
  # Stored chunks (shards, if any: the units of writing) of every variable: results are chunked
  # alike, so that Dask writes the chunks of a region in parallel.
  stored_chunks = [
    {
      name: var.encoding.get("shards") or var.encoding["chunks"]
      for name, var in xr.open_zarr(
        session.store, group=group, consolidated=False
      ).data_vars.items()
    }
    for group, _ in outputs
  ]

  def write(results: tuple[xr.Dataset, xr.Dataset], index: dict[str, int], label: str) -> None:
    writing = time.monotonic()
    # Explicit regions: "auto" maps a dimension without coordinate to index 0.
    region = {configs.time_dim: slice(0, configs.period)} | {
      d: slice(i, i + 1) for d, i in index.items()
    }
    for (group, steps), result, chunks in zip(outputs, results, stored_chunks, strict=True):
      if steps:
        result = result.arcomake.process(steps=steps)
      result = result.drop_attrs().chunk(
        {
          dim: size
          for name in result.data_vars
          for dim, size in zip(result[name].dims, chunks[str(name)], strict=True)
        }
      )
      result = result.drop_vars(
        [name for name, var in result.variables.items() if not set(var.dims) & set(region)]
      )
      with sharding.write_order(save.shard_order):  # arrays opened here: shards in that order
        to_icechunk(result, session, region=region, mode="r+", group=group)
    logger.info(f"{label} written in {time.monotonic() - writing:.0f} s")

  series = _series(dataset, split_dims)
  started = time.monotonic()
  # Each series is written in the background while the next one is read. Reads and writes use
  # local threads whatever --dask-setup says: results live in this process.
  with dask.config.set(scheduler="threads"), ThreadPoolExecutor(max_workers=1) as writer:
    pending = None
    for number, (name, index) in enumerate(series, start=1):
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
      if pending is not None:
        pending.result()  # one write at a time; errors surface here
      pending = writer.submit(write, (climatology, anomaly_std), index, f"{name} {index}")
      elapsed = time.monotonic() - started
      logger.info(
        f"Series {number}/{len(series)} ({name} {index}) computed; {elapsed / number:.0f} s per"
        f" series, {elapsed / number * (len(series) - number) / 60:.0f} min left"
      )
    if pending is not None:
      pending.result()
  run.commit(session, f"{label}: {len(series)} series", {METHOD_KEY: "streaming"})
  run.finish(configs.save.retention())
  logger.info("Climatology writes completed, one series at a time.")


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
  if bins == "day_of_year":
    labels, _, _ = _day_of_year_positions(da[time_dim], period)
  else:
    labels = np.arange(size) % period
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
      _welford_kernel()(
        count.reshape(period, -1),
        mean.reshape(period, -1),
        m2.reshape(period, -1),
        values.reshape(len(values), -1),
        labels[start : start + len(values)],
      )
  mean[count == 0] = np.nan
  return count, mean, m2


@functools.cache
def _welford_kernel():
  """The numba Welford kernel, compiled on first use (importing numba is slow)."""
  import numba

  @numba.njit(parallel=True, nogil=True)
  def welford_block(count, mean, m2, values, labels):  # pragma: no cover (compiled)
    """Add row j of `values` (steps × points) to bin `labels[j]`, skipping NaN (float64).

    One fused pass per step, parallel over points; `count`, `mean`, `m2` are (bins × points).
    """
    for j in range(values.shape[0]):
      b = labels[j]
      for i in numba.prange(values.shape[1]):
        x = values[j, i]
        if np.isfinite(x):
          n = count[b, i] + 1
          count[b, i] = n
          delta = x - mean[b, i]
          mean[b, i] += delta / n
          m2[b, i] += delta * (x - mean[b, i])

  return welford_block
