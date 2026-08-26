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
import warnings
from typing import Literal, get_args

import click
import dask
import numpy as np
import xarray as xr

from arcomake.checks import ValidationError, valid_time_coordinate
from arcomake.cli_utils import DictParamType, check_output_path, set_default_logger
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client, maybe_wait
from arcomake.dataset_utils import save_to_zarr
from arcomake.datetime_utils import may_parse_timedelta

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
  "input",
  required=True,
  type=click.Path(
    path_type=pathlib.Path, file_okay=True, dir_okay=True, exists=True, readable=True
  ),
)
@click.argument(
  "climatology_output",
  required=True,
  type=click.Path(path_type=pathlib.Path, file_okay=True, dir_okay=True, writable=True),
)
@click.argument(
  "anomaly_std_output",
  required=True,
  type=click.Path(path_type=pathlib.Path, file_okay=True, dir_okay=True, writable=True),
)
@click.option(
  "--time-dim",
  default="time",
  help="Name of the time dimension to average over.",
  show_default=True,
)
@click.option(
  "--in-chunks",
  default=None,
  show_default=True,
  type=DictParamType(),
  help="Dict containing chunking specs used when reading.",
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--out-chunks",
  default=None,
  show_default=True,
  type=DictParamType(),
  help="Dict containing chunking specs used when writing.",
)
@click.option(
  "--climatology-bin-dim",
  default="dayofyear",
  help="Name of the binning dimension of the resulting climatology.",
  show_default=True,
)
@click.option(
  "--window",
  default=15,
  type=click.IntRange(min=1),
  show_default=True,
  help="Size in days of the centered window over which statistics are computed for each bin.",
)
@click.option(
  "--compressor-name",
  "cname",
  default="lz4",
  show_default=True,
  help="Name of the compressor to use.",
)
@click.option(
  "--compressor-level", "clevel", default=1, show_default=True, help="Compressor level to use."
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
  "--calendar",
  default="365_day",
  type=click.Choice(["365_day", "366_day", "360_day"], case_sensitive=False),
  help="Fixed-day calendar the dataset is converted to before binning.",
)
@click.option(
  "--sync-step/--no-sync-step",
  "sync_step",
  default=False,
  help="Whether to persist the result in memory during climatology computation.",
  show_default=True,
)
@click.option(
  "--raise/--no-raise",
  "should_raise",
  help="Whether to raise an exception if validation fails",
  default=True,
  is_flag=True,
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
)
def compute_climatology(
  input: pathlib.Path,
  climatology_output: pathlib.Path,
  anomaly_std_output: pathlib.Path,
  time_dim: str = "time",
  in_chunks: dict[str, int | Literal["auto"]] | None = None,
  overwrite: bool = False,
  out_chunks: dict[str, int | Literal["auto"]] | None = None,
  climatology_bin_dim: str = "dayofyear",
  window: int = 15,
  cname: str = "lz4",
  clevel: int = 1,
  scheduler_type: SchedulerOptionType = "mpi",
  calendar: Literal["365_day", "366_day", "360_day"] = "365_day",
  sync_step: bool = False,
  should_raise: bool = False,
  log_level: str = "info",
):
  """
  Compute the climatology and anomaly std of a Zarr dataset and save them to Zarr.

  The time coordinate is validated (sorted, no missing dates or duplicates,
  uniformly spaced with a daily-multiple frequency dividing the calendar year)
  and the dataset is converted to the given fixed-day calendar to handle leap
  years. Data variables without the time dimension (e.g., static masks) are
  dropped. Two outputs are written, derived from ``output`` by appending
  ``_climatology`` and ``_anomaly_std`` to its stem: the climatological average
  over calendar-year bins, and the standard deviation of the anomalies with
  respect to that climatology.

  Parameters
  ----------
  input : pathlib.Path
      Path to the input Zarr dataset.
  climatology_output : pathlib.Path
      Path to the climatology output.
  anomaly_std_output : pathlib.Path
      Path to the anomaly std output.
  time_dim : str, default "time"
      Name of the time dimension to reduce or group by.
  in_chunks : dict[str, int] or None, default None
      Chunking specs used when reading the input dataset.
  overwrite : bool, default False
      Whether to overwrite the outputs if they already exist.
  out_chunks : dict[str, int] or None, default None
      Chunking specs used when writing the output dataset.
  climatology_bin_dim : str, default "dayofyear"
      Name of the binning dimension of the resulting climatology.
  window : int, default 1
      Size in days of the centered window over which averages and standard
      deviation are computed for each bin. ``window=1`` computes plain per-day
      statistics.
  cname : str, default "lz4"
      Name of the compressor used when writing the outputs.
  clevel : int, default 1
      Compression level used when writing the outputs.
  scheduler_type : SchedulerOptionType, default "mpi"
      Type of Dask scheduler to use.
  calendar : {"365_day", "366_day", "360_day"}, default "365_day"
      Fixed-day calendar the dataset is converted to before binning.
  sync_step : bool, default False
      Whether to persist the result in memory during climatology computation.
  should_raise : bool, default False
      Whether to raise an exception if time coordinate validation fails,
      instead of emitting a warning.
  log_level : {"debug", "info", "warning", "error", "critical"}, default "info"
      Logging verbosity.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  client = get_client(scheduler_type=scheduler_type)

  # Compute and check paths.
  if not input.exists():
    raise ValueError(f"Input path {input} does not exist")
  check_output_path(climatology_output, overwrite=overwrite)
  check_output_path(anomaly_std_output, overwrite=overwrite)

  logger.info(f"Opening input dataset from {input} with chunks={in_chunks}")
  # As the Dask graph tends to be huge, it's important to avoid inline_array=True,
  # see: https://docs.dask.org/en/latest/generated/dask.array.from_array.html#dask.array.from_array
  dataset = xr.open_dataset(input, engine="zarr", inline_array=False, chunks=in_chunks)

  # Drop static variables (e.g., masks)
  dataset = dataset.drop_vars(
    [name for name, var in dataset.data_vars.items() if time_dim not in var.dims]
  )

  # Get the number of days in the calendar year.
  if calendar == "360_day":
    n_days = 360
  elif calendar == "365_day":
    n_days = 365
  elif calendar == "366_day":
    n_days = 366
  else:
    raise ValueError(f"Unsupported calendar: {calendar}")

  # Check that the time coordinate is:
  #   1. sorted
  #   2. without missing dates or duplicates
  #   3. uniformly spaced, with a frequency that is a multiple of one day.
  #   4. the frequency in days divides the number of days in the calendar year.
  # Finally, it does not have to contain a whole number of years: incomplete years will have missing data.
  datetime_index = dataset[time_dim].to_index()
  freq = xr.infer_freq(datetime_index)
  if freq is None:
    raise ValueError("Could not infer frequency from time index")
  if not freq.endswith("D"):
    raise ValueError(f"Unsupported frequency: {freq}")
  # Pandas has troubles parsing unit abbreviations w/o a number
  freq = "1D" if freq == "D" else freq

  try:
    valid_time_coordinate(
      dataset,
      start_datetime=datetime_index.min().to_pydatetime(),
      end_datetime=datetime_index.max().to_pydatetime(),
      freq=freq,
      inclusive="both",
      time_dim=time_dim,
    )
  except ValidationError as exc:
    if should_raise:
      raise exc from None
    else:
      warnings.warn(f"Datetimes validation failed: {exc}")

  freq = may_parse_timedelta(freq)
  if n_days % freq.days != 0:
    raise ValueError(f"Invalid frequency: {freq}. Must be a multiple of {n_days} days.")
  step_size = n_days // freq.days

  # Convert the window size from days to time steps.
  if window % freq.days != 0:
    raise ValueError(f"Invalid window: {window} days. Must be a multiple of {freq.days} days.")
  window_size = window // freq.days

  # Here we handle leap years.
  # See: https://github.com/pydata/xarray/issues/1844#issuecomment-417855365
  dataset = dataset.convert_calendar(calendar)

  logger.info(f"Computing {freq.days}D-climatology ({calendar=})")
  # Welford's online algorithm yields both the climatological mean and the
  # unbiased sample variance of the anomalies in a single pass over the data.
  dataset_climatology, anomaly_var = _online_climatology(
    dataset,
    time_dim=time_dim,
    step_size=step_size,
    window=window_size,
    climatology_bin_dim=climatology_bin_dim,
    sync_step=sync_step,
  )
  save_configs = {
    "compressor": {"cname": cname, "clevel": clevel},
    "chunk": out_chunks,
  }
  _climatology_delayed_save = save_to_zarr(
    dataset_climatology, climatology_output, configs=save_configs
  )
  anomaly_std = xr.ufuncs.sqrt(anomaly_var)
  _anomaly_std_delayed_save = save_to_zarr(anomaly_std, anomaly_std_output, configs=save_configs)

  # Compute and save the climatology and anomaly std in parallel.
  dask.compute(_climatology_delayed_save, _anomaly_std_delayed_save)

  # Clean up.
  _climatology_delayed_save.close()
  _anomaly_std_delayed_save.close()
  dataset_climatology.close()
  anomaly_std.close()
  client.close()


def _online_climatology(
  dataset: xr.Dataset,
  step_size: int,
  window: int = 1,
  time_dim: str = "time",
  climatology_bin_dim: str = "dayofyear",
  sync_step=False,
) -> tuple[xr.Dataset, xr.Dataset]:
  """
  Average a dataset over consecutive windows of ``step_size`` steps along time.

  The dataset is split along the time dimension into consecutive windows of
  ``step_size`` steps (one per calendar year, given a fixed-day calendar), and
  the windows are combined elementwise using Welford's online algorithm, so
  that only one window is combined at a time. This yields both the running
  (streaming) mean and the unbiased sample variance across windows. The time
  dimension is replaced by ``climatology_bin_dim``, labelled with the day of
  year of the first window.

  Optionally, statistics can be computed over a ``window`` of steps centered on
  each bin, rather than over a single bin. In that case, for each yearly window
  and each bin, the ``window`` neighbouring steps (wrapped around the calendar
  year) are folded into the running statistics, so that each bin aggregates the
  values of a ``window``-step window centered on it, across all years. For
  example, with a centered window of 15 days on daily data, the statistics of
  the 8th of January are computed from the 1st to the 15th of January of every
  year.

  Parameters
  ----------
  dataset : xr.Dataset
      Input dataset, assumed to be sorted and uniformly spaced along the time
      dimension.
  step_size : int
      Number of time steps in each window (i.e., per calendar year).
  window : int, default 1
      Number of steps of the (centered) window over which statistics are
      computed for each bin. ``window=1`` reduces to the plain per-bin
      climatology.
  time_dim : str, default "time"
      Name of the time dimension to average over.
  climatology_bin_dim : str, default "dayofyear"
      Name of the binning dimension replacing the time dimension in the result.

  Returns
  -------
  tuple[xr.Dataset, xr.Dataset]
      A pair ``(mean, variance)``, each of size ``step_size`` along
      ``climatology_bin_dim``. ``mean`` is the climatological average across
      windows and ``variance`` is the unbiased sample variance across windows
      (i.e., normalised by ``n - 1``).
  """
  size = dataset.sizes[time_dim]
  bins = 1 + np.arange(step_size, dtype=int)
  # Offsets of a `window`-step window centered on each bin.
  half = window // 2
  offsets = range(-half, window - half)
  dataset = dataset.assign_coords({climatology_bin_dim: dataset[time_dim].dt.dayofyear})

  avg = None
  m2 = None
  counter = 0
  start = 0

  while start < size:
    end = min(start + step_size, size)
    value = dataset.isel({time_dim: slice(start, end)})
    value = value.drop_vars(time_dim)
    value = value.swap_dims({time_dim: climatology_bin_dim})
    value = value.reindex({climatology_bin_dim: bins}, copy=False)
    # Inner loop over the steps of the window centered on each bin. Each offset
    # contributes the value `offset` steps away (wrapped around the year), so
    # that every bin aggregates a `window`-step window centered on it.
    for offset in offsets:
      shifted = value.roll({climatology_bin_dim: -offset}, roll_coords=False)
      counter += 1
      if avg is None:
        avg = shifted
        # Welford's aggregated squared distance from the running mean (M2 accumulator).
        m2 = xr.zeros_like(avg)
        continue
      # Welford's online update for mean and squared-distance accumulator.
      delta = shifted - avg
      avg = avg + delta / float(counter)
      delta2 = shifted - avg
      m2 = m2 + delta * delta2 # ty: ignore
    if sync_step:
      avg = avg.persist() # ty: ignore
      m2 = m2.persist() # ty: ignore
      maybe_wait(avg)
      maybe_wait(m2)
    start = end

  # Unbiased sample variance (normalised by n - 1).
  # noinspection PyTypeChecker
  variance: xr.Dataset = m2 / float(counter - 1) # ty: ignore

  return avg, variance # ty: ignore
