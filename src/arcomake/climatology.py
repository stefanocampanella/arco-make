import logging
import pathlib
import warnings
from typing import Literal, get_args

import click
import numpy as np
import xarray as xr

from arcomake.checks import ValidationError, valid_time_coordinate
from arcomake.cli_utils import DictParamType, check_output_path, set_default_logger
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
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
  "--skipna/--no-skipna",
  default=False,
  help="Whether to skip NaNs when averaging.",
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
  in_chunks: dict[str, int] | None = None,
  overwrite: bool = False,
  out_chunks: dict[str, int] | None = None,
  climatology_bin_dim: str = "dayofyear",
  cname: str = "lz4",
  clevel: int = 1,
  scheduler_type: SchedulerOptionType = "mpi",
  calendar: Literal["365_day", "366_day", "360_day"] = "365_day",
  skipna: bool = False,
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
  cname : str, default "lz4"
      Name of the compressor used when writing the outputs.
  clevel : int, default 1
      Compression level used when writing the outputs.
  scheduler_type : SchedulerOptionType, default "mpi"
      Type of Dask scheduler to use.
  calendar : {"365_day", "366_day", "360_day"}, default "365_day"
      Fixed-day calendar the dataset is converted to before binning.
  skipna : bool, default False
      Whether to skip NaNs when averaging.
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

  # Here we handle leap years.
  # See: https://github.com/pydata/xarray/issues/1844#issuecomment-417855365
  dataset = dataset.convert_calendar(calendar)
  dataset = dataset.chunk({time_dim: step_size})

  logger.info(f"Computing {freq.days}D-climatology ({calendar=}, {skipna=})")
  dataset_climatology = climatological_average(
    dataset,
    time_dim=time_dim,
    step_size=step_size,
    climatology_bin_dim=climatology_bin_dim,
    skipna=skipna,
  )
  save_configs = {
    "compressor": {"cname": cname, "clevel": clevel},
    "chunk": out_chunks,
  }
  save_to_zarr(dataset_climatology, climatology_output, configs=save_configs)

  # Read back the climatology to compute the anomaly std.
  dataset_climatology = xr.open_dataset(climatology_output, engine="zarr", chunks=in_chunks)

  # Number of bins in the climatology (must match how climatology_bin_dim was constructed)
  nbins = dataset_climatology.sizes[climatology_bin_dim]

  # Day-of-year for every timestamp in the original dataset
  doy = dataset[time_dim].dt.dayofyear

  # Map each day-of-year to its N-day bin index (0-based), matching the
  # same binning convention used when the climatology was computed.
  # The modulo keeps late-year days that don't fill a full bin wrapped
  # consistently with the climatology's bin count.
  bin_index = ((doy - 1) // step_size) % nbins

  # bin_index is already a DataArray indexed by time_dim (inherited from doy),
  # so this is a vectorized/pointwise selection: the result will be indexed
  # by time_dim instead of climatology_bin_dim.
  dataset_climatology = dataset_climatology.sel(climatology_dim=bin_index)
  dataset_climatology = dataset_climatology.rename_dims({climatology_bin_dim: time_dim})
  dataset_climatology = dataset_climatology.chunk({time_dim: step_size})

  logger.info(f"Computing {freq.days}D-climatology anomaly std ({calendar=}, {skipna=})")
  anomaly = dataset - dataset_climatology
  anomaly_sq = anomaly * anomaly
  anomaly_var = climatological_average(
    anomaly_sq,
    time_dim=time_dim,
    step_size=step_size,
    climatology_bin_dim=climatology_bin_dim,
    skipna=skipna,
  )
  anomaly_std = xr.ufuncs.sqrt(anomaly_var)

  save_to_zarr(anomaly_std, anomaly_std_output, configs=save_configs)

  client.close()


def climatological_average(
  dataset: xr.Dataset,
  step_size: int,
  time_dim: str = "time",
  climatology_bin_dim: str = "dayofyear",
  skipna=False,
) -> xr.Dataset:
  """
  Average a dataset over consecutive windows of ``step_size`` steps along time.

  The dataset is split along the time dimension into consecutive windows of
  ``step_size`` steps (one per calendar year, given a fixed-day calendar), and
  the windows are averaged elementwise using a running (streaming) mean, so
  that only one window is combined at a time. The time dimension is replaced by
  ``climatology_bin_dim``, labelled with the day of year of the first window.

  Parameters
  ----------
  dataset : xr.Dataset
      Input dataset, assumed to be sorted and uniformly spaced along the time
      dimension.
  step_size : int
      Number of time steps in each window (i.e., per calendar year).
  time_dim : str, default "time"
      Name of the time dimension to average over.
  climatology_bin_dim : str, default "dayofyear"
      Name of the binning dimension replacing the time dimension in the result.
  skipna : bool, default False
      Whether to skip NaNs when averaging, replacing them with the running mean.

  Returns
  -------
  xr.Dataset
      Climatological average, of size ``step_size`` along ``climatology_bin_dim``.
  """
  size = dataset.sizes[time_dim]
  start = 0
  end = min(step_size, size)
  counter = 1
  dataset = dataset.assign_coords({climatology_bin_dim: dataset[time_dim].dt.dayofyear})
  avg = dataset.isel({time_dim: slice(start, end)})
  avg = avg.drop_vars(time_dim)
  avg = avg.swap_dims({time_dim: climatology_bin_dim})
  avg = avg.reindex({climatology_bin_dim: 1 + np.arange(step_size, dtype=int)}, copy=False)
  while True:
    start = end
    end = min(start + step_size, size)
    if start < size:
      counter += 1
      value = dataset.isel({time_dim: slice(start, end)})
      value = value.drop_vars(time_dim)
      value = value.swap_dims({time_dim: climatology_bin_dim})
      if skipna:
        value = value.reindex(
          {climatology_bin_dim: 1 + np.arange(step_size, dtype=int)}, copy=False
        )
        value = value.where(value.notnull(), avg)
      avg += (value - avg) / float(counter)
      avg.persist()
    else:
      break

  return avg
