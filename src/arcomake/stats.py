# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from typing import get_args

import click
import xarray as xr

from arcomake.cli_utils import DictParamType, check_output_path, set_default_logger
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import save_to_zarr

logger = logging.getLogger(__name__)


def mean(dataset: xr.Dataset, time_dim: str = "time", **kwargs) -> xr.Dataset:
  return dataset.mean(dim=time_dim, **kwargs)


def std(dataset: xr.Dataset, time_dim: str = "time", **kwargs) -> xr.Dataset:
  return dataset.std(dim=time_dim, **kwargs)


def diff_std(dataset: xr.Dataset, time_dim: str = "time", **kwargs) -> xr.Dataset:
  """
  Compute the standard deviation of one-step differences along the time dimension.

  Notes
  -----
  Given a sequence ``{x_i}_{i = 1, ..., N}``, the mean of the diffs
  ``{x_i - x_{i-1}}`` is the sum of a telescopic series divided by the number of
  terms, i.e. ``(x_N - x_1) / (N - 1)``, which becomes negligible for large N.
  Also, GraphCast-like models compute the increment between the present and next
  system state rescaled by diff_std, which amounts to standardizing the targets.
  Whatever the rationale, one can reasonably approximate the mean with zero here.
  """
  dataset_diff = dataset.diff(dim=time_dim)
  dataset_diff_var = (dataset_diff * dataset_diff).mean(dim=time_dim, **kwargs)
  dataset_diff_std = xr.ufuncs.sqrt(dataset_diff_var)
  return dataset_diff_std


StatsRegistry = {
  "mean": mean,
  "std": std,
  "diff_std": diff_std,
}


@click.command()
@click.argument("stats", required=True, type=click.Choice(StatsRegistry.keys()))
@click.argument(
  "input",
  required=True,
  type=click.Path(
    path_type=pathlib.Path, file_okay=True, dir_okay=True, exists=True, readable=True
  ),
)
@click.argument(
  "output",
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
  "--skipna/--no-skipna",
  default=False,
  help="Whether to skip NaNs when averaging.",
  show_default=True,
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
)
def compute_stats(
  stats: str,
  input: pathlib.Path,
  output: pathlib.Path,
  time_dim: str = "time",
  in_chunks: dict[str, int] | None = None,
  overwrite: bool = False,
  out_chunks: dict[str, int] | None = None,
  cname: str = "lz4",
  clevel: int = 1,
  scheduler_type: SchedulerOptionType = "mpi",
  skipna: bool = False,
  log_level: str = "info",
):
  """
  Compute a statistic of a Zarr dataset over time and save it to Zarr.

  Data variables without the time dimension (e.g., static masks) are dropped
  before computing the statistic. The output is overwritten if it already exists.

  Parameters
  ----------
  stats : {"mean", "std", "diff_std"}
      Statistic to compute.
  input : pathlib.Path
      Path to the input Zarr dataset.
  output : pathlib.Path
      Path to the output Zarr dataset.
  time_dim : str, default "time"
      Name of the time dimension to reduce over.
  in_chunks : dict[str, int] or None, default None
      Chunking specs used when reading the input dataset.
  out_chunks : dict[str, int] or None, default None
      Chunking specs used when writing the output dataset.
  skipna : bool, default False
      Whether to skip NaNs when computing the statistic.
  log_level : {"debug", "info", "warning", "error", "critical"}, default "info"
      Logging verbosity.
  scheduler_type : SchedulerOptionType, default "mpi"
      Type of Dask scheduler to use.
  cname : str, default "lz4"
      Name of the compressor used when writing the output.
  clevel : int, default 1
      Compression level used when writing the output.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  client = get_client(scheduler_type=scheduler_type)

  if stats not in StatsRegistry:
    raise ValueError(f"Invalid stats type: {stats}")

  # Check paths.
  if not input.exists():
    raise ValueError(f"Input path {input} does not exist")
  check_output_path(output, overwrite=overwrite)

  logger.info(f"Opening input dataset from {input} with chunks={in_chunks}")
  # As the Dask graph tends to be huge it's important to avoid inline_array=True,
  # see: https://docs.dask.org/en/latest/generated/dask.array.from_array.html#dask.array.from_array
  dataset = xr.open_dataset(input, engine="zarr", inline_array=False, chunks=in_chunks)
  dataset = dataset.drop_vars(
    [name for name, var in dataset.data_vars.items() if time_dim not in var.dims]
  )

  stats_ds = StatsRegistry[stats](dataset, time_dim=time_dim, skipna=skipna, keep_attrs=True)
  save_configs = {
    "compressor": {"cname": cname, "clevel": clevel},
    "chunk": out_chunks,
    "consolidated": True,
  }
  store = save_to_zarr(stats_ds, output, configs=save_configs, compute=True)

  # Clean up
  store.close()
  stats_ds.close()
  client.close()
