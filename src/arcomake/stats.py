# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from typing import Literal, get_args

import click
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from arcomake.cli_utils import (
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import ReadConfig, SaveConfig, save_to_zarr
from arcomake.processing import ProcessingStepConfig, process

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

StatsType = Literal["mean", "std", "diff_std"]


class StatsConfig(BaseModel):
  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  time_dim: str = "time"
  skipna: bool = Field(
    default=False,
    description="Whether to skip NaNs when computing the statistic.",
  )
  read: ReadConfig = Field(default_factory=ReadConfig)
  preprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig = Field(default_factory=SaveConfig)


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(
    path_type=pathlib.Path,
    resolve_path=True,
    exists=True,
    dir_okay=False,
  ),
)
@click.argument("stats", required=True, type=click.Choice(get_args(StatsType)))
@click.argument(
  "input_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, exists=True),
)
@click.argument(
  "output_path",
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
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
)
def compute_stats(
  config_path: pathlib.Path,
  stats: StatsType,
  input_path: pathlib.Path,
  output_path: pathlib.Path,
  overwrite: bool = False,
  scheduler_type: SchedulerOptionType = "mpi",
  log_level: str = "info",
):
  """
  Compute a statistic of a Zarr dataset over time and save it to Zarr.

  Parameters
  ----------
  config_path : pathlib.Path
      Path to TOML configuration file containing configuration options.
  stats : {"mean", "std", "diff_std"}
      Statistic to compute.
  input_path : pathlib.Path
      Path to the input Zarr dataset.
  output_path : pathlib.Path
      Path to the output Zarr dataset.
  overwrite : bool, default False
      Whether to overwrite existing output.
  scheduler_type : SchedulerOptionType, default "mpi"
      Type of Dask scheduler to use.
  log_level : {"debug", "info", "warning", "error", "critical"}, default "info"
      Logging verbosity.
  """

  # Set up logging.
  set_default_logger(log_level)

  if stats not in StatsRegistry:
    raise click.ClickException(f"Invalid stats type: {stats}")

  # Check output path.
  check_output_path(output_path, overwrite=overwrite)

  # Read configs
  configs = read_configs(config_path, schema=StatsConfig)

  # Set up Dask client.
  with get_client(scheduler_type=scheduler_type):
    try:
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with xr.open_dataset(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Preprocess input dataset
        if configs.preprocess:
          dataset = process(dataset=dataset, steps=configs.preprocess)
        # Compute stats
        with StatsRegistry[stats](
          dataset, time_dim=configs.time_dim, skipna=configs.skipna, keep_attrs=True
        ) as stats_ds:
          # Postprocess stats
          if configs.postprocess:
            stats_ds = process(dataset=stats_ds, steps=configs.postprocess)
          # Save stats
          store = save_to_zarr(
            dataset=stats_ds,
            path=output_path,
            configs=configs.save,
            compute=True,
          )
          # Close the store, see: https://github.com/pydata/xarray/issues/4076
          store.close()
    except Exception as exc:
      logger.exception(f"An error occurred while computing {stats}")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
