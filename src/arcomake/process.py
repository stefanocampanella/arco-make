# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from typing import get_args

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
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


class ProcessConfig(BaseModel):
  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  time_dim: str = "time"
  skipna: bool = Field(
    default=False,
    description="Whether to skip NaNs when computing the statistic.",
  )
  read: ReadConfig = Field(default_factory=ReadConfig)
  process: list[ProcessingStepConfig] = Field(default_factory=list)
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
def process(
  config_path: pathlib.Path,
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

  # Check output path.
  check_output_path(output_path, overwrite=overwrite)

  # Read configs
  configs = read_configs(config_path, schema=ProcessConfig)

  # Set up Dask client.
  with get_client(scheduler_type=scheduler_type):
    try:
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with xr.open_dataset(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Process dataset
        if configs.process:
          dataset = dataset.arcomake.process(steps=configs.process)
        # Save stats
        save_to_zarr(
          dataset=dataset,
          path=output_path,
          configs=configs.save,
          compute=True,
        )
    except Exception as exc:
      logger.exception("An error occurred while processing dataset")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
