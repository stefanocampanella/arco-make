# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from typing import get_args

import click
from pydantic import BaseModel, ConfigDict, Field

from arcomake.cli_utils import (
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import (
  ReadConfig,
  SaveConfig,
  open_archive,
  save_to_zarr,
)
from arcomake.processing import ProcessingStepConfig, process

logger = logging.getLogger(__name__)


class ArchiveConfig(BaseModel):
  model_config = ConfigDict(extra="allow")

  attrs_to_drop: list[str] | None = None
  read: ReadConfig


class UnpackConfig(BaseModel):
  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  time_dim: str = "time"
  attrs_to_drop: list[str] | None = None
  read: ReadConfig
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
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
  "output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, writable=True),
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
)
def unpack(
  config_path: pathlib.Path,
  input_path: pathlib.Path,
  output_path: pathlib.Path,
  overwrite: bool = False,
  scheduler_type: SchedulerOptionType = "mpi",
  log_level: str = "info",
):
  """
  Unpack a collection of zip files containing Zarr datasets into a single Zarr dataset.

  The command reads zipped Zarr fragments with xarray.open_mfdataset(engine='zarr'),
  optionally slices the time range, rechunks, and saves to a DirectoryStore at the
  given output path.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Check output path
  check_output_path(output_path, overwrite=overwrite)

  # Read configs
  configs = read_configs(config_path, schema=UnpackConfig)

  # Set up Dask client.
  with get_client(scheduler_type=scheduler_type):
    # Unpack archive
    try:
      with open_archive(
        input_path,
        time_dim=configs.time_dim,
        attrs_to_drop=configs.attrs_to_drop,
        **configs.read.model_dump(exclude_unset=True),
      ) as dataset:
        # Postproces unpacked dataset
        if configs.postprocess:
          dataset = process(dataset=dataset, steps=configs.postprocess)
        # Save unpacked dataset
        save_to_zarr(
          dataset=dataset,
          path=output_path,
          configs=configs.save,
          compute=True,
        )
    except Exception as exc:
      logger.exception("An error occurred while unpacking the archive")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
