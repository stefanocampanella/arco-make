# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from typing import get_args

import click
from pydantic import BaseModel, ConfigDict, Field

from arcomake import icechunk_utils
from arcomake.cli_utils import (
  check_aws_profile,
  check_destination,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_utils import DaskSetupType, get_client
from arcomake.dataset_utils import ReadConfig, SaveConfig, open_input, write_output
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


class ProcessConfig(BaseModel):
  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  read: ReadConfig = Field(default_factory=ReadConfig)
  process: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig = Field(default_factory=SaveConfig)


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
def process(
  configs_path: str,
  input_path: str,
  output_path: str,
  replace: bool = False,
  dask_setup: DaskSetupType = "mpi",
  use_frisky: bool = False,
  log_level: str = "info",
):
  """Apply recipe steps to a dataset and save the result to Zarr.

  CONFIGS_PATH is a TOML recipe with a [read] table (arguments of xarray.open_dataset), a list of
  [[process]] steps (see `arco-make steps`; e.g. a time mean) and a [save] table. INPUT_PATH and
  OUTPUT_PATH may be local paths or fsspec URLs.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  with get_client(dask_setup=dask_setup, use_frisky=use_frisky):
    try:
      # Read configs, then check the output path (with the recipe's storage options).
      configs = read_configs(configs_path, schema=ProcessConfig)
      check_aws_profile(input_path, output_path)
      check_destination(output_path, replace=replace, save_configs=configs.save)
      # Read input dataset
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with open_input(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Process dataset
        if configs.process:
          dataset = dataset.arcomake.process(steps=configs.process)
        # Save stats
        write_output(
          dataset, output_path, configs.save, message=f"process {input_path}", command="process"
        )
    except click.ClickException:
      raise
    except icechunk_utils.PromotionError as exc:
      raise click.ClickException(str(exc)) from exc
    except Exception as exc:
      logger.exception("An error occurred while processing dataset")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
