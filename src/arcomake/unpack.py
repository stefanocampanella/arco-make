# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from contextlib import nullcontext
from typing import get_args

import click
from dask.diagnostics import ProgressBar
from pydantic import BaseModel, ConfigDict, Field

from arcomake.cli_utils import (
  check_aws_profile,
  check_if_overwriting,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_utils import DaskSetupType, get_client
from arcomake.dataset_utils import (
  ReadConfig,
  SaveConfig,
  open_archive,
  safe_to_zarr,
)
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


class UnpackConfig(BaseModel):
  model_config = ConfigDict(extra="forbid", populate_by_name=True)

  glob: str
  time_dim: str = "time"
  attrs_to_drop: list[str] | None = None
  read: ReadConfig
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig


@click.command()
@click.argument(
  "configs_path",
  required=True,
  type=str,
  callback=check_path(),
)
@click.argument(
  "input_path",
  required=True,
  type=str,
  callback=check_path(file_okay=False),
)
@click.argument(
  "output_path",
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
)
@click.option(
  "--progress/--no-progress",
  "progress",
  help="Whether to display a progress bar",
  default=False,
  is_flag=True,
)
def unpack(
  configs_path: str,
  input_path: str,
  output_path: str,
  overwrite: bool = False,
  dask_setup: DaskSetupType = "mpi",
  log_level: str = "info",
  progress: bool = False,
  use_frisky: bool = False,
):
  """Join the pieces written by separate `download` runs into one Zarr dataset.

  INPUT_PATH is a directory (local or fsspec URL) holding the pieces; the recipe's `glob` selects
  them (e.g. "*.zarr" or "*.zip") and [read] gives the arguments to open each. Variables with the
  time dimension are concatenated in time order; static variables must be identical in every
  piece and are written once. Attributes listed in `attrs_to_drop` are removed before joining.
  The optional [[postprocess]] steps run on the joined dataset, which is saved per [save].
  """

  # Set up logging.
  set_default_logger(log_level)
  # Set up Dask client.
  with get_client(dask_setup=dask_setup, use_frisky=use_frisky):
    try:
      # Read configs, then check the output path (with the recipe's storage options).
      configs = read_configs(configs_path, schema=UnpackConfig)
      check_aws_profile(input_path, output_path)
      check_if_overwriting(output_path, overwrite=overwrite, save_configs=configs.save)
      # Unpack archive
      with open_archive(
        input_path,
        glob=configs.glob,
        time_dim=configs.time_dim,
        attrs_to_drop=configs.attrs_to_drop,
        **configs.read.model_dump(exclude_unset=True),
      ) as dataset:
        progress_bar = (
          ProgressBar if (progress and dask_setup in ["processes", "threads"]) else nullcontext
        )
        with progress_bar():
          # Postproces unpacked dataset
          if configs.postprocess:
            dataset = dataset.arcomake.process(steps=configs.postprocess)
          # Save unpacked dataset
          safe_to_zarr(
            dataset=dataset,
            destination=output_path,
            configs=configs.save,
            compute=True,
          )
    except click.ClickException:
      raise
    except Exception as exc:
      logger.exception("An error occurred while unpacking the archive")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
