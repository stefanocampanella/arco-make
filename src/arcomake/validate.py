# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib

import click
import xarray as xr
from pydantic import BaseModel, ConfigDict

from arcomake.checks import ChecksConfig, ValidationError
from arcomake.cli_utils import (
  read_configs,
  set_default_logger,
)
from arcomake.dataset_utils import ReadConfig

logger = logging.getLogger(__name__)


class ValidateConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  read: ReadConfig
  checks: ChecksConfig


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
@click.option("--fail/--no-fail", default=False)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
def validate(
  config_path: pathlib.Path,
  input_path: pathlib.Path,
  fail: bool = False,
  log_level: str = "info",
):
  """
  Validate a dataset against checks defined in a configuration file.

  The function reads the configuration file, opens the input dataset,
  and applies the validation procedure.
  """
  # Set up logging.
  set_default_logger(log_level)

  # Read configs
  configs = read_configs(config_path, schema=ValidateConfig)

  # Validate the dataset
  if configs.checks:
    try:
      with xr.open_dataset(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        dataset.arcomake.validate(
          checks=configs.checks,
          fail=fail,
        )
    except ValidationError as exc:
      raise click.ClickException(f"Validation failed: {exc}") from exc
    except Exception as exc:
      logger.exception("An error occurred during validation")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
