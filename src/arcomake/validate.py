# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging

import click
from pydantic import BaseModel, ConfigDict

from arcomake.checks import ChecksConfig, ValidationError
from arcomake.cli_utils import (
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dataset_utils import ReadConfig, open_input

logger = logging.getLogger(__name__)


class ValidateConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  read: ReadConfig
  checks: ChecksConfig


@click.command()
@click.argument("configs_path", required=True, type=str, callback=check_path())
@click.argument(
  "input_path",
  required=True,
  type=str,
  callback=check_path(),
)
@click.option("--fail/--no-fail", default=False)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
def validate(
  configs_path: str,
  input_path: str,
  fail: bool = False,
  log_level: str = "info",
):
  """Check a dataset against the [checks] of a recipe.

  Checks: `valid_time_coordinate` (complete, sorted, without duplicates, between `start` and
  `end` excluded), `valid_global_ecmwf_coordinates` (latitudes increasing within [-90, 90],
  longitudes within [0, 360)) and `ensure_no_nans` (no missing value inside each mask).
  Failed checks are warnings, or an error with --fail.
  """
  # Set up logging.
  set_default_logger(log_level)

  # Read configs
  configs = read_configs(configs_path, schema=ValidateConfig)

  # Validate the dataset
  if configs.checks:
    try:
      with open_input(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        dataset.arcomake.validate(
          checks=configs.checks,
          fail=fail,
        )
    except ValidationError as exc:
      raise click.ClickException(f"Validation failed: {exc}") from exc
    except Exception as exc:
      logger.exception("An error occurred during validation")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
