# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from datetime import datetime
from typing import Self

import click
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arcomake.checks import ChecksConfig, ValidationError
from arcomake.checks import validate as validate_dataset
from arcomake.cli_utils import (
  read_configs,
  set_default_logger,
)
from arcomake.dataset_utils import ReadConfig

logger = logging.getLogger(__name__)


class ValidateConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  start: datetime | None = Field(default=None)
  end: datetime | None = Field(default=None)
  read: ReadConfig
  checks: ChecksConfig

  @model_validator(mode="after")
  def validate_date_range(self) -> Self:
    if self.start is not None and self.end is not None and self.start > self.end:
      raise ValueError("start datetime must be before or equal to end datetime")
    return self


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
@click.option(
  "--start",
  "start_datetime",
  default=None,
  help="Override start datetime of the timeseries.",
  type=click.DateTime(),
)
@click.option(
  "--end",
  "end_datetime",
  default=None,
  help="Override end datetime of the timeseries.",
  type=click.DateTime(),
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
def validate(
  config_path: pathlib.Path,
  input_path: pathlib.Path,
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  log_level: str = "info",
):
  """
  Validate a dataset against checks defined in a configuration file.

  The function reads the configuration file, opens the input dataset,
  and applies the validation procedure.
  """
  # Set up logging.
  set_default_logger(log_level)

  # Read configs and inject start and end time.
  update_time_interval = {}
  if start_datetime is not None:
    update_time_interval["start"] = start_datetime
  if end_datetime is not None:
    update_time_interval["end"] = end_datetime
  configs = read_configs(config_path, inject=update_time_interval, schema=ValidateConfig)

  # Validate the dataset
  if configs.checks:
    try:
      with xr.open_dataset(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        validate_dataset(
          dataset=dataset,
          checks=configs.checks,
          start_datetime=start_datetime,
          end_datetime=end_datetime,
        )
    except ValidationError as exc:
      raise click.ClickException(f"Validation failed: {exc}") from exc
