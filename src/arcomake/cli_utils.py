# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
import tomllib
from typing import Any, TypeVar, overload

import click
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


@overload
def read_configs(
  path: str | pathlib.Path, schema: type[ModelT], inject: dict[str, Any] | None = None
) -> ModelT: ...


@overload
def read_configs(
  path: str | pathlib.Path, schema: None = None, inject: dict[str, Any] | None = None
) -> dict[str, Any]: ...


def read_configs(
  path: str | pathlib.Path,
  schema: type[ModelT] | None = None,
  inject: dict[str, Any] | None = None,
) -> ModelT | dict[str, Any]:
  path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
  logger.info(f"Reading configs from {path}")
  with path.open("rb") as config_file:
    configs = tomllib.load(config_file)
  if inject:
    logger.info(f"Updating configs with {inject}")
    configs.update(inject)
  if schema is not None:
    try:
      return schema.model_validate(configs)
    except ValidationError as exc:
      raise click.ClickException(f"Invalid config at {path}:\n{exc}") from exc
  return configs


def check_output_path(path: pathlib.Path, overwrite: bool = False) -> None:
  # If destination exists and should not overwrite, raise and exit.
  if path.exists():
    if overwrite:
      logger.info(f"Overwriting existing output destination {path}")
    else:
      raise click.ClickException(f"Output destination {path} already exists")
  # Ensure parent directory exists
  path.parent.mkdir(parents=True, exist_ok=True)


def set_default_logger(log_level: str = "info"):
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )
