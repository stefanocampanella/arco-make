# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import tomllib
from typing import Any, TypeVar, overload

import click
import fsspec
from pydantic import BaseModel, ValidationError

from arcomake.dataset_utils import SaveConfig

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_configs_path(ctx: click.Context, param: click.Parameter, value: str) -> str:
  fs, path = fsspec.url_to_fs(value)
  if not fs.exists(path):
    raise click.ClickException(f"Config path {path} does not exist")
  if not fs.isfile(path):
    raise click.ClickException(f"Config path {path} is not a file")
  return str(path)


@overload
def read_configs(
  path: str, schema: type[ModelT], inject: dict[str, Any] | None = None
) -> ModelT: ...


@overload
def read_configs(
  path: str, schema: None = None, inject: dict[str, Any] | None = None
) -> dict[str, Any]: ...


def read_configs(
  path: str,
  schema: type[ModelT] | None = None,
  inject: dict[str, Any] | None = None,
) -> ModelT | dict[str, Any]:
  logger.info(f"Reading configs from {path}")
  with fsspec.open(urlpath=path, mode="rb") as config_file:
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


def check_if_overwriting(
  path: str, overwrite: bool = False, save_configs: SaveConfig | None = None
) -> None:
  storage_options = getattr(save_configs, "storage_options", None) if save_configs else None
  fs, fs_path = fsspec.url_to_fs(path, **(storage_options or {}))
  try:
    exists = fs.exists(fs_path)
  except Exception as exc:
    logger.warning(f"Could not check existence of {path}: {exc}")
    exists = False

  if exists:
    if overwrite:
      logger.info(f"Overwriting existing output destination {path}")
    else:
      raise click.ClickException(f"Output destination {path} already exists")


def set_default_logger(log_level: str = "info"):
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )
