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


def check_path(exists=True, file_okay=True, dir_okay=True):

  def _check_path(ctx: click.Context, param: click.Parameter, value: str) -> str:
    check_aws_profile(value)  # checking existence opens a botocore session for s3:// paths
    fs, path = fsspec.url_to_fs(value)
    if exists and not fs.exists(path):
      raise click.ClickException(f"{path} does not exist")
    if not file_okay and fs.isfile(path):
      raise click.ClickException(f"{path} is a file")
    if not dir_okay and fs.isdir(path):
      raise click.ClickException(f"{path} is a directory")
    return value

  return _check_path


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
  if inject is not None:
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
    # Fail before any work: an unreachable destination would fail late, or be overwritten.
    raise click.ClickException(f"Could not check whether {path} exists: {exc}") from exc

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


def check_aws_profile(*paths: object, copernicus_marine: bool = False) -> None:
  """Fail before any work if `s3://` paths or Copernicus Marine parts meet an unknown AWS profile.

  Raises:
    click.ClickException: naming the profile (see `settings.check_aws_profile`).
  """
  from arcomake import settings

  try:
    settings.check_aws_profile(paths, copernicus_marine=copernicus_marine)
  except settings.SettingsError as error:
    raise click.ClickException(str(error)) from None
