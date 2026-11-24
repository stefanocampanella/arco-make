# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import tomllib
from collections.abc import Mapping
from typing import Any, TypeVar, overload

import click
import fsspec
import icechunk
from pydantic import BaseModel, ValidationError

from arcomake import icechunk_utils
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


def _merged(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
  """`base` with the values of `update`, tables merged key by key (CLI options injected into a
  recipe, e.g. one bound of a check, keep the rest of its table)."""
  merged = dict(base)
  for key, value in update.items():
    if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
      merged[key] = _merged(merged[key], value)
    else:
      merged[key] = value
  return merged


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
    configs = _merged(configs, inject)
  if schema is not None:
    try:
      return schema.model_validate(configs)
    except ValidationError as exc:
      raise click.ClickException(f"Invalid config at {path}:\n{exc}") from exc
  return configs


def check_destination(
  path: str,
  replace: bool = False,
  save_configs: SaveConfig | None = None,
  repository: icechunk.Repository | None = None,
) -> None:
  """Check an output path before any work: usable by Icechunk, reachable, and either absent or
  (with `replace`) an Icechunk repository, which the run writes a new version of. `repository`:
  the repository at `path`, if the command opened it already (`icechunk_utils.existing_repository`):
  not opened again."""
  if str(path).rstrip("/").lower().endswith(".zip"):
    # Zip outputs were removed with Zarr v3: a `.zip` name would silently become a directory.
    raise click.ClickException(f"{path}: zip outputs are not supported; write a directory or a URL")
  storage_options = getattr(save_configs, "storage_options", None) if save_configs else None
  # Fail before any work: a destination Icechunk cannot use (path, storage options, unknown S3
  # region) or cannot reach would fail at the first write, or be overwritten.
  try:
    storage = icechunk_utils.storage(path, storage_options)
    fs, fs_path = fsspec.url_to_fs(path, **icechunk_utils.fsspec_options(path, storage_options))
    exists = fs.exists(fs_path)
  except Exception as exc:
    raise click.ClickException(f"Could not check whether {path} exists: {exc}") from exc

  if not exists:
    return
  # Outputs are Icechunk repositories, replaced by a new version: anything else is left alone.
  try:
    if repository is None and icechunk.Repository.exists(storage):
      repository = icechunk_utils.open_repository(storage)
    # A repository whose `main` is still its first snapshot holds no version yet: a first run
    # failed there (its branch may hold what it wrote), and a retry needs no --replace.
    unwritten = repository is not None and icechunk_utils.has_no_version(repository)
  except Exception as exc:
    raise click.ClickException(f"Could not check whether {path} is a repository: {exc}") from exc
  if unwritten:
    logger.info(f"{path} holds no version yet: writing its first one")
    return
  if not replace:
    raise click.ClickException(
      f"Output destination {path} already exists (--replace writes a new version of it)"
    )
  if repository is None:
    raise click.ClickException(
      f"Output destination {path} exists and is not an Icechunk repository: remove it first"
    )
  logger.info(f"Writing a new version of {path} (the current one stays in its history)")


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
