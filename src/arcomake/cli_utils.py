# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
import tomllib
from typing import Any, TypeVar, overload, override

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
  if inject is not None:
    logger.info(f"Updating configs with {inject}")
    configs.update(inject)
  if schema is not None:
    try:
      return schema.model_validate(configs)
    except ValidationError as exc:
      raise click.ClickException(f"Invalid config at {path}:\n{exc}") from exc
  return configs


class DictParamType(click.ParamType[dict[str, int | float | bool | str]]):
  """Click ParamType that parses mappings like "a:1,b:2.5,c:true,d:x" into dict[str, int | float | bool | str].

  Rules:
  - Comma-separated items, each given as "key:value".
  - Keys are non-empty strings; surrounding whitespace is ignored.
  - Values are parsed, in order, as booleans ("true"/"false", case-insensitive),
    integers, floats, and finally strings; surrounding whitespace is ignored.
  - Empty string yields an empty dict.
  - Duplicate keys: later values overwrite earlier ones.

  Example:
    --param=a:1,b:2.5,c:true,d:x -> {"a": 1, "b": 2.5, "c": True, "d": "x"}
  """

  name = "dictionary"

  @staticmethod
  def _parse_value(val: str) -> int | float | bool | str:
    """Parse a string into an int, float, bool, or str (in that order)."""
    lowered = val.lower()
    if lowered == "true":
      return True
    if lowered == "false":
      return False
    try:
      return int(val)
    except ValueError:
      pass
    try:
      return float(val)
    except ValueError:
      pass
    return val

  @override
  def convert(self, value, param, ctx):  # type: ignore[override]
    if isinstance(value, dict):
      # Assume it's already a mapping of str->int | float | bool | str and perform minimal validation
      result = {}
      for k, v in value.items():
        if not isinstance(k, str) or k.strip() == "":
          self.fail(f"Invalid key in mapping: {k!r}", param, ctx)
        if not isinstance(v, (int, float, bool, str)):
          self.fail(f"Invalid value for key {k!r}: {v!r}", param, ctx)
        result[k.strip()] = v
      return result

    if not isinstance(value, str):
      self.fail(f"Expected string for {self.name.upper()}, got {type(value).__name__}", param, ctx)

    text = value.strip()
    # Special case: empty string is treated as empty dict
    if text == "":
      return {}

    items = [p for p in (s.strip() for s in text.split(",")) if p != ""]
    result = {}
    for item in items:
      if ":" not in item:
        self.fail(
          f"Invalid item {item!r}. Expected 'key:value' pairs separated by commas.",
          param,
          ctx,
        )
      key, val = item.split(":", 1)
      key = key.strip()
      val = val.strip()
      if key == "":
        self.fail("Empty key is not allowed in mapping.", param, ctx)
      result[key] = self._parse_value(val)
    return result


class ListParamType(click.ParamType[list[int | float | bool | str]]):
  """Click ParamType that parses lists like "foo,bar,1,2.5,true" into list[int | float | bool | str].

  Rules:
  - Comma-separated items; surrounding whitespace of each item is ignored.
  - Values are parsed, in order, as booleans ("true"/"false", case-insensitive),
    integers, floats, and finally strings.
  - Empty string yields an empty list.

  Example:
    --param=foo,bar,1,2.5,true -> ["foo", "bar", 1, 2.5, True]
  """

  name = "list"

  @staticmethod
  def _parse_value(val: str) -> int | float | bool | str:
    """Parse a string into an int, float, bool, or str (in that order)."""
    lowered = val.lower()
    if lowered == "true":
      return True
    if lowered == "false":
      return False
    try:
      return int(val)
    except ValueError:
      pass
    try:
      return float(val)
    except ValueError:
      pass
    return val

  @override
  def convert(self, value, param, ctx):  # type: ignore[override]
    if isinstance(value, (list, tuple)):
      # Assume it's already a sequence of int | float | bool | str and perform minimal validation
      result = []
      for v in value:
        if not isinstance(v, (int, float, bool, str)):
          self.fail(f"Invalid value in list: {v!r}", param, ctx)
        result.append(v)
      return result

    if not isinstance(value, str):
      self.fail(f"Expected string for {self.name.upper()}, got {type(value).__name__}", param, ctx)

    text = value.strip()
    # Special case: empty string is treated as empty list
    if text == "":
      return []

    items = [p for p in (s.strip() for s in text.split(",")) if p != ""]
    return [self._parse_value(item) for item in items]


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
