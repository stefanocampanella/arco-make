# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
import tomllib
from typing import Any, override

import click

logger = logging.getLogger(__name__)


def read_configs(path: str | pathlib.Path) -> dict[str, Any]:
  path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
  logger.info(f"Reading configs from {path}")
  with path.open("rb") as file:
     configs = tomllib.load(file)
  return configs


class DictParamType(click.ParamType):
  """Click ParamType that parses mappings like "a:1,b:2" into dict[str, int].

  Rules:
  - Comma-separated items, each as key:value.
  - Keys are non-empty strings; surrounding whitespace is ignored.
  - Values must be integers; surrounding whitespace is ignored.
  - Empty string yields an empty dict.
  - Duplicate keys: later values overwrite earlier ones.

  Example:
    --param=a:1,b:2,c:3  -> {"a": 1, "b": 2, "c": 3}
  """

  name = "dict"

  @override
  def convert(self, value, param, ctx):  # type: ignore[override]
    if isinstance(value, dict):
      # Assume it's already a mapping of str->int; perform minimal validation
      result = {}
      for k, v in value.items():
        if not isinstance(k, str) or k.strip() == "":
          self.fail(f"Invalid key in mapping: {k!r}", param, ctx)
        try:
          result[k.strip()] = int(v)
        except Exception:
          self.fail(f"Invalid integer value for key {k!r}: {v!r}", param, ctx)
      return result

    if not isinstance(value, str):
      self.fail(f"Expected string for {self.name}, got {type(value).__name__}", param, ctx)

    text = value.strip()
    if text == "":
      return {}

    items = [p for p in (s.strip() for s in text.split(",")) if p != ""]
    result: dict[str, int] = {}
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
      try:
        result[key] = int(val)
      except Exception:
        self.fail(f"Value for key {key!r} must be an integer, got {val!r}.", param, ctx)
    return result


def check_output_path(path: pathlib.Path, overwrite: bool = False) -> None:
  # If destination exists and should not overwrite, raise and exit.
  if path.exists():
    if overwrite:
      logger.info(f"Overwriting existing output destination {path}")
    else:
      raise ValueError(f"Output destination {path} already exists")
  # Ensure parent directory exists
  path.parent.mkdir(parents=True, exist_ok=True)


def set_default_logger(log_level: str = "info"):
  logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=log_level.upper(),
    force=True,
  )
