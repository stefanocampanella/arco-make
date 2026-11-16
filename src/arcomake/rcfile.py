# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The user's settings file, `~/.arcomakerc`: environment variables for every command.

It holds what arco-make and its libraries read from the environment: service credentials
(`COPERNICUSMARINE_SERVICE_USERNAME`, `CDSAPI_URL`, `CDSAPI_KEY`, `AWS_*`...), `TMPDIR`, Dask
settings (`DASK_*`). One `KEY=VALUE` per line, optionally after `export`, so that a shell can
`source` the same file::

  # ~/.arcomakerc (chmod 600: it holds secrets)
  export TMPDIR=/scratch/$USER/tmp
  export CDSAPI_URL=https://ewds.climate.copernicus.eu/api
  export CDSAPI_KEY='...'

Rules:
- Variables already in the environment win: the file only fills in what is missing.
- `ARCOMAKE_RC` names another file; set to an empty string, no file is read.
- Values in single quotes are literal; others may use `$VAR` and `${VAR}` (unknown names are kept
  as written). `#` starts a comment outside quotes. Other lines are skipped with a warning naming
  the line number only.
- Values are never logged. A file readable by other users is used, with a warning.
"""

import logging
import os
import pathlib
import re
import stat
import tempfile
import warnings
from collections.abc import MutableMapping

logger = logging.getLogger(__name__)

DEFAULT_PATH = pathlib.Path("~/.arcomakerc")
ENV_VAR = "ARCOMAKE_RC"

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def rc_path(environ: MutableMapping[str, str] | None = None) -> pathlib.Path | None:
  """The settings file to read: `$ARCOMAKE_RC`, or `~/.arcomakerc`; None if disabled."""
  environ = os.environ if environ is None else environ
  if ENV_VAR in environ:
    value = environ[ENV_VAR].strip()
    return pathlib.Path(value).expanduser() if value else None
  return DEFAULT_PATH.expanduser()


def _parse_value(raw: str, environ: MutableMapping[str, str]) -> str | None:
  raw = raw.strip()
  if raw.startswith("'"):
    end = raw.find("'", 1)
    return raw[1:end] if end > 0 else None
  if raw.startswith('"'):
    match = re.match(r'"((?:[^"\\]|\\.)*)"', raw)
    if match is None:
      return None
    value = re.sub(r"\\(.)", r"\1", match.group(1))
  else:
    value = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
  return _expand(value, environ)


def _expand(value: str, environ: MutableMapping[str, str]) -> str:
  def replace(match: re.Match[str]) -> str:
    name = match.group(1) or match.group(2)
    return environ.get(name, match.group(0))

  return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, value)


def parse(text: str, environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
  """The settings of a file's text, in order; `$VAR` expands from `environ` and earlier lines."""
  base = dict(os.environ if environ is None else environ)
  settings: dict[str, str] = {}
  for number, line in enumerate(text.splitlines(), start=1):
    if not line.strip() or line.lstrip().startswith("#"):
      continue
    match = _LINE.match(line)
    value = _parse_value(match.group(2), base | settings) if match else None
    if match is None or value is None:
      warnings.warn(f"arco-make settings file: line {number} is not KEY=VALUE; skipped")
      continue
    settings[match.group(1)] = value
  return settings


def load(
  path: pathlib.Path | None = None, environ: MutableMapping[str, str] | None = None
) -> list[str]:
  """Set the variables of the settings file that `environ` lacks; return their names.

  Args:
    path: The file; by default `rc_path()`. A missing file is not an error.
    environ: The environment to fill; by default `os.environ`.
  """
  environ = os.environ if environ is None else environ
  path = rc_path(environ) if path is None else path
  if path is None or not path.is_file():
    return []
  mode = path.stat().st_mode
  if mode & (stat.S_IRWXG | stat.S_IRWXO):
    warnings.warn(
      f"{path} is accessible to other users; it may hold secrets: run `chmod 600 {path}`"
    )
  settings = parse(path.read_text(), environ)
  added = [name for name in settings if name not in environ]
  for name in added:
    environ[name] = settings[name]
  logger.debug(f"Settings file {path}: set {', '.join(added) or 'nothing'}")
  if environ is os.environ:
    _apply_cached_settings(added)
  return added


def _apply_cached_settings(names: list[str]) -> None:
  """Libraries read some variables once, at import: make them see the new values."""
  if "TMPDIR" in names:
    tempfile.tempdir = None  # `tempfile.gettempdir()` caches its first answer.
  if any(name.startswith("DASK_") for name in names):
    import dask.config

    dask.config.refresh()
