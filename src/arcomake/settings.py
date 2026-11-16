# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""User settings: credentials and machine settings from the environment and `~/.arcomakerc`.

The settings are the environment variables that arco-make's libraries read (Copernicus Marine and
CDS/EWDS credentials, the default S3 profile, `TMPDIR`, `DASK_*`). `ArcoMakeSettings` reads them
from the environment and from a dotenv file, `~/.arcomakerc` (or the file named by `ARCOMAKE_RC`;
an empty value disables it). `export KEY=VALUE` lines are allowed, so job scripts can `source` the
same file::

  # ~/.arcomakerc (chmod 600: it holds secrets)
  export TMPDIR=/scratch/${USER}/tmp
  export AWS_PROFILE=my-bucket-profile
  export CDSAPI_URL=https://ewds.climate.copernicus.eu/api
  export CDSAPI_KEY='...'

`apply()` then exports to the process environment the values that came from the file: variables
already in the environment win. Secrets are `SecretStr`, never logged. Keys that are not settings
are ignored with a warning naming the key (`DASK_*` keys go to Dask's configuration).
"""

import logging
import os
import pathlib
import stat
import tempfile
import warnings
from collections.abc import MutableMapping

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_PATH = pathlib.Path("~/.arcomakerc")
ENV_VAR = "ARCOMAKE_RC"


class ArcoMakeSettings(BaseSettings):
  """Settings read from the environment, then from the settings file (the environment wins)."""

  model_config = SettingsConfigDict(case_sensitive=True, extra="allow", env_file=None)

  tmpdir: str | None = Field(
    default=None,
    validation_alias="TMPDIR",
    description="Temporary files (checkpoints, downloads): on disk, not in RAM",
  )
  aws_profile: str | None = Field(
    default=None,
    validation_alias="AWS_PROFILE",
    description="Default profile of `s3://` paths (recipes' `storage_options` override it)",
  )
  # botocore reads AWS_DEFAULT_PROFILE first, then AWS_PROFILE.
  aws_default_profile: str | None = Field(default=None, validation_alias="AWS_DEFAULT_PROFILE")
  aws_config_file: str | None = Field(default=None, validation_alias="AWS_CONFIG_FILE")
  aws_shared_credentials_file: str | None = Field(
    default=None, validation_alias="AWS_SHARED_CREDENTIALS_FILE"
  )
  google_application_credentials: str | None = Field(
    default=None, validation_alias="GOOGLE_APPLICATION_CREDENTIALS"
  )
  copernicusmarine_service_username: str | None = Field(
    default=None, validation_alias="COPERNICUSMARINE_SERVICE_USERNAME"
  )
  copernicusmarine_service_password: SecretStr | None = Field(
    default=None, validation_alias="COPERNICUSMARINE_SERVICE_PASSWORD"
  )
  cdsapi_url: str | None = Field(default=None, validation_alias="CDSAPI_URL", pattern=r"^https?://")
  cdsapi_key: SecretStr | None = Field(default=None, validation_alias="CDSAPI_KEY")


def settings_path(environ: MutableMapping[str, str] | None = None) -> pathlib.Path | None:
  """The settings file: `$ARCOMAKE_RC`, or `~/.arcomakerc`; None if disabled."""
  environ = os.environ if environ is None else environ
  if ENV_VAR in environ:
    value = environ[ENV_VAR].strip()
    return pathlib.Path(value).expanduser() if value else None
  return DEFAULT_PATH.expanduser()


def read(path: pathlib.Path | None) -> ArcoMakeSettings:
  """The settings from the environment and `path` (if it exists).

  Raises:
    pydantic.ValidationError: for an invalid value; the message names the key, not the secret.
  """
  env_file = path if path is not None and path.is_file() else None
  if env_file is not None and env_file.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
    warnings.warn(
      f"{env_file} is accessible to other users; it may hold secrets: run `chmod 600 {env_file}`"
    )
  return ArcoMakeSettings(_env_file=env_file)


def apply(settings: ArcoMakeSettings, environ: MutableMapping[str, str] | None = None) -> list[str]:
  """Export the settings the environment lacks (they came from the file); return their names."""
  environ = os.environ if environ is None else environ
  values: dict[str, str] = {}
  for name, field in ArcoMakeSettings.model_fields.items():
    value = getattr(settings, name)
    if value is not None:
      alias = str(field.validation_alias)
      values[alias] = value.get_secret_value() if isinstance(value, SecretStr) else value
  for key, value in (settings.model_extra or {}).items():
    if key.startswith("DASK_"):
      values[key] = str(value)
    else:
      warnings.warn(f"arco-make settings file: {key!r} is not a setting; ignored")
  added = [key for key in values if key not in environ]
  for key in added:
    environ[key] = values[key]
  logger.debug(f"Settings from the file: {', '.join(added) or 'none'}")
  if environ is os.environ:
    _refresh_cached_settings(added)
  return added


def load() -> list[str]:
  """Read the settings file and export what the environment lacks; return the names exported."""
  path = settings_path()
  if path is None or not path.is_file():
    return []
  return apply(read(path))


def missing_aws_profile() -> str | None:
  """The profile botocore will use, if it cannot find it; else None.

  botocore takes `AWS_DEFAULT_PROFILE`, then `AWS_PROFILE`. Every botocore session reads it, the
  Copernicus Marine toolbox's included: it signs nothing and uses its own endpoint, but creating
  its session fails on an unknown profile, after minutes of retries. A container without
  `~/.aws` is the usual cause.
  """
  import botocore.session

  session = botocore.session.Session()
  profile = session.get_config_variable("profile")
  if profile is None or profile in session.available_profiles:
    return None
  return str(profile)


def _refresh_cached_settings(names: list[str]) -> None:
  """Libraries read some variables once, at import: make them see the new values."""
  if "TMPDIR" in names:
    tempfile.tempdir = None  # `tempfile.gettempdir()` caches its first answer.
  if any(name.startswith("DASK_") for name in names):
    import dask.config

    dask.config.refresh()
