# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""User settings: credentials and machine settings from the environment and `~/.arcomakerc`.

The settings are the environment variables that arco-make's libraries read (Copernicus Marine and
CDS/EWDS credentials, AWS profile and credentials, ecCodes paths, `TMPDIR`, `DASK_*`).
`ArcoMakeSettings` reads them from the environment and from a dotenv file, `~/.arcomakerc` (or the
file named by `ARCOMAKE_RC`; an empty value disables it). `export KEY=VALUE` lines are allowed, so
job scripts can `source` the same file::

  # ~/.arcomakerc (chmod 600: it holds secrets)
  export TMPDIR=/scratch/${USER}/tmp
  export AWS_PROFILE=my-bucket-profile
  export CDSAPI_URL=https://ewds.climate.copernicus.eu/api
  export CDSAPI_KEY='...'

`apply()` then exports to the process environment the values that came from the file: variables
already in the environment win, and empty values count as unset. Values are checked whether they
come from the environment or the file; secrets are `SecretStr`, never logged or shown in errors.
Keys that are not settings are ignored with a warning naming the key (`DASK_*` keys go to Dask's
configuration).
"""

import logging
import os
import pathlib
import stat
import tempfile
import warnings
from collections.abc import Iterable, MutableMapping

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_PATH = pathlib.Path("~/.arcomakerc")
ENV_VAR = "ARCOMAKE_RC"


class SettingsError(Exception):
  """Invalid settings; the message names the variables, never their values."""


class ArcoMakeSettings(BaseSettings):
  """Settings read from the environment, then from the settings file (the environment wins)."""

  model_config = SettingsConfigDict(
    case_sensitive=True, extra="allow", env_file=None, env_ignore_empty=True
  )

  tmpdir: str | None = Field(
    default=None,
    validation_alias="TMPDIR",
    description="Temporary files (checkpoints, downloads): on disk, not in RAM",
  )
  # botocore reads AWS_DEFAULT_PROFILE first, then AWS_PROFILE: the default profile of `s3://`
  # paths (recipes' `storage_options` override it).
  aws_profile: str | None = Field(default=None, validation_alias="AWS_PROFILE")
  aws_default_profile: str | None = Field(default=None, validation_alias="AWS_DEFAULT_PROFILE")
  aws_config_file: str | None = Field(default=None, validation_alias="AWS_CONFIG_FILE")
  aws_shared_credentials_file: str | None = Field(
    default=None, validation_alias="AWS_SHARED_CREDENTIALS_FILE"
  )
  aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
  aws_secret_access_key: SecretStr | None = Field(
    default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
  )
  aws_session_token: SecretStr | None = Field(default=None, validation_alias="AWS_SESSION_TOKEN")
  aws_region: str | None = Field(default=None, validation_alias="AWS_REGION")
  aws_default_region: str | None = Field(default=None, validation_alias="AWS_DEFAULT_REGION")
  aws_endpoint_url: str | None = Field(
    default=None, validation_alias="AWS_ENDPOINT_URL", pattern=r"^https?://"
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
  eccodes_dir: str | None = Field(default=None, validation_alias="ECCODES_DIR")
  eccodes_definition_path: str | None = Field(
    default=None, validation_alias="ECCODES_DEFINITION_PATH"
  )
  eccodes_samples_path: str | None = Field(default=None, validation_alias="ECCODES_SAMPLES_PATH")

  @field_validator("tmpdir")
  @classmethod
  def _writable_directory(cls, value: str | None) -> str | None:
    # Python's tempfile silently falls back to /tmp (often RAM on HPC nodes) otherwise.
    if value is not None and not (os.path.isdir(value) and os.access(value, os.W_OK | os.X_OK)):
      raise ValueError("not an existing, writable directory")
    return value


def settings_path(environ: MutableMapping[str, str] | None = None) -> pathlib.Path | None:
  """The settings file: `$ARCOMAKE_RC`, or `~/.arcomakerc`; None if disabled."""
  environ = os.environ if environ is None else environ
  if ENV_VAR in environ:
    value = environ[ENV_VAR].strip()
    return pathlib.Path(value).expanduser() if value else None
  return DEFAULT_PATH.expanduser()


def read(path: pathlib.Path | None) -> ArcoMakeSettings:
  """The settings from the environment and `path` (if it is a file).

  Raises:
    SettingsError: for invalid values (named by variable, never shown) or an unreadable file.
  """
  env_file = path if path is not None and path.is_file() else None
  if env_file is not None and env_file.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
    warnings.warn(
      f"{env_file} is accessible to other users; it may hold secrets: run `chmod 600 {env_file}`"
    )
  try:
    return ArcoMakeSettings(_env_file=env_file)
  except ValidationError as error:
    problems = "; ".join(
      f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in error.errors()
    )
    raise SettingsError(f"Invalid settings: {problems}") from None
  except OSError as error:
    raise SettingsError(f"Cannot read the settings file {env_file}: {error.strerror}") from None


def apply(settings: ArcoMakeSettings, environ: MutableMapping[str, str] | None = None) -> list[str]:
  """Export the settings the environment lacks (they came from the file); return their names."""
  environ = os.environ if environ is None else environ
  values: dict[str, str] = {}
  for name, field in ArcoMakeSettings.model_fields.items():
    value = getattr(settings, name)
    if value is not None:
      alias = str(field.validation_alias)
      values[alias] = value.get_secret_value() if isinstance(value, SecretStr) else value
  dask_values: dict[str, str] = {}
  for key, value in (settings.model_extra or {}).items():
    if key.startswith("DASK_"):
      dask_values[key] = str(value)
    else:
      warnings.warn(f"arco-make settings file: {key!r} is not a setting; ignored")
  values |= dask_values
  added = [key for key in values if values[key] != "" and environ.get(key, "") == ""]
  for key in added:
    environ[key] = values[key]
  logger.debug(f"Settings from the file: {', '.join(added) or 'none'}")
  if environ is os.environ:
    _refresh_cached_settings(added, {k: v for k, v in dask_values.items() if k in added})
  return added


def load() -> list[str]:
  """Check the settings and export what the environment lacks from the file; return the names.

  The environment's values are checked even without a file. An explicit `ARCOMAKE_RC` naming a
  missing file is reported (the default file may be absent).

  Raises:
    SettingsError: for invalid settings or an unreadable file.
  """
  path = settings_path()
  if path is not None and not path.is_file() and os.environ.get(ENV_VAR, "").strip():
    warnings.warn(f"{ENV_VAR} names {path}, which is not a file: no settings file read")
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


def check_aws_profile(paths: Iterable[object] = (), copernicus_marine: bool = False) -> None:
  """Fail early if botocore's profile is unknown and something here will use botocore.

  Botocore is used by `s3://` paths and by Copernicus Marine parts; other runs are not checked.

  Raises:
    SettingsError: naming the profile and the variables that select it.
  """
  uses_s3 = any(str(path).lower().startswith(("s3://", "s3a://")) for path in paths)
  if not (uses_s3 or copernicus_marine):
    return
  profile = missing_aws_profile()
  if profile is not None:
    raise SettingsError(
      f"The AWS profile {profile!r} (AWS_DEFAULT_PROFILE or AWS_PROFILE) is not configured "
      "(AWS_CONFIG_FILE, ~/.aws/config). Fix or unset it."
    )


def _refresh_cached_settings(names: list[str], dask_values: dict[str, str]) -> None:
  """Libraries read some variables once, at import: make them see the new values."""
  if "TMPDIR" in names:
    tempfile.tempdir = None  # `tempfile.gettempdir()` caches its first answer.
  if dask_values:
    import dask.config

    # Merge only these keys: `dask.config.refresh()` would drop values set in this process.
    dask.config.update(dask.config.config, dask.config.collect_env(dask_values))
