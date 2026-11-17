# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""User settings (`~/.arcomakerc`, or `$ARCOMAKE_RC`). Only temporary files are read."""

import os
import pathlib
import tempfile

import dask.config
import pytest
from click.testing import CliRunner

from arcomake import cli, settings

SETTINGS = [
  "TMPDIR",
  "AWS_PROFILE",
  "AWS_DEFAULT_PROFILE",
  "AWS_CONFIG_FILE",
  "AWS_SHARED_CREDENTIALS_FILE",
  "CDSAPI_URL",
  "CDSAPI_KEY",
  "COPERNICUSMARINE_SERVICE_PASSWORD",
  "COPERNICUSMARINE_MAX_POOL_CONNECTIONS",
  "DASK_ARCOMAKE_TEST",
]


@pytest.fixture
def rc(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
  """Write a settings file (mode 600) and point `ARCOMAKE_RC` at it; the settings start unset.

  Each variable is set, then deleted, so that monkeypatch records it and removes whatever
  `settings.load()` exports into the process environment after the test.
  """
  for name in SETTINGS:
    monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv(name)
  monkeypatch.setattr(tempfile, "tempdir", None)

  def write(text: str, mode: int = 0o600) -> pathlib.Path:
    path = tmp_path / "arcomakerc"
    path.write_text(text)
    path.chmod(mode)
    monkeypatch.setenv("ARCOMAKE_RC", str(path))
    return path

  return write


def test_settings_file_is_shell_compatible(rc, tmp_path: pathlib.Path):
  """`export`, quotes, comments and `${VAR}`, as a shell would read the same file."""
  (tmp_path / "t").mkdir()
  rc(
    "# comment\n"
    f"export TMPDIR={tmp_path}/t  # trailing comment\n"
    "export AWS_PROFILE='bucket-profile'\n"
    'CDSAPI_URL="https://ewds.example/api"\n'
    "CDSAPI_KEY=${AWS_PROFILE}-key\n"
  )
  assert sorted(settings.load()) == ["AWS_PROFILE", "CDSAPI_KEY", "CDSAPI_URL", "TMPDIR"]
  assert os.environ["TMPDIR"] == f"{tmp_path}/t"
  assert os.environ["AWS_PROFILE"] == "bucket-profile"
  assert os.environ["CDSAPI_KEY"] == "bucket-profile-key"


def test_the_environment_wins_and_empty_values_are_unset(rc, monkeypatch: pytest.MonkeyPatch):
  """The file only fills in what the environment lacks; `KEY=` sets nothing (a template line
  must not hide the libraries' own credential files)."""
  monkeypatch.setenv("AWS_PROFILE", "from-env")
  rc("AWS_PROFILE=from-file\nCDSAPI_KEY=\nCDSAPI_URL=https://x.example\n")
  assert settings.load() == ["CDSAPI_URL"]
  assert os.environ["AWS_PROFILE"] == "from-env" and "CDSAPI_KEY" not in os.environ


def test_invalid_values_are_reported_without_the_value(rc):
  """Errors name the variable, never its value; the CLI shows them as a message, not a trace."""
  rc("CDSAPI_URL=ftp://secret-host\nCDSAPI_KEY=secret-xyz\n")
  result = CliRunner().invoke(cli, ["steps"])
  assert result.exit_code != 0 and "CDSAPI_URL" in result.output
  assert "secret" not in result.output and "Traceback" not in result.output


def test_copernicus_marine_pool_size_is_a_positive_integer(rc, monkeypatch: pytest.MonkeyPatch):
  """The toolbox reads its pool size from the environment with `int()`, falling back to 10
  without a message: only what `int()` reads as a positive integer is accepted."""
  rc("COPERNICUSMARINE_MAX_POOL_CONNECTIONS=32\n")
  assert settings.load() == ["COPERNICUSMARINE_MAX_POOL_CONNECTIONS"]
  assert os.environ["COPERNICUSMARINE_MAX_POOL_CONNECTIONS"] == "32"
  for bad in ("0", "many", "32.0", "-4"):
    monkeypatch.delenv("COPERNICUSMARINE_MAX_POOL_CONNECTIONS", raising=False)
    rc(f"COPERNICUSMARINE_MAX_POOL_CONNECTIONS={bad}\n")
    with pytest.raises(settings.SettingsError, match="COPERNICUSMARINE_MAX_POOL_CONNECTIONS"):
      settings.load()
  rc("")
  monkeypatch.setenv("COPERNICUSMARINE_MAX_POOL_CONNECTIONS", "32.0")  # from the environment
  with pytest.raises(settings.SettingsError, match="COPERNICUSMARINE_MAX_POOL_CONNECTIONS"):
    settings.load()


def test_environment_values_are_checked_without_a_file(monkeypatch: pytest.MonkeyPatch, tmp_path):
  """Whether a command runs does not depend on a settings file existing."""
  monkeypatch.setenv("ARCOMAKE_RC", "")
  monkeypatch.setenv("TMPDIR", str(tmp_path / "missing"))
  with pytest.raises(settings.SettingsError, match="TMPDIR: .*writable directory"):
    settings.load()


def test_secrets_are_hidden():
  """Secrets never appear in the representation."""
  s = settings.ArcoMakeSettings(CDSAPI_KEY="secret-xyz", AWS_SECRET_ACCESS_KEY="aws-xyz")  # ty: ignore[unknown-argument]
  assert "secret-xyz" not in repr(s) and "aws-xyz" not in repr(s)


def test_unknown_keys_are_ignored_and_dask_keys_merged(rc):
  """A key that is not a setting is reported by name (never its value) and not exported;
  `DASK_*` keys are merged into Dask's configuration without dropping values set in-process."""
  rc("SOMETHING_ELSE=secret-value-xyz\nDASK_ARCOMAKE_TEST=1\n")
  with dask.config.set({"arcomake-kept": 2}):
    with pytest.warns(UserWarning, match="'SOMETHING_ELSE' is not a setting") as record:
      added = settings.load()
    assert dask.config.get("arcomake-test") == 1 and dask.config.get("arcomake-kept") == 2
  assert added == ["DASK_ARCOMAKE_TEST"] and "SOMETHING_ELSE" not in os.environ
  assert all("secret-value-xyz" not in str(w.message) for w in record)


def test_missing_files(tmp_path, monkeypatch: pytest.MonkeyPatch):
  """`ARCOMAKE_RC=""` disables the file; an explicitly named file that is missing is reported."""
  monkeypatch.setenv("ARCOMAKE_RC", "")
  assert settings.settings_path() is None and settings.load() == []
  monkeypatch.setenv("ARCOMAKE_RC", str(tmp_path / "absent"))
  with pytest.warns(UserWarning, match="not a file"):
    assert settings.load() == []


def test_files_readable_by_others_are_reported(rc):
  """The file may hold secrets: access by other users is reported (chmod 600)."""
  rc("CDSAPI_KEY=k\n", mode=0o644)
  with pytest.warns(UserWarning, match="chmod 600"):
    assert settings.load() == ["CDSAPI_KEY"]


def test_unknown_aws_profile_fails_runs_that_use_botocore(rc, tmp_path, monkeypatch):
  """An unknown AWS profile fails, before any work, the runs with `s3://` paths or Copernicus
  Marine parts (which create botocore sessions); local runs and `--help` are not affected."""
  config = tmp_path / "aws-config"
  config.write_text("[profile known]\nregion = eu-west-1\n")
  monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
  monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
  rc("AWS_DEFAULT_PROFILE=unknown\n")
  recipe = tmp_path / "process.toml"
  recipe.write_text('[read]\nengine = "zarr"\n\n[[process]]\nname = "mean"\ndim = "time"\n')
  out = str(tmp_path / "out.zarr")
  remote = CliRunner().invoke(
    cli, ["process", "--dask-setup", "threads", str(recipe), "s3://b/x.zarr", out]
  )
  assert remote.exit_code != 0 and "AWS profile 'unknown'" in remote.output
  local = CliRunner().invoke(
    cli, ["process", "--dask-setup", "threads", str(recipe), str(tmp_path / "in.zarr"), out]
  )
  assert "AWS profile" not in local.output
  assert CliRunner().invoke(cli, ["process", "--help"]).exit_code == 0


def test_cli_loads_the_settings_file(rc, tmp_path: pathlib.Path):
  """Every command reads the file first; TMPDIR from it applies to Python's temporary files."""
  tmpdir = tmp_path / "temporary-files"
  tmpdir.mkdir(exist_ok=True)
  rc(f"export TMPDIR={tmpdir}\nexport AWS_PROFILE=p\n")
  result = CliRunner().invoke(cli, ["steps"])
  assert result.exit_code == 0, result.output
  assert os.environ["AWS_PROFILE"] == "p"
  assert tempfile.gettempdir() == str(tmpdir)
