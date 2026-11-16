# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""User settings (`~/.arcomakerc`, or `$ARCOMAKE_RC`). Only temporary files are read."""

import os
import pathlib
import tempfile

import pydantic
import pytest
from click.testing import CliRunner

from arcomake import cli, settings

SETTINGS = [
  "TMPDIR",
  "AWS_PROFILE",
  "AWS_DEFAULT_PROFILE",
  "CDSAPI_URL",
  "CDSAPI_KEY",
  "COPERNICUSMARINE_SERVICE_PASSWORD",
  "DASK_ARCOMAKE_TEST",
]


@pytest.fixture
def rc(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
  """Write a settings file (mode 600) and point `ARCOMAKE_RC` at it; the settings start unset."""
  for name in SETTINGS:
    monkeypatch.delenv(name, raising=False)
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
  assert os.environ["CDSAPI_URL"] == "https://ewds.example/api"
  assert os.environ["CDSAPI_KEY"] == "bucket-profile-key"


def test_the_environment_wins(rc, monkeypatch: pytest.MonkeyPatch):
  """The file only fills in what the environment lacks."""
  monkeypatch.setenv("AWS_PROFILE", "from-env")
  rc("AWS_PROFILE=from-file\nCDSAPI_KEY=k\n")
  assert settings.load() == ["CDSAPI_KEY"]
  assert os.environ["AWS_PROFILE"] == "from-env"


def test_secrets_are_hidden():
  """Secrets never appear in the representation or in validation errors."""
  s = settings.ArcoMakeSettings(CDSAPI_KEY="secret-xyz", COPERNICUSMARINE_SERVICE_PASSWORD="pw-xyz")  # ty: ignore[unknown-argument]
  assert "secret-xyz" not in repr(s) and "pw-xyz" not in repr(s)
  with pytest.raises(pydantic.ValidationError) as error:
    settings.ArcoMakeSettings(CDSAPI_URL="ftp://secret-host", CDSAPI_KEY="secret-xyz")  # ty: ignore[unknown-argument]
  assert "CDSAPI_URL" in str(error.value) and "secret-xyz" not in str(error.value)


def test_unknown_keys_are_ignored_with_their_name_only(rc):
  """A key that is not a setting is reported by name (never its value) and not exported;
  `DASK_*` keys go to Dask."""
  rc("SOMETHING_ELSE=secret-value-xyz\nDASK_ARCOMAKE_TEST=1\n")
  with pytest.warns(UserWarning, match="'SOMETHING_ELSE' is not a setting") as record:
    added = settings.load()
  assert added == ["DASK_ARCOMAKE_TEST"] and "SOMETHING_ELSE" not in os.environ
  assert all("secret-value-xyz" not in str(w.message) for w in record)
  import dask.config

  assert dask.config.get("arcomake-test") == 1


def test_disabled_or_missing_files_set_nothing(tmp_path, monkeypatch: pytest.MonkeyPatch):
  """`ARCOMAKE_RC=""` disables the file; a missing file is not an error."""
  monkeypatch.setenv("ARCOMAKE_RC", "")
  assert settings.settings_path() is None and settings.load() == []
  monkeypatch.setenv("ARCOMAKE_RC", str(tmp_path / "absent"))
  assert settings.load() == []


def test_files_readable_by_others_are_reported(rc):
  """The file may hold secrets: access by other users is reported (chmod 600)."""
  rc("CDSAPI_KEY=k\n", mode=0o644)
  with pytest.warns(UserWarning, match="chmod 600"):
    assert settings.load() == ["CDSAPI_KEY"]


def test_unknown_aws_profile_fails_data_commands_at_once(rc, tmp_path, monkeypatch):
  """An `AWS_PROFILE` botocore cannot find fails data commands before any work.

  The Copernicus Marine toolbox creates a botocore session for every read (unsigned, to its own
  endpoint): with an unknown profile it failed after minutes of retries.
  """
  config = tmp_path / "aws-config"
  config.write_text("[profile known]\nregion = eu-west-1\n")
  monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
  monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
  rc("AWS_PROFILE=unknown\n")
  result = CliRunner().invoke(cli, ["process", "--help"])
  assert result.exit_code != 0 and "AWS profile 'unknown'" in result.output
  assert CliRunner().invoke(cli, ["steps"]).exit_code == 0
  monkeypatch.setenv("AWS_PROFILE", "known")
  assert CliRunner().invoke(cli, ["process", "--help"]).exit_code == 0
  # botocore takes AWS_DEFAULT_PROFILE first: it is checked too.
  monkeypatch.setenv("AWS_DEFAULT_PROFILE", "unknown-default")
  result = CliRunner().invoke(cli, ["process", "--help"])
  assert result.exit_code != 0 and "'unknown-default'" in result.output


def test_cli_loads_the_settings_file(rc, tmp_path: pathlib.Path):
  """Every command reads the file first; TMPDIR from it applies to Python's temporary files."""
  tmpdir = tmp_path / "temporary-files"
  tmpdir.mkdir(exist_ok=True)
  rc(f"export TMPDIR={tmpdir}\nexport AWS_PROFILE=p\n")
  result = CliRunner().invoke(cli, ["steps"])
  assert result.exit_code == 0, result.output
  assert os.environ["AWS_PROFILE"] == "p"
  assert tempfile.gettempdir() == str(tmpdir)
