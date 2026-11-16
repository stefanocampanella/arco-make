# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The user's settings file (`~/.arcomakerc`, or `$ARCOMAKE_RC`). Only temporary files are read."""

import os
import pathlib
import tempfile

import pytest
from click.testing import CliRunner

from arcomake import cli, rcfile


def write_rc(path: pathlib.Path, text: str, mode: int = 0o600) -> pathlib.Path:
  path.write_text(text)
  path.chmod(mode)
  return path


def test_parse_shell_compatible_lines():
  """`export`, quotes, comments and `$VAR` expansion, as a shell would read the same file."""
  text = (
    "# comment\n"
    "export A=plain  # trailing comment\n"
    "B='lit $HOME # kept'\n"
    'C="x y \\"z\\""\n'
    "D=$A/sub\n"
    "E=${UNKNOWN_ARCOMAKE_TEST}/k\n"
    "\n"
  )
  assert rcfile.parse(text, environ={}) == {
    "A": "plain",
    "B": "lit $HOME # kept",
    "C": 'x y "z"',
    "D": "plain/sub",
    "E": "${UNKNOWN_ARCOMAKE_TEST}/k",
  }


def test_unparsable_lines_are_skipped_without_their_content():
  """A line that is not KEY=VALUE is skipped; the warning names its number, never its text."""
  with pytest.warns(UserWarning, match="line 2 is not KEY=VALUE") as record:
    settings = rcfile.parse("A=1\nsource secret-token-xyz\n", environ={})
  assert settings == {"A": "1"}
  assert all("secret-token-xyz" not in str(w.message) for w in record)


def test_load_fills_in_only_missing_variables(tmp_path: pathlib.Path):
  """The environment wins: the file only sets what is missing."""
  path = write_rc(tmp_path / "rc", "export KEEP=from-file\nexport NEW=from-file\n")
  environ = {"KEEP": "from-env", "ARCOMAKE_RC": str(path)}
  assert rcfile.load(environ=environ) == ["NEW"]
  assert environ["KEEP"] == "from-env" and environ["NEW"] == "from-file"


def test_disabled_or_missing_files_are_not_errors(tmp_path: pathlib.Path):
  """`ARCOMAKE_RC=""` disables the file; a missing file sets nothing."""
  assert rcfile.rc_path({"ARCOMAKE_RC": ""}) is None
  assert rcfile.load(environ={"ARCOMAKE_RC": ""}) == []
  assert rcfile.load(environ={"ARCOMAKE_RC": str(tmp_path / "absent")}) == []


def test_files_readable_by_others_are_used_with_a_warning(tmp_path: pathlib.Path):
  """The file may hold secrets: other users' access is reported (chmod 600)."""
  path = write_rc(tmp_path / "rc", "A=1\n", mode=0o644)
  environ = {"ARCOMAKE_RC": str(path)}
  with pytest.warns(UserWarning, match="chmod 600"):
    assert rcfile.load(environ=environ) == ["A"]


def test_cli_loads_the_settings_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
  """Every command reads the file first; TMPDIR from it applies to Python's temporary files."""
  tmpdir = tmp_path / "temporary-files"
  tmpdir.mkdir(exist_ok=True)
  path = write_rc(tmp_path / "rc", f"export TMPDIR={tmpdir}\nexport ARCOMAKE_TEST_SETTING=1\n")
  monkeypatch.setenv("ARCOMAKE_RC", str(path))
  monkeypatch.delenv("TMPDIR", raising=False)
  monkeypatch.delenv("ARCOMAKE_TEST_SETTING", raising=False)
  monkeypatch.setattr(tempfile, "tempdir", None)
  result = CliRunner().invoke(cli, ["steps"])
  assert result.exit_code == 0, result.output
  assert os.environ["ARCOMAKE_TEST_SETTING"] == "1"
  assert tempfile.gettempdir() == str(tmpdir)
