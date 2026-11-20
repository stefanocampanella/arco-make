# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Recipes without dates (given by the command) and `download --extend`: the days after an
output's last appended as a new version, with the recipe it was written with."""

import importlib
import pathlib

import icechunk
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
from tests.conftest import make_static_dataset, open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command
START, MIDDLE, END = "2020-02-26T00:00:00", "2020-02-28T00:00:00", "2020-03-02T00:00:00"


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  """The sample recipe without its dates."""
  path = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  text = path.read_text()
  dates = "start = 2020-02-26T00:00:00\nend = 2020-03-02T00:00:00\n"
  assert text.count(dates) == 1
  path.write_text(text.replace(dates, ""))
  return path


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def versions(output: pathlib.Path) -> int:
  repo = icechunk.Repository.open(icechunk_utils.storage(output))
  return sum(1 for _ in repo.ancestry(branch="main"))


def test_a_recipe_without_dates_takes_them_from_the_command(recipe, tmp_path):
  missing = run(recipe, tmp_path / "out")
  assert missing.exit_code != 0 and "give --start and --end" in missing.output
  result = run(recipe, tmp_path / "out", f"--start={START}", f"--end={END}")
  assert result.exit_code == 0, result.output
  assert open_output(tmp_path / "out").sizes["time"] == 5


def test_an_extension_equals_one_download(recipe, tmp_path):
  whole = run(recipe, tmp_path / "whole", f"--start={START}", f"--end={END}")
  assert whole.exit_code == 0, whole.output
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  before = versions(output)
  extended = run(recipe, output, "--extend", f"--end={END}")
  assert extended.exit_code == 0, extended.output
  xr.testing.assert_identical(open_output(output), open_output(tmp_path / "whole"))
  assert versions(output) > before  # the earlier version stays in the history
  again = run(recipe, output, "--extend", f"--end={END}")
  assert again.exit_code == 0 and "Nothing to extend" in again.output


def test_an_extension_keeps_the_recipe(recipe, tmp_path):
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  recipe.write_text(recipe.read_text().replace('title = "Synthetic sample dataset"', 'title = "x"'))
  other = run(recipe, output, "--extend", f"--end={END}")
  assert other.exit_code != 0 and "another recipe" in other.output


@pytest.mark.parametrize("option", [f"--start={MIDDLE}", "--replace"])
def test_an_extension_sets_its_own_start(recipe, tmp_path, option):
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  refused = run(recipe, output, "--extend", option, f"--end={END}")
  assert refused.exit_code != 0 and "not with --replace nor --start" in refused.output


def test_an_extension_compares_the_static_variables(recipe, tmp_path, sources):
  """Appends do not rewrite them: changed static variables fail the extension."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  static = make_static_dataset()
  static["deptho"] = static.deptho + 1.0
  static.to_zarr(sources / "static.zarr", mode="w", zarr_format=2)
  failed = run(recipe, output, "--extend", f"--end={END}")
  assert failed.exit_code != 0 and "differs from the stored one" in failed.output


def test_a_failed_extension_resumes(recipe, tmp_path, monkeypatch):
  whole = run(recipe, tmp_path / "whole", f"--start={START}", f"--end={END}")
  assert whole.exit_code == 0, whole.output
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  real = download_module.maybe_checkpointing_download_and_process

  def failing(configs, start, end, **kwargs):
    if str(start).startswith("2020-03-01"):
      raise RuntimeError("simulated failure")
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", failing)
  assert run(recipe, output, "--extend", f"--end={END}").exit_code != 0
  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", real)
  resumed = run(recipe, output, "--extend", "--resume", f"--end={END}")
  assert resumed.exit_code == 0, resumed.output
  xr.testing.assert_identical(open_output(output), open_output(tmp_path / "whole"))
