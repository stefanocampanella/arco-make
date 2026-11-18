# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Blocks of a download (6.2): each block's checkpoints are removed with it, and a failed block
keeps the blocks before it, which `--resume` continues."""

import datetime
import importlib
import pathlib

import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
from arcomake.temporary_utils import get_current_temp_registry
from tests.conftest import open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  return render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def temporary_stores() -> list[pathlib.Path]:
  registry = get_current_temp_registry()
  assert registry is not None
  return list(registry.job_dir.rglob("*.zarr"))


def test_checkpoints_go_with_their_block(
  recipe, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """Checkpoints (the sample recipe checkpoints its ocean dataset daily) are removed at the end
  of their block: a long run does not keep every block's checkpoints on disk until it ends."""
  left: list[list[pathlib.Path]] = []
  real_committed = download_module._committed

  def committed(run, interval):  # called as each block starts: what did earlier blocks leave?
    left.append(temporary_stores())
    return real_committed(run, interval)

  monkeypatch.setattr(download_module, "_committed", committed)
  result = run(recipe, tmp_path / "out.zarr")
  assert result.exit_code == 0, result.output
  assert left == [[], [], []]


def test_a_failed_block_keeps_those_before_it(
  recipe, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """A block that fails (not transiently) stops the run with the blocks before it committed on
  the run's branch; `--resume` writes the rest, and the output equals a clean run's."""
  real = download_module.maybe_checkpointing_download_and_process
  fail = {"start": datetime.datetime(2020, 2, 28)}

  def failing(configs, start, end, **kwargs):
    if start == fail["start"]:
      raise RuntimeError("simulated failure")
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", failing)
  output = tmp_path / "out.zarr"
  failed = run(recipe, output)
  assert failed.exit_code != 0 and "--resume continues from there" in failed.output
  waiting = icechunk_utils.waiting_run(output, None, "download")
  assert waiting is not None and waiting.blocks == ("2020-02-26T00:00/2020-02-28T00:00",)
  fail["start"] = None
  resumed = run(recipe, output, "--resume")
  assert resumed.exit_code == 0, resumed.output
  clean = tmp_path / "clean.zarr"
  assert run(recipe, clean).exit_code == 0
  xr.testing.assert_identical(open_output(output).load(), open_output(clean).load())


@pytest.mark.parametrize("step", ["2D", "1D"])
def test_the_time_coverage_is_the_outputs(recipe, tmp_path: pathlib.Path, step: str):
  """xarray replaces a store's attributes with each appended block's: the time coverage is still
  the whole output's, from the first block's start to the last's end, with its resolution even
  when a block has one day (2-day blocks over 5 days end with one; 1-day blocks always do)."""
  text = recipe.read_text().replace('step = "2D"', f'step = "{step}"', 1)
  recipe.write_text(text + '\n[[postprocess]]\nname = "coverage_attrs"\n')
  output = tmp_path / "out.zarr"
  result = run(recipe, output)
  assert result.exit_code == 0, result.output
  attrs = open_output(output).attrs
  assert attrs["time_coverage_start"] == "2020-02-26T00:00:00Z"
  assert attrs["time_coverage_end"] == "2020-03-01T00:00:00Z"
  assert attrs["time_coverage_resolution"] == "P1D"


def test_blocks_option_is_gone(recipe, tmp_path: pathlib.Path):
  """`--blocks N` was removed: a block twice as long is faster, with less memory."""
  result = run(recipe, tmp_path / "out.zarr", "--blocks", "2")
  assert result.exit_code == 2 and "No such option '--blocks'" in result.output
