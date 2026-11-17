# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`download --blocks N` reads and processes N blocks at once (6.2): each is staged on local disk
and committed in order, so the output is the one of a run with one block at a time, metadata
and encodings included."""

import datetime
import importlib
import pathlib
import threading

import pytest
import xarray as xr
import zarr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
from tests.conftest import open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  return render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def array_metadata(path: pathlib.Path) -> dict[str, dict]:
  store = icechunk_utils.readonly_store(path, None)
  group = zarr.open_group(store, mode="r")
  return {name: array.metadata.to_dict() for name, array in group.arrays()}


def test_blocks_at_once_write_the_same_output(recipe, tmp_path: pathlib.Path):
  one, three = tmp_path / "one.zarr", tmp_path / "three.zarr"
  assert run(recipe, one).exit_code == 0
  result = run(recipe, three, "--blocks", "3")
  assert result.exit_code == 0, result.output
  xr.testing.assert_identical(open_output(one).load(), open_output(three).load())
  assert array_metadata(one) == array_metadata(three)


def test_blocks_run_at_once(recipe, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
  """Two blocks are read at the same time: each waits for the other to start."""
  real = download_module.maybe_checkpointing_download_and_process
  both_started = threading.Barrier(2, timeout=30)
  starts: set[datetime.datetime] = set()

  def meeting(configs, start, end, **kwargs):
    if start not in starts and start < datetime.datetime(2020, 3, 1):
      starts.add(start)
      both_started.wait()  # BrokenBarrierError if the blocks ran one after the other
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", meeting)
  result = run(recipe, tmp_path / "out.zarr", "--blocks", "2")
  assert result.exit_code == 0, result.output


def test_a_failed_block_keeps_those_before_it(
  recipe, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  real = download_module.maybe_checkpointing_download_and_process
  fail = {"start": datetime.datetime(2020, 2, 28)}

  def failing(configs, start, end, **kwargs):
    if start == fail["start"]:
      raise RuntimeError("simulated failure")
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", failing)
  output = tmp_path / "out.zarr"
  failed = run(recipe, output, "--blocks", "2")
  assert failed.exit_code != 0 and "--resume continues from there" in failed.output
  waiting = icechunk_utils.waiting_run(output, None, "download")
  assert waiting is not None and waiting.blocks == ("2020-02-26T00:00/2020-02-28T00:00",)
  fail["start"] = None
  resumed = run(recipe, output, "--resume", "--blocks", "2")
  assert resumed.exit_code == 0, resumed.output
  clean = tmp_path / "clean.zarr"
  assert run(recipe, clean).exit_code == 0
  xr.testing.assert_identical(open_output(output).load(), open_output(clean).load())
