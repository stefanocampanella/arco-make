# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`download --resume` continues a failed download (6.2, requirements 1-3): the blocks it
committed are kept (each commit records its interval and the recipe), the others written; a
plain rerun does not start beside a failed run, and `--replace` starts over."""

import datetime
import importlib
import pathlib

import icechunk
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
from tests.conftest import open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command
SECOND_BLOCK = datetime.datetime(2020, 2, 28)


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  return render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch):
  """The block starts each dataset is read for; `fail` makes a block fail (not transiently)."""
  real = download_module.maybe_checkpointing_download_and_process
  state = {"starts": [], "fail": None}

  def tracked(configs, start, end, **kwargs):
    state["starts"].append(start)
    if start == state["fail"]:
      raise RuntimeError("simulated failure")
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", tracked)
  return state


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def branches(path) -> set[str]:
  return icechunk.Repository.open(icechunk_utils.storage(path)).list_branches()


def test_resume_writes_only_the_missing_blocks(recipe, reads, tmp_path: pathlib.Path):
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  failed = run(recipe, output)
  assert failed.exit_code != 0 and "--resume continues from there" in failed.output
  assert "Traceback" in failed.output and "simulated failure" in failed.output  # logged
  # A plain rerun does not start beside the failed run.
  refused = run(recipe, output)
  assert refused.exit_code != 0 and "--resume continues it" in refused.output
  reads["fail"], reads["starts"] = None, []
  resumed = run(recipe, output, "--resume")
  assert resumed.exit_code == 0, resumed.output
  assert datetime.datetime(2020, 2, 26) not in reads["starts"]  # the first block is not read again
  assert branches(output) == {"main"}
  clean = tmp_path / "clean.zarr"
  assert run(recipe, clean).exit_code == 0
  xr.testing.assert_identical(open_output(output).load(), open_output(clean).load())


def test_resume_needs_the_same_recipe(recipe, reads, tmp_path: pathlib.Path):
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output).exit_code != 0
  reads["fail"] = None
  other = run(recipe, output, "--resume", "--end=2020-03-01T00:00:00")
  assert other.exit_code != 0 and "another recipe" in other.output


def test_resume_with_other_read_settings(recipe, reads, tmp_path: pathlib.Path):
  """Request timeouts, checkpointing and retries change how blocks are read, not what is
  written: a run that failed on timeouts resumes with longer ones."""
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output).exit_code != 0
  reads["fail"] = None
  text = recipe.read_text()
  edited = text.replace(
    '[datasets.ocean.checkpointing]\nstep = "1D"', '[datasets.ocean.checkpointing]\nstep = "2D"'
  ).replace("[[datasets.ocean.parts]]\n", "[[datasets.ocean.parts]]\ntimeout = 600\n")
  assert edited.count("timeout = 600") >= 1 and 'step = "2D"\n\n[[datasets.ocean.parts]]' in edited
  recipe.write_text(edited + "\n[retries]\nattempts = 2\n")
  resumed = run(recipe, output, "--resume")
  assert resumed.exit_code == 0, resumed.output


def test_nothing_to_resume(recipe, tmp_path: pathlib.Path):
  result = run(recipe, tmp_path / "out.zarr", "--resume")
  assert result.exit_code != 0 and "Nothing to resume" in result.output


def test_replace_starts_over(recipe, reads, tmp_path: pathlib.Path):
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output).exit_code != 0
  reads["fail"] = None
  started_over = run(recipe, output, "--replace")
  assert started_over.exit_code == 0, started_over.output
  assert open_output(output).sizes["time"] == 5


def test_a_run_behind_main_cannot_resume(recipe, reads, tmp_path: pathlib.Path):
  """A failed run started from a version of `main` that has since been replaced cannot be
  promoted: --resume refuses it."""
  output = tmp_path / "out.zarr"
  assert run(recipe, output).exit_code == 0  # a first version
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output, "--replace").exit_code != 0  # a failed run, from that version
  reads["fail"] = None
  icechunk_utils.write(xr.Dataset({"x": ("t", [1.0])}), output, message="main moves")
  result = run(recipe, output, "--resume")
  assert result.exit_code != 0 and "main changed" in result.output
  # It does not block a new version either.
  replaced = run(recipe, output, "--replace")
  assert replaced.exit_code == 0, replaced.output
