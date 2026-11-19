# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Static datasets (no time dimension once processed: bathymetry, masks, fractions) are read and
processed once per download run, not once per block; a resumed run checks that its static
variables are those stored by the first block."""

import collections
import datetime
import importlib
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner, Result

from arcomake import cli
from tests.conftest import make_static_dataset, open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command
SECOND_BLOCK = datetime.datetime(2020, 2, 28)


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  return render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch):
  """How often each dataset is read and processed; `fail` makes a block fail."""
  real = download_module.maybe_checkpointing_download_and_process
  state = {"count": collections.Counter(), "fail": None}

  def tracked(configs, start, end, **kwargs):
    state["count"][kwargs["name"]] += 1
    if start == state["fail"]:
      raise RuntimeError("simulated failure")
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", tracked)
  return state


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def test_static_datasets_are_processed_once_per_run(recipe, reads, tmp_path: pathlib.Path):
  """Three blocks: the time-varying datasets are read for each, the static one for the first."""
  result = run(recipe, tmp_path / "out.zarr")
  assert result.exit_code == 0, result.output
  assert reads["count"] == {"atmosphere": 3, "ocean": 3, "ocean_static": 1}
  out = open_output(tmp_path / "out.zarr")
  assert "time" not in out.sea_mask.dims and out.sea_mask.any()


def test_a_run_that_writes_the_first_block_does_not_check_static_variables(
  recipe, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
  """The comparison is for resumed runs: a run's later blocks reuse what its first one wrote."""
  calls = []
  monkeypatch.setattr(download_module, "_check_static", lambda *args: calls.append(args))
  assert run(recipe, tmp_path / "out.zarr").exit_code == 0
  assert calls == []


def test_resume_with_packed_static_variables_goes_on(recipe, reads, tmp_path: pathlib.Path):
  """A static variable packed into integers (scale_factor) is stored rounded: the computed one
  is compared after the same packing, not as computed."""
  packed = recipe.read_text() + (
    '\n[save.encoding.deptho]\ndtype = "int16"\nscale_factor = 0.3\n_FillValue = -32768\n'
  )
  recipe.write_text(packed)
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output).exit_code != 0
  reads["fail"] = None
  resumed = run(recipe, output, "--resume")
  assert resumed.exit_code == 0, resumed.output


def test_resume_checks_the_static_variables(recipe, reads, sources, tmp_path: pathlib.Path):
  """A resumed run computes the static datasets again (its first block is not the output's) and
  refuses to go on if they differ from the stored ones: appends do not rewrite them."""
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output).exit_code != 0
  reads["fail"] = None
  # The bathymetry source changes between the runs.
  static = make_static_dataset()
  static["deptho"] = static.deptho + 1.0
  static.to_zarr(sources / "static.zarr", mode="w", zarr_format=2)
  resumed = run(recipe, output, "--resume")
  assert resumed.exit_code != 0
  assert "Static variable" in resumed.output and "differs from the stored one" in resumed.output


def test_resume_with_the_same_static_variables_goes_on(recipe, reads, tmp_path: pathlib.Path):
  output = tmp_path / "out.zarr"
  reads["fail"] = SECOND_BLOCK
  assert run(recipe, output).exit_code != 0
  reads["fail"], reads["count"] = None, collections.Counter()
  resumed = run(recipe, output, "--resume")
  assert resumed.exit_code == 0, resumed.output
  assert reads["count"] == {"atmosphere": 2, "ocean": 2, "ocean_static": 1}
  assert np.isfinite(open_output(output).deptho.values).any()
