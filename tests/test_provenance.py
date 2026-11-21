# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Every commit records what made it: arco-make's version and commit, the environment, the
recipe's fingerprint; in the commit's metadata, without naming the machine or its paths."""

import os
import pathlib
import socket
import subprocess

import icechunk
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli, icechunk_utils, provenance
from arcomake.dataset_utils import write_output
from tests.conftest import render_recipe


def commits(path) -> list:
  repo = icechunk.Repository.open(icechunk_utils.storage(path))
  return [s for s in repo.ancestry(branch="main") if s.message != "Repository initialized"]


def test_the_environment_names_no_machine():
  recorded = provenance.environment()
  assert recorded["arcomake_version"]
  assert recorded["packages"]["xarray"] == xr.__version__
  assert recorded["cpus"] == os.cpu_count()
  assert len(recorded["distributions_sha256"]) == 64
  text = repr(recorded)
  assert socket.gethostname() not in text
  assert str(pathlib.Path.home()) not in text and os.environ.get("USER", "\0") not in text


def test_a_write_records_provenance_and_recipe_in_the_metadata(tmp_path: pathlib.Path):
  path = tmp_path / "out.zarr"
  write_output(
    xr.Dataset({"x": ("t", np.arange(3.0))}), path, message="a short message", recipe="abc123"
  )
  (snapshot,) = commits(path)
  assert snapshot.message == "a short message"
  assert snapshot.metadata[provenance.PROVENANCE_KEY] == provenance.environment()
  assert snapshot.metadata[icechunk_utils.RECIPE_KEY] == "abc123"


def test_every_download_block_records_provenance(tmp_path: pathlib.Path, sources: pathlib.Path):
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  output = tmp_path / "out"
  result = CliRunner().invoke(
    cli, ["download", "--dask-setup", "threads", str(recipe), str(output)]
  )
  assert result.exit_code == 0, result.output
  blocks = commits(output)
  assert len(blocks) == 3
  for snapshot in blocks:
    assert snapshot.metadata[provenance.PROVENANCE_KEY]["arcomake_version"]
    assert len(snapshot.metadata[icechunk_utils.RECIPE_KEY]) == 16


@pytest.mark.parametrize(
  ("env", "file", "expected"), [("abc", None, "abc"), (None, "def\n", "def"), ("abc", "def", "abc")]
)
def test_the_commit_comes_from_the_environment_or_the_image(
  monkeypatch, tmp_path, env, file, expected
):
  """`ARCOMAKE_COMMIT` first, then the file the image build writes."""
  build_file = tmp_path / "BUILD_COMMIT"
  if file is not None:
    build_file.write_text(file)
  monkeypatch.setattr(provenance, "BUILD_COMMIT_FILE", build_file)
  if env is None:
    monkeypatch.delenv("ARCOMAKE_COMMIT", raising=False)
  else:
    monkeypatch.setenv("ARCOMAKE_COMMIT", env)
  assert provenance._commit() == expected


def test_a_checkout_gives_its_git_commit(monkeypatch, tmp_path):
  monkeypatch.delenv("ARCOMAKE_COMMIT", raising=False)
  monkeypatch.setattr(provenance, "BUILD_COMMIT_FILE", tmp_path / "missing")
  commit = provenance._commit()
  if commit is None:
    pytest.skip("not a git checkout")
  assert len(commit.removesuffix("-dirty")) == 40


@pytest.mark.parametrize("split", [False, True])
def test_climatology_outputs_record_provenance_and_recipe(tmp_path: pathlib.Path, split: bool):
  """Both paths of `compute-climatology` (one graph, one series at a time)."""
  time = np.arange("2020-01-01", "2021-01-01", dtype="datetime64[D]").astype("datetime64[ns]")
  values = np.random.default_rng(0).standard_normal((time.size, 2)).astype("float32")
  source = tmp_path / "in.zarr"
  xr.Dataset({"x": (("time", "depth"), values)}, coords={"time": time}).to_zarr(source)
  recipe = tmp_path / "climatology.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 1\nweighting = "constant"\n'
    + ('split_dims = ["depth"]\n' if split else "")
    + '\n[read]\nengine = "zarr"\n\n[save]\n'
  )
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  path = ["--streaming"] if split else []
  args = ["compute-climatology", *path, "--dask-setup", "threads", str(recipe), str(source)]
  result = CliRunner().invoke(cli, [*args, str(clim), str(std)])
  assert result.exit_code == 0, result.output
  for output in (clim, std):
    (snapshot,) = commits(output)
    assert snapshot.metadata[provenance.PROVENANCE_KEY] == provenance.environment()
    assert len(snapshot.metadata[icechunk_utils.RECIPE_KEY]) == 16


def test_no_commit_of_another_repository(monkeypatch, tmp_path):
  """Installed (not editable) inside another project's checkout, git would name that project's
  commit: only arco-make's own checkout counts."""
  monkeypatch.delenv("ARCOMAKE_COMMIT", raising=False)
  monkeypatch.setattr(provenance, "BUILD_COMMIT_FILE", tmp_path / "missing")
  project = tmp_path / "project"
  package = project / ".venv" / "lib" / "site-packages" / "arcomake"
  package.mkdir(parents=True)
  subprocess.run(["git", "init", "-q", str(project)], check=True)
  subprocess.run(
    [
      "git",
      "-C",
      str(project),
      "-c",
      "user.name=t",
      "-c",
      "user.email=t@t",
      "commit",
      "-q",
      "--allow-empty",
      "-m",
      "x",
    ],
    check=True,
  )
  monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))
  assert provenance._commit() is None


def test_a_run_carries_the_recipe_not_write_output(tmp_path):
  run = icechunk_utils.Run(tmp_path / "out", None, "write-1", recipe="abc")
  with pytest.raises(ValueError, match="carries its recipe"):
    write_output(xr.Dataset({"x": ("t", [1.0])}), run, message="m", recipe="abc")
