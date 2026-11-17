# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Runs write on a branch of their own; `main` moves only when a run succeeds (`--replace` writes
a new version of an existing output)."""

import pathlib

import icechunk
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli, icechunk_utils
from arcomake.dataset_utils import write_output
from tests.conftest import open_output, render_recipe


def repository(path) -> icechunk.Repository:
  return icechunk.Repository.open(icechunk_utils.storage(path))


def dataset(value: float) -> xr.Dataset:
  return xr.Dataset({"x": ("t", np.full(4, value))}, coords={"t": np.arange(4)})


def test_a_failed_download_leaves_main_as_it_was(
  tmp_path: pathlib.Path, sources: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """A download that fails after some blocks leaves `main` at the previous version: its blocks
  stay on the run's branch, which the error names."""
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  output = tmp_path / "out.zarr"
  args = ["download", "--dask-setup", "threads", str(recipe), str(output)]
  assert CliRunner().invoke(cli, args).exit_code == 0
  before = repository(output).lookup_branch("main")
  previous = open_output(output).load()

  import importlib

  download = importlib.import_module("arcomake.download")  # the module, not the command

  calls = []
  real = download.maybe_checkpointing_download_and_process

  def failing_on_the_second_block(*a, **kw):
    calls.append(a[1])
    if len({str(c) for c in calls}) > 1:
      raise TimeoutError("simulated read timeout")
    return real(*a, **kw)

  monkeypatch.setattr(
    download, "maybe_checkpointing_download_and_process", failing_on_the_second_block
  )
  result = CliRunner().invoke(cli, [*args[:1], "--replace", *args[1:]])
  assert result.exit_code != 0 and "`main` is unchanged" in result.output
  repo = repository(output)
  assert repo.lookup_branch("main") == before
  xr.testing.assert_identical(open_output(output).load(), previous)
  (branch,) = [b for b in repo.list_branches() if b.startswith("run/download-")]
  assert branch in result.output
  assert next(iter(repo.ancestry(branch=branch))).message.startswith("download [2020-02-26")


def test_a_successful_run_leaves_no_branch(tmp_path: pathlib.Path):
  """After a run, `main` holds its output and its branch is gone; the version it replaced stays
  in `main`'s history."""
  path = tmp_path / "out.zarr"
  first = write_output(dataset(1), path, message="first")
  write_output(dataset(2), path, message="second")
  repo = repository(path)
  assert repo.list_branches() == {"main"}
  xr.testing.assert_identical(open_output(path).load(), dataset(2))
  assert first in [s.id for s in repo.ancestry(branch="main")]


def test_main_moved_during_a_run_is_not_overwritten(tmp_path: pathlib.Path):
  """If `main` changes while a run writes, the run is not promoted (the other version stays) and
  its output stays on its branch."""
  path = tmp_path / "out.zarr"
  write_output(dataset(1), path, message="first")
  run = icechunk_utils.Run(path, None, icechunk_utils.run_label("test"))
  run.write(dataset(2), message="slow run")
  write_output(dataset(3), path, message="meanwhile")
  with pytest.raises(icechunk_utils.PromotionError, match=run.branch):
    run.finish()
  xr.testing.assert_identical(open_output(path).load(), dataset(3))
  assert run.branch in repository(path).list_branches()


def run_labels(path) -> list[str]:
  labels = []
  for snapshot in repository(path).ancestry(branch="main"):
    label = (snapshot.metadata or {}).get(icechunk_utils.RUN_KEY)
    if label is not None and label not in labels:
      labels.append(label)
  return labels


def test_a_failed_first_run_needs_no_replace(tmp_path: pathlib.Path):
  """A repository whose `main` holds no version yet (a first run failed there) is written again
  without --replace."""
  from arcomake.cli_utils import check_destination

  path = tmp_path / "out.zarr"
  failed = icechunk_utils.Run(path, None, icechunk_utils.run_label("download"))
  failed.write(dataset(1), message="first block, then a failure")
  check_destination(str(path))
  write_output(dataset(2), path, message="v1")
  import click

  with pytest.raises(click.ClickException, match="--replace"):
    check_destination(str(path))


def test_other_errors_of_the_promotion_are_not_conflicts(tmp_path, monkeypatch):
  """Only a change of `main` is reported as one; other errors (network, credentials) as they
  are."""
  path = tmp_path / "out.zarr"
  run = icechunk_utils.Run(path, None, icechunk_utils.run_label("test"))
  run.write(dataset(1), message="v1")

  def unreachable(*args, **kwargs):
    raise OSError("connection reset")

  monkeypatch.setattr(icechunk.Repository, "reset_branch", unreachable)
  with pytest.raises(OSError, match="connection reset"):
    run.finish()


def test_a_promoted_run_succeeds_even_if_its_branch_stays(tmp_path, monkeypatch, caplog):
  """Once `main` has moved the run has succeeded: an error deleting its branch is logged."""
  path = tmp_path / "out.zarr"
  run = icechunk_utils.Run(path, None, icechunk_utils.run_label("test"))
  tip = run.write(dataset(1), message="v1")

  def broken(*args, **kwargs):
    raise OSError("store unavailable")

  monkeypatch.setattr(icechunk.Repository, "delete_branch", broken)
  assert run.finish() == tip
  assert repository(path).lookup_branch("main") == tip
  assert "could not be deleted" in caplog.text


def test_process_runs_are_labelled_process(tmp_path: pathlib.Path):
  """Runs carry the command's name in their branch and commit metadata."""
  write_output(dataset(1), tmp_path / "in.zarr", message="input")
  recipe = tmp_path / "process.toml"
  recipe.write_text('[read]\nengine = "icechunk"\n\n[[process]]\nname = "mean"\ndim = "t"\n')
  out = tmp_path / "out.zarr"
  args = ["process", "--dask-setup", "threads", str(recipe), str(tmp_path / "in.zarr"), str(out)]
  assert CliRunner().invoke(cli, args).exit_code == 0
  assert run_labels(out)[0].startswith("process-")
