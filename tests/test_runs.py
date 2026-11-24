# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Runs write on a branch of their own; `main` moves only when a run succeeds (`--replace` writes
a new version of an existing output); the recipe's retention expires the versions replaced."""

import datetime
import pathlib
from types import SimpleNamespace

import icechunk
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner
from pydantic import ValidationError

from arcomake import cli, dataset_utils, icechunk_utils
from arcomake.dataset_utils import SaveConfig, write_output
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
      raise RuntimeError("simulated failure")  # not transient: no retry
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


def test_keep_versions_expires_older_versions(tmp_path: pathlib.Path, no_grace):
  """`keep_versions = 2`: after a third run, the first version is expired and its data deleted;
  the two newest stay readable."""
  path = tmp_path / "out.zarr"
  save = SaveConfig(keep_versions=2)
  snapshots = [write_output(dataset(v), path, save, message=f"v{v}") for v in (1, 2, 3)]
  assert len(run_labels(path)) == 2
  for snapshot, value in zip(snapshots[1:], (2, 3), strict=True):
    store = icechunk_utils.readonly_store(path, snapshot_id=snapshot)
    xr.testing.assert_identical(xr.open_zarr(store, consolidated=False).load(), dataset(value))
  with pytest.raises(Exception):  # noqa: B017 (icechunk's error type for a deleted snapshot)
    xr.open_zarr(icechunk_utils.readonly_store(path, snapshot_id=snapshots[0])).load()


def test_without_retention_every_version_stays(tmp_path: pathlib.Path):
  """Without keep_history or keep_versions nothing is expired."""
  path = tmp_path / "out.zarr"
  for value in (1, 2, 3):
    write_output(dataset(value), path, message=f"v{value}")
  assert len(run_labels(path)) == 3


@pytest.mark.parametrize(
  ("save", "message"),
  [
    ({"keep_history": "30D", "keep_versions": 2}, "not both"),
    ({"keep_versions": 0}, "greater than or equal to 1"),
    ({"keep_history": "-1D"}, "not negative"),
  ],
)
def test_invalid_retention_fails_when_loaded(save: dict, message: str):
  with pytest.raises(ValidationError, match=message):
    SaveConfig.model_validate(save)


def test_retention_from_the_recipe():
  """`keep_history` takes durations as recipes write them; `retention()` is None without one."""
  assert SaveConfig.model_validate({"keep_history": "30D"}).keep_history.days == 30  # type: ignore[union-attr]
  assert SaveConfig().retention() is None
  assert isinstance(SaveConfig(keep_versions=3).retention(), icechunk_utils.Retention)


def test_retention_errors_do_not_fail_a_promoted_run(tmp_path, monkeypatch, caplog):
  """The run is promoted before the retention applies: an error there is logged, not raised."""
  path = tmp_path / "out.zarr"
  write_output(dataset(1), path, message="v1")

  def broken(*args, **kwargs):
    raise OSError("store unavailable")

  monkeypatch.setattr(icechunk.Repository, "expire_snapshots", broken)
  write_output(dataset(2), path, SaveConfig(keep_versions=1), message="v2")
  xr.testing.assert_identical(open_output(path).load(), dataset(2))
  assert "retention could not be applied" in caplog.text
  assert dataset_utils  # imported for the module under test


@pytest.fixture
def no_grace(monkeypatch: pytest.MonkeyPatch):
  """Collect at once (the grace period protects other runs' uncommitted data for days)."""
  monkeypatch.setattr(icechunk_utils, "GRACE", datetime.timedelta(0))


def test_keep_history_expires_versions_replaced_before_the_limit(tmp_path, no_grace):
  """`keep_history`: a version is kept while it was replaced less than that long ago (here a
  limit of zero: only the current version stays); branches of failed runs older than the limit
  and the grace period are deleted."""
  path = tmp_path / "out.zarr"
  write_output(dataset(1), path, message="v1")
  failed = icechunk_utils.Run(path, None, icechunk_utils.run_label("failed"))
  failed.write(dataset(9), message="never finished")
  write_output(dataset(2), path, SaveConfig(keep_history="0s"), message="v2")
  assert len(run_labels(path)) == 1
  assert repository(path).list_branches() == {"main"}
  xr.testing.assert_identical(open_output(path).load(), dataset(2))


def snapshot(label: str, days_ago: float) -> SimpleNamespace:
  written = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_ago)
  return SimpleNamespace(metadata={icechunk_utils.RUN_KEY: label}, written_at=written)


class FakeHistory:
  """`main`'s history, newest first, counting the snapshots read."""

  def __init__(self, snapshots):
    self.snapshots, self.read = snapshots, 0

  def ancestry(self, branch):
    for item in self.snapshots:
      self.read += 1
      yield item


def test_keep_history_counts_from_the_replacement():
  """A version written long ago but replaced recently is kept: the age that counts is since it
  was replaced (when the next version was written), not since it was written."""
  retention = icechunk_utils.Retention(keep_history=datetime.timedelta(days=30))
  recent_replacement = FakeHistory([snapshot("c", 0), snapshot("b", 60)])
  assert retention.limit(recent_replacement) is None  # type: ignore[arg-type]
  history = FakeHistory([snapshot("c", 0), snapshot("b", 60), snapshot("a", 90)])
  b = history.snapshots[1]
  assert retention.limit(history) == b.written_at  # type: ignore[arg-type]  # a: replaced 60 days ago


def test_keep_versions_reads_only_the_history_it_needs():
  """Finding the versions to keep reads `main`'s history only until one more run is found."""
  blocks = [snapshot(label, i) for i, label in enumerate("ccccbbbbaaaa")]
  history = FakeHistory(blocks)
  limit = icechunk_utils.Retention(keep_versions=1).limit(history)  # type: ignore[arg-type]
  assert limit == blocks[0].written_at and history.read == 5


def test_another_run_in_progress_is_left_alone(tmp_path: pathlib.Path):
  """Retention after one run keeps the branch, and the data, of another run still writing to the
  same repository (younger than the grace period)."""
  path = tmp_path / "out.zarr"
  write_output(dataset(1), path, message="v1")
  other = icechunk_utils.Run(path, None, icechunk_utils.run_label("other"))
  other.write(dataset(7), message="still writing")
  write_output(dataset(2), path, SaveConfig(keep_versions=1), message="v2")
  assert other.branch in repository(path).list_branches()
  store = repository(path).readonly_session(branch=other.branch).store
  xr.testing.assert_identical(xr.open_zarr(store, consolidated=False).load(), dataset(7))


def test_nothing_expired_means_no_collection(tmp_path, no_grace, monkeypatch):
  """Garbage collection scans the whole repository: it runs only when snapshots were expired."""
  calls = []
  monkeypatch.setattr(icechunk.Repository, "garbage_collect", lambda *a, **kw: calls.append(a))
  path = tmp_path / "out.zarr"
  for value in (1, 2):
    write_output(dataset(value), path, SaveConfig(keep_versions=5), message=f"v{value}")
  assert calls == []


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


@pytest.mark.parametrize("value", ["30", 30])
def test_keep_history_needs_a_unit(value):
  """A bare number would be read as nanoseconds or seconds and expire nearly the whole history."""
  with pytest.raises(ValidationError, match="with its unit"):
    SaveConfig.model_validate({"keep_history": value})


def test_labels_are_read_back_by_command_and_start():
  """Only `parse_label` takes labels apart: a command may contain dashes (`download-extend`),
  and branches that are not runs' are ignored."""
  label = icechunk_utils.run_label("download-extend")
  parsed = icechunk_utils.parse_label(label)
  assert parsed is not None
  command, started = parsed
  assert command == "download-extend"
  assert abs(datetime.datetime.now() - started) < datetime.timedelta(days=1)
  assert icechunk_utils.parse_label("main") is None
  assert icechunk_utils.parse_label("download-20200101") is None


def test_a_waiting_run_is_one_of_the_commands_asked_for(tmp_path: pathlib.Path):
  """`download` does not match a failed `download-extend` run (a shared prefix), and the newest
  of the commands asked for is the waiting run, with its command."""
  path = tmp_path / "out"
  write_output(dataset(0.0), path, message="first")
  icechunk_utils.Run(path, None, icechunk_utils.run_label("download"))
  icechunk_utils.Run(path, None, icechunk_utils.run_label("download-extend"))
  repo = icechunk_utils.existing_repository(path)
  waiting = icechunk_utils.waiting_run(repo, "download")
  assert waiting is not None and waiting.command == "download"
  newest = icechunk_utils.waiting_run(repo, "download", "download-extend")
  assert newest is not None and newest.command == "download-extend"
  assert icechunk_utils.waiting_run(repo, "process") is None
  assert icechunk_utils.waiting_run(None, "download") is None
