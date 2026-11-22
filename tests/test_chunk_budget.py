# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The memory budget of planned chunks: given (`download --memory-limit`, `ARCOMAKE_MEMORY_LIMIT`)
or detected, shared among Dask's threads."""

import importlib
import logging
import pathlib

import dask
import dask.system
import distributed.system
import pytest
from click.testing import CliRunner

from arcomake import cli
from arcomake.chunk_plan import Budget, detect_budget
from tests.conftest import render_recipe

GB = 10**9


def test_a_given_limit_is_shared_among_threads():
  with dask.config.set(scheduler="threads", num_workers=5):
    assert detect_budget(8 * GB) == Budget(8 * GB, 5, "given")


def test_the_limit_is_detected(monkeypatch: pytest.MonkeyPatch):
  """Without a limit: the memory the process may use (distributed's: RAM, cgroup, rlimit) and
  the CPUs it may use."""
  monkeypatch.delenv("ARCOMAKE_MEMORY_LIMIT", raising=False)
  monkeypatch.setattr(distributed.system, "MEMORY_LIMIT", 64 * GB)
  monkeypatch.setattr(dask.system, "CPU_COUNT", 12)
  with dask.config.set(scheduler="threads", num_workers=None):
    assert detect_budget() == Budget(64 * GB, 12, "detected")


def test_the_limit_of_the_settings_is_used_by_every_command(monkeypatch: pytest.MonkeyPatch):
  """`ARCOMAKE_MEMORY_LIMIT` reached only `download` (its option's envvar): the climatology sized
  its series and copy threads for the whole machine."""
  monkeypatch.setenv("ARCOMAKE_MEMORY_LIMIT", "40GB")
  monkeypatch.setattr(distributed.system, "MEMORY_LIMIT", 64 * GB)
  with dask.config.set(scheduler="threads", num_workers=4):
    assert detect_budget() == Budget(40 * GB, 4, "ARCOMAKE_MEMORY_LIMIT")
    assert detect_budget(8 * GB).memory_limit == 8 * GB  # an explicit limit wins


def test_the_synchronous_scheduler_is_one_thread():
  with dask.config.set(scheduler="synchronous"):
    assert detect_budget(GB).threads == 1


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  return render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))


@pytest.mark.parametrize("value", ["lots", "0", "-2GB"])
def test_the_option_takes_an_amount_of_memory(recipe, tmp_path: pathlib.Path, value: str):
  args = ["download", "--memory-limit", value, str(recipe), str(tmp_path / "out")]
  result = CliRunner().invoke(cli, args)
  assert result.exit_code == 2
  assert "is not an amount of memory" in result.output
  # From the environment, the settings refuse it first, naming the variable.
  result = CliRunner(env={"ARCOMAKE_MEMORY_LIMIT": value}).invoke(cli, args[:1] + args[3:])
  assert result.exit_code != 0 and "ARCOMAKE_MEMORY_LIMIT" in result.output
  assert not (tmp_path / "out").exists()


def test_download_logs_its_budget(recipe, tmp_path: pathlib.Path, caplog, monkeypatch):
  module = importlib.import_module("arcomake.download")  # `arcomake.download` is the command
  monkeypatch.setattr(module, "set_default_logger", lambda *args, **kwargs: None)  # keeps caplog
  args = ["download", "--dask-setup", "threads", str(recipe), str(tmp_path / "out")]
  with caplog.at_level(logging.INFO, logger="arcomake"):
    result = CliRunner(env={"ARCOMAKE_MEMORY_LIMIT": "8GB"}).invoke(cli, args)
  assert result.exit_code == 0, result.output
  assert "7.45 GiB (ARCOMAKE_MEMORY_LIMIT)" in caplog.text
