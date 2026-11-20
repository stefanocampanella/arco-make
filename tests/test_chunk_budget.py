# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The memory budget of planned chunks: given (`download --memory-limit`, `ARCOMAKE_MEMORY_LIMIT`)
or detected, shared among Dask's threads and the blocks at once."""

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


def test_a_given_limit_is_shared_among_threads_and_blocks():
  with dask.config.set(scheduler="threads", num_workers=5):
    assert detect_budget(8 * GB, blocks=2) == Budget(8 * GB, 5, 2, "given")


def test_the_limit_is_detected(monkeypatch: pytest.MonkeyPatch):
  """Without a limit: the memory the process may use (distributed's: RAM, cgroup, rlimit) and
  the CPUs it may use."""
  monkeypatch.setattr(distributed.system, "MEMORY_LIMIT", 64 * GB)
  monkeypatch.setattr(dask.system, "CPU_COUNT", 12)
  with dask.config.set(scheduler="threads", num_workers=None):
    assert detect_budget() == Budget(64 * GB, 12, 1, "detected")


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
  assert "7.45 GiB (given)" in caplog.text
