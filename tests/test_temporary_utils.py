# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Temporary data never outlive the process (intent 9)."""

import os
import pathlib

from arcomake.temporary_utils import TempDirectoryRegistry, get_current_temp_registry


def test_registry_is_scoped_and_cleans_up(tmp_path: pathlib.Path):
  """Inside the context the registry is current; on exit its directories are deleted."""
  with TempDirectoryRegistry(base_dir=tmp_path) as registry:
    assert get_current_temp_registry() is registry
    directory = registry.create_temp_dir(suffix=".zarr")
    (directory / "file").write_text("data")
    assert directory.is_relative_to(registry.job_dir)
  assert get_current_temp_registry() is None
  assert not directory.exists()
  assert not registry.job_dir.exists()


def test_registry_cleans_up_on_error(tmp_path: pathlib.Path):
  """An exception inside the context still removes the temporary data."""
  try:
    with TempDirectoryRegistry(base_dir=tmp_path) as registry:
      job_dir = registry.job_dir
      raise RuntimeError("failure")
  except RuntimeError:
    pass
  assert not job_dir.exists()


def test_stale_directories_of_dead_processes_are_removed(tmp_path: pathlib.Path):
  """Directories left by killed runs (dead PID) are removed; those of live processes are kept."""
  dead = tmp_path / "pid_999999999_abcdef_x"
  live = tmp_path / f"pid_{os.getppid()}_abcdef_x"
  dead.mkdir()
  live.mkdir()
  TempDirectoryRegistry.clean_stale_directories(tmp_path)
  assert not dead.exists()
  assert live.exists()
