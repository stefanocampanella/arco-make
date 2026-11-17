# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Files over HTTP are fetched once per run: every block of a download reads the same file (the
GloFAS static file was fetched again for every block, and an outage of its server on 16 Nov
failed the block it met)."""

import pathlib
import threading

import numpy as np
import pytest
import xarray as xr

from arcomake import backends
from arcomake.backends import NetCDFOverHTTP
from arcomake.temporary_utils import TempDirectoryRegistry
from tests.conftest import FakeResponse


def test_a_file_is_fetched_once_per_registry(tmp_path: pathlib.Path):
  calls = []

  def fetch(path: pathlib.Path) -> None:
    calls.append(path)
    path.write_text("content")

  with TempDirectoryRegistry(base_dir=tmp_path) as registry:
    first = registry.cached_file("https://example.org/a.nc", "a.nc", fetch)
    second = registry.cached_file("https://example.org/a.nc", "a.nc", fetch)
    assert first == second and len(calls) == 1
  assert not first.exists()  # removed with the registry


def test_a_failed_fetch_is_tried_again(tmp_path: pathlib.Path):
  attempts = []

  def flaky(path: pathlib.Path) -> None:
    attempts.append(path)
    path.write_text("partial")
    if len(attempts) == 1:
      raise TimeoutError("server stalled")

  with TempDirectoryRegistry(base_dir=tmp_path) as registry:
    with pytest.raises(TimeoutError):
      registry.cached_file("key", "f.nc", flaky)
    assert not attempts[0].exists()  # the partial file is not kept
    path = registry.cached_file("key", "f.nc", flaky)
    assert path.read_text() == "partial" and len(attempts) == 2


def test_netcdf_over_http_downloads_once_per_run(tmp_path, monkeypatch: pytest.MonkeyPatch):
  """Two blocks opening the same URL make one request."""
  source = tmp_path / "static.nc"
  xr.Dataset({"uparea": ("x", np.arange(3.0))}).to_netcdf(source, engine="netcdf4")
  payload = source.read_bytes()
  requests_made = []

  def fake_get(url, **kwargs):
    requests_made.append(url)
    return FakeResponse(payload)

  monkeypatch.setattr(backends.requests, "get", fake_get)
  with TempDirectoryRegistry(base_dir=tmp_path / "tmp"):
    for _ in range(2):
      with NetCDFOverHTTP().open_dataset("https://example.org/static.nc", timeout=60) as ds:
        assert ds.uparea.values.tolist() == [0.0, 1.0, 2.0]
  assert requests_made == ["https://example.org/static.nc"]


def test_fetches_of_other_keys_do_not_wait(tmp_path: pathlib.Path):
  """A slow server holds up only the fetches of its own file."""
  other_done = threading.Event()

  def slow(path: pathlib.Path) -> None:
    assert other_done.wait(timeout=10), "the other fetch waited for this one"
    path.write_text("slow")

  def fast(path: pathlib.Path) -> None:
    path.write_text("fast")
    other_done.set()

  with TempDirectoryRegistry(base_dir=tmp_path) as registry:
    slow_fetch = threading.Thread(target=registry.cached_file, args=("slow", "s.nc", slow))
    slow_fetch.start()
    registry.cached_file("fast", "f.nc", fast)
    slow_fetch.join(timeout=10)
    assert not slow_fetch.is_alive() and registry.cached_file("slow", "s.nc", slow).exists()
