# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Files over HTTP are fetched once per run: every block of a download reads the same file (the
GloFAS static file was fetched again for every block, and an outage of its server on 16 Nov
failed the block it met)."""

import pathlib

import numpy as np
import pytest
import xarray as xr

from arcomake import backends
from arcomake.backends import NetCDFOverHTTP
from arcomake.temporary_utils import TempDirectoryRegistry


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


class FakeResponse:
  def __init__(self, payload: bytes):
    self.payload = payload

  def __enter__(self):
    return self

  def __exit__(self, *args):
    return False

  def raise_for_status(self):
    pass

  def iter_content(self, chunk_size: int):
    for start in range(0, len(self.payload), chunk_size):
      yield self.payload[start : start + chunk_size]


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
