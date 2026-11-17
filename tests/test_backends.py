# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Remote backends, with the network replaced by fakes (no credentials needed)."""

import functools
import http.server
import pathlib
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime

import numpy as np
import pytest
import requests
import xarray as xr

from arcomake import backends
from arcomake.backends import CopernicusMarine, EarlyWarningDataStore
from arcomake.temporary_utils import TempDirectoryRegistry


def test_ewds_requests_split_by_month():
  """EWDS requests group consecutive days of the same month and year (one request each)."""
  groups = EarlyWarningDataStore._consecutive_dates_by_month(
    datetime(2020, 2, 27), datetime(2020, 3, 2)
  )
  assert [[d.day for d in g] for g in groups] == [[27, 28, 29], [1]]


def test_cdsapi_client_receives_explicit_url(monkeypatch: pytest.MonkeyPatch):
  """`url` given to `get_cdsapi_client` reaches `cdsapi.Client`."""
  received = {}

  class FakeClient:
    def __init__(self, **kwargs):
      received.update(kwargs)

  monkeypatch.setattr(backends.cdsapi, "Client", FakeClient)
  EarlyWarningDataStore.get_cdsapi_client(url="https://example.invalid/api")
  assert received.get("url") == "https://example.invalid/api"


def test_copernicus_marine_drop_variables(monkeypatch: pytest.MonkeyPatch):
  """`drop_variables` removes variables, as for xarray's own backends."""
  fake = xr.Dataset({"a": ("x", np.zeros(2)), "b": ("x", np.ones(2))})
  monkeypatch.setattr("copernicusmarine.open_dataset", lambda **kwargs: fake)
  out = CopernicusMarine().open_dataset("cm://some-dataset", drop_variables=["b"])
  assert list(out.data_vars) == ["a"]


def test_listing_the_backends_does_not_import_the_toolbox():
  """The toolbox reads COPERNICUSMARINE_MAX_POOL_CONNECTIONS at import: xarray listing its
  backends (any `open_dataset`) must not import it before the settings file is applied."""
  code = (
    "import sys, xarray, arcomake; xarray.backends.list_engines(); "
    "sys.exit('copernicusmarine' in sys.modules)"
  )
  assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_copernicus_marine_requires_dataset_id():
  """A `cm://` URL without a dataset id is rejected before any request."""
  with pytest.raises(ValueError, match="dataset ID"):
    CopernicusMarine().open_dataset("cm://")


def test_ewds_download_stays_in_registry(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
  """Every file of a request, the zip included, is inside the registry's directory (intent 9)."""
  seen: list[pathlib.Path] = []

  class FakeClient:
    def retrieve(self, name, request, target):
      seen.append(pathlib.Path(target))
      with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("data.grib", b"")

  monkeypatch.setattr(
    EarlyWarningDataStore, "get_cdsapi_client", staticmethod(lambda **kwargs: FakeClient())
  )
  monkeypatch.setattr(backends.xr, "open_mfdataset", lambda paths, **kwargs: xr.Dataset())
  with TempDirectoryRegistry(base_dir=tmp_path / "registry") as registry:
    EarlyWarningDataStore()._process_request("cems-glofas-historical", {})
    assert seen and all(path.is_relative_to(registry.job_dir) for path in seen)


@pytest.mark.parametrize("time_dim", ["time", "valid_time"])
def test_ewds_one_day_requests_use_time_dim(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, time_dim: str
):
  """A one-day request (no time dimension in the GRIB) is stamped along the recipe's time_dim."""
  day = xr.Dataset(
    {"dis": (("latitude", "longitude"), np.ones((2, 2)))},
    coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
  )
  monkeypatch.setattr(
    EarlyWarningDataStore,
    "_process_requests",
    lambda self, name, requests, **kwargs: [day for _ in requests],
  )
  with TempDirectoryRegistry(base_dir=tmp_path):
    out = EarlyWarningDataStore().open_dataset(
      "ewds://cems-glofas-historical",
      start_datetime=datetime(2020, 2, 29),
      end_datetime=datetime(2020, 3, 2),
      system_version="v",
      hydrological_model="m",
      product_type="p",
      timespan="t",
      variable="dis",
      time_dim=time_dim,
    )
  assert out.dis.dims == (time_dim, "latitude", "longitude")
  assert [str(t)[:10] for t in out[time_dim].values] == ["2020-02-29", "2020-03-01"]


def test_ewds_requests_open_grib_files_one_at_a_time(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """Requests download in parallel, but their GRIB files are opened one at a time.

  ecCodes' definition parser is not thread-safe: concurrent opens crashed a download
  ("fatal flex scanner internal error", segmentation fault) for an interval spanning two months.
  """
  active, peak = 0, 0
  guard = threading.Lock()

  class FakeClient:
    def retrieve(self, name, request, target):
      with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("data.grib", b"")

  def fake_open(paths, **kwargs):
    nonlocal active, peak
    with guard:
      active += 1
      peak = max(peak, active)
    time.sleep(0.05)
    with guard:
      active -= 1
    return xr.Dataset()

  monkeypatch.setattr(
    EarlyWarningDataStore, "get_cdsapi_client", staticmethod(lambda **kwargs: FakeClient())
  )
  monkeypatch.setattr(backends.xr, "open_mfdataset", fake_open)
  with TempDirectoryRegistry(base_dir=tmp_path):
    EarlyWarningDataStore()._process_requests("cems-glofas-historical", [{}] * 4, max_workers=4)
  assert peak == 1


class QuietHandler(http.server.SimpleHTTPRequestHandler):
  def log_message(self, format, *args):
    pass


@pytest.fixture
def http_root(tmp_path: pathlib.Path):
  """A local HTTP server for `tmp_path / "www"`; yields the directory and its base URL."""
  root = tmp_path / "www"
  root.mkdir()
  handler = functools.partial(QuietHandler, directory=str(root))
  with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_netcdf_over_http_reads_the_file(tmp_path: pathlib.Path, http_root):
  """The `netcdfoverhttp` engine downloads the file into the registry and opens it unchanged."""
  root, url = http_root
  expected = xr.Dataset(
    {"a": (("y", "x"), np.arange(6.0).reshape(2, 3)), "b": ("x", np.ones(3))},
    coords={"x": [0.0, 0.5, 1.0], "y": [10.0, 20.0]},
  )
  expected.to_netcdf(root / "static.nc", engine="netcdf4")
  with TempDirectoryRegistry(base_dir=tmp_path / "registry") as registry:
    out = xr.open_dataset(f"{url}/static.nc", engine="netcdfoverhttp", drop_variables=["b"])
    xr.testing.assert_identical(out.load(), expected.drop_vars("b"))
    out.close()
    downloaded = list(registry.job_dir.rglob("*.nc"))
  assert len(downloaded) == 1


def test_netcdf_over_http_error_fails(tmp_path: pathlib.Path, http_root):
  """An HTTP error response fails the read."""
  _, url = http_root
  with (
    TempDirectoryRegistry(base_dir=tmp_path / "registry"),
    pytest.raises(requests.HTTPError, match="404"),
  ):
    xr.open_dataset(f"{url}/missing.nc", engine="netcdfoverhttp")
