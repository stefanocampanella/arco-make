# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Remote backends, with the network replaced by fakes (no credentials needed)."""

import pathlib
import zipfile
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from arcomake import backends
from arcomake.backends import CopernicusMarine, EarlyWarningDataStore
from arcomake.temporary_utils import TempDirectoryRegistry


def test_ewds_requests_split_by_month():
  """EWDS requests group consecutive days of the same month and year (one request each)."""
  groups = EarlyWarningDataStore._consecutive_dates_with_same_month_or_year(
    datetime(2020, 2, 27), datetime(2020, 3, 2)
  )
  assert [[d.day for d in g] for g in groups] == [[27, 28, 29], [1]]


@pytest.mark.xfail(reason="B7: an explicit cdsapi URL is never passed to the client", strict=True)
def test_cdsapi_client_receives_explicit_url(monkeypatch: pytest.MonkeyPatch):
  """`url` given to `get_cdsapi_client` reaches `cdsapi.Client`."""
  received = {}

  class FakeClient:
    def __init__(self, **kwargs):
      received.update(kwargs)

  monkeypatch.setattr(backends.cdsapi, "Client", FakeClient)
  EarlyWarningDataStore.get_cdsapi_client(url="https://example.invalid/api")
  assert received.get("url") == "https://example.invalid/api"


@pytest.mark.xfail(
  reason="B8: drop_variables is accepted and ignored by the Copernicus backend", strict=True
)
def test_copernicus_marine_drop_variables(monkeypatch: pytest.MonkeyPatch):
  """`drop_variables` removes variables, as for xarray's own backends."""
  fake = xr.Dataset({"a": ("x", np.zeros(2)), "b": ("x", np.ones(2))})
  monkeypatch.setattr(backends.cm, "open_dataset", lambda **kwargs: fake)
  out = CopernicusMarine().open_dataset("cm://some-dataset", drop_variables=["b"])
  assert list(out.data_vars) == ["a"]


def test_copernicus_marine_requires_dataset_id():
  """A `cm://` URL without a dataset id is rejected before any request."""
  with pytest.raises(ValueError, match="dataset ID"):
    CopernicusMarine().open_dataset("cm://")


@pytest.mark.xfail(
  reason="B16: the EWDS zip is downloaded to the system temporary directory", strict=True
)
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
