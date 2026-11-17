# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Every request to a source has a timeout (6.2): a stalled connection fails the block, which is
retried, instead of hanging until the job's time limit (N7)."""

import datetime

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from arcomake import backends, dataset_utils
from arcomake.backends import EarlyWarningDataStore, NetCDFOverHTTP
from arcomake.dataset_utils import (
  DEFAULT_TIMEOUT,
  CopernicusMarinePartConfig,
  DatasetConfig,
  EarlyWarningDataStorePartConfig,
  NetCDFOverHTTPPartConfig,
  ZarrPartConfig,
  download_and_process,
)
from arcomake.temporary_utils import TempDirectoryRegistry

START = datetime.datetime(2020, 1, 1)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
  """The arguments every part is opened with (no source is read)."""
  calls: list[dict] = []

  def fake_open_input(**kwargs):
    calls.append(kwargs)
    return xr.Dataset({"x": ("time", np.zeros(1))}, coords={"time": [np.datetime64(START, "ns")]})

  monkeypatch.setattr(dataset_utils, "open_input", fake_open_input)
  return calls


def read(part) -> None:
  download_and_process(DatasetConfig(parts=[part]), START, START + datetime.timedelta(days=1))


def test_gcs_zarr_parts_get_a_request_timeout(opened):
  """gcsfs waits forever by default: Zarr parts on GCS get `requests_timeout`, the part's
  `timeout` or `DEFAULT_TIMEOUT`; a value in the recipe's storage options wins."""
  read(ZarrPartConfig(filename_or_obj="gs://b/era5.zarr"))
  read(ZarrPartConfig.model_validate({"filename_or_obj": "gs://b/era5.zarr", "timeout": 30}))
  read(
    ZarrPartConfig.model_validate(
      {"filename_or_obj": "gs://b/era5.zarr", "storage_options": {"requests_timeout": 600}}
    )
  )
  assert [c["storage_options"]["requests_timeout"] for c in opened] == [DEFAULT_TIMEOUT, 30, 600]


def test_s3_zarr_parts_get_botocore_timeouts_when_set(opened):
  """On S3 botocore already times out (60 s); a part's `timeout` sets its connect and read
  timeouts."""
  read(ZarrPartConfig(filename_or_obj="s3://b/x.zarr"))
  read(ZarrPartConfig.model_validate({"filename_or_obj": "s3://b/x.zarr", "timeout": 45}))
  assert "storage_options" not in opened[0]
  assert opened[1]["storage_options"]["config_kwargs"] == {
    "connect_timeout": 45,
    "read_timeout": 45,
  }


def test_http_and_ewds_parts_pass_the_timeout_to_their_backend(opened):
  read(NetCDFOverHTTPPartConfig(filename_or_obj="https://example.org/static.nc"))
  read(
    EarlyWarningDataStorePartConfig.model_validate(
      {"filename_or_obj": "ewds://cems-glofas-historical", "timeout": 300}
    )
  )
  assert [c["timeout"] for c in opened] == [DEFAULT_TIMEOUT, 300]


def test_local_and_other_parts_are_left_alone(opened):
  read(ZarrPartConfig(filename_or_obj="/data/local.zarr"))
  assert "storage_options" not in opened[0] and "timeout" not in opened[0]


def test_copernicus_marine_parts_take_no_timeout():
  """The toolbox sets its own timeouts and retries: a `timeout` there is a recipe error."""
  with pytest.raises(ValidationError, match="timeout"):
    CopernicusMarinePartConfig.model_validate({"filename_or_obj": "cm://x", "timeout": 30})


def test_http_backend_requests_have_a_timeout(monkeypatch: pytest.MonkeyPatch):
  """`requests.get` without a timeout hangs on a stalled connection."""
  seen = {}

  class Stop(Exception):
    pass

  def fake_get(url, **kwargs):
    seen.update(kwargs)
    raise Stop

  monkeypatch.setattr(backends.requests, "get", fake_get)
  with TempDirectoryRegistry(), pytest.raises(Stop):
    NetCDFOverHTTP().open_dataset("https://example.org/static.nc", timeout=90)
  assert seen["timeout"] == (30.0, 90)


def test_ewds_requests_pass_the_timeout_to_cdsapi(monkeypatch: pytest.MonkeyPatch):
  seen = {}

  def fake_process_requests(self, dataset_name, requests, max_workers=None, **kwargs):
    seen.update(kwargs)
    return []

  monkeypatch.setattr(EarlyWarningDataStore, "_process_requests", fake_process_requests)
  monkeypatch.setattr(backends, "get_current_temp_registry", lambda: object())
  with pytest.raises(ValueError):  # no datasets to concatenate: the request was never made
    EarlyWarningDataStore().open_dataset(
      "ewds://cems-glofas-historical",
      start_datetime=START,
      end_datetime=START + datetime.timedelta(days=1),
      system_version="v",
      hydrological_model="m",
      product_type="p",
      timespan="t",
      variable="v",
      timeout=200,
    )
  assert seen == {"timeout": 200}
