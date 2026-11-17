# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Every request to a source has a timeout (6.2): a stalled connection fails the block, which is
retried, instead of hanging until the job's time limit (N7)."""

import datetime
import http.server
import threading
import time

import aiohttp
import gcsfs
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from arcomake import backends, dataset_utils

try:  # gcsfs falls back to its base filesystem without it
  from gcsfs.extended_gcsfs import BucketType, ExtendedGcsFileSystem
except ImportError:
  ExtendedGcsFileSystem = None
from arcomake.backends import EarlyWarningDataStore, NetCDFOverHTTP
from arcomake.dataset_utils import (
  DEFAULT_TIMEOUT,
  CopernicusMarinePartConfig,
  DatasetConfig,
  EarlyWarningDataStorePartConfig,
  GenericPartConfig,
  NetCDFOverHTTPPartConfig,
  ZarrPartConfig,
  download_and_process,
  gcs_timeout,
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
  """gcsfs waits forever by default: Zarr parts on GCS get `requests_timeout`, from the part's
  `timeout` or `DEFAULT_TIMEOUT`."""
  read(ZarrPartConfig(filename_or_obj="gs://b/era5.zarr"))
  read(ZarrPartConfig.model_validate({"filename_or_obj": "gs://b/era5.zarr", "timeout": 30}))
  assert [c["storage_options"]["requests_timeout"] for c in opened] == [
    gcs_timeout(DEFAULT_TIMEOUT),
    gcs_timeout(30),
  ]
  assert gcs_timeout(DEFAULT_TIMEOUT) == aiohttp.ClientTimeout(
    total=None, sock_connect=30, sock_read=120
  )


def test_gcsfs_requests_timeout_is_refused():
  """gcsfs's `requests_timeout`, a number, bounds a whole request and is not retried by gcsfs:
  the recipe is told to set the part's `timeout` (elsewhere it is left alone)."""
  with pytest.raises(ValidationError, match="set the part's `timeout` instead"):
    ZarrPartConfig.model_validate(
      {"filename_or_obj": "gs://b/era5.zarr", "storage_options": {"requests_timeout": 600}}
    )
  with pytest.raises(ValidationError, match="requests_timeout"):
    GenericPartConfig.model_validate(
      {"filename_or_obj": "gcs://b/x.nc", "storage_options": {"requests_timeout": 60}}
    )
  ZarrPartConfig.model_validate(
    {"filename_or_obj": "s3://b/x.zarr", "storage_options": {"requests_timeout": 600}}
  )


class StallingGCS(http.server.BaseHTTPRequestHandler):
  """A Google Cloud Storage download endpoint whose first answer for each object stalls."""

  stall = 2.0
  seen: dict[str, int]

  def do_GET(self):
    count = self.seen[self.path] = self.seen.get(self.path, 0) + 1
    try:
      if count == 1:
        time.sleep(self.stall)  # nothing sent: the client waits for data
      body = b"chunk"
      self.send_response(200)
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):  # the client gave up on this one
      pass

  def log_message(self, format, *args):
    pass


@pytest.fixture
def stalling_gcs(monkeypatch: pytest.MonkeyPatch):
  """The endpoint URL of a local `StallingGCS`, and the requests it saw per path.

  gcsfs's default filesystem (the one `gs://` URLs get) first asks a bucket's layout over gRPC,
  which this endpoint does not serve: it is told the bucket is a plain one."""
  if ExtendedGcsFileSystem is not None and issubclass(gcsfs.GCSFileSystem, ExtendedGcsFileSystem):

    async def plain(self, bucket):
      return BucketType.NON_HIERARCHICAL

    monkeypatch.setattr(ExtendedGcsFileSystem, "_get_bucket_type", plain)
  seen: dict[str, int] = {}
  handler = type("Handler", (StallingGCS,), {"seen": seen})
  with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


def _gcs(endpoint: str, requests_timeout) -> gcsfs.GCSFileSystem:
  return gcsfs.GCSFileSystem(
    token="anon", endpoint_url=endpoint, requests_timeout=requests_timeout, skip_instance_cache=True
  )


def _part_timeout(seconds: float):
  options: dict = {"filename_or_obj": "gs://bucket/data.zarr"}
  dataset_utils._with_timeout(options, "zarr", seconds)
  return options["storage_options"]["requests_timeout"]


def test_a_stalled_gcs_answer_is_requested_again(stalling_gcs):
  """With arco-make's timeout, gcsfs asks again for a stalled object (one ERA5 chunk took 212 s
  on 17 Nov while its block waited for it alone): the read succeeds, the block goes on."""
  endpoint, seen = stalling_gcs
  assert _gcs(endpoint, _part_timeout(1.0)).cat_file("bucket/data.zarr/0.0") == b"chunk"
  assert list(seen.values()) == [2]


def test_a_total_timeout_is_not_retried_by_gcsfs(stalling_gcs):
  """Why not a number: as a total, the timeout raises a plain `TimeoutError`, which gcsfs does
  not retry, and the whole block fails (and is done again)."""
  endpoint, seen = stalling_gcs
  with pytest.raises(TimeoutError) as raised:
    _gcs(endpoint, 1.0).cat_file("bucket/data.zarr/0.0")
  assert not isinstance(raised.value, aiohttp.ClientError)
  assert list(seen.values()) == [1]


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


def test_icechunk_inputs_are_left_alone(opened):
  """Icechunk storages are not fsspec: gcsfs or botocore options would be refused."""
  read(
    GenericPartConfig.model_validate(
      {"filename_or_obj": "gs://b/repo", "engine": "icechunk", "storage_options": {"token": "anon"}}
    )
  )
  read(
    GenericPartConfig.model_validate(
      {"filename_or_obj": "s3://b/repo", "engine": "icechunk", "timeout": 30}
    )
  )
  assert opened[0]["storage_options"] == {"token": "anon"}
  assert "storage_options" not in opened[1]
