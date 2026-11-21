# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Outputs are Icechunk repositories: storages from paths, versions, the `icechunk` engine."""

import pathlib

import icechunk
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli, icechunk_utils
from arcomake.dataset_utils import SaveConfig, open_input, write_output, write_outputs_together
from tests.conftest import open_output, render_recipe


def test_download_commits_one_block_at_a_time(tmp_path: pathlib.Path, sources: pathlib.Path):
  """`download` commits each block, named by its interval: the history tells which are in."""
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  output = tmp_path / "out.zarr"
  result = CliRunner().invoke(
    cli, ["download", "--dask-setup", "threads", str(recipe), str(output)]
  )
  assert result.exit_code == 0, result.output
  repo = icechunk.Repository.open(icechunk_utils.storage(output))
  messages = [snapshot.message for snapshot in repo.ancestry(branch="main")]
  assert messages == [
    "download [2020-03-01T00:00, 2020-03-02T00:00)",
    "download [2020-02-28T00:00, 2020-03-01T00:00)",
    "download [2020-02-26T00:00, 2020-02-28T00:00)",
    "Repository initialized",
  ]


def test_the_icechunk_engine_reads_any_version(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`engine = "icechunk"` reads the tip of `main`, or a tag, branch or snapshot."""
  path = tmp_path / "out.zarr"
  first = write_output(ocean_ds, path, message="first")
  write_output(ocean_ds + 1, path, message="second")
  icechunk.Repository.open(icechunk_utils.storage(path)).create_tag("v1", snapshot_id=first)
  latest = open_input(str(path), engine="icechunk", chunks={})
  xr.testing.assert_allclose(latest.load(), ocean_ds + 1)
  for version in ({"snapshot_id": first}, {"tag": "v1"}):
    old = open_input(str(path), engine="icechunk", chunks={}, **version)
    xr.testing.assert_allclose(old.load(), ocean_ds)
  # `consolidated`, which recipes give for Zarr inputs, is accepted and ignored.
  open_input(str(path), engine="icechunk", consolidated=True).close()


@pytest.mark.parametrize("scheduler", ["synchronous", "threads", "processes"])
def test_outputs_written_together_with_any_local_scheduler(ocean_ds, tmp_path, scheduler):
  """Each output is written in its own commit, whatever the scheduler: Icechunk's writer
  collects the writes of other processes too (they once went through this process's sessions,
  with threads instead of processes)."""
  import dask

  outputs = [(ocean_ds, tmp_path / "a.zarr"), (ocean_ds * 2, tmp_path / "b.zarr")]
  with dask.config.set(scheduler=scheduler):
    assert len(write_outputs_together(outputs, SaveConfig(), label="both")) == 2
  xr.testing.assert_allclose(open_output(tmp_path / "a.zarr").load(), ocean_ds)
  xr.testing.assert_allclose(open_output(tmp_path / "b.zarr").load(), ocean_ds * 2)


def test_s3_and_gcs_storages_from_storage_options():
  """Recipes give fsspec-like options; only those Icechunk can honour are accepted."""
  s3 = icechunk_utils.storage("s3://bucket/some/prefix", {"region": "eu-south-1", "anon": True})
  assert isinstance(s3, icechunk.Storage)
  gcs = icechunk_utils.storage("gs://bucket/prefix", {"token": "anon"})
  assert isinstance(gcs, icechunk.Storage)
  with pytest.raises(ValueError, match=r"\['client_kwargs'\] are not understood"):
    icechunk_utils.storage("s3://bucket/prefix", {"client_kwargs": {}})
  with pytest.raises(ValueError, match="local paths, s3:// or gs://"):
    icechunk_utils.storage("https://example.org/repo")


def test_existence_checks_get_fsspec_options():
  """The region of an S3 repository becomes s3fs's `client_kwargs.region_name`."""
  options = icechunk_utils.fsspec_options("s3://b/p", {"region": "eu-south-1", "profile": "x"})
  assert options == {"profile": "x", "client_kwargs": {"region_name": "eu-south-1"}}
  assert icechunk_utils.fsspec_options("/local/path", {"region": "r"}) == {"region": "r"}


def test_s3_credentials_come_from_botocore(monkeypatch: pytest.MonkeyPatch):
  """S3 credentials are resolved by botocore (profiles icechunk's own client cannot read), with
  an expiry, so that icechunk asks again before temporary credentials lapse."""
  import botocore.credentials

  monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
  monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-example")
  monkeypatch.delenv("AWS_PROFILE", raising=False)
  monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
  assert botocore.credentials  # imported for the environment provider
  credentials = icechunk_utils._botocore_credentials(None)
  assert isinstance(credentials, icechunk.S3StaticCredentials)
  assert credentials.access_key_id == "AKIDEXAMPLE" and credentials.expires_after is not None


class FakeS3:
  """botocore's S3 client, answering HeadBucket as S3 does for a bucket in another region."""

  calls: list = []

  def __init__(self, **kwargs):
    FakeS3.calls.append(kwargs)

  def head_bucket(self, Bucket):  # noqa: N803 (botocore's name)
    import botocore.exceptions

    headers = {"x-amz-bucket-region": "eu-south-1"}
    raise botocore.exceptions.ClientError(
      {"Error": {"Code": "301"}, "ResponseMetadata": {"HTTPHeaders": headers}}, "HeadBucket"
    )


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch):
  import botocore.session

  FakeS3.calls = []
  icechunk_utils._bucket_region.cache_clear()
  monkeypatch.setattr(
    botocore.session.Session, "create_client", lambda self, service, **kwargs: FakeS3(**kwargs)
  )
  monkeypatch.setattr(
    botocore.session.Session,
    "get_config_variable",
    lambda self, name: "us-east-1" if name == "region" else None,
  )
  yield FakeS3
  icechunk_utils._bucket_region.cache_clear()


def test_the_bucket_region_wins_over_the_profile(fake_s3):
  """The region S3 reports for the bucket (also in error answers) wins over the profile's: a
  client signing for the profile's region would be refused by a bucket elsewhere."""
  assert icechunk_utils._bucket_region("b", None, None, False) == "eu-south-1"


def test_the_region_is_asked_at_the_endpoint_and_anonymously(fake_s3):
  """The question goes to the recipe's endpoint, unsigned for anonymous repositories."""
  import botocore

  icechunk_utils.storage("s3://b/p", {"anon": True, "endpoint_url": "https://s3.example.org"})
  (call,) = fake_s3.calls
  assert call["endpoint_url"] == "https://s3.example.org"
  assert call["config"].signature_version is botocore.UNSIGNED


def test_unusable_destinations_fail_before_any_work(fake_s3):
  """Options or paths Icechunk cannot use fail at the output check, as a message (no download
  runs first, no traceback)."""
  import click

  from arcomake.cli_utils import check_destination

  bad = SaveConfig(storage_options={"client_kwargs": {"region_name": "x"}})
  with pytest.raises(click.ClickException, match="not understood"):
    check_destination("s3://b/out.zarr", save_configs=bad)
  with pytest.raises(click.ClickException, match="local paths, s3:// or gs://"):
    check_destination("https://example.org/out.zarr")


def test_gcs_tokens(tmp_path: pathlib.Path):
  """GCS: "anon", or a service account file; anything else is an error, not a silent switch to
  the environment's credentials."""
  account = tmp_path / "sa.json"
  account.write_text("{}")
  assert isinstance(icechunk_utils.storage("gs://b/p", {"token": str(account)}), icechunk.Storage)
  with pytest.raises(ValueError, match="service account file"):
    icechunk_utils.storage("gs://b/p", {"token": "google_default"})


@pytest.mark.parametrize("split", ["", 'split_dims = ["lat"]\n'], ids=["graph", "series"])
def test_climatology_outputs_share_their_commit_message(tmp_path: pathlib.Path, split: str):
  """Both outputs of a compute-climatology run carry the same message: their commits are not
  atomic together, and a run whose second commit failed shows."""
  import numpy as np
  import pandas as pd

  time = pd.date_range("2019-01-01", periods=730, freq="1D")
  values = np.random.default_rng(0).random((time.size, 1))
  ds = xr.Dataset({"x": (("time", "lat"), values)}, coords={"time": time, "lat": [0.0]})
  write_output(ds, tmp_path / "in.zarr", message="input")
  recipe = tmp_path / "clim.toml"
  recipe.write_text(
    f'period = 365\nwindow = 1\nweighting = "constant"\n{split}[read]\nengine = "icechunk"\n\n[save]\n'
  )
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  args = [str(recipe), str(tmp_path / "in.zarr"), str(clim), str(std)]
  path = ["--streaming"] if split else []
  result = CliRunner().invoke(cli, ["compute-climatology", *path, "--dask-setup", "threads", *args])
  assert result.exit_code == 0, result.output
  messages = [
    next(iter(icechunk.Repository.open(icechunk_utils.storage(p)).ancestry(branch="main"))).message
    for p in (clim, std)
  ]
  assert messages[0] == messages[1] and messages[0].startswith("compute-climatology-")


def test_s3_credentials_are_resolved_once_while_valid(monkeypatch: pytest.MonkeyPatch):
  """Resolving credentials again for every client or fork icechunk creates made S3 writes
  several times slower: they are reused while more than a minute of validity is left."""
  import botocore.session

  sessions = []
  original = botocore.session.Session.__init__

  def counting_init(self, *args, **kwargs):
    sessions.append(kwargs.get("profile"))
    original(self, *args, **kwargs)

  monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
  monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-example")
  monkeypatch.delenv("AWS_PROFILE", raising=False)
  monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
  monkeypatch.setattr(icechunk_utils, "_CREDENTIALS", {})
  monkeypatch.setattr(botocore.session.Session, "__init__", counting_init)
  first = icechunk_utils._botocore_credentials(None)
  assert icechunk_utils._botocore_credentials(None) is first
  assert sessions == [None]


def test_s3_stream_timeout_option(fake_s3):
  """`network_stream_timeout_seconds` is accepted (default `STREAM_TIMEOUT`, shorter than
  icechunk's 60 s: stalled requests are retried sooner)."""
  options = {"region": "eu-south-1", "anon": True, "network_stream_timeout_seconds": 5}
  assert isinstance(icechunk_utils.storage("s3://b/p", options), icechunk.Storage)
  assert icechunk_utils.STREAM_TIMEOUT < 60
