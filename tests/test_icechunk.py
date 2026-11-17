# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Outputs are Icechunk repositories: storages from paths, versions, the `icechunk` engine."""

import pathlib

import icechunk
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli, dataset_utils, icechunk_utils
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


def test_outputs_written_in_turn_on_a_distributed_scheduler(ocean_ds, tmp_path, monkeypatch):
  """On a distributed client each output is written (and computed) in its own commit."""
  monkeypatch.setattr(dataset_utils, "_distributed_client", lambda: True)
  outputs = [(ocean_ds, tmp_path / "a.zarr"), (ocean_ds * 2, tmp_path / "b.zarr")]
  assert len(write_outputs_together(outputs, SaveConfig(), message="both")) == 2
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
