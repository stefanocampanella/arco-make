# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Outputs are Icechunk repositories: storages from paths, versions, the `icechunk` engine."""

import pathlib

import icechunk
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli, icechunk_utils
from arcomake.dataset_utils import SaveConfig, open_input, write_output, write_tree
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
def test_groups_written_in_one_commit_with_any_local_scheduler(ocean_ds, tmp_path, scheduler):
  """`write_tree` writes datasets into the groups of one repository in one commit, whatever the
  scheduler: Icechunk's writer collects the writes of other processes too (they once went
  through this process's sessions, with threads instead of processes)."""
  import dask

  path = tmp_path / "tree.zarr"
  with dask.config.set(scheduler=scheduler):
    tip = write_tree({"a": ocean_ds, "b": ocean_ds * 2}, path, SaveConfig(), label="both")
  xr.testing.assert_allclose(open_output(path, "a").load(), ocean_ds)
  xr.testing.assert_allclose(open_output(path, "b").load(), ocean_ds * 2)
  history = list(icechunk.Repository.open(icechunk_utils.storage(path)).ancestry(branch="main"))
  assert history[0].id == tip and history[0].message == "both" and len(history) == 2


def test_groups_nest_under_the_save_group(ocean_ds, tmp_path):
  """With `[save] group`, `write_tree`'s groups are under it, as `write_output` writes there."""
  path = tmp_path / "tree.zarr"
  write_tree({"a": ocean_ds}, path, SaveConfig.model_validate({"group": "g"}), label="x")
  xr.testing.assert_allclose(open_output(path, "g/a").load(), ocean_ds)


def test_read_tables_read_one_group(ocean_ds, tmp_path):
  """`[read] group` reads one group of an input, e.g. the climatology of a statistics repository
  (for `validate`, `process`), whose root holds no variables."""
  from arcomake.dataset_utils import ReadConfig

  path = tmp_path / "statistics.zarr"
  write_tree({"climatology": ocean_ds, "anomaly_std": ocean_ds * 2}, path, SaveConfig(), label="x")
  read = ReadConfig.model_validate({"engine": "icechunk", "group": "anomaly_std", "chunks": {}})
  with open_input(str(path), **read.model_dump(exclude_unset=True)) as dataset:
    xr.testing.assert_allclose(dataset.load(), ocean_ds * 2)


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
def test_climatology_groups_are_one_commit(tmp_path: pathlib.Path, split: str):
  """Both groups of a compute-climatology run are in one commit of one repository (both paths):
  the climatology and the anomaly std always come from the same run."""
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
  output = tmp_path / "statistics.zarr"
  args = [str(recipe), str(tmp_path / "in.zarr"), str(output)]
  path = ["--streaming"] if split else []
  result = CliRunner().invoke(cli, ["compute-climatology", *path, "--dask-setup", "threads", *args])
  assert result.exit_code == 0, result.output
  history = list(icechunk.Repository.open(icechunk_utils.storage(output)).ancestry(branch="main"))
  assert len(history) == 2 and history[0].message.startswith("compute-climatology-")
  tree = xr.open_datatree(icechunk_utils.readonly_store(output), engine="zarr", consolidated=False)
  assert {"/climatology", "/anomaly_std"} <= set(tree.groups)


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


@pytest.mark.parametrize("create", [False, True], ids=["read", "write"])
def test_requests_waiting_for_an_answer_are_retried_then_fail(monkeypatch, create: bool):
  """A server that accepts connections and never answers fails an open, to read or to write,
  after the retries (`open_repository`'s read timeout, or its per-attempt one for writing runs);
  with icechunk's defaults it waited forever (the stream timeout only covers answers in
  progress), and a read from S3 once waited 985 s."""
  import socket
  import threading

  server = socket.socket()
  server.bind(("127.0.0.1", 0))
  server.listen(64)
  held = []

  def accept():
    while True:
      try:
        held.append(server.accept()[0])  # never answered
      except OSError:
        return

  threading.Thread(target=accept, daemon=True).start()
  monkeypatch.setattr(icechunk_utils, "READ_TIMEOUT_MS", 300)
  monkeypatch.setattr(icechunk_utils, "MAX_TRIES", 2)
  monkeypatch.setattr(icechunk_utils, "WRITE_ATTEMPT_TIMEOUT_MS", 300)
  storage = icechunk.s3_storage(
    bucket="b",
    prefix="p",
    region="us-east-1",
    endpoint_url=f"http://127.0.0.1:{server.getsockname()[1]}",
    allow_http=True,
    access_key_id="x",
    secret_access_key="y",
  )
  outcome = []

  def open_it():
    try:
      icechunk_utils.open_repository(storage, create=create)
    except icechunk.IcechunkError as exc:
      outcome.append(exc)

  opening = threading.Thread(target=open_it, daemon=True)
  opening.start()
  opening.join(60)
  server.close()
  assert outcome and "timeout" in str(outcome[0])
  assert len(held) > 1  # retried


class ServedRepository:
  """An S3 endpoint serving, to GET requests, a repository made on local disk; once `stall` is
  set, it accepts requests and never answers them (`held`). Enough for icechunk to check, open
  and read the repository."""

  def __init__(self, root: pathlib.Path):
    import hashlib
    import http.server
    import threading

    icechunk.Repository.create(icechunk.local_filesystem_storage(str(root)))
    self.stall, self.closing, self.held = threading.Event(), threading.Event(), []
    served = self

    class Handler(http.server.BaseHTTPRequestHandler):
      protocol_version = "HTTP/1.1"

      def do_GET(self):  # noqa: N802 (http.server's name)
        if served.stall.is_set():
          served.held.append(self.path)
          served.closing.wait()
          return
        file = root / self.path.split("?")[0].removeprefix("/b/p/")
        body = file.read_bytes() if file.is_file() else b""
        self.send_response(200 if file.is_file() else 404)
        self.send_header("ETag", f'"{hashlib.md5(body).hexdigest()}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def log_message(self, *args):
        pass

    self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.server.daemon_threads = True
    threading.Thread(target=self.server.serve_forever, daemon=True).start()

  def storage(self, path=None, storage_options=None) -> icechunk.Storage:
    """A new storage of the endpoint (`icechunk_utils.storage`'s signature)."""
    return icechunk.s3_storage(
      bucket="b",
      prefix="p",
      region="us-east-1",
      endpoint_url=f"http://127.0.0.1:{self.server.server_port}",
      allow_http=True,
      access_key_id="x",
      secret_access_key="y",
    )

  def close(self):
    self.closing.set()
    self.server.shutdown()
    self.server.server_close()


@pytest.fixture
def served(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
  repository = ServedRepository(tmp_path / "repository")
  monkeypatch.setattr(icechunk_utils, "storage", repository.storage)
  yield repository
  repository.close()


def stalled_read(served: ServedRepository, repo: icechunk.Repository, wait: float = 20) -> str:
  """What a read of `repo` raises once the endpoint stalls ("still waiting" after `wait` s)."""
  import threading

  served.stall.set()
  outcome = ["still waiting"]

  def read():
    try:
      repo.lookup_branch("main")
      outcome[0] = "answered"
    except icechunk.IcechunkError as exc:
      outcome[0] = str(exc)

  reading = threading.Thread(target=read, daemon=True)
  reading.start()
  reading.join(wait)
  return outcome[0]


def test_a_checked_repository_reads_with_the_read_settings(served, monkeypatch):
  """`existing_repository` asks whether a repository exists, then opens it: on one storage, the
  opening kept icechunk's defaults (the client is built at the first request, with its settings),
  and a read the endpoint never answered waited forever."""
  monkeypatch.setattr(icechunk_utils, "READ_TIMEOUT_MS", 300)
  monkeypatch.setattr(icechunk_utils, "MAX_TRIES", 2)
  repo = icechunk_utils.existing_repository("s3://b/p")
  assert repo is not None
  assert "read timeout" in stalled_read(served, repo)
  assert len(served.held) == 2  # MAX_TRIES


def test_a_run_writes_with_the_write_settings(served, monkeypatch):
  """A run opens the repository to write on a storage of its own: a repository opened to read
  and reopened kept the read settings (a read timeout cuts uploads of large shards), and one asked
  whether it exists first kept icechunk's defaults (no bound on an attempt: an upload left
  unanswered held a production download for 3 h, logbook 28 Nov)."""
  monkeypatch.setattr(icechunk_utils, "READ_TIMEOUT_MS", 60_000)
  monkeypatch.setattr(icechunk_utils, "WRITE_ATTEMPT_TIMEOUT_MS", 300)
  monkeypatch.setattr(icechunk_utils, "MAX_TRIES", 3)
  checked = icechunk_utils.existing_repository("s3://b/p")  # as `download` checks first
  assert checked is not None
  waiting = icechunk_utils.WaitingRun("l", "download", "main", "x", (), None, True)
  run = icechunk_utils.Run("s3://b/p", None, "l", waiting=waiting)
  assert "operation attempt timeout" in stalled_read(served, run.repo)
  assert len(served.held) == 3  # MAX_TRIES


@pytest.mark.xfail(
  strict=True,
  reason="icechunk 2.3.1 builds a storage's client at its first request, with that request's"
  " settings, and keeps it: hence a storage per opening (`icechunk_utils._config`)",
)
@pytest.mark.parametrize("first", ["exists", "reopen"])
def test_icechunk_applies_the_settings_of_each_opening(served, first: str):
  """Pins icechunk's behaviour: should it apply each opening's settings, this fails (XPASS)."""
  settings = icechunk.StorageSettings(
    timeouts=icechunk.StorageTimeoutSettings(operation_attempt_timeout_ms=300),
    retries=icechunk.StorageRetriesSettings(max_tries=1),
  )
  config = icechunk.RepositoryConfig(storage=settings)
  storage = served.storage()
  if first == "exists":
    assert icechunk.Repository.exists(storage)
    repo = icechunk.Repository.open(storage, config=config)
  else:
    slow = icechunk.StorageSettings(
      timeouts=icechunk.StorageTimeoutSettings(read_timeout_ms=60_000)
    )
    opened = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig(storage=slow))
    repo = opened.reopen(config=config)
  assert "operation attempt timeout" in stalled_read(served, repo, wait=5)
