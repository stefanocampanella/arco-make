# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Transient errors redo the block, with growing waits; other errors stop at once (6.2,
requirement 4), and the log says which is which."""

import datetime
import importlib
import logging
import pathlib

import numcodecs
import numpy as np
import pytest
import requests
from click.testing import CliRunner
from pydantic import ValidationError

from arcomake import cli, icechunk_utils, retries
from arcomake.retries import RetryConfig, is_transient, with_retries
from tests.conftest import open_output, render_recipe


def http_error(status: int) -> requests.HTTPError:
  response = requests.Response()
  response.status_code = status
  return requests.HTTPError(f"{status}", response=response)


def raised_from(error: BaseException, cause: BaseException) -> BaseException:
  try:
    try:
      raise cause
    except BaseException as inner:
      raise error from inner
  except BaseException as outer:
    return outer


@pytest.mark.parametrize(
  "error",
  [
    TimeoutError("read timed out"),
    ConnectionResetError("reset by peer"),
    requests.ConnectionError("no route"),
    requests.ReadTimeout("slow"),
    http_error(503),
    http_error(429),
    raised_from(ValueError("dask task failed"), TimeoutError("gcs read")),
    ExceptionGroup("tasks", [TimeoutError("one")]),
  ],
  ids=repr,
)
def test_transient_errors(error):
  assert is_transient(error)


@pytest.mark.parametrize(
  "error",
  [ValueError("bad recipe"), KeyError("thetao"), http_error(404), http_error(403), RuntimeError()],
  ids=repr,
)
def test_errors_that_would_recur(error):
  assert not is_transient(error)


def blosc_error() -> RuntimeError:
  """What numcodecs raises for a Blosc chunk cut short (e.g. a read that ended early)."""
  data = numcodecs.Blosc(cname="lz4").encode(np.arange(10_000, dtype="f4"))
  try:
    numcodecs.Blosc().decode(data[: len(data) // 2])
  except RuntimeError as error:
    return error
  raise AssertionError("a truncated chunk decompressed")


def test_a_chunk_that_does_not_decompress_is_transient():
  """A source chunk that does not decompress was most likely cut short on the way: the block is
  read again (seen once reading a source, on a block that passed before and after)."""
  assert is_transient(blosc_error())
  assert is_transient(raised_from(ValueError("dask task failed"), blosc_error()))
  assert not is_transient(RuntimeError("error during something else"))


def test_botocore_errors():
  import botocore.exceptions

  timeout = botocore.exceptions.ReadTimeoutError(endpoint_url="https://s3")
  assert is_transient(timeout)
  slow_down = botocore.exceptions.ClientError(
    {"Error": {"Code": "SlowDown"}, "ResponseMetadata": {"HTTPStatusCode": 503}}, "GetObject"
  )
  denied = botocore.exceptions.ClientError(
    {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject"
  )
  assert is_transient(slow_down) and not is_transient(denied)


# What icechunk raised in a production download (logbook, 28 Nov), and for a stalled endpoint.
ICECHUNK_NETWORK_ERRORS = [
  "object store error dispatch failure: io error: client error (Connect): Connection reset by"
  " peer (os error 104)",
  "object store error dispatch failure: timeout: client error (Connect): HTTP connect timeout"
  " occurred after 3.1s: timed out",
  "object store error dispatch failure: timeout: HTTP read timeout occurred after 300ms: timed out",
  "object store error request has timed out: operation attempt timeout (single attempt) occurred"
  " after 400ms",
  'object store error service error: unhandled error (SlowDown): Error { code: "SlowDown" }',
  # A stalled stream (`network_stream_timeout_seconds`), as aws-smithy words it.
  "object store error minimum throughput was specified at 1 B/s, but throughput of 0 B/s was"
  " observed",
]


@pytest.mark.parametrize("message", ICECHUNK_NETWORK_ERRORS)
def test_icechunk_network_errors_are_transient(message):
  """Icechunk's storage errors carry only their message: a network failure that outlasted its
  own retries redoes the block (it failed the whole run, logbook 28 Nov)."""
  import icechunk

  assert is_transient(icechunk.StorageError(message))
  assert is_transient(raised_from(ValueError("dask task failed"), icechunk.StorageError(message)))


def test_icechunk_errors_that_would_recur():
  import icechunk

  denied = "object store error service error: unhandled error (AccessDenied): Error { code: x }"
  assert not is_transient(icechunk.StorageError(denied))
  assert not is_transient(icechunk.StorageError("storage error: Object should have an etag"))
  assert not is_transient(icechunk.ConflictError("dispatch failure"))  # not a storage error
  # Only the error chain is read: the context icechunk appends holds names and arguments.
  context = f'{denied}\n\ncontext:\n   0: icechunk::store::set\n           with key="timeout"'
  assert not is_transient(icechunk.StorageError(context))
  # An answer that does not parse would recur.
  assert not is_transient(icechunk.StorageError("object store error response error: unparseable"))


def test_icechunk_kinds_of_the_network_are_transient():
  """Icechunk's own throttling error, whatever its message; and I/O errors."""
  import icechunk

  throttled = icechunk.StorageError("storage throttled the request: x", "throttled")
  assert is_transient(throttled)
  assert is_transient(icechunk.StorageError("anything", "io"))
  assert not is_transient(icechunk.StorageError("anything", "object-store"))


def test_an_icechunk_connection_refused_is_transient():
  """As icechunk raises it (one try, to a port nobody listens on)."""
  import socket

  import icechunk

  with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
  storage = icechunk.s3_storage(
    bucket="b",
    prefix="p",
    region="us-east-1",
    endpoint_url=f"http://127.0.0.1:{port}",
    allow_http=True,
    access_key_id="x",
    secret_access_key="y",
  )
  retries = icechunk.StorageRetriesSettings(max_tries=1)
  config = icechunk.RepositoryConfig(storage=icechunk.StorageSettings(retries=retries))
  with pytest.raises(icechunk.StorageError) as raised:
    icechunk.Repository.open(storage, config=config)
  assert is_transient(raised.value)


def test_waits_grow_up_to_the_limit():
  config = RetryConfig(attempts=6, wait="1min", max_wait="5min")  # type: ignore[arg-type]
  assert [w.total_seconds() / 60 for w in config.waits()] == [1, 2, 4, 5, 5]


def failing(errors: list[BaseException]):
  calls = []

  def action():
    calls.append(1)
    if errors:
      raise errors.pop(0)
    return "done"

  return action, calls


def test_transient_errors_are_retried_with_waits(caplog):
  action, calls = failing([TimeoutError("one"), TimeoutError("two")])
  slept = []
  config = RetryConfig(attempts=3, wait="10s")  # type: ignore[arg-type]
  with caplog.at_level(logging.WARNING):
    assert with_retries(action, config, "Block X", sleep=slept.append) == "done"
  assert len(calls) == 3 and slept == [10, 20]
  assert "transient: attempt 2/3" in caplog.text


def test_other_errors_are_not_retried(caplog):
  action, calls = failing([ValueError("missing variable")])
  with pytest.raises(ValueError), caplog.at_level(logging.ERROR):
    with_retries(action, RetryConfig(), "Block X", sleep=lambda s: pytest.fail("slept"))
  assert len(calls) == 1 and "not transient: no retry" in caplog.text


def test_retries_end(caplog):
  action, calls = failing([TimeoutError(str(i)) for i in range(5)])
  with pytest.raises(TimeoutError), caplog.at_level(logging.ERROR):
    with_retries(action, RetryConfig(attempts=3), "Block X", sleep=lambda s: None)
  assert len(calls) == 3 and "3 attempts failed" in caplog.text


def test_retries_in_the_recipe():
  config = RetryConfig.model_validate({"attempts": 2, "wait": "30s", "max_wait": "2min"})
  assert config.wait == datetime.timedelta(seconds=30)
  with pytest.raises(ValidationError):
    RetryConfig.model_validate({"attempts": 0})
  with pytest.raises(ValidationError):
    RetryConfig.model_validate({"tries": 3})


@pytest.mark.parametrize(
  "error",
  [
    TimeoutError("simulated GCS read timeout"),
    pytest.param(ICECHUNK_NETWORK_ERRORS[0], id="icechunk-reset"),
  ],
)
def test_a_block_failing_once_is_redone(
  tmp_path: pathlib.Path, sources: pathlib.Path, monkeypatch: pytest.MonkeyPatch, error
):
  """A download whose second block fails once on the network retries it and completes."""
  import icechunk

  if isinstance(error, str):
    error = icechunk.StorageError(error)
  download = importlib.import_module("arcomake.download")  # the module, not the command
  real = download.maybe_checkpointing_download_and_process
  failures = []

  def flaky(configs, start, end, **kwargs):
    if start == datetime.datetime(2020, 2, 28) and not failures:
      failures.append(start)
      raise error
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download, "maybe_checkpointing_download_and_process", flaky)
  monkeypatch.setattr(retries.time, "sleep", lambda seconds: None)
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  output = tmp_path / "out.zarr"
  result = CliRunner().invoke(
    cli, ["download", "--dask-setup", "threads", str(recipe), str(output)]
  )
  assert result.exit_code == 0, result.output
  # The command's own log handler writes to its output.
  assert failures and f"{type(error).__name__}: {str(error)[:20]}" in result.output
  assert "(transient: attempt 2/" in result.output
  assert open_output(output).sizes["time"] == 5


def test_a_commit_that_landed_is_not_redone(
  tmp_path: pathlib.Path, sources: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """The second block's commit lands, then its answer times out: the retry must not append the
  block again (duplicate times)."""
  real_write = icechunk_utils.Run.write
  lost = []

  def write_then_time_out(self, dataset, **kwargs):
    snapshot = real_write(self, dataset, **kwargs)
    if kwargs["mode"] == "a-" and not lost:
      lost.append(snapshot)
      raise TimeoutError("answer lost after the commit")
    return snapshot

  monkeypatch.setattr(icechunk_utils.Run, "write", write_then_time_out)
  monkeypatch.setattr(retries.time, "sleep", lambda seconds: None)
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  output = tmp_path / "out.zarr"
  args = ["download", "--dask-setup", "threads"]
  result = CliRunner().invoke(cli, [*args, str(recipe), str(output)])
  assert result.exit_code == 0, result.output
  assert lost and "already committed by the previous attempt" in result.output
  times = open_output(output).time.values
  assert len(times) == 5 and len(set(times)) == 5


def test_expected_errors_are_raised_unlogged(caplog: pytest.LogCaptureFixture):
  """An error the caller handles as an outcome (a forecast run not published yet) is raised at
  once, without the error line other errors get."""

  class Outcome(Exception):
    pass

  def action():
    raise Outcome("not yet")

  with caplog.at_level(logging.INFO), pytest.raises(Outcome):
    with_retries(action, RetryConfig(), "block", sleep=lambda s: None, expected=(Outcome,))
  assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_block_timeout_in_the_recipe():
  """Absent by default; a duration; positive. Left out of the fingerprint with `[retries]`: a run
  resumes with one added."""
  assert RetryConfig().block_timeout is None
  config = RetryConfig.model_validate({"block_timeout": "1h"})
  assert config.block_timeout == datetime.timedelta(hours=1)
  with pytest.raises(ValidationError):
    RetryConfig.model_validate({"block_timeout": "0s"})


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
  """The exit statuses a deadline ended the process with (it keeps running in tests), and the
  temporary directories cleaned up first (0)."""
  statuses: list[int] = []
  monkeypatch.setattr(retries, "_exit", statuses.append)
  monkeypatch.setattr(
    retries.temporary_utils, "cleanup_all_active_registries", lambda: statuses.append(0)
  )
  return statuses


def test_a_block_past_its_deadline_ends_the_process(exits, caplog, capfd):
  """Every thread's stack on standard error (where it waits), the reason in the log, the
  temporary directories cleaned up, the exit status of a timeout."""
  import threading

  blocked = threading.Event()
  with retries.deadline(datetime.timedelta(seconds=0.2), "Block [2020-01-01, 2020-02-01)"):
    blocked.wait(2)  # stands for a wait nothing else ends
  assert exits == [0, retries.BLOCK_TIMEOUT_EXIT]
  assert "Block [2020-01-01, 2020-02-01): still running after 0:00:00.200000" in caplog.text
  stderr = capfd.readouterr().err
  assert "thread(s)" in stderr and "blocked.wait(2)" in stderr


def test_the_process_ends_whatever_fails_or_hangs_before(exits, monkeypatch):
  """A broken standard stream (the detached run's) or a cleanup stuck on a lock the hung thread
  holds must not keep the process alive: the exit comes after `TIDY_SECONDS` at most."""
  import io
  import threading

  broken = io.StringIO()
  broken.close()  # flushing it raises
  monkeypatch.setattr(retries.sys, "stdout", broken)
  monkeypatch.setattr(retries, "_stacks", lambda: 1 / 0)  # the grouped stacks fail too
  stuck = threading.Event()
  monkeypatch.setattr(retries.temporary_utils, "cleanup_all_active_registries", stuck.wait)
  monkeypatch.setattr(retries, "TIDY_SECONDS", 0.5)
  with retries.deadline(datetime.timedelta(seconds=0.1), "a block"):
    threading.Event().wait(1.5)
  stuck.set()
  assert exits == [retries.BLOCK_TIMEOUT_EXIT]


def test_the_child_processes_end_with_the_process(exits):
  """Workers of a process-based Dask setup would outlive `os._exit`, holding their memory."""
  import subprocess
  import sys as sys_module
  import time as time_module

  child = subprocess.Popen([sys_module.executable, "-c", "import time; time.sleep(60)"])
  try:
    with retries.deadline(datetime.timedelta(seconds=0.1), "a block"):
      time_module.sleep(1)
    assert child.wait(timeout=10) != 0  # killed
  finally:
    child.kill()


def test_a_block_within_its_deadline_or_without_one_goes_on(exits):
  import time as time_module

  with retries.deadline(datetime.timedelta(seconds=0.2), "a block"):
    pass
  with retries.deadline(None, "a block"):
    time_module.sleep(0.3)
  time_module.sleep(0.3)  # the first timer would have fired by now
  assert exits == []


def test_a_download_whose_block_outlasts_the_deadline_ends(
  tmp_path: pathlib.Path, sources: pathlib.Path, monkeypatch: pytest.MonkeyPatch, exits
):
  """Each attempt of a block has `[retries] block_timeout`: the second block, too slow, ends the
  process (here it goes on, `_exit` replaced, and completes)."""
  import time as time_module

  download = importlib.import_module("arcomake.download")  # the module, not the command
  real = download.maybe_checkpointing_download_and_process

  def slow(configs, start, end, **kwargs):
    if start == datetime.datetime(2020, 2, 28):
      time_module.sleep(8)
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download, "maybe_checkpointing_download_and_process", slow)
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  recipe.write_text(recipe.read_text() + '\n[retries]\nblock_timeout = "5s"\n')
  result = CliRunner().invoke(
    cli, ["download", "--dask-setup", "threads", str(recipe), str(tmp_path / "out")]
  )
  assert result.exit_code == 0, result.output
  # The slow block (others take well under the timeout, but a loaded machine may stretch them).
  assert exits.count(retries.BLOCK_TIMEOUT_EXIT) >= 1
  expired = [line for line in result.output.splitlines() if "still running after 0:00:05" in line]
  assert any("2020-02-28" in line for line in expired), result.output
