# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Transient errors redo the block, with growing waits; other errors stop at once (6.2,
requirement 4), and the log says which is which."""

import datetime
import importlib
import logging
import pathlib

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


def test_a_block_failing_once_is_redone(
  tmp_path: pathlib.Path, sources: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """A download whose second block times out once retries it and completes."""
  download = importlib.import_module("arcomake.download")  # the module, not the command
  real = download.maybe_checkpointing_download_and_process
  failures = []

  def flaky(configs, start, end, **kwargs):
    if start == datetime.datetime(2020, 2, 28) and not failures:
      failures.append(start)
      raise TimeoutError("simulated GCS read timeout")
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
  assert failures and "TimeoutError: simulated GCS read timeout (transient" in result.output
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
