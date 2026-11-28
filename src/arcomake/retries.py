# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Transient failures and the retries of a block (6.2, requirement 4).

A block reads all its sources in one computation, so a failure cannot be retried at the source
that caused it: the block is redone. Only transient errors are retried (timeouts, dropped
connections, server and throttling answers); others (a recipe or data error, a missing variable)
would recur, and fail at once. The clients' own retries come first: gcsfs retries a request 6
times, cdsapi up to 500, the Copernicus Marine toolbox 9.
"""

import asyncio
import contextlib
import datetime
import errno
import logging
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Generator, Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arcomake import temporary_utils
from arcomake.datetime_utils import may_parse_timedelta

logger = logging.getLogger(__name__)

# HTTP statuses worth another try: request timeout, throttling, server errors.
TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}
TRANSIENT_ERRNOS = {
  errno.ECONNRESET,
  errno.ECONNABORTED,
  errno.ECONNREFUSED,
  errno.ETIMEDOUT,
  errno.EPIPE,
  errno.ENETUNREACH,
  errno.EHOSTUNREACH,
}
# A compressed chunk that does not decompress: most likely cut short on the way (a read ending
# early without an error), so read again; a chunk corrupt at the source fails once the attempts
# are spent. numcodecs raises it as a bare RuntimeError (zarr 2 and 3 alike).
DECOMPRESSION_ERRORS = ("error during blosc decompression",)
# Icechunk's storage errors (`icechunk.StorageError`) carry only a message, the chain of the Rust
# errors: worth another try are those of the network (the request not sent or not answered:
# "dispatch failure", a connection refused or reset, any timeout; an answer cut short), and S3's
# answers asking to try again. Others (access denied, a missing object, a malformed answer)
# would recur. Icechunk retries a request first (`icechunk_utils.MAX_TRIES`): such an error is a
# network failure that outlasted them, or a connect timeout of its defaults (logbook, 28 Nov).
ICECHUNK_TRANSIENT = (
  "dispatch failure",
  "timed out",
  "timeout",
  "response error",
  "connection closed",
  "SlowDown",
  "ServiceUnavailable",
  "InternalError",
  "RequestTimeout",
  "Throttling",
)


# Exit status of a download whose block outlasted `[retries] block_timeout` (as coreutils'
# `timeout`): the block is not committed, `--resume` writes it again.
BLOCK_TIMEOUT_EXIT = 124


class RetryConfig(BaseModel):
  """The `[retries]` table of a download recipe: how often a block is redone after a transient
  error, waiting `wait`, doubled at each attempt up to `max_wait`; and how long an attempt may
  take, `block_timeout` (none by default), past which the process ends (`deadline`)."""

  model_config = ConfigDict(extra="forbid")

  attempts: int = Field(default=5, ge=1)
  wait: datetime.timedelta = datetime.timedelta(minutes=1)
  max_wait: datetime.timedelta = datetime.timedelta(minutes=30)
  block_timeout: datetime.timedelta | None = None

  @field_validator("block_timeout")
  @classmethod
  def _positive(cls, value: datetime.timedelta | None) -> datetime.timedelta | None:
    if value is not None and value <= datetime.timedelta(0):
      raise ValueError("block_timeout must be positive")
    return value

  @field_validator("wait", "max_wait", "block_timeout", mode="before")
  @classmethod
  def _parse_duration(cls, value: Any) -> Any:
    return may_parse_timedelta(value) if isinstance(value, str) else value

  def waits(self) -> Iterator[datetime.timedelta]:
    """The waits before the second, third... attempt."""
    for retry in range(self.attempts - 1):
      wait = datetime.timedelta(seconds=self.wait.total_seconds() * 2**retry)
      yield min(wait, self.max_wait)


def _causes(error: BaseException) -> Iterator[BaseException]:
  """`error`, the errors it was raised from or during, and those grouped in it."""
  seen: set[int] = set()
  stack = [error]
  while stack:
    current = stack.pop()
    if id(current) in seen:
      continue
    seen.add(id(current))
    yield current
    stack.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    if isinstance(current, BaseExceptionGroup):
      stack.extend(current.exceptions)


def _status(error: BaseException) -> int | None:
  """The HTTP status an error carries, if any (requests, aiohttp, gcsfs, botocore)."""
  response = getattr(error, "response", None)
  for value in (
    getattr(response, "status_code", None),
    getattr(error, "status", None),
    getattr(error, "code", None),
    (response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
    if isinstance(response, dict)
    else None,
  ):
    if isinstance(value, int):
      return value
  return None


def is_transient(error: BaseException) -> bool:
  """Whether `error` (or an error it comes from) is worth another try: a timeout, a dropped or
  refused connection, an HTTP answer of `TRANSIENT_STATUSES`, a chunk that does not decompress
  (`DECOMPRESSION_ERRORS`), or an icechunk storage error of the network (`ICECHUNK_TRANSIENT`)."""
  for cause in _causes(error):
    if isinstance(cause, TimeoutError | asyncio.TimeoutError | socket.timeout | ConnectionError):
      return True
    if type(cause) is RuntimeError and str(cause).startswith(DECOMPRESSION_ERRORS):
      return True
    if isinstance(cause, OSError) and cause.errno in TRANSIENT_ERRNOS:
      return True
    name = f"{type(cause).__module__}.{type(cause).__qualname__}"
    if name.startswith(
      ("aiohttp.client_exceptions.ClientConnect", "aiohttp.client_exceptions.Server")
    ):
      return True
    if name in (
      "aiohttp.client_exceptions.ClientPayloadError",
      "requests.exceptions.ConnectionError",
      "requests.exceptions.Timeout",
      "requests.exceptions.ReadTimeout",
      "requests.exceptions.ConnectTimeout",
      "requests.exceptions.ChunkedEncodingError",
      "botocore.exceptions.EndpointConnectionError",
      "botocore.exceptions.ConnectionClosedError",
      "botocore.exceptions.ReadTimeoutError",
      "botocore.exceptions.ConnectTimeoutError",
    ):
      return True
    if name == "icechunk.StorageError" and any(text in str(cause) for text in ICECHUNK_TRANSIENT):
      return True
    if _status(cause) in TRANSIENT_STATUSES:
      return True
  return False


# How the process ends at a deadline (replaced in tests).
_exit = os._exit


def _stacks() -> str:
  """Every thread's Python stack, threads with the same stack together (most are idle workers)."""
  names = {thread.ident: thread.name for thread in threading.enumerate()}
  stacks: dict[str, list[str]] = {}
  for ident, frame in sys._current_frames().items():
    stack = "".join(traceback.format_stack(frame))
    stacks.setdefault(stack, []).append(names.get(ident, str(ident)))
  return "\n".join(
    f"{len(threads)} thread(s) ({', '.join(threads[:5])}{', ...' if len(threads) > 5 else ''}):"
    f"\n{stack}"
    for stack, threads in sorted(stacks.items(), key=lambda item: -len(item[1]))
  )


@contextlib.contextmanager
def deadline(timeout: datetime.timedelta | None, what: str) -> Generator[None]:
  """End the process if the block is still running after `timeout` (none: no limit): log every
  thread's stack, clean up the temporary directories, exit with `BLOCK_TIMEOUT_EXIT`.

  For a wait nothing else ends: a request without a bound, a lock never released (on 27 Nov 2026
  a production download stood idle 3 h in a block, logbook 28 Nov). Exiting is the only sure
  stop: a thread blocked in a library's native code, or on a lock, cannot be interrupted, and a
  retry in the same process would share whatever holds it. The block is not committed; a run
  stopped this way resumes with `--resume` (a script that reruns it does so on its own)."""
  if timeout is None:
    yield
    return
  lock = threading.Lock()
  done = False

  def expire() -> None:
    with lock:
      if done:
        return
      logger.error(
        f"{what}: not committed after {timeout} ([retries] block_timeout): ending the process"
        f" (exit status {BLOCK_TIMEOUT_EXIT}); --resume writes it again. Every thread's stack:\n"
        f"{_stacks()}"
      )
      temporary_utils.cleanup_all_active_registries()
      # `os._exit` flushes nothing: the log (the root logger's handlers) and what was printed.
      for stream in (*logging.getLogger().handlers, sys.stdout, sys.stderr):
        stream.flush()
      _exit(BLOCK_TIMEOUT_EXIT)

  timer = threading.Timer(timeout.total_seconds(), expire)
  timer.daemon = True
  timer.start()
  try:
    yield
  finally:
    with lock:
      done = True
    timer.cancel()


def with_retries[T](
  action: Callable[[], T],
  config: RetryConfig,
  what: str,
  sleep: Callable[[float], Any] | None = None,
  expected: tuple[type[Exception], ...] = (),
) -> T:
  """Run `action`, again after each transient error (`is_transient`) as `config` says; other
  errors, and the last transient one, are raised. The log tells which is which. `expected`
  errors are outcomes the caller handles (a forecast run not published yet): raised unlogged.
  Each attempt is bounded by `config.block_timeout` (`deadline`)."""
  waits = config.waits()
  attempt = 1
  while True:
    try:
      with deadline(config.block_timeout, what):
        return action()
    except expected:
      raise
    except Exception as error:
      summary = f"{type(error).__name__}: {str(error)[:200]}"
      if not is_transient(error):
        logger.error(f"{what}: {summary} (not transient: no retry)")
        raise
      wait = next(waits, None)
      if wait is None:
        logger.error(f"{what}: {summary} (transient, but {config.attempts} attempts failed)")
        raise
      logger.warning(
        f"{what}: {summary} (transient: attempt {attempt + 1}/{config.attempts} in {wait})"
      )
      (sleep or time.sleep)(wait.total_seconds())
      attempt += 1
