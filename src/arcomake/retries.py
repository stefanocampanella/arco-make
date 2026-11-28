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
import datetime
import errno
import logging
import socket
import time
from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class RetryConfig(BaseModel):
  """The `[retries]` table of a download recipe: how often a block is redone after a transient
  error, waiting `wait`, doubled at each attempt up to `max_wait`."""

  model_config = ConfigDict(extra="forbid")

  attempts: int = Field(default=5, ge=1)
  wait: datetime.timedelta = datetime.timedelta(minutes=1)
  max_wait: datetime.timedelta = datetime.timedelta(minutes=30)

  @field_validator("wait", "max_wait", mode="before")
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


def with_retries[T](
  action: Callable[[], T],
  config: RetryConfig,
  what: str,
  sleep: Callable[[float], Any] | None = None,
  expected: tuple[type[Exception], ...] = (),
) -> T:
  """Run `action`, again after each transient error (`is_transient`) as `config` says; other
  errors, and the last transient one, are raised. The log tells which is which. `expected`
  errors are outcomes the caller handles (a forecast run not published yet): raised unlogged."""
  waits = config.waits()
  attempt = 1
  while True:
    try:
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
