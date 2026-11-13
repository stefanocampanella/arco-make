# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Date intervals: the blocks of a download are right-open and cover the range exactly."""

from datetime import datetime, timedelta

import pytest

from arcomake.datetime_utils import (
  DateInterval,
  IterableDateInterval,
  may_parse_datetime,
  may_parse_timedelta,
)


def test_iterable_date_interval_tiles_the_range():
  """Consecutive blocks share their boundary, and the last block is cut at the end (intent 5)."""
  blocks = list(
    IterableDateInterval(datetime(2020, 1, 1), datetime(2020, 1, 10), timedelta(days=4))
  )
  assert [(b.start.day, b.end.day) for b in blocks] == [(1, 5), (5, 9), (9, 10)]


def test_iterable_date_interval_empty_and_len():
  """An empty range has no blocks; `len` and indexing agree with iteration."""
  assert (
    list(IterableDateInterval(datetime(2020, 1, 1), datetime(2020, 1, 1), timedelta(days=1))) == []
  )
  blocks = IterableDateInterval(datetime(2020, 1, 1), datetime(2020, 1, 3), timedelta(days=1))
  assert len(blocks) == 2
  assert blocks[-1] == DateInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))


def test_date_interval_to_list_excludes_end():
  """Daily stamps of a right-open interval exclude the end."""
  interval = DateInterval(datetime(2020, 2, 28), datetime(2020, 3, 1))
  assert interval.to_list(freq="1D") == [datetime(2020, 2, 28), datetime(2020, 2, 29)]


def test_parse_helpers():
  """Recipe strings like "128D" and ISO dates are accepted; other types are rejected."""
  assert may_parse_timedelta("128D") == timedelta(days=128)
  assert may_parse_timedelta(timedelta(hours=1)) == timedelta(hours=1)
  assert may_parse_datetime("2020-01-01T00:00:00") == datetime(2020, 1, 1)
  with pytest.raises(ValueError):
    may_parse_timedelta(3)  # type: ignore[arg-type]
