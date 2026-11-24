# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, SupportsIndex, override

import numpy as np
import pandas as pd

InclusiveOptions = Literal["left", "right", "neither", "both"]

logger = logging.getLogger(__name__)


def naive_utc(value: datetime | None, what: str) -> datetime | None:
  """`value` without its time zone, if it has one and it is UTC.

  The sources' stamps have no zone and are read as UTC: a UTC time is that stamp, another offset
  would be converted to a different one than written.

  Raises:
    ValueError: for another offset, starting with `what`.
  """
  if value is None or value.tzinfo is None:
    return value
  if value.utcoffset() != timedelta(0):
    raise ValueError(f"{what}: without a time zone, or in UTC")
  return value.replace(tzinfo=None)


@dataclass
class DateInterval:
  """Represents a right open date interval."""

  start: datetime
  end: datetime

  @override
  def __repr__(self):
    return f"[{self.start.isoformat()}, {self.end.isoformat()})"

  def to_slice(self) -> slice:
    return slice(self.start, self.end)

  def to_datetime_index(
    self, freq: str | timedelta, inclusive: InclusiveOptions = "left"
  ) -> pd.DatetimeIndex:
    """Convert the DateInterval to a Pandas DatetimeIndex spaced using the given frequency.

    Args:
      freq (str | timedelta): Frequency to use for spacing the datetime64[ns] values.
      inclusive (InclusiveOptions): Whether the start and end dates are included.

    Returns:
      pd.DatetimeIndex: DatetimeIndex from start to end (inclusive) with the given step.
    """
    # Create a Pandas date range from start to end (inclusive) with the given step
    return pd.date_range(start=self.start, end=self.end, freq=freq, inclusive=inclusive)

  def to_numpy(self, freq: str | timedelta, inclusive: InclusiveOptions = "left") -> np.ndarray:
    """Convert the DateInterval to a numpy array of datetime64[ns] spaced using the given frequency.

    Args:
      freq (str | timedelta): Frequency to use for spacing the datetime64[ns] values.
      inclusive (InclusiveOptions): Whether the start and end dates are included.

    Returns:
      np.ndarray: Array of datetime64[ns] values from start to end (inclusive) with the given step.
    """

    # Convert to numpy array with datetime64[ns] dtype
    return self.to_datetime_index(freq, inclusive=inclusive).to_numpy(dtype="datetime64[ns]")

  def to_list(self, freq: str | timedelta, inclusive: InclusiveOptions = "left") -> list[datetime]:
    """Convert to a list of datetime objects spaced using the given frequency.

    Args:
      freq (str | timedelta): Frequency to use for spacing the datetime64[ns] values.
      inclusive (InclusiveOptions): Whether the start and end dates are included.

    Returns:
      list[datetime]: List of datetime objects from start to end (inclusive) with the given step.
    """
    return self.to_datetime_index(freq, inclusive=inclusive).to_pydatetime().tolist()


@dataclass
class IterableDateInterval(Iterable[DateInterval]):
  """A right-open interval [start, end), iterated as consecutive blocks of length `step`.

  Blocks share their boundaries and the last one is cut at `end`, so the blocks cover the
  interval exactly once.

  Attributes:
    start (datetime.datetime): The start date of the interval.
    end (datetime.datetime): The (excluded) end date of the interval.
    step (datetime.timedelta): Length of each block (no default).
  """

  start: datetime
  end: datetime
  step: timedelta

  @override
  def __iter__(self) -> Iterator[DateInterval]:

    def _date_iterator() -> Iterator[DateInterval]:
      left = self.start
      while left < self.end:
        right = min(left + self.step, self.end)
        yield DateInterval(left, right)
        left = right

    return _date_iterator()

  def __getitem__(self, item: SupportsIndex) -> DateInterval:
    return [interval for interval in self][item]

  def __len__(self) -> int:
    return sum(1 for _ in self)

  @override
  def __repr__(self) -> str:
    intervals = list(self)
    intervals_str = ",".join([str(interval) for interval in intervals[:3]])  # Show first 3
    if len(self) > 3:
      intervals_str += f", ..., {intervals[-1]}"
    return intervals_str


def may_parse_datetime(date: datetime | str) -> datetime:
  if isinstance(date, datetime):
    return date
  elif isinstance(date, str):
    return datetime.fromisoformat(date)
  else:
    raise ValueError(f"Invalid date format: {date}. Must be a datetime or a string in ISO format.")


def may_parse_timedelta(delta: timedelta | str) -> timedelta:
  if isinstance(delta, timedelta):
    return delta
  elif isinstance(delta, str):
    parsed_timedelta = pd.Timedelta(delta)
    if isinstance(parsed_timedelta, pd.Timedelta):
      return parsed_timedelta.to_pytimedelta()
    else:
      raise ValueError(f"Invalid timedelta format: {delta}. Must be a valid timedelta string.")
  else:
    raise ValueError(
      f"Invalid timedelta format: {delta}. Must be a timedelta or a string in ISO format."
    )
