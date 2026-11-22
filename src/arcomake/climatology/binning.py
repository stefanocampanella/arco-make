# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Which bin each time step falls in, and the weights of the window over bins: the
definitions both paths share."""

import logging
from typing import NamedTuple

import numpy as np
import xarray as xr

from arcomake.climatology.config import BinningType, WeightingType

logger = logging.getLogger(__name__)


def window_weights(
  offsets: list[int], weighting: WeightingType, scale: float | None = None
) -> np.ndarray:
  """
  Compute the weights of the steps of the (centered) window.

  The weight of a step depends only on its (signed) distance in steps from the
  center of the window, and is therefore constant from one year to the next.

  Parameters
  ----------
  offsets : list[int]
      Signed offsets of the window steps relative to its center.
  weighting : {"constant", "gaussian", "geometric"}
      Weighting scheme. ``"constant"`` gives equal weights (plain average),
      ``"gaussian"`` uses a Gaussian smoothing kernel, and ``"geometric"`` uses
      weights decaying geometrically with the distance from the center.
  scale : float or None, default None
      Scale of the kernel, in step units. For ``"gaussian"`` it is the standard
      deviation of the kernel, and for ``"geometric"`` it is the per-step decay
      ratio (in ``(0, 1]``). If ``None``, a default derived from the window size
      is used: half the window half-width for ``"gaussian"``, and the ratio
      reaching one tenth at the window edge for ``"geometric"``. Ignored for
      ``"constant"``.

  Returns
  -------
  np.ndarray
      Array of weights aligned with ``offsets``. Only the relative magnitude of
      the weights matters, hence they are left unnormalised.
  """
  positions = np.asarray(offsets, dtype=float)
  half = int(np.abs(positions).max()) if positions.size else 0
  if weighting == "constant" or half == 0:
    return np.ones_like(positions)
  if weighting == "gaussian":
    sigma = scale if scale is not None else half / 2.0
    if sigma <= 0:
      raise ValueError(f"Gaussian scale (standard deviation) must be positive, got {sigma}")
    return np.exp(-0.5 * (positions / sigma) ** 2)
  if weighting == "geometric":
    ratio = scale if scale is not None else 0.1 ** (1.0 / half)
    if not 0.0 < ratio <= 1.0:
      raise ValueError(f"Geometric scale (decay ratio) must be in (0, 1], got {ratio}")
    return ratio ** np.abs(positions)
  raise ValueError(f"Unsupported weighting scheme: {weighting}")


# Days before each month in a 365-day year: the day-of-year bins of `bins = "day_of_year"`.
_DAYS_BEFORE_MONTH = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# Feb 29 counts in Feb 28's bin (0-based day 58 of a 365-day year).
FEB_29_BIN = 58


def day_of_year_bins(time: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
  """The 0-based day of a 365-day year of each time stamp, and which stamps are Feb 29.

  Days after Feb 29 in a leap year take the bin of the same date in other years; Feb 29 itself
  takes Feb 28's bin (`FEB_29_BIN`). Works for numpy and cftime dates with standard months.
  """
  month = np.asarray(time.dt.month.values)
  day = np.asarray(time.dt.day.values)
  is_feb_29 = (month == 2) & (day == 29)
  if np.any((day > _DAYS_IN_MONTH[month - 1]) & ~is_feb_29):
    raise ValueError("day_of_year bins need a calendar with standard month lengths")
  bins = _DAYS_BEFORE_MONTH[month - 1] + day - 1
  bins[is_feb_29] = FEB_29_BIN
  return bins, is_feb_29


class TimeBins(NamedTuple):
  """Where the time steps go: their bin, which are Feb 29 (in Feb 28's bin, apart from the
  regular days), and their position on a grid of whole periods (bin k of period p at
  p * period + k)."""

  labels: np.ndarray
  is_feb_29: np.ndarray
  position: np.ndarray


def time_bins(time: xr.DataArray, period: int, bins: BinningType) -> TimeBins:
  """The bins of the time steps `time`, for both paths.

  Raises:
    ValueError: day-of-year bins with a period other than 365, or with two steps on the same day
      (sub-daily data); position bins over a partial period, where leap days and partial years
      would shift every later bin.
  """
  if bins == "day_of_year":
    if period != 365:
      raise ValueError(f"day_of_year bins need period = 365, got {period}")
    labels, is_feb_29 = day_of_year_bins(time)
    years = np.asarray(time.dt.year.values)
    position = (years - years.min()) * period + labels
    regular = ~is_feb_29
    if np.unique(position[regular]).size != np.count_nonzero(regular):
      raise ValueError("day_of_year bins need at most one time step per day")
    return TimeBins(labels, is_feb_29, position)
  size = time.size
  if size % period != 0:
    raise ValueError(
      f"position bins need a whole number of periods ({size} steps, period {period}): leap"
      ' days and partial years shift every later bin; use bins = "day_of_year" for daily data'
    )
  steps = np.arange(size)
  return TimeBins(steps % period, np.zeros(size, dtype=bool), steps)
