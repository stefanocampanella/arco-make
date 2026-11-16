# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Climatology and anomaly variance against a direct NumPy computation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arcomake.climatology import _compute_climatology, _window_weights

PERIOD = 6
N_PERIODS = 5


def series(seed: int = 0) -> xr.Dataset:
  """N_PERIODS periods of PERIOD steps at 3 points, with a seasonal cycle and noise."""
  rng = np.random.default_rng(seed)
  steps = np.arange(PERIOD * N_PERIODS)
  cycle = np.sin(2 * np.pi * steps / PERIOD)[:, None]
  values = cycle + rng.standard_normal((steps.size, 3))
  return xr.Dataset({"x": (("time", "point"), values)}, coords={"time": steps})


def reference(values: np.ndarray, window: int, weighting: str) -> tuple[np.ndarray, np.ndarray]:
  """Weighted mean and unbiased (reliability-weights) variance of each bin, ignoring NaN.

  Bin b pools the samples of bins b + k (k in the centred window, wrapping around the period)
  from every period, each with the weight of its offset k.
  """
  folded = values.reshape(N_PERIODS, PERIOD, -1)
  half = window // 2
  offsets = list(range(-half, window - half))
  weights = _window_weights(offsets, weighting)  # type: ignore[arg-type]
  mean = np.empty(folded.shape[1:])
  var = np.empty(folded.shape[1:])
  for b in range(PERIOD):
    for p in range(folded.shape[2]):
      x = np.concatenate([folded[:, (b + k) % PERIOD, p] for k in offsets])
      w = np.repeat(weights, N_PERIODS)
      valid = ~np.isnan(x)
      x, w = x[valid], w[valid]
      v1, v2 = w.sum(), (w**2).sum()
      mean[b, p] = (w * x).sum() / v1
      var[b, p] = (w * (x - mean[b, p]) ** 2).sum() / (v1 - v2 / v1)
  return mean, var


@pytest.mark.parametrize(
  ("window", "weighting"), [(1, "constant"), (3, "constant"), (3, "gaussian"), (5, "geometric")]
)
def test_climatology_matches_numpy(window: int, weighting: str):
  """On complete data the climatology equals the pooled weighted statistics of each bin."""
  ds = series()
  mean, var = _compute_climatology(ds, period=PERIOD, window=window, weighting=weighting)  # type: ignore[arg-type]
  ref_mean, ref_var = reference(ds.x.values, window, weighting)
  np.testing.assert_allclose(mean.x.values, ref_mean, rtol=1e-12)
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)
  np.testing.assert_array_equal(mean.time.values, np.arange(PERIOD))


def test_climatology_variance_with_a_missing_year():
  """A value missing in one period does not bias the variance of its bin."""
  ds = series()
  values = ds.x.values.copy()
  values[2 * PERIOD + 1, 0] = np.nan  # bin 1, period 2, point 0
  ds = ds.assign(x=(ds.x.dims, values))
  _, var = _compute_climatology(ds, period=PERIOD, window=1)
  _, ref_var = reference(values, 1, "constant")
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)


def test_climatology_window_does_not_spread_nan():
  """A bin missing in every period leaves the smoothed neighbours finite."""
  ds = series()
  values = ds.x.values.copy()
  values[3::PERIOD, 0] = np.nan  # bin 3 always missing at point 0
  ds = ds.assign(x=(ds.x.dims, values))
  mean, _ = _compute_climatology(ds, period=PERIOD, window=3)
  ref_mean, _ = reference(values, 3, "constant")
  np.testing.assert_allclose(mean.x.values, ref_mean, rtol=1e-12)


def test_climatology_with_an_incomplete_last_period():
  """A partial last period counts in the bins it covers only (no integer number of periods)."""
  ds = series()
  values = ds.x.values.copy()
  values[-2:] = np.nan  # as if the last two steps were missing
  _, var = _compute_climatology(ds.isel(time=slice(0, -2)), period=PERIOD, window=3)
  _, ref_var = reference(values, 3, "constant")
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)


def daily(start: str, end: str, seed: int = 0) -> xr.Dataset:
  rng = np.random.default_rng(seed)
  time = pd.date_range(start, end, freq="1D", inclusive="left")
  return xr.Dataset({"x": ("time", rng.standard_normal(time.size))}, coords={"time": time})


def test_day_of_year_bins_put_feb_29_with_feb_28():
  """Feb 29 counts in Feb 28's bin; later days of a leap year take the same date's bin."""
  ds = daily("2019-01-01", "2021-01-01")  # 2020 is a leap year
  mean, var = _compute_climatology(ds, period=365, bins="day_of_year")
  x = ds.x.to_series()
  feb_28 = x[(x.index.month == 2) & (x.index.day.isin([28, 29]))]
  np.testing.assert_allclose(mean.x.values[58], feb_28.mean(), rtol=1e-12)
  np.testing.assert_allclose(var.x.values[58], feb_28.var(ddof=1), rtol=1e-12)
  mar_1 = x[(x.index.month == 3) & (x.index.day == 1)]
  np.testing.assert_allclose(mean.x.values[59], mar_1.mean(), rtol=1e-12)
  dec_31 = x[(x.index.month == 12) & (x.index.day == 31)]
  np.testing.assert_allclose(mean.x.values[364], dec_31.mean(), rtol=1e-12)
  assert mean.sizes["time"] == 365


def test_day_of_year_bins_match_position_bins_without_leap_days():
  """Without Feb 29 and with whole years, day-of-year bins are position bins."""
  ds = daily("2021-01-01", "2024-01-01")
  by_day, var_day = _compute_climatology(ds, period=365, window=5, bins="day_of_year")
  by_position, var_position = _compute_climatology(ds, period=365, window=5)
  np.testing.assert_allclose(by_day.x.values, by_position.x.values, rtol=1e-12)
  np.testing.assert_allclose(var_day.x.values, var_position.x.values, rtol=1e-12)


def test_climatology_keeps_the_input_dtype():
  """float32 inputs give float32 statistics (counts are integers internally)."""
  ds = series().astype("float32")
  mean, var = _compute_climatology(ds, period=PERIOD, window=3)
  assert mean.x.dtype == var.x.dtype == np.float32


def test_flox_gives_the_same_statistics():
  """The flox implementation agrees with the folded one, Feb 29 included."""
  pytest.importorskip("flox")
  ds = daily("2019-03-01", "2021-07-01").chunk(time=100)
  values = ds.x.values.copy()
  values[10:40] = np.nan
  ds = ds.assign(x=("time", values))
  mean, var = _compute_climatology(ds, period=365, window=7, bins="day_of_year")
  mean_f, var_f = _compute_climatology(ds, period=365, window=7, bins="day_of_year", use_flox=True)
  np.testing.assert_allclose(mean_f.x.values, mean.x.values, rtol=1e-10)
  np.testing.assert_allclose(var_f.x.values, var.x.values, rtol=1e-10)


def test_window_weights():
  """Constant weights are ones; Gaussian and geometric weights decay symmetrically from the centre."""
  np.testing.assert_array_equal(_window_weights([-1, 0, 1], "constant"), [1, 1, 1])
  gaussian = _window_weights([-2, -1, 0, 1, 2], "gaussian")
  assert gaussian[2] == 1.0 and gaussian[1] == gaussian[3] > gaussian[0] == gaussian[4]
  geometric = _window_weights([-2, -1, 0, 1, 2], "geometric")
  np.testing.assert_allclose(geometric[0], 0.1)
  with pytest.raises(ValueError):
    _window_weights([-1, 0, 1], "geometric", scale=2.0)
