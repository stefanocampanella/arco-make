# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Climatology and anomaly variance against a direct NumPy computation."""

import numpy as np
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


@pytest.mark.xfail(reason="B6: bins count every period even where a period is missing", strict=True)
def test_climatology_variance_with_a_missing_year():
  """A value missing in one period does not bias the variance of its bin."""
  ds = series()
  values = ds.x.values.copy()
  values[2 * PERIOD + 1, 0] = np.nan  # bin 1, period 2, point 0
  ds = ds.assign(x=(ds.x.dims, values))
  _, var = _compute_climatology(ds, period=PERIOD, window=1)
  _, ref_var = reference(values, 1, "constant")
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)


@pytest.mark.xfail(
  reason="B6: an all-NaN bin spreads NaN to its neighbours through the window", strict=True
)
def test_climatology_window_does_not_spread_nan():
  """A bin missing in every period leaves the smoothed neighbours finite."""
  ds = series()
  values = ds.x.values.copy()
  values[3::PERIOD, 0] = np.nan  # bin 3 always missing at point 0
  ds = ds.assign(x=(ds.x.dims, values))
  mean, _ = _compute_climatology(ds, period=PERIOD, window=3)
  ref_mean, _ = reference(values, 3, "constant")
  np.testing.assert_allclose(mean.x.values, ref_mean, rtol=1e-12)


def test_window_weights():
  """Constant weights are ones; Gaussian and geometric weights decay symmetrically from the centre."""
  np.testing.assert_array_equal(_window_weights([-1, 0, 1], "constant"), [1, 1, 1])
  gaussian = _window_weights([-2, -1, 0, 1, 2], "gaussian")
  assert gaussian[2] == 1.0 and gaussian[1] == gaussian[3] > gaussian[0] == gaussian[4]
  geometric = _window_weights([-2, -1, 0, 1, 2], "geometric")
  np.testing.assert_allclose(geometric[0], 0.1)
  with pytest.raises(ValueError):
    _window_weights([-1, 0, 1], "geometric", scale=2.0)
