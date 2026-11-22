# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The numba kernels: the window over bins (both paths) and the Welford update
(streaming)."""

import functools
import logging
import threading

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def window_bins_last(
  count: np.ndarray,
  mean: np.ndarray,
  m2: np.ndarray,
  offsets: list[int],
  weights: np.ndarray,
  dtype: npt.DTypeLike = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
  """The window step on per-bin statistics with the bins on the last axis, by the numba kernel
  (`_window_kernel`): climatology and variance in `dtype`, computed in float64."""
  shape = mean.shape
  bins = shape[-1]
  climatology = np.empty((mean.size // max(bins, 1), bins), dtype=dtype)
  variance = np.empty_like(climatology)
  _window_kernel()(
    *(np.ascontiguousarray(a.reshape(-1, bins)) for a in (count, mean, m2)),
    np.asarray(offsets, dtype=np.int64),
    np.asarray(weights, dtype=np.float64),
    climatology,
    variance,
  )
  return climatology.reshape(shape), variance.reshape(shape)


_WINDOW_KERNEL_LOCK = threading.Lock()


def _window_kernel():
  """The numba window kernel (one for all threads: numba compiles it once, at its first call).

  Dask's threads ask for it at once on their first chunks: `functools.cache` alone let each of
  them make (and compile) its own."""
  with _WINDOW_KERNEL_LOCK:
    return _new_window_kernel()


@functools.cache
def _new_window_kernel():
  """The numba window kernel, made on first use (importing numba is slow)."""
  import numba

  @numba.njit(nogil=True)
  def window_rows(count, mean, m2, offsets, weights, climatology, variance):  # pragma: no cover
    """Row by row (a point's bins): bin i pools bins i + offsets[j] (wrapping around), weighted
    by weights[j], skipping empty bins; reliability-weights variance. In float64, in the order
    of the former NumPy implementation (same bits)."""
    points, bins = mean.shape
    for p in range(points):
      for i in range(bins):
        v1 = 0.0
        v2 = 0.0
        total = 0.0
        for j in range(offsets.shape[0]):
          b = (i + offsets[j]) % bins
          n = np.float64(count[p, b])
          if n > 0:
            w = weights[j]
            v1 += w * n
            v2 += w * w * n
            total += w * (n * np.float64(mean[p, b]))
        if v1 > 0:
          c = total / v1
          total = 0.0
          for j in range(offsets.shape[0]):
            b = (i + offsets[j]) % bins
            n = np.float64(count[p, b])
            if n > 0:
              d = np.float64(mean[p, b]) - c
              total += weights[j] * (d * d * n + np.float64(m2[p, b]))
          denominator = v1 - v2 / v1
          climatology[p, i] = c
          variance[p, i] = total / denominator if denominator > 0 else np.nan
        else:
          climatology[p, i] = np.nan
          variance[p, i] = np.nan

  return window_rows


# Series computed at once take turns in the Welford kernel: it is parallel already, and numba's
# workqueue threading layer (used where neither TBB nor OpenMP is installed) aborts when parallel
# kernels are launched from several threads at once. Updating a block takes a small part of the
# time reading it.
WELFORD_LOCK = threading.Lock()


@functools.cache
def welford_kernel():
  """The numba Welford kernel, compiled on first use (importing numba is slow)."""
  import numba

  @numba.njit(parallel=True, nogil=True)
  def welford_block(count, mean, m2, values, labels):  # pragma: no cover (compiled)
    """Add row j of `values` (steps × points) to bin `labels[j]`, skipping NaN (float64).

    One fused pass per step, parallel over points; `count`, `mean`, `m2` are (bins × points).
    """
    for j in range(values.shape[0]):
      b = labels[j]
      for i in numba.prange(values.shape[1]):
        x = values[j, i]
        if np.isfinite(x):
          n = count[b, i] + 1
          count[b, i] = n
          delta = x - mean[b, i]
          mean[b, i] += delta / n
          m2[b, i] += delta * (x - mean[b, i])

  return welford_block
