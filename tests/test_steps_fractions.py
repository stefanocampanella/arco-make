# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Steps for coarsened ocean datasets: fractions and masks, fills along depth, products and
ratios, values kept where a condition holds, quantities per unit area (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu


def fraction_ds() -> xr.Dataset:
  values = np.array([0.0, 1e-8, 2e-6, 0.25, 0.5, 0.75, 1.0], dtype="float32")
  return xr.Dataset({"f": ("x", values, {"units": "1"})})


def test_fraction_mask_cleans_round_off_and_adds_any_water_mask():
  """Below the tolerance a fraction is round-off: it becomes 0; the default mask is `f > 0`."""
  out = pu.fraction_mask(fraction_ds(), fraction="f", mask_name="m")
  expected = np.array([0.0, 0.0, 2e-6, 0.25, 0.5, 0.75, 1.0], dtype="float32")
  np.testing.assert_array_equal(out.f.values, expected)
  assert out.f.dtype == np.float32 and out.f.attrs == {"units": "1"}
  np.testing.assert_array_equal(out.m.values, [False, False, True, True, True, True, True])


@pytest.mark.parametrize(
  ("threshold", "inclusive", "expected"),
  [
    (0.5, False, [False, False, False, False, False, True, True]),
    (0.5, True, [False, False, False, False, True, True, True]),
  ],
)
def test_fraction_mask_threshold_and_inclusive(threshold, inclusive, expected):
  out = pu.fraction_mask(
    fraction_ds(), fraction="f", mask_name="m", threshold=threshold, inclusive=inclusive
  )
  np.testing.assert_array_equal(out.m.values, expected)


def test_fraction_mask_settable_tolerance():
  """`tolerance = 0` keeps every value: the round-off cell is then in the mask."""
  out = pu.fraction_mask(fraction_ds(), fraction="f", mask_name="m", tolerance=0.0)
  assert out.f.values[1] == np.float32(1e-8) and out.m.values[1]


def test_fraction_mask_without_mask_name_only_cleans():
  out = pu.fraction_mask(fraction_ds(), fraction="f")
  assert set(out.data_vars) == {"f"}


@pytest.mark.parametrize(
  "kwargs", [{"threshold": 1.0}, {"threshold": 0.0, "inclusive": True}, {"tolerance": -1.0}]
)
def test_fraction_mask_rejects_meaningless_arguments(kwargs):
  """A threshold of 1 or an inclusive 0 make every cell sea or none; a negative tolerance is
  meaningless."""
  with pytest.raises(ValueError):
    pu.fraction_mask(fraction_ds(), fraction="f", mask_name="m", **kwargs)
