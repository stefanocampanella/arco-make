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


def column_ds(chunked: bool = False) -> xr.Dataset:
  """Two columns of five levels: gaps under valid values, a column starting with a gap."""
  t = np.array([[1.0, np.nan, 3.0, np.nan, np.nan], [np.nan, 2.0, np.nan, np.nan, np.nan]])
  mask = np.array([[True, True, True, True, False], [True, True, True, False, False]])
  ds = xr.Dataset(
    {"t": (("x", "depth"), t, {"units": "degC"}), "m": (("x", "depth"), mask)},
    coords={"depth": [1.0, 2.0, 3.0, 4.0, 5.0]},
  )
  return ds.chunk(depth=2) if chunked else ds


@pytest.mark.parametrize("chunked", [False, True])
def test_fill_from_above_inside_the_mask(chunked: bool):
  """Each gap inside the mask takes the nearest value above; a gap with nothing above stays."""
  out = pu.fill_from_above(column_ds(chunked), variables=["t"], depth_dim="depth", mask_name="m")
  expected = [[1.0, 1.0, 3.0, 3.0, np.nan], [np.nan, 2.0, 2.0, np.nan, np.nan]]
  np.testing.assert_array_equal(out.t.values, expected)
  assert out.t.attrs == {"units": "degC"} and out.t.dims == ("x", "depth")


def test_fill_from_above_without_mask_fills_every_gap_below_a_value():
  out = pu.fill_from_above(column_ds(), variables=["t"], depth_dim="depth")
  expected = [[1.0, 1.0, 3.0, 3.0, 3.0], [np.nan, 2.0, 2.0, 2.0, 2.0]]
  np.testing.assert_array_equal(out.t.values, expected)


def test_fill_from_above_then_zero_for_columns_without_values():
  """The recipe's pair: from above, then `masked_fill` with 0 for what has nothing above."""
  ds = pu.fill_from_above(column_ds(), variables=["t"], depth_dim="depth", mask_name="m")
  out = pu.masked_fill(ds, variables=["t"], fill_value=0.0, mask_name="m")
  assert out.t.values[1, 0] == 0.0 and np.isnan(out.t.values[0, 4])
