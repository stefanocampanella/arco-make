# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Mask steps, filling and bottom values (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu
from tests.conftest import BATHYMETRY_PROFILE, DEPTHS


def test_get_sea_mask_is_bathymetry_below_level_centre(static_ds: xr.Dataset):
  """With threshold 0.5 a cell is sea at a level when the bathymetry is deeper than its centre."""
  out = pu.get_sea_mask(static_ds, bathymetry="deptho", depth_dim="depth", mask_name="mask")
  expected = static_ds.deptho.values[None] > DEPTHS[:, None, None]
  np.testing.assert_array_equal(out["mask"].values, expected)
  assert out["mask"].dims == ("depth", "latitude", "longitude")
  assert not out["mask"].isel(longitude=0).any(), "land (NaN bathymetry) is never sea"


def test_get_sea_mask_threshold_moves_the_cut(static_ds: xr.Dataset):
  """Threshold 1 cuts at the top face of each level, 0 at the bottom face (faces 0, 1, 3, 7 m)."""
  top = pu.get_sea_mask(static_ds, depth_dim="depth", threshold=1.0, mask_name="m")["m"]
  bottom = pu.get_sea_mask(static_ds, depth_dim="depth", threshold=0.0, mask_name="m")["m"]
  deptho = static_ds.deptho.values
  np.testing.assert_array_equal(top.values, deptho[None] > np.array([0, 1, 3])[:, None, None])
  np.testing.assert_array_equal(bottom.values, deptho[None] > np.array([1, 3, 7])[:, None, None])


def test_get_sea_mask_default_dimension_is_the_depth_coordinate_one(static_ds: xr.Dataset):
  """Without `depth_dim`, the mask's vertical dimension is that of the depth coordinate."""
  ds = static_ds.rename(depth="level")
  out = pu.get_sea_mask(ds, depth_coordinate="level", mask_name="m")
  assert out["m"].dims == ("level", "latitude", "longitude")


def test_get_sea_mask_rejects_threshold_outside_unit_interval(static_ds: xr.Dataset):
  """The threshold is a fraction of the cell height."""
  with pytest.raises(ValueError):
    pu.get_sea_mask(static_ds, depth_dim="depth", threshold=1.5)


def test_get_notnull_and_is_positive_masks(static_ds: xr.Dataset):
  """Masks from a variable: defined where not null, or where strictly positive."""
  notnull = pu.get_notnull_mask(static_ds.copy(), variable="deptho", mask_name="m")["m"]
  positive = pu.is_positive_mask(static_ds.copy(), variable="deptho", mask_name="m")["m"]
  np.testing.assert_array_equal(notnull.isel(latitude=0).values, ~np.isnan(BATHYMETRY_PROFILE))
  np.testing.assert_array_equal(positive.isel(latitude=0).values, BATHYMETRY_PROFILE > 0)


@pytest.mark.parametrize(
  ("step", "kwargs"),
  [
    (pu.get_notnull_mask, {"variable": "deptho", "mask_name": "m"}),
    (pu.is_positive_mask, {"variable": "deptho", "mask_name": "m"}),
    (pu.get_sea_mask, {"bathymetry": "deptho", "depth_dim": "depth", "mask_name": "m"}),
  ],
)
def test_mask_steps_do_not_modify_their_input(static_ds: xr.Dataset, step, kwargs):
  """A step returns a new dataset; the caller's dataset does not gain the mask (intent 1)."""
  step(static_ds, **kwargs)
  assert "m" not in static_ds


def test_apply_mask(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """`apply_mask` sets the listed variables to NaN outside the mask, and leaves the others."""
  ds = pu.get_notnull_mask(xr.merge([ocean_ds, static_ds]), variable="deptho", mask_name="m")
  ds = ds.assign(zos=ds.zos.fillna(0.0))
  out = pu.apply_mask(ds, mask_name="m", variables=["zos"])
  assert out.zos.isel(longitude=0).isnull().all()
  assert out.zos.isel(longitude=slice(1, None)).notnull().all()
  xr.testing.assert_identical(out.thetao, ds.thetao)


def test_apply_mask_default_leaves_the_mask_boolean(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """Without `variables` every data variable is masked, except the mask itself."""
  ds = pu.get_notnull_mask(xr.merge([ocean_ds, static_ds]), variable="deptho", mask_name="m")
  out = pu.apply_mask(ds, mask_name="m")
  xr.testing.assert_identical(out.m, ds.m)
  assert out.zos.isel(longitude=0).isnull().all()


def test_masked_fill_constant_inside_mask_only(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """NaN inside the mask are filled; NaN outside the mask stay NaN; valid values are unchanged."""
  ds = pu.get_notnull_mask(xr.merge([ocean_ds, static_ds]), variable="deptho", mask_name="m")
  out = pu.masked_fill(ds, variables=["thetao"], fill_value=-1.0, mask_name="m")
  inside = ds.m.broadcast_like(ds.thetao)
  assert out.thetao.where(~inside).isel(longitude=0).isnull().all()
  filled = ds.thetao.isnull() & inside
  assert (out.thetao.where(filled) == -1.0).sum() == filled.sum()
  xr.testing.assert_equal(out.thetao.where(ds.thetao.notnull()), ds.thetao)


def test_masked_fill_from_variable_and_lower_dimensional_mask(ocean_ds, static_ds):
  """The fill value may name another variable; a 3D mask is reduced to its first level for 2D data."""
  ds = xr.merge([ocean_ds, static_ds])
  ds = pu.get_sea_mask(ds, bathymetry="deptho", depth_dim="depth", mask_name="m")
  ds = pu.get_bottom(ds, depth_dim="depth", variables={"thetao": "thetao_bottom"}, replace=False)
  out = pu.masked_fill(ds, ["thetao"], fill_value={"thetao": "thetao_bottom"}, mask_name="m")
  filled = ds.thetao.isnull() & ds.m
  expected = ds.thetao_bottom.broadcast_like(ds.thetao).where(filled)
  xr.testing.assert_equal(out.thetao.where(filled), expected.transpose(*ds.thetao.dims))
  out2 = pu.masked_fill(ds.assign(zos=ds.zos * np.nan), ["zos"], fill_value=0.0, mask_name="m")
  assert (out2.zos.where(ds.m.isel(depth=0)) == 0.0).sum() == ds.m.isel(
    depth=0
  ).sum() * ds.time.size


def test_masked_fill_requires_existing_mask(ocean_ds: xr.Dataset):
  """A missing mask is an error rather than a silent no-op."""
  with pytest.raises(ValueError, match="does not exist"):
    pu.masked_fill(ocean_ds, ["thetao"], fill_value=0.0, mask_name="absent")


def test_get_bottom_values(ocean_ds: xr.Dataset):
  """The bottom value is the deepest non-NaN value of each column; all-NaN columns stay NaN."""
  out = pu.get_bottom(ocean_ds, depth_dim="depth", variables={"thetao": "tb"}, replace=False)
  values = ocean_ds.thetao.transpose(..., "depth").values
  expected = np.full(values.shape[:-1], np.nan)
  for index in np.ndindex(values.shape[:-1]):
    valid = values[index][~np.isnan(values[index])]
    if valid.size:
      expected[index] = valid[-1]
  np.testing.assert_array_equal(out.tb.transpose("time", "latitude", "longitude").values, expected)
  assert "depth" not in out.tb.dims
  xr.testing.assert_identical(out.thetao, ocean_ds.thetao)


def test_get_bottom_replace(ocean_ds: xr.Dataset):
  """With `replace = true` the variable itself is replaced by its bottom value."""
  out = pu.get_bottom(ocean_ds, depth_dim="depth", variables=["thetao"])
  assert out.thetao.dims == ("time", "latitude", "longitude")


def test_get_bottom_without_replace_needs_a_mapping(ocean_ds: xr.Dataset):
  """Without `replace`, the new names must be given."""
  with pytest.raises(ValueError):
    pu.get_bottom(ocean_ds, depth_dim="depth", variables=["thetao"], replace=False)
