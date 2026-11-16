# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Regridding and resampling steps (processing_utils)."""

import numpy as np
import xarray as xr

from arcomake import processing_utils as pu

# A 1-degree grid whose cells contain exactly 2 x 2 source cells.
GRID = {
  "south": -1.5,
  "north": 1.5,
  "west": 0.5,
  "east": 3.5,
  "resolution_lat": 1.0,
  "resolution_lon": 1.0,
}


def _block_mean(values: np.ndarray) -> np.ndarray:
  """Mean over non-overlapping 2 x 2 blocks of the last two axes."""
  *lead, ny, nx = values.shape
  return values.reshape(*lead, ny // 2, 2, nx // 2, 2).mean(axis=(-3, -1))


def test_regrid_target_coordinates_are_float32(atmosphere_ds: xr.Dataset):
  """The target grid has float32 coordinates (as published), built from the recipe's grid table."""
  out = pu.regrid(atmosphere_ds, grid=GRID, method="conservative")
  np.testing.assert_array_equal(out.latitude.values, [-1.5, -0.5, 0.5, 1.5])
  np.testing.assert_array_equal(out.longitude.values, [0.5, 1.5, 2.5, 3.5])
  assert out.latitude.dtype == np.float32 and out.longitude.dtype == np.float32


def test_conservative_regrid_of_complete_field_is_block_mean(atmosphere_ds: xr.Dataset):
  """On aligned grids, conservative regridding of a field without NaN is the mean of each block."""
  out = pu.regrid(atmosphere_ds.isel(time=slice(0, 3)), grid=GRID, method="conservative")
  expected = _block_mean(atmosphere_ds.t2m.isel(time=slice(0, 3)).values)
  np.testing.assert_allclose(out.t2m.values, expected, rtol=1e-6)


def test_conservative_regrid_nan_rules(static_ds: xr.Dataset):
  """`skipna = false` counts land as 0 (water volume conserved), `skipna = true` averages sea only.

  This pins the convention the arco-ocean recipes rely on for the bathymetry (findings C7):
  a coastal cell with one land column of two is half as deep with `skipna = false`.
  """
  kwargs_false = {"skipna": False}
  kwargs_true = {"skipna": True, "nan_threshold": 1.0}
  no_skip = pu.regrid(static_ds, grid=GRID, method="conservative", kwargs=kwargs_false)
  skip = pu.regrid(static_ds, grid=GRID, method="conservative", kwargs=kwargs_true)
  # First target column: source columns 0 (land, NaN) and 1 (1.5 m).
  np.testing.assert_allclose(no_skip.deptho.isel(longitude=0).values, 0.75)
  np.testing.assert_allclose(skip.deptho.isel(longitude=0).values, 1.5)
  # Fully wet target columns agree.
  np.testing.assert_allclose(no_skip.deptho.isel(longitude=1).values, 3.0)
  np.testing.assert_allclose(skip.deptho.isel(longitude=1).values, 3.0)


def test_regrid_does_not_modify_its_arguments(atmosphere_ds: xr.Dataset):
  """The arguments of a step are not changed by the step (intent 1 extended to arguments)."""
  kwargs = {"skipna": True}
  pu.regrid(atmosphere_ds.isel(time=0), grid=GRID, method="conservative", kwargs=kwargs)
  assert kwargs == {"skipna": True}


def test_resample_daily_mean_and_sum(atmosphere_ds: xr.Dataset):
  """Hourly to daily with left-closed, left-labelled bins: day D is the hours 00..23 of D."""
  mean = pu.resample(atmosphere_ds, reduce="mean", time="1D", label="left", closed="left")
  total = pu.resample(atmosphere_ds, reduce="sum", time="1D", label="left", closed="left")
  assert mean.sizes["time"] == 6
  np.testing.assert_allclose(
    mean.t2m.isel(time=0).values, atmosphere_ds.t2m.isel(time=slice(0, 24)).mean("time").values
  )
  np.testing.assert_allclose(
    total.tp.isel(time=1).values, atmosphere_ds.tp.isel(time=slice(24, 48)).sum("time").values
  )
  assert str(mean.time.values[0])[:10] == "2020-02-26"
