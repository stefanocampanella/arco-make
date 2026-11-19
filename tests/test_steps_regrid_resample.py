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


def _points(static_ds: xr.Dataset) -> xr.Dataset:
  """1 at each sea point of the static grid, NaN on land: points to count."""
  wet = static_ds.deptho.notnull()
  return xr.Dataset({"n": wet.astype("float64").where(wet)})


def _grid(south: float, north: float, west: float, east: float) -> dict[str, float]:
  return {"south": south, "north": north, "west": west, "east": east}


def test_stat_sum_on_a_dataset_beyond_the_source(static_ds: xr.Dataset):
  """`stat` sums the points of each target cell; target cells beyond the source are NaN.

  xarray-regrid 0.4.2 fails here without a `fill_value` (it reads `.dtype` of the Dataset): the
  step passes NaN unless the recipe gives one.
  """
  grid = _grid(-3.5, 3.5, -0.5, 3.5) | {"resolution_lat": 1.0, "resolution_lon": 1.0}
  points = _points(static_ds)
  kwargs = {"method": "sum", "skipna": True}
  out = pu.regrid(points, grid=grid, method="stat", kwargs=kwargs)
  assert float(out.n.sum()) == float(points.n.sum())
  assert out.n.sel(latitude=-3.5).isnull().all() and out.n.sel(longitude=-0.5).isnull().all()
  given = pu.regrid(points, grid=grid, method="stat", kwargs=kwargs | {"fill_value": 0.0})
  assert float(given.n.sel(latitude=-3.5).sum()) == 0.0


def test_target_cells_centred_beyond_the_source_are_empty(static_ds: xr.Dataset):
  """xarray-regrid blanks the target cells whose centre lies outside the source's centres, even
  where they overlap source points, which are then lost, whatever the method. Checks of totals
  rely on this; if xarray-regrid changes, this test tells."""
  grid = _grid(-2.0, 2.0, 0.0, 4.0) | {"resolution_lat": 1.0, "resolution_lon": 1.0}
  points = _points(static_ds)
  out = pu.regrid(points, grid=grid, method="stat", kwargs={"method": "sum", "skipna": True})
  assert out.n.sel(latitude=-2.0).isnull().all(), "centre -2 < -1.75, the first source centre"
  lost = float(points.n.sel(latitude=-1.75).sum())
  assert lost > 0 and float(out.n.sum()) <= float(points.n.sum()) - lost
