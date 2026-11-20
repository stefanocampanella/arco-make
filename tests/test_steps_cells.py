# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Cell geometry steps: CF bounds and cell areas (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu

RADIUS = 6_378_100.0


def global_grid(resolution: float = 1.0) -> xr.Dataset:
  """A global grid with centres on the poles (half-width polar rows), as the coarse outputs."""
  lat = np.arange(-90.0, 90.0 + resolution / 2, resolution, dtype="float32")
  lon = np.arange(0.0, 360.0, resolution, dtype="float32")
  depth = np.array([0.5, 1.6, 2.7], dtype="float32")
  return xr.Dataset(
    {
      "seafrac": (("depth", "lat", "lon"), np.full((3, lat.size, lon.size), 0.5, "float32")),
      "e3t": ("depth", np.array([1.0, 1.2, 1.0], "float32"), {"units": "m"}),
    },
    coords={
      "depth": ("depth", depth, {"units": "m"}),
      "lat": ("lat", lat, {"units": "degrees_north"}),
      "lon": ("lon", lon, {"units": "degrees_east"}),
    },
  )


def test_cell_bounds_horizontal():
  """Midway between centres, contiguous, latitudes clipped to the poles; the coordinates name
  their bounds and keep their other attributes."""
  ds = global_grid()
  out = pu.cell_bounds(ds, latitude_dim="lat", longitude_dim="lon")
  np.testing.assert_array_equal(
    out.lat_bnds.values[[0, 1, -1]], [[-90, -89.5], [-89.5, -88.5], [89.5, 90]]
  )
  np.testing.assert_array_equal(out.lon_bnds.values[[0, -1]], [[-0.5, 0.5], [358.5, 359.5]])
  for name in ("lat_bnds", "lon_bnds"):
    bounds = out[name].values
    np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])
    assert out[name].dims[1] == "bnds" and out[name].dtype == np.float32
  assert out.lat.attrs == {"units": "degrees_north", "bounds": "lat_bnds"}
  assert "depth_bnds" not in out and "bounds" not in out.depth.attrs
  # Coordinates: steps acting on every data variable (regrid, apply_mask) leave them alone.
  assert {"lat_bnds", "lon_bnds"} <= set(out.coords) and set(out.data_vars) == set(ds.data_vars)
  assert "bounds" not in ds.lat.attrs  # the input is unchanged


def test_cell_bounds_depth_from_thicknesses():
  ds = global_grid()
  out = pu.cell_bounds(
    ds, latitude_dim="lat", longitude_dim="lon", depth_dim="depth", thickness="e3t"
  )
  np.testing.assert_allclose(out.depth_bnds.values, [[0.0, 1.0], [1.0, 2.2], [2.2, 3.2]], rtol=1e-6)
  assert out.depth.attrs == {"units": "m", "bounds": "depth_bnds"}
  assert "bounds" not in ds.depth.attrs


def test_cell_bounds_refuses_levels_outside_their_layers():
  """Thicknesses that do not match the levels (another product's, or levels cut at the top)."""
  ds = global_grid().isel(depth=slice(1, None))
  with pytest.raises(ValueError, match=r"levels \[1.6.*\] lie outside their layers"):
    pu.cell_bounds(ds, latitude_dim="lat", longitude_dim="lon", depth_dim="depth", thickness="e3t")


@pytest.mark.parametrize(
  "arguments",
  [{"depth_dim": "depth"}, {"thickness": "e3t"}, {"depth_dim": "lat", "thickness": "seafrac"}],
)
def test_cell_bounds_refuses_incomplete_depth_arguments(arguments):
  with pytest.raises(ValueError, match="cell_bounds"):
    pu.cell_bounds(global_grid(), latitude_dim="lat", longitude_dim="lon", **arguments)


@pytest.mark.parametrize("resolution", [1.0, 0.25])
def test_cell_area_covers_the_sphere(resolution: float):
  ds = global_grid(resolution)
  out = pu.cell_area(ds, latitude_dim="lat", longitude_dim="lon", radius=RADIUS)
  assert out.areacello.dims == ("lat", "lon") and out.areacello.dtype == np.float64
  assert out.areacello.attrs == {"standard_name": "cell_area", "units": "m2"}
  np.testing.assert_allclose(float(out.areacello.sum()), 4 * np.pi * RADIUS**2, rtol=1e-12)


def test_cell_area_agrees_with_the_bounds():
  """The area of each cell between its bounds: R^2 x longitude width x |sin difference|."""
  ds = pu.cell_bounds(global_grid(), latitude_dim="lat", longitude_dim="lon")
  out = pu.cell_area(ds, variable="area", latitude_dim="lat", longitude_dim="lon", radius=RADIUS)
  lat = np.deg2rad(ds.lat_bnds.values.astype("float64"))
  lon = np.deg2rad(ds.lon_bnds.values.astype("float64"))
  bands, widths = np.abs(np.diff(np.sin(lat), axis=-1)), np.diff(lon, axis=-1)
  expected = RADIUS**2 * bands * widths.T
  np.testing.assert_allclose(out.area.values, expected, rtol=1e-12)


def test_sea_volumes_from_fraction_thickness_and_area():
  """`products` of the three gives each cell's sea volume, broadcast over depth, lat and lon."""
  ds = pu.cell_area(global_grid(), latitude_dim="lat", longitude_dim="lon", radius=RADIUS)
  out = pu.products(ds, volcello=["seafrac", "e3t", "areacello"])
  assert out.volcello.dims == ("depth", "lat", "lon")
  total = float(out.volcello.sum())
  np.testing.assert_allclose(total, 0.5 * 3.2 * 4 * np.pi * RADIUS**2, rtol=1e-6)


def test_cell_bounds_of_integer_coordinates_are_float():
  """Midway edges of integer centres are not truncated."""
  ds = xr.Dataset({"v": (("y", "x"), np.zeros((3, 3)))}, coords={"y": [0, 1, 2], "x": [10, 20, 30]})
  out = pu.cell_bounds(ds, latitude_dim="y", longitude_dim="x")
  np.testing.assert_array_equal(out.y_bnds.values, [[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]])
  np.testing.assert_array_equal(out.x_bnds.values[0], [5.0, 15.0])
  assert out.y_bnds.dtype == np.float64
