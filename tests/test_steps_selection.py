# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Steps that select, reorder or rename (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu


def test_sel_with_mapping_is_a_slice(ocean_ds: xr.Dataset):
  """A `{start, stop}` table in a recipe selects an inclusive label range, like `slice`."""
  out = pu.sel(ocean_ds, latitude={"start": -0.75, "stop": 0.75})
  np.testing.assert_array_equal(out.latitude.values, [-0.75, -0.25, 0.25, 0.75])


def test_sel_with_list_and_scalar(ocean_ds: xr.Dataset):
  """Lists select labels; scalars select one label and drop the dimension."""
  out = pu.sel(ocean_ds, depth=[0.5, 5.0], longitude=0.25)
  assert out.sizes["depth"] == 2
  assert "longitude" not in out.dims


def test_sel_rejects_unknown_slice_keys(ocean_ds: xr.Dataset):
  """A typo in a slice table (e.g. `end` for `stop`) is an error, not a silent full selection."""
  with pytest.raises(ValueError, match="Invalid selection"):
    pu.sel(ocean_ds, latitude={"start": 0.0, "end": 1.0})


def test_isel_with_mapping(ocean_ds: xr.Dataset):
  """`isel` accepts the same `{start, stop, step}` tables, by position."""
  out = pu.isel(ocean_ds, longitude={"start": 1, "stop": 7, "step": 2})
  np.testing.assert_array_equal(out.longitude.values, ocean_ds.longitude.values[1:7:2])


def test_flip_reverses_a_dimension(ocean_ds: xr.Dataset):
  """`flip` reverses coordinate and data together (used to make ERA5 latitudes increasing)."""
  out = pu.flip(ocean_ds, dim="latitude")
  np.testing.assert_array_equal(out.latitude.values, ocean_ds.latitude.values[::-1])
  xr.testing.assert_identical(out.sortby("latitude"), ocean_ds)


def test_select_variables(ocean_ds: xr.Dataset):
  """Only the listed data variables are kept, with the coordinates of their dimensions."""
  out = pu.select_variables(ocean_ds, ["zos"])
  assert list(out.data_vars) == ["zos"]
  assert set(out.coords) == {"time", "latitude", "longitude"}


def test_transpose(ocean_ds: xr.Dataset):
  """`transpose` takes the dimension order as a list."""
  out = pu.transpose(ocean_ds, ["longitude", "latitude", "depth", "time"])
  assert out["thetao"].dims == ("longitude", "latitude", "depth", "time")


def test_drop_static_vars(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """Variables without the time dimension are dropped."""
  out = pu.drop_static_vars(xr.merge([ocean_ds, static_ds]))
  assert set(out.data_vars) == {"thetao", "zos"}


def test_rename_coordinates_ignores_missing_names(ocean_ds: xr.Dataset):
  """Renaming skips names that are not coordinates, so one step serves several sources."""
  out = pu.rename_coordinates(ocean_ds, {"latitude": "lat", "longitude": "lon", "absent": "x"})
  assert {"lat", "lon"} <= set(out.dims)


def test_rename_coordinates_with_new_index(ocean_ds: xr.Dataset):
  """`set_new_coordinate = {new = "auto"}` replaces the index by 0..n-1 and keeps the old values."""
  out = pu.rename_coordinates(ocean_ds, {"depth": "level"}, set_new_coordinate={"level": "auto"})
  np.testing.assert_array_equal(out.level.values, [0, 1, 2])
  np.testing.assert_array_equal(out.depth.values, ocean_ds.depth.values)
  assert out.depth.dims == ("level",)
