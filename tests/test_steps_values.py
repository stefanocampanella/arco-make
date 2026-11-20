# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Steps that change values, types or attributes (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu


def test_astype_whole_dataset(ocean_ds: xr.Dataset):
  """`dtype` with `casting` converts every data variable (the form used by the recipes)."""
  out = pu.astype(ocean_ds, dtype="float32", casting="same_kind")
  assert all(da.dtype == np.float32 for da in out.data_vars.values())


def test_astype_dtype_alone(ocean_ds: xr.Dataset):
  """`dtype` alone converts too; it must not be a silent no-op."""
  out = pu.astype(ocean_ds, dtype="float32")
  assert out.thetao.dtype == np.float32


def test_astype_per_variable(ocean_ds: xr.Dataset):
  """A table per variable converts that variable only."""
  out = pu.astype(ocean_ds, zos={"dtype": "float32", "casting": "same_kind"})
  assert out.zos.dtype == np.float32


def test_astype_per_variable_keeps_other_variables(ocean_ds: xr.Dataset):
  """Variables not listed are kept unchanged."""
  out = pu.astype(ocean_ds, zos={"dtype": "float32", "casting": "same_kind"})
  assert set(out.data_vars) == {"thetao", "zos"}
  xr.testing.assert_identical(out.thetao, ocean_ds.thetao)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"dtype": "float32", "zos": {"dtype": "float32"}},  # both forms at once
    {"casting": "same_kind"},  # casting alone
    {"absent": {"dtype": "float32"}},  # not a data variable
  ],
)
def test_astype_rejects_ambiguous_arguments(ocean_ds: xr.Dataset, kwargs: dict):
  """Arguments that would be ignored or misread are errors."""
  with pytest.raises(ValueError):
    pu.astype(ocean_ds, **kwargs)


def test_clip_negative(atmosphere_ds: xr.Dataset):
  """Listed variables are clipped at 0 (e.g. precipitation); others are untouched; NaN stay NaN."""
  ds = atmosphere_ds.assign(tp=atmosphere_ds.tp.where(atmosphere_ds.tp > -1e-3))
  out = pu.clip_negative(ds, ["tp"])
  assert float(out.tp.min()) >= 0.0
  assert int(out.tp.isnull().sum()) == int(ds.tp.isnull().sum())
  xr.testing.assert_identical(out.t2m, ds.t2m)


def test_rescale(ocean_ds: xr.Dataset):
  """Variables are multiplied by the given factors."""
  out = pu.rescale(ocean_ds, zos=100.0)
  xr.testing.assert_allclose(out.zos, 100.0 * ocean_ds.zos)
  xr.testing.assert_identical(out.thetao, ocean_ds.thetao)


def test_assign_and_drop_var_attrs(ocean_ds: xr.Dataset):
  """Attributes of variables and coordinates are replaced or dropped, without touching the input."""
  out = pu.assign_var_attrs(ocean_ds, latitude={"units": "degrees_north"})
  assert out.latitude.attrs == {"units": "degrees_north"}
  assert ocean_ds.latitude.attrs == {}
  out = pu.drop_var_attrs(ocean_ds, ["thetao"])
  assert out.thetao.attrs == {}
  assert ocean_ds.thetao.attrs == {"units": "degC"}


@pytest.mark.parametrize("unit", ["degrees", "radians"])
def test_sin_cos_averages_directions(unit: str):
  """Directions of 350 and 10 degrees average to 0 (north) through sine and cosine, where the
  angles average to 180; missing angles stay missing, and the angle is replaced."""
  degrees = np.array([[350.0, 10.0], [90.0, np.nan]])
  angles = degrees if unit == "degrees" else np.deg2rad(degrees)
  ds = xr.Dataset(
    {"VMDR": (("time", "x"), angles.T, {"long_name": "mean wave direction"})},
    coords={"time": [0, 1], "x": [0, 1]},
  )
  out = pu.sin_cos(ds, "VMDR", "mwd_sin", "mwd_cos", unit=unit)
  assert set(out.data_vars) == {"mwd_sin", "mwd_cos"}
  mean = np.rad2deg(np.arctan2(out.mwd_sin.mean("time"), out.mwd_cos.mean("time")))
  assert float(mean[0]) == pytest.approx(0.0, abs=1e-9)
  naive = (
    float(np.rad2deg(ds.VMDR.isel(x=0).mean()))
    if unit == "radians"
    else float(ds.VMDR.isel(x=0).mean())
  )
  assert naive == pytest.approx(180.0)  # what averaging the angles gives
  assert float(out.mwd_sin[0, 1]) == pytest.approx(1.0) and np.isnan(out.mwd_cos[1, 1])
  assert out.mwd_sin.attrs == {"units": "1", "long_name": "sine of mean wave direction"}
  assert "VMDR" in pu.sin_cos(ds, "VMDR", "s", "c", unit=unit, keep=True)


def test_coverage_attrs_come_from_the_coordinates(ocean_ds: xr.Dataset):
  """The extent attributes follow the coordinates, not what a recipe typed (it said longitudes
  -180 to 180 over coordinates 0 to 359.75, finding 15)."""
  ds = ocean_ds.assign_coords(depth=ocean_ds.depth.assign_attrs(units="m", positive="down"))
  attrs = pu.coverage_attrs(ds, "latitude", "longitude", vertical_dim="depth").attrs
  assert attrs["geospatial_lat_min"] == -1.75 and attrs["geospatial_lat_max"] == 1.75
  assert attrs["geospatial_lon_min"] == 0.25 and attrs["geospatial_lon_max"] == 3.75
  assert attrs["geospatial_lat_resolution"] == "0.5 degree"
  assert attrs["geospatial_lon_units"] == "degrees_east"
  assert (attrs["geospatial_vertical_min"], attrs["geospatial_vertical_max"]) == (0.5, 5.0)
  assert attrs["geospatial_vertical_positive"] == "down"
  times = ds.time.to_index()
  assert attrs["time_coverage_start"] == f"{times[0]:%Y-%m-%dT%H:%M:%S}Z"
  assert attrs["time_coverage_end"] == f"{times[-1]:%Y-%m-%dT%H:%M:%S}Z"
  assert attrs["time_coverage_resolution"] == "P1D"


@pytest.mark.parametrize(
  ("lon", "resolution"),
  [
    (np.arange(0.0, 10.0, 0.1), "0.1 degree"),
    (np.arange(0.0, 360.0, 1 / 12).astype("float32"), "0.0833333 degree"),
    (np.arange(0.0, 360.0, 0.01).astype("float32"), "0.01 degree"),
    (np.array([0.0, 1.0, 3.0]), None),
  ],
  ids=["tenth", "twelfth-float32", "hundredth-float32", "irregular"],
)
def test_coverage_resolution_of_inexact_grids(lon: np.ndarray, resolution):
  """Spacings not exact in binary (0.1, a float32 1/12 degree) are regular; uneven ones are not."""
  ds = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": lon})
  attrs = pu.coverage_attrs(ds, time_dim=None).attrs
  assert attrs.get("geospatial_lon_resolution") == resolution
