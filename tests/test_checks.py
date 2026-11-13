# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Validation checks and the `validate` accessor."""

import warnings
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from arcomake import checks
from arcomake.checks import ChecksConfig, ValidationError


def global_grid(
  lat_name: str = "lat", lon_name: str = "lon", decreasing: bool = False
) -> xr.Dataset:
  lat = np.arange(-90.0, 90.1, 45.0)
  return xr.Dataset(
    {"a": ((lat_name, lon_name), np.zeros((5, 4)))},
    coords={lat_name: lat[::-1] if decreasing else lat, lon_name: np.arange(0.0, 360.0, 90.0)},
  )


def test_valid_time_coordinate_accepts_complete_daily_axis(ocean_ds: xr.Dataset):
  """A daily axis from start to end (excluded) passes, Feb 29 included."""
  checks.valid_time_coordinate(ocean_ds, datetime(2020, 2, 26), datetime(2020, 3, 3))


@pytest.mark.parametrize(
  ("selection", "message"),
  [
    ([0, 1, 3, 4, 5], "Missing"),
    ([0, 1, 1, 2, 3, 4, 5], "Duplicate"),
    ([1, 0, 2, 3, 4, 5], "not sorted"),
  ],
)
def test_valid_time_coordinate_reports_problems(ocean_ds: xr.Dataset, selection, message):
  """Gaps, duplicates and disorder are reported as validation failures."""
  with pytest.raises(ValidationError, match=message):
    checks.valid_time_coordinate(
      ocean_ds.isel(time=selection), datetime(2020, 2, 26), datetime(2020, 3, 3)
    )


def test_valid_time_coordinate_cftime_without_deprecation():
  """cftime axes are checked with the non-deprecated API."""
  times = xr.date_range("2001-01-01", periods=3, freq="1D", calendar="noleap", use_cftime=True)
  ds = xr.Dataset(coords={"time": times})
  with warnings.catch_warnings():
    warnings.simplefilter("error", FutureWarning)
    warnings.simplefilter("error", DeprecationWarning)
    checks.valid_time_coordinate(ds, datetime(2001, 1, 1), datetime(2001, 1, 4))


def test_valid_time_coordinate_cftime_calendar():
  """A complete `noleap` axis passes: Feb 29 is not expected in that calendar."""
  times = xr.date_range(
    "2020-02-27", periods=3, freq="1D", calendar="noleap", use_cftime=True
  )  # Feb 27, Feb 28, Mar 1
  ds = xr.Dataset(coords={"time": times})
  checks.valid_time_coordinate(ds, datetime(2020, 2, 27), datetime(2020, 3, 2))
  with pytest.raises(ValidationError, match="Missing"):
    checks.valid_time_coordinate(ds.isel(time=[0, 2]), datetime(2020, 2, 27), datetime(2020, 3, 2))


def test_valid_global_ecmwf_coordinates_accepts_increasing_latitudes():
  """Increasing latitudes in [-90, 90] and longitudes in [0, 360) pass."""
  checks.valid_global_ecmwf_coordinates(global_grid(), latitude_dim="lat", longitude_dim="lon")


@pytest.mark.parametrize(
  ("grid", "message"),
  [
    (global_grid(decreasing=True), "decreasing"),
    (global_grid().assign_coords(lat=np.linspace(-95, 90, 5)), r"outside the \[-90, 90\]"),
    (global_grid().assign_coords(lon=np.arange(-180.0, 180.0, 90.0)), r"\[0, 360\)"),
  ],
)
def test_valid_global_ecmwf_coordinates_failure_is_a_validation_error(grid, message):
  """Rejected coordinates are validation failures, so `--no-fail` reports them instead of aborting."""
  with pytest.raises(ValidationError, match=message):
    checks.valid_global_ecmwf_coordinates(grid, latitude_dim="lat", longitude_dim="lon")


def test_validate_without_fail_warns_on_bad_coordinates():
  """With `fail=False` the accessor turns the failure into a warning (B11 aborted here)."""
  config = ChecksConfig.model_validate({"valid_global_ecmwf_coordinates": {}})
  with pytest.warns(UserWarning, match="decreasing"):
    global_grid(decreasing=True).arcomake.validate(config)


def test_ecmwf_check_defaults_agree():
  """The defaults of the recipe table and of the function are the same."""
  model_defaults = checks.ValidGlobalECMWFCoordinatesCheck().model_dump()
  ds = global_grid(
    lat_name=model_defaults["latitude_dim"], lon_name=model_defaults["longitude_dim"]
  )
  checks.valid_global_ecmwf_coordinates(ds)


def test_ensure_no_nans_respects_masks(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """NaN outside a variable's mask are allowed; NaN inside it are reported, per day."""
  ds = xr.merge([ocean_ds, static_ds])
  ds["m"] = ds.deptho.notnull()
  ds = ds.drop_vars("deptho").assign(zos=ds.zos.fillna(0.0).where(ds.m))
  checks.ensure_no_nans(ds[["zos", "m"]], m=["zos"])
  broken = ds.assign(zos=ds.zos.where(ds.time != ds.time[2]))
  with pytest.raises(ValidationError, match="2020-02-28"):
    checks.ensure_no_nans(broken[["zos", "m"]], m=["zos"])


def test_validate_accessor_warns_or_raises(ocean_ds: xr.Dataset):
  """Failed checks warn by default and raise with `fail=True`."""
  config = ChecksConfig.model_validate(
    {
      "valid_time_coordinate": {
        "start": datetime(2020, 2, 26),
        "end": datetime(2020, 3, 10),
        "freq": "1D",
      }
    }
  )
  with pytest.warns(UserWarning, match="valid_time_coordinate failed"):
    ocean_ds.arcomake.validate(config)
  with pytest.raises(ValidationError):
    ocean_ds.arcomake.validate(config, fail=True)
