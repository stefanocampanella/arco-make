# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`rename` and `coordinate_tolerance` of a part: a static file of the same grid from another
publisher (other coordinate names, coordinates off by rounding) merged with a dataset's other
parts, which are merged only on equal coordinates."""

import datetime
import pathlib

import numpy as np
import pydantic
import pytest
import xarray as xr

from arcomake.dataset_utils import DatasetConfig, GenericPartConfig, download_and_process

START = datetime.datetime(2020, 1, 1)
LAT = np.arange(-59.975, -59.0, 0.05)
LON = np.arange(-179.975, -179.0, 0.05)


@pytest.fixture
def files(tmp_path: pathlib.Path) -> tuple[str, str]:
  """A field on (latitude, longitude), and a static field on (lat, lon) whose coordinates differ
  by 4e-12, as GloFAS's discharge and its drainage directions do."""
  rng = np.random.default_rng(0)
  field = xr.Dataset(
    {"q": (("latitude", "longitude"), rng.random((LAT.size, LON.size)))},
    coords={"latitude": LAT, "longitude": LON},
  )
  static = xr.Dataset(
    {"ldd": (("lat", "lon"), rng.integers(1, 10, (LAT.size, LON.size)).astype("float32"))},
    coords={"lat": LAT + 4e-12, "lon": LON - 4e-12},
  )
  static.lat.attrs = {"units": "degrees_north"}
  paths = tmp_path / "q.nc", tmp_path / "ldd.nc"
  field.to_netcdf(paths[0])
  static.to_netcdf(paths[1])
  return str(paths[0]), str(paths[1])


def dataset(files: tuple[str, str], **static_options) -> xr.Dataset:
  configs = DatasetConfig(
    parts=[
      GenericPartConfig(filename_or_obj=files[0]),
      GenericPartConfig(filename_or_obj=files[1], **static_options),
    ]
  )
  return download_and_process(configs, START, START + datetime.timedelta(days=1)).load()


def test_renamed_part_with_tolerance_takes_the_first_parts_coordinates(files):
  ds = dataset(files, rename={"lat": "latitude", "lon": "longitude"}, coordinate_tolerance=1e-9)
  assert set(ds.data_vars) == {"q", "ldd"}
  np.testing.assert_array_equal(ds.latitude.values, LAT)
  np.testing.assert_array_equal(ds.longitude.values, LON)
  assert ds.ldd.notnull().all()


def test_without_tolerance_parts_merge_only_on_equal_coordinates(files):
  with pytest.raises(ValueError):
    dataset(files, rename={"lat": "latitude", "lon": "longitude"})


def test_a_difference_above_the_tolerance_fails(files):
  with pytest.raises(ValueError, match="differs from an earlier part's"):
    dataset(files, rename={"lat": "latitude", "lon": "longitude"}, coordinate_tolerance=1e-13)


def test_a_missing_coordinate_value_fails(files, tmp_path: pathlib.Path):
  """NaN differs by NaN, which no tolerance check catches: the coordinates were replaced."""
  static = xr.open_dataset(files[1]).load()
  static = static.assign_coords(lat=np.where(np.arange(LAT.size) == 3, np.nan, static.lat))
  path = tmp_path / "nan.nc"
  static.to_netcdf(path)
  with pytest.raises(ValueError, match="missing values"):
    dataset(
      (files[0], str(path)),
      rename={"lat": "latitude", "lon": "longitude"},
      coordinate_tolerance=1e-9,
    )


def test_missing_values_at_the_same_positions_are_matched(files, tmp_path: pathlib.Path):
  """Equal missing values merge without the tolerance too: they must not fail with it."""
  paths = []
  for number, path in enumerate(files):
    ds = xr.open_dataset(path).load()
    lat = "latitude" if number == 0 else "lat"
    ds = ds.assign_coords({lat: np.where(np.arange(LAT.size) == 3, np.nan, ds[lat])})
    paths.append(str(tmp_path / f"nan{number}.nc"))
    ds.to_netcdf(paths[-1])
  ds = dataset(
    tuple(paths), rename={"lat": "latitude", "lon": "longitude"}, coordinate_tolerance=1e-9
  )
  assert np.isnan(ds.latitude.values[3]) and ds.ldd.notnull().all()


def test_a_tolerance_matching_no_coordinate_fails(files):
  """Without `rename`, no coordinate of the part is an earlier part's: nothing would be snapped."""
  with pytest.raises(ValueError, match="no index coordinate of the part matches"):
    dataset(files, coordinate_tolerance=1e-9)


def test_tolerance_on_the_first_part_is_refused():
  """It has no earlier part to take coordinates from."""
  with pytest.raises(pydantic.ValidationError, match="first part"):
    DatasetConfig(
      parts=[
        GenericPartConfig(filename_or_obj="a.nc", coordinate_tolerance=1e-9),
        GenericPartConfig(filename_or_obj="b.nc"),
      ]
    )


def test_tolerance_must_be_positive():
  with pytest.raises(pydantic.ValidationError):
    GenericPartConfig(filename_or_obj="x.nc", coordinate_tolerance=0.0)
