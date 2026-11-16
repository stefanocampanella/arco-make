# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Synthetic datasets shared by the tests.

Everything here is small, deterministic and offline: no network, no credentials.

The ocean grid is 8 x 8 cells of 0.5 degrees around the equator, with three depth levels whose
faces are at 0, 1, 3 and 7 m (centres 0.5, 2 and 5 m). The first longitude column is land; the
bathymetry deepens eastwards, so deeper levels have fewer sea cells (a coastline and a slope).
Below the bathymetry and on land the 3D fields are NaN, as in a real ocean reanalysis.
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arcomake  # noqa: F401  (registers the `arcomake` accessor)

DATA_DIR = pathlib.Path(__file__).parent / "data"


def pytest_addoption(parser: pytest.Parser) -> None:
  parser.addoption(
    "--live",
    action="store_true",
    help="run the live checks (marker `network`): needs the network and service credentials",
  )


@pytest.fixture(autouse=True)
def no_user_settings_file(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
  """Tests never read the user's `~/.arcomakerc` (it holds secrets); live checks may."""
  if request.node.get_closest_marker("network") is None:
    monkeypatch.setenv("ARCOMAKE_RC", "")
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
      monkeypatch.delenv(name, raising=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
  """Skip the live checks unless `--live` is given, even when they are named explicitly."""
  if config.getoption("--live"):
    return
  skip = pytest.mark.skip(reason="live check: run with --live (network and credentials)")
  for item in items:
    if "network" in item.keywords:
      item.add_marker(skip)


LATITUDES = np.arange(-1.75, 2.0, 0.5)
LONGITUDES = np.arange(0.25, 4.0, 0.5)
DEPTHS = np.array([0.5, 2.0, 5.0])
# One land column (NaN), then a slope from 1.5 m to 7.5 m.
BATHYMETRY_PROFILE = np.array([np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
# 2020 is a leap year: the daily axis crosses Feb 29.
DAILY_TIMES = pd.date_range("2020-02-26", periods=6, freq="1D")


def make_static_dataset() -> xr.Dataset:
  """Bathymetry `deptho` (NaN on land) and the depth coordinate of the levels."""
  deptho = np.tile(BATHYMETRY_PROFILE, (LATITUDES.size, 1))
  return xr.Dataset(
    {"deptho": (("latitude", "longitude"), deptho, {"units": "m"})},
    coords={"latitude": LATITUDES, "longitude": LONGITUDES, "depth": ("depth", DEPTHS)},
  )


def make_ocean_dataset(times: pd.DatetimeIndex = DAILY_TIMES) -> xr.Dataset:
  """Daily 3D `thetao` (NaN below the bathymetry) and 2D `zos`, with attributes and encoding."""
  rng = np.random.default_rng(0)
  shape = (times.size, DEPTHS.size, LATITUDES.size, LONGITUDES.size)
  thetao = 10.0 + rng.standard_normal(shape)
  deptho = np.tile(BATHYMETRY_PROFILE, (LATITUDES.size, 1))
  wet = DEPTHS[:, None, None] < deptho[None, :, :]  # (depth, lat, lon); False on land (NaN)
  thetao = np.where(wet[None], thetao, np.nan)
  zos = np.where(np.isnan(deptho)[None], np.nan, 0.1 * rng.standard_normal(shape[:1] + shape[2:]))
  ds = xr.Dataset(
    {
      "thetao": (("time", "depth", "latitude", "longitude"), thetao, {"units": "degC"}),
      "zos": (("time", "latitude", "longitude"), zos, {"units": "m"}),
    },
    coords={"time": times, "depth": DEPTHS, "latitude": LATITUDES, "longitude": LONGITUDES},
    attrs={"title": "synthetic ocean"},
  )
  ds["thetao"].encoding = {"dtype": "float32"}
  return ds


def make_atmosphere_dataset(start: str = "2020-02-26", days: int = 6) -> xr.Dataset:
  """Hourly 2D `t2m` and `tp` (with a few negative values, as accumulated fields sometimes have)."""
  times = pd.date_range(start, periods=24 * days, freq="1h")
  rng = np.random.default_rng(1)
  shape = (times.size, LATITUDES.size, LONGITUDES.size)
  return xr.Dataset(
    {
      "t2m": (("time", "latitude", "longitude"), 280.0 + rng.standard_normal(shape)),
      "tp": (("time", "latitude", "longitude"), 1e-3 * rng.standard_normal(shape)),
    },
    coords={"time": times, "latitude": LATITUDES, "longitude": LONGITUDES},
  )


@pytest.fixture
def static_ds() -> xr.Dataset:
  return make_static_dataset()


@pytest.fixture
def ocean_ds() -> xr.Dataset:
  return make_ocean_dataset()


@pytest.fixture
def atmosphere_ds() -> xr.Dataset:
  return make_atmosphere_dataset()


@pytest.fixture
def sources(tmp_path: pathlib.Path) -> pathlib.Path:
  """Write the synthetic sources as local Zarr stores and return their directory."""
  directory = tmp_path / "sources"
  directory.mkdir()
  make_ocean_dataset().to_zarr(directory / "ocean.zarr", mode="w")
  make_static_dataset().to_zarr(directory / "static.zarr", mode="w")
  make_atmosphere_dataset().to_zarr(directory / "atmosphere.zarr", mode="w")
  return directory


def render_recipe(name: str, destination: pathlib.Path, **substitutions: str) -> pathlib.Path:
  """Copy a recipe from `tests/data`, replacing `@KEY@` placeholders, and return its path."""
  text = (DATA_DIR / name).read_text()
  for key, value in substitutions.items():
    text = text.replace(f"@{key}@", value)
  path = destination / name
  path.write_text(text)
  return path


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
  """Keep the temporary files of arco-make (TempDirectoryRegistry, checkpoints) inside tmp_path."""
  directory = tmp_path / "tmp"
  directory.mkdir()
  monkeypatch.setattr(tempfile, "tempdir", str(directory))
  return directory
