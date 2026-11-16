# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Live checks: one tiny download per remote backend, against the real services.

Marked `network`: they need the network and, for Copernicus Marine and EWDS, credentials from the
environment. They are skipped by default and never run on pull requests (see
`.github/workflows/live-services.yml`); run them with `pytest -m network tests/live`.

The services revise their data, so these tests check structure and plausible values, not exact
numbers. A failure means a service, a client library or arco-make's use of them changed.
"""

import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli

RECIPES = pathlib.Path(__file__).parent / "recipes"

pytestmark = pytest.mark.network


@dataclass(frozen=True)
class Expected:
  variable: str
  low: float
  high: float
  has_time: bool = True
  needs: tuple[str, ...] = ()


CASES = {
  # ERA5 2 m temperature over the Adriatic, a January day, in kelvin.
  "zarr_gcs": Expected("2m_temperature", 250.0, 300.0),
  # GLORYS surface temperature, °C.
  "copernicusmarine": Expected(
    "thetao",
    5.0,
    25.0,
    needs=("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"),
  ),
  # GloFAS river discharge, m³/s (NaN over the sea).
  "earlywarningdatastore": Expected("avg_dis", 0.0, 1e5, needs=("CDSAPI_URL", "CDSAPI_KEY")),
  # GloFAS upstream area, m² (NaN over the sea).
  "netcdfoverhttp": Expected("uparea", 0.0, 1e13, has_time=False),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_download_from_live_service(name: str, tmp_path: pathlib.Path):
  """One day from the service: the variable is there, on a non-empty grid, with plausible values."""
  expected = CASES[name]
  missing = [var for var in expected.needs if not os.environ.get(var)]
  if missing:
    pytest.fail(f"credentials not set: {', '.join(missing)}")
  output = tmp_path / f"{name}.zarr"
  result = CliRunner().invoke(
    cli,
    [
      "download",
      "--dask-setup",
      "threads",
      "--no-progress",
      str(RECIPES / f"{name}.toml"),
      str(output),
    ],
  )
  assert result.exit_code == 0, f"download failed: {type(result.exception).__name__}"

  ds = xr.open_zarr(output)
  assert expected.variable in ds.data_vars
  da = ds[expected.variable]
  if expected.has_time:
    assert ds.sizes["time"] == 1
  assert all(size > 0 for size in da.sizes.values())
  values = da.values
  finite = values[np.isfinite(values)]
  assert finite.size > 0, "every value is NaN"
  assert expected.low <= finite.min() and finite.max() <= expected.high, (
    f"values outside [{expected.low}, {expected.high}]: {finite.min()}..{finite.max()}"
  )
