# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Live checks: one tiny download per remote backend, against the real services.

Marked `network` and skipped unless pytest gets `--live`: they need the network and, for
Copernicus Marine and EWDS, credentials (environment variables, `~/.arcomakerc`, or the services'
own credential files). They never run on pull requests (see `.github/workflows/live-services.yml`); run them with
`pytest --live tests/live`.

The services revise their data, so these tests check structure, completeness and plausible values,
not exact numbers. A failure means a service, a client library or arco-make's use of them changed.
"""

import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli, rcfile

RECIPES = pathlib.Path(__file__).parent / "recipes"

pytestmark = pytest.mark.network


@dataclass(frozen=True)
class Expected:
  variable: str
  low: float
  high: float
  # Fraction of finite values in the box: 1 where the variable has no mask; land or sea cells
  # are NaN otherwise. A lost chunk (N8) or a broken mask lowers it.
  min_finite: float
  max_finite: float = 1.0
  has_time: bool = True
  # Environment variables with the credentials, or a credentials file of the service.
  credentials: tuple[str, ...] = ()
  credentials_file: str | None = None


CASES = {
  # ERA5 2 m temperature over the Adriatic, a January day, in kelvin: defined everywhere.
  "zarr_gcs": Expected("2m_temperature", 250.0, 300.0, min_finite=1.0),
  # GLORYS surface temperature, °C, NaN over land.
  "copernicusmarine": Expected(
    "thetao",
    5.0,
    25.0,
    min_finite=0.35,  # 0.41 measured
    max_finite=0.5,
    credentials=("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"),
    credentials_file="~/.copernicusmarine/.copernicusmarine-credentials",
  ),
  # GloFAS river discharge, m³/s, NaN over the sea.
  "earlywarningdatastore": Expected(
    "avg_dis",
    0.0,
    1e5,
    min_finite=0.55,  # 0.62 measured
    max_finite=0.7,
    credentials=("CDSAPI_URL", "CDSAPI_KEY"),
    credentials_file="~/.cdsapirc",
  ),
  # GloFAS upstream area, m², NaN over the sea.
  "netcdfoverhttp": Expected("uparea", 0.0, 1e13, min_finite=0.55, max_finite=0.7, has_time=False),
}


def credentials_available(expected: Expected) -> bool:
  if all(os.environ.get(var) for var in expected.credentials):
    return True
  return expected.credentials_file is not None and (
    pathlib.Path(expected.credentials_file).expanduser().exists()
  )


def failure_report(result: Result) -> str:
  """The chain of exception types; locally, also the end of the command's output.

  In CI the output stays out of the (public) log: services put temporary tokens in URLs and
  messages.
  """
  types, exc, seen = [], result.exception, set()
  while exc is not None and id(exc) not in seen:
    seen.add(id(exc))
    types.append(type(exc).__name__)
    exc = exc.__cause__ or exc.__context__
  report = f"download failed: {' <- '.join(types)}"
  if not os.environ.get("CI"):
    report += "\n" + "\n".join(result.output.strip().splitlines()[-20:])
  return report


@pytest.mark.parametrize("name", sorted(CASES))
def test_download_from_live_service(name: str, tmp_path: pathlib.Path):
  """One day from the service: the variable, on a non-empty grid, complete, with plausible values."""
  expected = CASES[name]
  rcfile.load()  # credentials may come from ~/.arcomakerc, as for the CLI
  if expected.credentials and not credentials_available(expected):
    pytest.fail(f"no credentials: set {', '.join(expected.credentials)}")
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
  assert result.exit_code == 0, failure_report(result)

  ds = xr.open_zarr(output)
  assert expected.variable in ds.data_vars
  da = ds[expected.variable]
  if expected.has_time:
    assert ds.sizes["time"] == 1
  assert all(size > 0 for size in da.sizes.values())
  values = da.values
  finite = values[np.isfinite(values)]
  fraction = finite.size / values.size
  assert expected.min_finite <= fraction <= expected.max_finite, (
    f"finite fraction {fraction:.3f} outside [{expected.min_finite}, {expected.max_finite}]"
  )
  assert expected.low <= finite.min() and finite.max() <= expected.high, (
    f"values outside [{expected.low}, {expected.high}]: {finite.min()}..{finite.max()}"
  )
