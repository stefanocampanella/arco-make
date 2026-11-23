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

import datetime
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest
from click.testing import CliRunner, Result

from arcomake import cli, settings
from arcomake.backends import CopernicusMarine, ECMWFOpenData
from arcomake.temporary_utils import TempDirectoryRegistry
from tests.conftest import open_output

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
  settings.load()  # credentials may come from ~/.arcomakerc, as for the CLI
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

  ds = open_output(output)
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


def test_copernicusmarine_returns_lazy_arrays():
  """The backend gets lazily indexed arrays from the toolbox, so the planned chunks decide the reads.

  It passes `chunk_size_limit=False`, which the toolbox does not document (it accepts -1 or a
  positive limit): today a falsy limit opens the store without Dask (`chunks=None`), and download
  then chunks the variables as planned (`chunk_plan`). The toolbox's default returns Dask arrays of
  its own chunking instead (time 2, all 50 depths for this dataset), which the plan would only
  rechunk. If a toolbox release changes that, this check fails. The backend is called
  directly, with the recipes' service and version: `xarray.open_dataset` would wrap or rechunk
  what it returns.
  """
  expected = CASES["copernicusmarine"]
  settings.load()
  if not credentials_available(expected):
    pytest.fail(f"no credentials: set {', '.join(expected.credentials)}")
  ds = CopernicusMarine().open_dataset(
    "cm://cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao", "zos"],
    start_datetime="2012-01-01",
    end_datetime="2012-01-04",
    service="arco-geo-series",
    dataset_version="202311",
  )
  assert {name: ds[name].chunks for name in ("thetao", "zos")} == {"thetao": None, "zos": None}


def test_ecmwf_open_data_reads_a_recent_run(tmp_path: pathlib.Path):
  """A recent IFS run from AWS's copy of ECMWF's open data (anonymous): a day of 2 m temperature
  and precipitation along valid time, 3-hourly, precipitation as increments (plausible daily
  totals), on the global 0.25 deg grid. A run three days old is published whatever the hour."""
  run = datetime.datetime.combine(
    datetime.date.today() - datetime.timedelta(days=3), datetime.time()
  )
  day = datetime.timedelta(days=1)
  with TempDirectoryRegistry(base_dir=tmp_path):
    ds = ECMWFOpenData().open_dataset(
      "s3://ecmwf-forecasts",
      run_datetime=run,
      start_datetime=run,
      end_datetime=run + day,
      param=["2t", "tp"],
    )
  assert dict(ds.sizes) == {"time": 9, "latitude": 721, "longitude": 1440}
  assert float(ds.t2m.min()) > 180.0 and float(ds.t2m.max()) < 340.0
  total = ds.tp.isel(time=slice(0, 8)).sum("time")  # the day's eight 3-hour increments
  assert float(total.min()) > -1e-6 and 0.0005 < float(total.mean()) < 0.01  # metres a day
  assert bool(ds.tp.isel(time=-1).isnull().all())  # no increment after the last step
