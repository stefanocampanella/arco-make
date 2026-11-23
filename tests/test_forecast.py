# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Forecast downloads: runs read along valid time, stored along their initialization."""

import datetime
import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import backends, cli, dataset_utils, forecast, icechunk_utils
from arcomake.cli_utils import read_configs
from arcomake.dataset_utils import (
  CopernicusMarinePartConfig,
  DatasetConfig,
  EarlyWarningDataStorePartConfig,
  download_and_process,
)
from tests.conftest import open_output

download_module = importlib.import_module("arcomake.download")  # the module, not the command
DAY = datetime.timedelta(days=1)
RUN = datetime.datetime(2026, 11, 21)


def _ewds_part(**extra) -> EarlyWarningDataStorePartConfig:
  return EarlyWarningDataStorePartConfig.model_validate(
    {
      "filename_or_obj": "ewds://cems-glofas-forecast",
      "system_version": "operational",
      "hydrological_model": "lisflood",
      "product_type": "control_forecast",
      "variable": "river_discharge_in_the_last_24_hours",
      **extra,
    }
  )


def test_run_parts_get_the_run_and_the_shifted_valid_times(monkeypatch: pytest.MonkeyPatch):
  """A forecast block over [R, R + 10 days) of valid time gives the run's parts the run date and
  the block moved by their `time_offset`: GloFAS's daily discharge, stamped at the end of its
  day, is leads 24 to 240 h, stamped R to R + 9 days once shifted back."""
  received = {}
  valid = pd.date_range(RUN + DAY, periods=10, freq="1D")

  def fake_open_input(**kwargs):
    received.update(kwargs)
    return xr.Dataset({"dis24": ("time", np.ones(10))}, coords={"time": valid})

  monkeypatch.setattr(dataset_utils, "open_input", fake_open_input)
  config = DatasetConfig(parts=[_ewds_part(time_offset="-24h")])
  out = download_and_process(config, RUN, RUN + 10 * DAY, run_datetime=RUN)
  assert received["run_datetime"] == RUN
  assert (received["start_datetime"], received["end_datetime"]) == (RUN + DAY, RUN + 11 * DAY)
  assert out.time.values[0] == np.datetime64(RUN) and out.time.size == 10


def test_parts_read_no_run_without_one(monkeypatch: pytest.MonkeyPatch):
  """A historical block gives no run date: the EWDS part reads the historical product."""
  received = {}

  def fake_open_input(**kwargs):
    received.update(kwargs)
    time = pd.date_range(RUN, periods=1, freq="1D")
    return xr.Dataset({"dis24": ("time", np.ones(1))}, coords={"time": time})

  monkeypatch.setattr(dataset_utils, "open_input", fake_open_input)
  download_and_process(DatasetConfig(parts=[_ewds_part(timespan="day")]), RUN, RUN + DAY)
  assert "run_datetime" not in received


def test_ecmwf_open_data_parts_get_the_run_and_the_block(monkeypatch: pytest.MonkeyPatch):
  """An `ecmwfopendata` part needs its engine named (its URL is any copy of the tree), and gets
  the run and the block's valid times."""
  received = {}

  def fake_open_input(**kwargs):
    received.update(kwargs)
    time = pd.date_range(RUN, periods=8, freq="3h")
    return xr.Dataset({"t2m": ("time", np.ones(8))}, coords={"time": time})

  monkeypatch.setattr(dataset_utils, "open_input", fake_open_input)
  part = {"filename_or_obj": "s3://ecmwf-forecasts", "param": ["2t"]}
  with pytest.raises(ValueError):
    DatasetConfig.model_validate({"parts": [part]})  # a generic part needs no `param`
  config = DatasetConfig.model_validate(
    {"forecast": "runs", "parts": [part | {"engine": "ecmwfopendata"}]}
  )
  download_and_process(config, RUN, RUN + DAY, run_datetime=RUN)
  assert (received["run_datetime"], received["start_datetime"]) == (RUN, RUN)
  assert received["end_datetime"] == RUN + DAY and received["param"] == ["2t"]


def test_engines_without_runs_are_refused():
  """A dataset none of whose parts reads forecast runs cannot be read as one."""
  part = CopernicusMarinePartConfig.model_validate({"filename_or_obj": "cm://some-dataset"})
  with pytest.raises(ValueError, match="no part reads forecast runs"):
    download_and_process(DatasetConfig(parts=[part]), RUN, RUN + DAY, run_datetime=RUN)


def test_other_parts_of_a_runs_dataset_are_read_as_in_any_block(monkeypatch: pytest.MonkeyPatch):
  """A "runs" dataset may have parts that read no runs (GloFAS's drainage network, a static
  file): they get no run, and are read as in any block."""
  received: list[dict] = []

  def fake_open_input(**kwargs):
    received.append(kwargs)
    if kwargs.get("engine") == "netcdfoverhttp":
      return xr.Dataset({"ldd": ("x", np.ones(2))})
    valid = pd.date_range(RUN + DAY, periods=10, freq="1D")
    return xr.Dataset({"dis24": ("time", np.ones(10))}, coords={"time": valid})

  monkeypatch.setattr(dataset_utils, "open_input", fake_open_input)
  static = {"filename_or_obj": "https://example.org/ldd.nc", "engine": "netcdfoverhttp"}
  config = DatasetConfig.model_validate(
    {"forecast": "runs", "parts": [_ewds_part(time_offset="-24h"), static]}
  )
  out = download_and_process(config, RUN, RUN + 10 * DAY, run_datetime=RUN)
  assert "run_datetime" in received[0] and "run_datetime" not in received[1]
  assert set(out.data_vars) == {"dis24", "ldd"}


def test_runs_need_engines_that_read_runs():
  with pytest.raises(ValueError, match="no part reads forecast runs"):
    DatasetConfig.model_validate(
      {"forecast": "runs", "parts": [{"filename_or_obj": "x.zarr", "engine": "zarr"}]}
    )


# Forecast downloads end to end: a run source (EWDS parts, answered here), a best-estimate series
# (a local Zarr store, overwritten by each "release") and a static dataset.

FIRST = datetime.datetime(2026, 11, 1)
LAT, LON = [0.0, 1.0], [10.0, 11.0, 12.0]

RECIPE = """
[forecast]
horizon = "3D"

[datasets.atmos]
forecast = "runs"
flag = "atmos_forecast"

[[datasets.atmos.parts]]
filename_or_obj = "ewds://fake-forecast"
system_version = "operational"
hydrological_model = "m"
product_type = "control_forecast"
variable = "v"

[[datasets.atmos.postprocess]]
name = "resample"
time = "1D"
label = "left"
closed = "left"
reduce = "mean"

[datasets.ocean]
flag = "ocean_forecast"

[[datasets.ocean.parts]]
filename_or_obj = "@SERIES@"
engine = "zarr"

[datasets.static]

[[datasets.static.parts]]
filename_or_obj = "@STATIC@"
engine = "zarr"

[save]
chunks = {init_time = 1}
"""


def atmos_value(run: datetime.datetime, valid: np.datetime64) -> float:
  """The run source's value: the run's day number x 100 + the lead in hours."""
  lead = (valid - np.datetime64(run, "ns")) / np.timedelta64(1, "h")
  return (run - FIRST).days * 100.0 + lead


def write_series(path, end: datetime.datetime) -> None:
  """The best-estimate series as a release ending at `end` (exclusive) leaves it: daily values,
  the day number since FIRST."""
  time = pd.date_range(FIRST - 5 * DAY, end - DAY, freq="1D")
  days = ((time - pd.Timestamp(FIRST)) / pd.Timedelta("1D")).to_numpy()
  values = np.broadcast_to(days[:, None, None], (time.size, len(LAT), len(LON)))
  ds = xr.Dataset(
    {"sst": (("time", "lat", "lon"), values.copy())},
    coords={"time": time, "lat": LAT, "lon": LON},
  )
  ds.to_zarr(path, mode="w", consolidated=False)


@pytest.fixture
def forecast_recipe(tmp_path, monkeypatch: pytest.MonkeyPatch):
  """The recipe, its series and static stores; the EWDS answered by `atmos_value` every 6 hours,
  except for the runs in `unavailable` (returned, for a test to fill)."""
  write_series(tmp_path / "series.zarr", FIRST + 5 * DAY)  # the release of run 3 (Nov 3)
  xr.Dataset(
    {"depth": (("lat", "lon"), np.ones((len(LAT), len(LON))))},
    coords={"lat": LAT, "lon": LON},
  ).to_zarr(tmp_path / "static.zarr", mode="w", consolidated=False)
  recipe = tmp_path / "forecast.toml"
  text = RECIPE.replace("@SERIES@", str(tmp_path / "series.zarr"))
  recipe.write_text(text.replace("@STATIC@", str(tmp_path / "static.zarr")))
  unavailable: set[datetime.datetime] = set()
  real_open = dataset_utils.open_input

  def open_input(filename_or_obj, **kwargs):
    if kwargs.get("engine") != "earlywarningdatastore":
      return real_open(filename_or_obj, **kwargs)
    run = kwargs["run_datetime"]
    if run in unavailable:
      raise backends.RunUnavailable(f"fake-forecast has no run of {run:%Y-%m-%d}")
    valid = pd.date_range(
      kwargs["start_datetime"], kwargs["end_datetime"], freq="6h", inclusive="left"
    )
    values = np.array([atmos_value(run, v) for v in valid.to_numpy()])
    data = np.broadcast_to(values[:, None, None], (valid.size, len(LAT), len(LON)))
    return xr.Dataset(
      {"t2m": (("time", "lat", "lon"), data.copy())},
      coords={"time": valid, "lat": LAT, "lon": LON},
    )

  monkeypatch.setattr(dataset_utils, "open_input", open_input)
  return recipe, unavailable


def download(recipe, output, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def test_runs_stored_along_init_time(forecast_recipe, tmp_path):
  """Three runs: lead days 0-2 each, valid time = init + lead, the run source's daily means, the
  series by valid time, the static dataset once. The series is the last run's own forecast only
  (its release reaches that run's horizon, not beyond): the earlier runs are flagged."""
  recipe, _ = forecast_recipe
  result = download(recipe, tmp_path / "out", "--start=2026-11-01", "--end=2026-11-04")
  assert result.exit_code == 0, result.output
  out = open_output(tmp_path / "out")
  assert dict(out.sizes) == {"init_time": 3, "lead_time": 3, "lat": 2, "lon": 3}
  assert out.t2m.dims == ("init_time", "lead_time", "lat", "lon")
  assert out.depth.dims == ("lat", "lon")
  inits = pd.date_range(FIRST, periods=3, freq="1D").to_numpy()
  np.testing.assert_array_equal(out.init_time.values, inits)
  assert (out.lead_time.values / np.timedelta64(1, "D")).tolist() == [0, 1, 2]
  np.testing.assert_array_equal(
    out.valid_time.values, inits[:, None] + out.lead_time.values[None, :]
  )
  for i, init in enumerate(inits):
    run = pd.Timestamp(init).to_pydatetime()
    for lead in range(3):
      day = init + np.timedelta64(lead, "D")
      hours = [day + np.timedelta64(h, "h") for h in (0, 6, 12, 18)]
      expected = np.mean([atmos_value(run, h) for h in hours])
      assert float(out.t2m.isel(init_time=i, lead_time=lead, lat=0, lon=0)) == expected
      assert float(out.sst.isel(init_time=i, lead_time=lead, lat=0, lon=0)) == i + lead
  assert out.atmos_forecast.values.all()
  assert out.ocean_forecast.values[:, 0].tolist() == [False, False, True]


def test_time_coverage_spans_every_run(forecast_recipe, tmp_path):
  """`coverage_attrs` gives the output's coverage: the valid times of every run stored, from the
  first run's start to the last run's last lead day."""
  recipe, _ = forecast_recipe
  steps = '\n[[postprocess]]\nname = "coverage_attrs"\n'
  recipe.write_text(recipe.read_text().replace("\n[save]", f"{steps}\n[save]"))
  result = download(recipe, tmp_path / "out", "--start=2026-11-01", "--end=2026-11-04")
  assert result.exit_code == 0, result.output
  attrs = open_output(tmp_path / "out").attrs
  assert attrs["time_coverage_start"] == "2026-11-01T00:00:00Z"
  assert attrs["time_coverage_end"] == "2026-11-05T00:00:00Z"
  assert attrs["time_coverage_resolution"] == "P1D"


def test_a_run_its_source_lacks_is_nan_and_flagged(forecast_recipe, tmp_path):
  """A run the source does not have (not published, not archived) leaves its dataset NaN in that
  run, flagged; the other datasets are written."""
  recipe, unavailable = forecast_recipe
  unavailable.add(FIRST + DAY)
  result = download(recipe, tmp_path / "out", "--start=2026-11-01", "--end=2026-11-04")
  assert result.exit_code == 0, result.output
  out = open_output(tmp_path / "out")
  assert out.t2m.isel(init_time=1).isnull().all()
  assert out.t2m.isel(init_time=[0, 2]).notnull().all()
  assert out.atmos_forecast.values[:, 0].tolist() == [True, False, True]
  assert out.sst.isel(init_time=1).notnull().all()


def test_the_first_run_needs_every_dataset(forecast_recipe, tmp_path):
  recipe, unavailable = forecast_recipe
  unavailable.add(FIRST)
  result = download(recipe, tmp_path / "out", "--start=2026-11-01", "--end=2026-11-03")
  assert result.exit_code != 0 and "The first run" in result.output


def test_an_extension_adds_the_runs_after_the_last(forecast_recipe, tmp_path):
  """`--extend` adds the runs after the last stored one, a missed day included, and equals one
  download of them all made at the same time; a run stored as its own forecast keeps its flag."""
  recipe, _ = forecast_recipe
  output = tmp_path / "out"
  first = download(recipe, output, "--start=2026-11-01", "--end=2026-11-04")
  assert first.exit_code == 0, first.output
  write_series(tmp_path / "series.zarr", FIRST + 7 * DAY)  # two releases later (Nov 5)
  extended = download(recipe, output, "--extend", "--end=2026-11-06")
  assert extended.exit_code == 0, extended.output
  out = open_output(output)
  assert out.sizes["init_time"] == 5
  assert out.ocean_forecast.values[:, 0].tolist() == [False, False, True, False, True]
  whole = download(recipe, tmp_path / "whole", "--start=2026-11-01", "--end=2026-11-06")
  assert whole.exit_code == 0, whole.output
  reference = open_output(tmp_path / "whole")
  xr.testing.assert_identical(
    out.drop_vars("ocean_forecast"), reference.drop_vars("ocean_forecast")
  )


def test_a_series_ending_before_the_horizon_fails(forecast_recipe, tmp_path):
  """A run whose series does not reach its horizon (its release not out yet) fails, naming the
  dataset, before writing."""
  recipe, _ = forecast_recipe
  output = tmp_path / "out"
  assert download(recipe, output, "--start=2026-11-01", "--end=2026-11-02").exit_code == 0
  write_series(tmp_path / "series.zarr", FIRST + 3 * DAY)  # a day short of run 2's horizon
  result = download(recipe, output, "--extend", "--end=2026-11-03")
  assert result.exit_code != 0
  assert "different valid times: ocean to 2026-11-03T00:00" in result.output
  assert open_output(output).sizes["init_time"] == 1


def test_a_run_shorter_than_those_stored_fails(forecast_recipe, tmp_path):
  """A run whose datasets all end early (here the run source lacks the run, and the series ends a
  day short) has fewer lead times than those stored: it fails, before writing."""
  recipe, unavailable = forecast_recipe
  output = tmp_path / "out"
  assert download(recipe, output, "--start=2026-11-01", "--end=2026-11-02").exit_code == 0
  unavailable.add(FIRST + DAY)
  write_series(tmp_path / "series.zarr", FIRST + 3 * DAY)
  result = download(recipe, output, "--extend", "--end=2026-11-03")
  assert result.exit_code != 0 and "other lead times than those stored" in result.output
  assert open_output(output).sizes["init_time"] == 1


def test_a_failed_run_resumes(forecast_recipe, tmp_path, monkeypatch: pytest.MonkeyPatch):
  """A run that fails leaves the stored ones on the run's branch; `--resume` writes the rest."""
  recipe, _ = forecast_recipe
  real = download_module.as_run
  failures = []

  def failing(block, run, *args, **kwargs):
    if run == FIRST + 2 * DAY and not failures:
      failures.append(run)
      raise RuntimeError("boom")
    return real(block, run, *args, **kwargs)

  monkeypatch.setattr(download_module, "as_run", failing)
  output = tmp_path / "out"
  failed = download(recipe, output, "--start=2026-11-01", "--end=2026-11-04")
  assert failed.exit_code != 0 and "--resume continues" in failed.output
  resumed = download(recipe, output, "--resume", "--start=2026-11-01", "--end=2026-11-04")
  assert resumed.exit_code == 0, resumed.output
  assert open_output(output).sizes["init_time"] == 3


def test_plan_prints_the_first_run(forecast_recipe, tmp_path):
  recipe, _ = forecast_recipe
  args = ("--plan", "--start=2026-11-01", "--end=2026-11-03")
  result = download(recipe, tmp_path / "out", *args)
  assert result.exit_code == 0, result.output
  assert "t2m (init_time, lead_time, lat, lon)" in result.output


@pytest.mark.parametrize(
  ("old", "new", "message"),
  [
    ('[forecast]\nhorizon = "3D"\n', 'step = "1D"\n', "are for forecast recipes"),
    ("[forecast]\n", 'step = "1D"\n[forecast]\n', "every instead"),
    ("", "", None),
  ],
  ids=["no-forecast-table", "step-and-forecast", "valid"],
)
def test_forecast_recipes_are_checked(old, new, message, tmp_path):
  path = tmp_path / "r.toml"
  path.write_text(RECIPE.replace(old, new) if old else RECIPE)
  if message is None:
    assert read_configs(str(path), schema=download_module.DownloadConfig).forecast is not None
    return
  with pytest.raises(Exception, match=message):
    read_configs(str(path), schema=download_module.DownloadConfig)


# A rolling window of runs: each new run drops the oldest beyond `window`.


def with_window(recipe, window: int) -> None:
  text = recipe.read_text()
  lines = [line for line in text.splitlines() if not line.startswith("window = ")]
  text = "\n".join(lines) + "\n"
  recipe.write_text(text.replace('horizon = "3D"\n', f'horizon = "3D"\nwindow = {window}\n'))


def test_a_window_keeps_the_last_runs(forecast_recipe, tmp_path):
  """With `window = 2`, four runs leave the last two: their values, initialization and valid
  times as a download of those two alone; the static dataset stays."""
  recipe, _ = forecast_recipe
  with_window(recipe, 2)
  write_series(tmp_path / "series.zarr", FIRST + 6 * DAY)  # the release of run 4 (Nov 4)
  output = tmp_path / "out"
  result = download(recipe, output, "--start=2026-11-01", "--end=2026-11-05")
  assert result.exit_code == 0, result.output
  out = open_output(output)
  np.testing.assert_array_equal(
    out.init_time.values, pd.date_range(FIRST + 2 * DAY, periods=2, freq="1D").to_numpy()
  )
  assert out.ocean_forecast.values[:, 0].tolist() == [False, True]
  alone = download(recipe, tmp_path / "alone", "--start=2026-11-03", "--end=2026-11-05")
  assert alone.exit_code == 0, alone.output
  xr.testing.assert_identical(out, open_output(tmp_path / "alone"))


def test_a_window_bounds_the_storage(forecast_recipe, tmp_path, monkeypatch: pytest.MonkeyPatch):
  """Runs shifted out of the window go with the versions that held them: with one version kept
  (and no grace period), the repository holds as many chunk objects after each later run."""
  monkeypatch.setattr(icechunk_utils, "GRACE", datetime.timedelta(0))
  # Chunks this small are inlined in the manifests: stored as objects, they can be counted.
  config = icechunk_utils.icechunk.RepositoryConfig
  monkeypatch.setattr(
    icechunk_utils.icechunk,
    "RepositoryConfig",
    lambda **kwargs: config(inline_chunk_threshold_bytes=0, **kwargs),
  )
  recipe, _ = forecast_recipe
  with_window(recipe, 2)
  recipe.write_text(recipe.read_text().replace("[save]\n", "[save]\nkeep_versions = 1\n"))
  write_series(tmp_path / "series.zarr", FIRST + 9 * DAY)
  output = tmp_path / "out"

  def chunks() -> int:
    return sum(1 for f in (output / "chunks").rglob("*") if f.is_file())

  assert download(recipe, output, "--start=2026-11-01", "--end=2026-11-04").exit_code == 0
  counts = [chunks()]
  for end in ("2026-11-05", "2026-11-06", "2026-11-07"):
    extended = download(recipe, output, "--extend", f"--end={end}")
    assert extended.exit_code == 0, extended.output
    counts.append(chunks())
  assert open_output(output).sizes["init_time"] == 2
  # A version is a command's last commit: its earlier commits (here the first command's three
  # runs) go with the next command's retention; from then on, as many objects after each run.
  assert counts[1] > 0 and counts[1:] == [counts[1]] * 3, counts


def test_a_window_grows_and_shrinks(forecast_recipe, tmp_path):
  """The window is out of the recipe's fingerprint: lengthened, runs are appended until it is
  full; shortened, the oldest runs beyond it go at the next run."""
  recipe, _ = forecast_recipe
  write_series(tmp_path / "series.zarr", FIRST + 8 * DAY)
  output = tmp_path / "out"
  with_window(recipe, 2)
  assert download(recipe, output, "--start=2026-11-01", "--end=2026-11-04").exit_code == 0
  assert open_output(output).sizes["init_time"] == 2
  with_window(recipe, 3)
  grown = download(recipe, output, "--extend", "--end=2026-11-05")
  assert grown.exit_code == 0, grown.output
  inits = open_output(output).init_time.values
  np.testing.assert_array_equal(inits, pd.date_range(FIRST + DAY, periods=3, freq="1D").to_numpy())
  with_window(recipe, 2)
  shrunk = download(recipe, output, "--extend", "--end=2026-11-06")
  assert shrunk.exit_code == 0, shrunk.output
  out = open_output(output)
  expected = pd.date_range(FIRST + 3 * DAY, periods=2, freq="1D").to_numpy()
  np.testing.assert_array_equal(out.init_time.values, expected)
  np.testing.assert_array_equal(
    out.valid_time.values, expected[:, None] + out.lead_time.values[None, :]
  )
  run = pd.Timestamp(expected[0]).to_pydatetime()
  hours = [np.datetime64(run) + np.timedelta64(h, "h") for h in (0, 6, 12, 18)]
  assert float(out.t2m.isel(init_time=0, lead_time=0, lat=0, lon=0)) == np.mean(
    [atmos_value(run, h) for h in hours]
  )


def test_a_failed_rolled_run_resumes(forecast_recipe, tmp_path, monkeypatch: pytest.MonkeyPatch):
  """A run failing once the window is full leaves `main` as it was; `--resume` completes it."""
  recipe, _ = forecast_recipe
  with_window(recipe, 2)
  write_series(tmp_path / "series.zarr", FIRST + 6 * DAY)
  real = download_module.as_run
  failures = []

  def failing(block, run, *args, **kwargs):
    if run == FIRST + 3 * DAY and not failures:
      failures.append(run)
      raise RuntimeError("boom")
    return real(block, run, *args, **kwargs)

  monkeypatch.setattr(download_module, "as_run", failing)
  output = tmp_path / "out"
  failed = download(recipe, output, "--start=2026-11-01", "--end=2026-11-05")
  assert failed.exit_code != 0
  resumed = download(recipe, output, "--resume", "--start=2026-11-01", "--end=2026-11-05")
  assert resumed.exit_code == 0, resumed.output
  inits = open_output(output).init_time.values
  np.testing.assert_array_equal(
    inits, pd.date_range(FIRST + 2 * DAY, periods=2, freq="1D").to_numpy()
  )


@pytest.mark.parametrize(
  "save",
  ["chunks = {lat = 1}", "chunks = 1", "chunks = {init_time = 1}\nshards = {init_time = 2}"],
  ids=["no-init-chunks", "not-a-table", "shards-of-two-runs"],
)
def test_a_window_needs_one_run_per_object(save, tmp_path):
  path = tmp_path / "r.toml"
  text = RECIPE.replace('horizon = "3D"\n', 'horizon = "3D"\nwindow = 2\n')
  path.write_text(text.replace("chunks = {init_time = 1}", save))
  with pytest.raises(Exception, match="one run per stored object"):
    read_configs(str(path), schema=download_module.DownloadConfig)


def test_a_run_keeps_the_static_coordinates():
  """Coordinates without time on dimensions of no variable (cell bounds on `bnds`) stay in the
  run: they were lost, and the next run found the stored ones missing."""
  block = xr.Dataset(
    {"t": (("time", "lat"), np.ones((3, 2))), "depth": ("lat", np.ones(2))},
    coords={
      "time": pd.date_range(FIRST, periods=3, freq="1D"),
      "lat": LAT,
      "lat_bnds": (("lat", "bnds"), np.array([[-0.5, 0.5], [0.5, 1.5]])),
    },
  )
  config = forecast.ForecastConfig(horizon="3D")
  run = forecast.as_run(block, FIRST, config, "time", {})
  assert run.lat_bnds.dims == ("lat", "bnds")
  np.testing.assert_array_equal(run.lat_bnds.values, block.lat_bnds.values)
  assert run.t.dims == ("init_time", "lead_time", "lat") and run.depth.dims == ("lat",)


def test_stored_variables_missing_without_a_missing_source_fail():
  """NaN fills only the datasets whose source had no such run: a run lacking stored variables
  otherwise is an error."""
  dims = ("init_time", "lead_time")
  stored = xr.Dataset(
    {"a": (dims, np.ones((1, 2))), "b": (dims, np.ones((1, 2)))},
    coords={
      "init_time": [np.datetime64(FIRST, "ns")],
      "lead_time": np.array([0, 1], "timedelta64[D]").astype("timedelta64[ns]"),
    },
  )
  run = stored[["a"]].assign_coords(init_time=[np.datetime64(FIRST + DAY, "ns")])
  with pytest.raises(ValueError, match="every source had it"):
    forecast.with_missing(run, stored, "init_time", [])
  filled = forecast.with_missing(run, stored, "init_time", ["source_of_b"])
  assert filled.b.isnull().all() and filled.a.notnull().all()


def test_runs_less_than_a_day_apart(forecast_recipe, tmp_path):
  """Runs every 12 hours keep their initialization and valid times (the units are explicit: from
  the first run's one value, xarray would infer days), appended and rolled."""
  recipe, _ = forecast_recipe
  text = recipe.read_text()
  atmos_only = text[: text.index("[datasets.ocean]")]
  atmos_only = atmos_only.replace('horizon = "3D"\n', 'horizon = "1D"\nevery = "12h"\nwindow = 3\n')
  atmos_only = atmos_only.replace('time = "1D"', 'time = "6h"')
  recipe.write_text(atmos_only + "\n[save]\nchunks = {init_time = 1}\n")
  output = tmp_path / "out"
  result = download(recipe, output, "--start=2026-11-01", "--end=2026-11-03")
  assert result.exit_code == 0, result.output
  out = open_output(output)
  expected = pd.date_range("2026-11-01T12:00", periods=3, freq="12h").to_numpy()
  np.testing.assert_array_equal(out.init_time.values, expected)
  assert (out.lead_time.values / np.timedelta64(1, "h")).tolist() == [0, 6, 12, 18]
  np.testing.assert_array_equal(
    out.valid_time.values, expected[:, None] + out.lead_time.values[None, :]
  )


def test_an_end_off_the_runs_grid_is_refused(forecast_recipe, tmp_path):
  recipe, _ = forecast_recipe
  result = download(recipe, tmp_path / "out", "--start=2026-11-01", "--end=2026-11-03T06:00:00")
  assert result.exit_code != 0 and "not a whole number of runs" in result.output


def test_checks_leave_out_a_missing_source(forecast_recipe, tmp_path):
  """`ensure_no_nans` leaves out the variables of a source without the run (stored as NaN,
  flagged), and checks the others."""
  recipe, unavailable = forecast_recipe
  checks = '\n[checks.ensure_no_nans]\ndepth = ["t2m", "sst"]\n'
  recipe.write_text(recipe.read_text().replace("\n[save]", f"{checks}\n[save]"))
  unavailable.add(FIRST + DAY)
  result = download(recipe, tmp_path / "out", "--start=2026-11-01", "--end=2026-11-03")
  assert result.exit_code == 0, result.output
  assert open_output(tmp_path / "out").t2m.isel(init_time=1).isnull().all()


def test_engines_reading_only_runs_need_a_runs_dataset():
  part = {"filename_or_obj": "s3://ecmwf-forecasts", "engine": "ecmwfopendata", "param": "2t"}
  with pytest.raises(ValueError, match="reads forecast runs only"):
    DatasetConfig.model_validate({"parts": [part]})


def test_static_parts_say_nothing_of_a_series_end(tmp_path):
  """A part read at one instant (`static_time`) is left out of the series' ends."""
  time = pd.date_range(FIRST - 5 * DAY, periods=30, freq="1D")
  xr.Dataset(
    {"mask": (("time", "lat"), np.ones((30, 2)))}, coords={"time": time, "lat": LAT}
  ).to_zarr(tmp_path / "m.zarr", consolidated=False)
  part = {"filename_or_obj": str(tmp_path / "m.zarr"), "engine": "zarr", "static_time": FIRST}
  config = DatasetConfig.model_validate({"parts": [part]})
  ends: list = []
  download_and_process(config, FIRST, FIRST + 3 * DAY, part_ends=ends)
  assert ends == []
