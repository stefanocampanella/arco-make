# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Climatology and anomaly variance against a direct NumPy computation."""

import datetime

import dask
import dask.array
import numpy as np
import pandas as pd
import pydantic
import pytest
import xarray as xr

from arcomake.climatology import ClimatologyConfig
from arcomake.climatology.binning import window_weights
from arcomake.climatology.graph import _compute_climatology
from arcomake.climatology.staging import _staging_chunks
from arcomake.climatology.streamed import _compute_series, _read_block

PERIOD = 6
N_PERIODS = 5


def series(seed: int = 0) -> xr.Dataset:
  """N_PERIODS periods of PERIOD steps at 3 points, with a seasonal cycle and noise."""
  rng = np.random.default_rng(seed)
  steps = np.arange(PERIOD * N_PERIODS)
  cycle = np.sin(2 * np.pi * steps / PERIOD)[:, None]
  values = cycle + rng.standard_normal((steps.size, 3))
  return xr.Dataset({"x": (("time", "point"), values)}, coords={"time": steps})


def reference(values: np.ndarray, window: int, weighting: str) -> tuple[np.ndarray, np.ndarray]:
  """Weighted mean and unbiased (reliability-weights) variance of each bin, ignoring NaN.

  Bin b pools the samples of bins b + k (k in the centred window, wrapping around the period)
  from every period, each with the weight of its offset k.
  """
  folded = values.reshape(N_PERIODS, PERIOD, -1)
  half = window // 2
  offsets = list(range(-half, window - half))
  weights = window_weights(offsets, weighting)  # type: ignore[arg-type]
  mean = np.empty(folded.shape[1:])
  var = np.empty(folded.shape[1:])
  for b in range(PERIOD):
    for p in range(folded.shape[2]):
      x = np.concatenate([folded[:, (b + k) % PERIOD, p] for k in offsets])
      w = np.repeat(weights, N_PERIODS)
      valid = ~np.isnan(x)
      x, w = x[valid], w[valid]
      v1, v2 = w.sum(), (w**2).sum()
      mean[b, p] = (w * x).sum() / v1
      var[b, p] = (w * (x - mean[b, p]) ** 2).sum() / (v1 - v2 / v1)
  return mean, var


@pytest.mark.parametrize(
  ("window", "weighting"), [(1, "constant"), (3, "constant"), (3, "gaussian"), (5, "geometric")]
)
def test_climatology_matches_numpy(window: int, weighting: str):
  """On complete data the climatology equals the pooled weighted statistics of each bin."""
  ds = series()
  mean, var = _compute_climatology(ds, period=PERIOD, window=window, weighting=weighting)  # type: ignore[arg-type]
  ref_mean, ref_var = reference(ds.x.values, window, weighting)
  np.testing.assert_allclose(mean.x.values, ref_mean, rtol=1e-12)
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)
  np.testing.assert_array_equal(mean.time.values, np.arange(PERIOD))


def test_climatology_variance_with_a_missing_year():
  """A value missing in one period does not bias the variance of its bin."""
  ds = series()
  values = ds.x.values.copy()
  values[2 * PERIOD + 1, 0] = np.nan  # bin 1, period 2, point 0
  ds = ds.assign(x=(ds.x.dims, values))
  _, var = _compute_climatology(ds, period=PERIOD, window=1)
  _, ref_var = reference(values, 1, "constant")
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)


def test_climatology_window_does_not_spread_nan():
  """A bin missing in every period leaves the smoothed neighbours finite."""
  ds = series()
  values = ds.x.values.copy()
  values[3::PERIOD, 0] = np.nan  # bin 3 always missing at point 0
  ds = ds.assign(x=(ds.x.dims, values))
  mean, _ = _compute_climatology(ds, period=PERIOD, window=3)
  ref_mean, _ = reference(values, 3, "constant")
  np.testing.assert_allclose(mean.x.values, ref_mean, rtol=1e-12)


def test_climatology_with_missing_last_steps():
  """Steps missing at the end of the last period (NaN) count in no bin."""
  ds = series()
  values = ds.x.values.copy()
  values[-2:] = np.nan
  _, var = _compute_climatology(ds.assign(x=(ds.x.dims, values)), period=PERIOD, window=3)
  _, ref_var = reference(values, 3, "constant")
  np.testing.assert_allclose(var.x.values, ref_var, rtol=1e-12)


@pytest.mark.parametrize("path", ["graph", "streaming"])
def test_position_bins_need_whole_periods(path: str):
  """Position bins refuse a partial period: leap days or partial years would shift later bins.
  Streaming did not check, and binned the steps of the partial period."""
  partial = series().isel(time=slice(0, -2))
  with pytest.raises(ValueError, match="whole number of periods"):
    if path == "graph":
      _compute_climatology(partial, period=PERIOD)
    else:
      _compute_series(partial, PERIOD, 1, "constant", None, "time", "position")


def daily(start: str, end: str, seed: int = 0) -> xr.Dataset:
  rng = np.random.default_rng(seed)
  time = pd.date_range(start, end, freq="1D", inclusive="left")
  return xr.Dataset({"x": ("time", rng.standard_normal(time.size))}, coords={"time": time})


def test_day_of_year_bins_put_feb_29_with_feb_28():
  """Feb 29 counts in Feb 28's bin; later days of a leap year take the same date's bin."""
  ds = daily("2019-01-01", "2021-01-01")  # 2020 is a leap year
  mean, var = _compute_climatology(ds, period=365, bins="day_of_year")
  x = ds.x.to_series()
  feb_28 = x[(x.index.month == 2) & (x.index.day.isin([28, 29]))]
  np.testing.assert_allclose(mean.x.values[58], feb_28.mean(), rtol=1e-12)
  np.testing.assert_allclose(var.x.values[58], feb_28.var(ddof=1), rtol=1e-12)
  mar_1 = x[(x.index.month == 3) & (x.index.day == 1)]
  np.testing.assert_allclose(mean.x.values[59], mar_1.mean(), rtol=1e-12)
  dec_31 = x[(x.index.month == 12) & (x.index.day == 31)]
  np.testing.assert_allclose(mean.x.values[364], dec_31.mean(), rtol=1e-12)
  assert mean.sizes["time"] == 365


def test_day_of_year_bins_match_position_bins_without_leap_days():
  """Without Feb 29 and with whole years, day-of-year bins are position bins."""
  ds = daily("2021-01-01", "2024-01-01")
  by_day, var_day = _compute_climatology(ds, period=365, window=5, bins="day_of_year")
  by_position, var_position = _compute_climatology(ds, period=365, window=5)
  np.testing.assert_allclose(by_day.x.values, by_position.x.values, rtol=1e-12)
  np.testing.assert_allclose(var_day.x.values, var_position.x.values, rtol=1e-12)


def test_climatology_keeps_the_input_dtype():
  """float32 inputs give float32 statistics (counts are integers internally)."""
  ds = series().astype("float32")
  mean, var = _compute_climatology(ds, period=PERIOD, window=3)
  assert mean.x.dtype == var.x.dtype == np.float32


def test_in_memory_series_equal_the_lazy_statistics():
  """One series at a time (S1): one read pass, then the window in NumPy, by latitude bands."""
  rng = np.random.default_rng(1)
  time = pd.date_range("2019-03-01", "2021-07-01", freq="1D", inclusive="left")
  values = rng.standard_normal((time.size, 70, 3)).astype("float32")
  values[10:40, :5] = np.nan  # gaps
  values[:, 7] = np.nan  # a row without data
  assert time[365] == pd.Timestamp("2020-02-29")
  values[365, 20:30] = np.nan  # Feb 29 missing where Feb 28 has values
  ds = xr.Dataset({"x": (("time", "lat", "lon"), values)}, coords={"time": time}).chunk(time=50)
  args = {"period": 365, "window": 7, "weighting": "gaussian", "weighting_scale": 2.0}
  mean, var = _compute_climatology(ds, bins="day_of_year", **args)
  mean_s, var_s = _compute_series(ds, time_dim="time", bins="day_of_year", **args)
  # Welford in float64 against float32 reductions: equal to float32 precision.
  np.testing.assert_allclose(mean_s.x.values, mean.x.values, rtol=1e-5, atol=1e-6, equal_nan=True)
  np.testing.assert_allclose(var_s.x.values, var.x.values, rtol=1e-5, atol=1e-6, equal_nan=True)
  assert mean_s.x.dtype == np.float32 and mean_s.x.dims == ("time", "lat", "lon")


def test_read_blocks_follow_the_input_chunks():
  """Blocks of about 1 GB, a whole number of the input's time chunks (or shards), at least one."""
  da = xr.DataArray(np.zeros((1000, 721, 1440), dtype="float32"), dims=("time", "lat", "lon"))
  assert _read_block(da, "time") == 240  # 4.15 MB per step
  assert _read_block(da.chunk(time=50), "time") == 200
  assert _read_block(da.chunk(time=500), "time") == 500  # never less than one chunk


def test_window_weights():
  """Constant weights are ones; Gaussian and geometric weights decay symmetrically from the centre."""
  np.testing.assert_array_equal(window_weights([-1, 0, 1], "constant"), [1, 1, 1])
  gaussian = window_weights([-2, -1, 0, 1, 2], "gaussian")
  assert gaussian[2] == 1.0 and gaussian[1] == gaussian[3] > gaussian[0] == gaussian[4]
  geometric = window_weights([-2, -1, 0, 1, 2], "geometric")
  np.testing.assert_allclose(geometric[0], 0.1)
  with pytest.raises(ValueError):
    window_weights([-1, 0, 1], "geometric", scale=2.0)


@pytest.mark.parametrize(
  ("table", "message"),
  [
    ({"reference_period": {"start": "2019-01-01", "end": "1993-01-01"}}, "before end"),
    (
      {
        "reference_period": {
          "start": "1993-01-01T00:00:00+02:00",
          "end": "2019-01-01",
        }
      },
      "without a time zone, or in UTC",
    ),
    ({"bins": "day_of_year", "period": 366}, "needs period = 365"),
  ],
)
def test_climatology_recipes_are_checked_when_loaded(table: dict, message: str):
  """Inconsistent settings fail when the recipe is read, not after the cluster has started."""
  base = {"period": 365, "window": 1, "weighting": "constant", "read": {}, "save": {}}
  with pytest.raises(pydantic.ValidationError, match=message):
    ClimatologyConfig.model_validate(base | table)


def test_series_at_once_fit_in_memory():
  """As many series at a time as the largest fits in the budget's memory, at least one, at most
  the series and the threads; a series holds the most of its phases: reading, the window,
  writing."""
  from arcomake.chunk_plan import USABLE, Budget
  from arcomake.climatology.streamed import _read_block, _series, _series_at_once, _series_bytes

  time = pd.date_range("2001-01-01", periods=730, freq="1D")
  ds = xr.Dataset(
    {
      "t3": (
        ("time", "depth", "x"),
        dask.array.zeros((730, 3, 1000), dtype="float32", chunks=(10, 3, -1)),
      ),
      "t2": (("time", "x"), dask.array.zeros((730, 1000), dtype="float64", chunks=(10, -1))),
    },
    coords={"time": time},
  )
  series = _series(ds, ["depth"])
  assert len(series) == 4
  t3 = _series_bytes(ds.t3, {"depth": 0}, 365, "time")
  t2 = _series_bytes(ds.t2, {}, 365, "time")
  assert _read_block(ds.t3.isel(depth=slice(0, 1)), "time") > 730  # about 1 GB: the whole series
  reading = 365 * 1000 * 20 + 3 * 730 * 1000 * 4
  window = 365 * 1000 * (20 + 2 * 4) * 2  # statistics, results; one band of all 1000 points
  assert t3 == max(reading, window, 365 * 1000 * 4 * 4)
  assert t2 > t3  # float64: larger results and blocks
  at_once = [
    _series_at_once(ds, series, 365, "time", Budget(int(usable / USABLE) + 1, threads))
    for usable, threads in [(t2 // 2, 32), (2 * t2, 32), (100 * t2, 32), (100 * t2, 3)]
  ]
  assert at_once == [1, 2, 4, 3]


def test_in_memory_series_reject_sub_daily_data():
  """The per-series path checks day-of-year bins like the Dask path (one step per day)."""
  hourly = xr.Dataset(
    {"x": ("time", np.zeros(48))},
    coords={"time": pd.date_range("2020-01-01", periods=48, freq="1h")},
  )
  with pytest.raises(ValueError, match="one time step per day"):
    _compute_series(
      hourly,
      period=365,
      window=1,
      weighting="constant",
      weighting_scale=None,
      time_dim="time",
      bins="day_of_year",
    )


def test_read_blocks_follow_stored_chunks_without_dask():
  """Without Dask chunks, blocks follow the stored chunks the zarr backend reports."""
  da = xr.DataArray(np.zeros((1000, 721, 1440), dtype="float32"), dims=("time", "lat", "lon"))
  da.encoding["preferred_chunks"] = {"time": 50, "lat": 721, "lon": 1440}
  assert _read_block(da, "time") == 200


def test_the_window_does_not_tile_the_grid():
  """Folded data are chunked once, every bin in a chunk: the window needs no rechunk. Left to
  Dask, its rechunk to whole bins cut the 1 deg grid into 7 x 7 tiles (747,727 tasks for one
  variable of 49 levels)."""
  time = pd.date_range("2013-03-01", periods=8, freq="1D")
  values = np.zeros((8, 3, 181, 360), dtype="float32")
  ds = xr.Dataset(
    {"t": (("time", "depth", "lat", "lon"), values)},
    coords={
      "time": time,
      "depth": [0.5, 1.5, 2.5],
      "lat": np.arange(-90.0, 91.0),
      "lon": np.arange(360.0),
    },
  ).chunk({"time": 8, "depth": 1, "lat": -1, "lon": -1})
  mean, _ = _compute_climatology(ds, 365, 29, "gaussian", 7.0, "time", "day_of_year")
  assert mean.t.chunks == ((365,), (1, 1, 1), (181,), (360,))
  assert len(mean.t.__dask_graph__()) < 3000


def test_the_statistics_of_a_level_read_only_that_level():
  """Each level's statistics need only that level's values, so levels are computed one after the
  other. A two-pass variance tied to a checkpoint of every level's mean
  (`dask.graph_manipulation.bind`) computed the statistics of all levels before any window and
  held them (3.5 GiB for 10 levels of 8 days at 1 deg, against 0.44 GiB now)."""
  read = set()

  def tagged(block: np.ndarray, block_info: dict | None = None) -> np.ndarray:
    if block_info is not None:
      read.add(block_info[None]["chunk-location"][1])
    return block

  values = dask.array.zeros((400, 2, 4), chunks=(50, 1, 4), dtype="float32")
  values = values.map_blocks(tagged, meta=np.array((), dtype="float32"))
  time = pd.date_range("2019-01-01", periods=400, freq="1D")
  ds = xr.Dataset({"t": (("time", "depth", "x"), values)}, coords={"time": time})
  mean, var = _compute_climatology(ds, 365, 7, "gaussian", 2.0, "time", "day_of_year")
  for statistic in (mean, var):
    read.clear()
    statistic.t.isel(depth=0).compute()
    assert read == {0}


def test_threads_share_one_window_kernel(monkeypatch):
  """Threads asking for the kernel at once (Dask's, on their first chunks) get the same one,
  made (and so compiled) once."""
  import threading
  import time

  import numba

  from arcomake import climatology

  made = []
  njit = numba.njit

  def slow_njit(*args, **kwargs):
    made.append(1)
    time.sleep(0.05)  # widens the window in which other threads miss the cache
    return njit(*args, **kwargs)

  monkeypatch.setattr(numba, "njit", slow_njit)
  climatology.kernels._window_kernel.__wrapped__.cache_clear()
  start = threading.Barrier(8)
  kernels = []

  def ask() -> None:
    start.wait()
    kernels.append(climatology.kernels._window_kernel())

  threads = [threading.Thread(target=ask) for _ in range(8)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  # Later tests make their own, with numba's njit.
  climatology.kernels._window_kernel.__wrapped__.cache_clear()
  assert len(made) == 1 and len({id(kernel) for kernel in kernels}) == 1


def test_variables_chunked_differently_share_the_window_tiles():
  """Variables chunked differently (`Dataset.chunks` raised for them) or of different types get
  the same tiles for the window, and so do Feb 29's statistics (combining them in the input's
  tiles cut every bin's statistics into the intersections of both tilings)."""
  time = pd.date_range("2019-03-01", "2021-03-01", freq="1D", inclusive="left")  # Feb 29, 2020
  rng = np.random.default_rng(3)
  t = rng.standard_normal((time.size, 30, 20)).astype("float32")
  z = rng.standard_normal((time.size, 30, 20))  # float64: other "auto" tiles
  dims = ("time", "lat", "lon")
  source = xr.Dataset({"t": (dims, t), "z": (dims, z)}, coords={"time": time})
  ds = source.assign(t=source.t.chunk(time=50, lat=7), z=source.z.chunk(time=50, lat=11))
  with dask.config.set({"array.chunk-size": "64KiB"}):
    statistics = _compute_climatology(ds, 365, 5, "gaussian", 2.0, "time", "day_of_year")
  expected = _compute_climatology(source, 365, 5, "gaussian", 2.0, "time", "day_of_year")
  for out, reference in zip(statistics, expected, strict=True):
    tilings = {out[name].chunksizes["lat"] for name in out.data_vars}
    assert len(tilings) == 1
    (lat,) = tilings
    assert len(lat) > 1 and len(set(lat[:-1])) == 1  # tiled, regularly: no intersections
    xr.testing.assert_allclose(out.compute(), reference)


DIMS = ("time", "depth", "lat", "lon")
MIB = 2**20


def test_staging_one_bin_chunks_are_grouped_and_split_by_level():
  """One-bin, one-level stored chunks at 0.25 deg: the copy takes 32 bins of a level per task
  (128 MiB), and the store the same, by level even where Dask holds all levels (each of its
  chunks read once, not once per level)."""
  stored, copy = _staging_chunks(
    DIMS, (365, 49, 721, 1440), (365, 49, 92, 1440), (1, 1, 721, 1440), 4, "time", 128 * MIB
  )
  assert copy == (32, 1, 721, 1440)
  assert stored == (32, 1, 92, 1440)


def test_staging_shards_spanning_levels_are_copied_one_per_task():
  """Shards of 8 bins and all levels (1.6 GB at 0.25 deg) are copied one per task; the store
  keeps Dask's level and tiles (they do not divide the shard's)."""
  stored, copy = _staging_chunks(
    DIMS, (365, 49, 721, 1440), (365, 1, 92, 1440), (8, 49, 721, 1440), 4, "time", 128 * MIB
  )
  assert copy == (8, 49, 721, 1440)
  assert stored == (8, 1, 92, 1440)
  # Without a stored layout, Dask's chunks throughout.
  assert _staging_chunks(DIMS, (365, 3, 4, 4), (365, 1, 4, 4), None, 4, "time", MIB) == (
    (365, 1, 4, 4),
    None,
  )


def test_temporary_store_warns_about_memory_and_disk(monkeypatch, caplog):
  """Stored units above a task's share of memory, and a store larger than the free disk, are
  warned about before computing."""
  import collections
  import logging

  from arcomake import climatology
  from arcomake.chunk_plan import Budget
  from arcomake.dataset_utils import SaveConfig
  from arcomake.temporary_utils import TempDirectoryRegistry

  monkeypatch.setattr(climatology.staging, "detect_budget", lambda: Budget(64 * 1024, 1))
  usage = collections.namedtuple("usage", "total used free")
  monkeypatch.setattr(climatology.staging.shutil, "disk_usage", lambda path: usage(1, 1, 0))
  time = np.arange(8)
  ds = xr.Dataset({"t": (("time", "x"), np.zeros((8, 1000), "float32"))}, coords={"time": time})
  save = SaveConfig.model_validate({"chunks": {"time": 8, "x": -1}})
  with caplog.at_level(logging.WARNING), TempDirectoryRegistry():
    climatology.staging.stage({"g": ds.chunk()}, save, "time")
  assert "may not fit" in caplog.text and "chunks" in caplog.text


def test_copies_run_on_local_threads_at_most_their_cap(monkeypatch):
  """The copy runs in this process on local threads, capped: Dask's distributed schedulers ignore
  `num_workers`, and every copy task holds a whole stored unit."""
  from arcomake import climatology
  from arcomake.dataset_utils import SaveConfig
  from arcomake.temporary_utils import TempDirectoryRegistry

  seen = {}

  def write_tree(groups, run, save, **kwargs):
    seen["scheduler"] = dask.config.get("scheduler")
    seen["threads"] = dask.config.get("num_workers")

  monkeypatch.setattr(climatology.staging, "write_tree", write_tree)
  ds = xr.Dataset({"t": (("time", "x"), np.zeros((8, 10), "float32"))}, coords={"time": range(8)})
  with TempDirectoryRegistry(), dask.config.set(scheduler="processes"):
    stage = climatology.staging.stage({"g": ds.chunk()}, SaveConfig(), "time")
    stage.write()
    stage.threads = 3
    copy = climatology.staging.copy
    copy(stage, None, SaveConfig(), label="x", metadata={})  # type: ignore[arg-type]
  assert seen == {"scheduler": "threads", "threads": 3}


def test_copy_threads_count_the_chunks_a_task_reads(monkeypatch):
  """A copy task reads whole chunks of the store: where they are larger than the stored unit (it
  does not divide Dask's chunks), the threads are counted for them."""
  from arcomake import climatology
  from arcomake.chunk_plan import COPIES, Budget
  from arcomake.dataset_utils import SaveConfig
  from arcomake.temporary_utils import TempDirectoryRegistry

  ds = xr.Dataset({"t": (("time", "x"), np.zeros((8, 1000), "float32"))}, coords={"time": range(8)})
  save = SaveConfig.model_validate({"chunks": {"time": 8, "x": 300}})  # 300 does not divide 1000
  whole = 8 * 1000 * 4  # the store's chunk: Dask's, the whole of x
  # Room for 2 tasks holding the store's chunks (5 holding the stored units of 300).
  budget = Budget(int(2.5 * COPIES * whole / 0.8), 32)
  monkeypatch.setattr(climatology.staging, "detect_budget", lambda: budget)
  with TempDirectoryRegistry():
    stage = climatology.staging.stage({"g": ds.chunk()}, save, "time")
  assert stage.threads == 2


def test_copy_threads_hold_their_units():
  """The copy runs with as many threads as hold `COPIES` times their stored unit within the
  budget (32 copy tasks of 1.6 GB shards did not fit in 110 GB), at least one."""
  from arcomake import climatology
  from arcomake.chunk_plan import COPIES, Budget

  shard = 8 * 49 * 721 * 1440 * 4  # 1.6 GB
  budget = Budget(110 * 2**30, 32)
  threads = climatology.staging._copy_threads([shard], budget)
  assert threads is not None and 1 <= threads < 32
  assert threads * COPIES * shard <= budget.usable
  assert climatology.staging._copy_threads([2**20], budget) is None  # small units: all of Dask's
  assert climatology.staging._copy_threads([budget.usable], budget) == 1


def test_recipe_dates_in_utc_are_stamps_of_the_time_axis():
  """A reference period in UTC is the naive stamp (it was refused), as in download recipes and
  the time check, whose dates in another offset are refused (they were compared with the naive
  time axis)."""
  from arcomake.checks import ValidTimeCoordinateCheck

  utc = datetime.datetime(1993, 1, 1, tzinfo=datetime.UTC)
  base = {"period": 365, "window": 1, "weighting": "constant", "read": {}, "save": {}}
  configs = ClimatologyConfig.model_validate(
    base | {"reference_period": {"start": utc, "end": "2019-01-01"}}
  )
  assert configs.reference_period is not None
  assert configs.reference_period.start == datetime.datetime(1993, 1, 1)
  assert ValidTimeCoordinateCheck(start=utc).start == datetime.datetime(1993, 1, 1)
  with pytest.raises(pydantic.ValidationError, match="without a time zone, or in UTC"):
    ValidTimeCoordinateCheck(end="2020-01-01T00:00:00+02:00")  # ty: ignore[invalid-argument-type]
