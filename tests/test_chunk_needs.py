# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Steps declare what they need from their input's chunks, and the needs of a dataset's steps are
gathered before its parts are chunked (6.1, point 3)."""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arcomake.chunk_plan import Budget, Needs, plan, shapes
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.steps import needs_for


def steps(*tables: dict) -> list[ProcessingStepConfig]:
  return [ProcessingStepConfig.model_validate(table) for table in tables]


def hourly(times) -> xr.Dataset:
  return xr.Dataset(
    {"t2m": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 4), np.float32))},
    coords={"time": times, "latitude": [0.0, 1.0, 2.0], "longitude": [0.0, 1.0, 2.0, 3.0]},
  )


HOURS = pd.date_range("2013-01-01", periods=48, freq="h")


@pytest.mark.parametrize(
  ("frequency", "times", "multiple"),
  [
    ("1D", HOURS, {"time": 24}),
    ("1 D", HOURS, {"time": 24}),
    ("24h", HOURS, {"time": 24}),
    ("1D", pd.date_range("2013-01-01", periods=16, freq="3h"), {"time": 8}),
    ("6h", HOURS, {"time": 6}),
    ("1MS", HOURS, {}),  # months: bins of no fixed length
    ("1W", HOURS, {}),
    ("1D", HOURS.delete(5), {}),  # a missing hour: steps are not evenly spaced
    ("1D", HOURS[:1], {}),
    ("30min", HOURS, {}),  # bins shorter than a step
    (
      "1D",
      xr.date_range("2013-01-01", periods=48, freq="h", calendar="noleap", use_cftime=True),
      {"time": 24},
    ),
  ],
)
def test_resampling_needs_whole_bins(frequency, times, multiple):
  """A fixed frequency over evenly spaced stamps: chunks of whole bins (24 hourly steps a day)."""
  table = {"name": "resample", "time": frequency, "reduce": "mean", "label": "left"}
  assert needs_for(steps(table), hourly(times)) == Needs(multiple=multiple)


def test_cftime_stamps_are_supported():
  times = [cftime.DatetimeNoLeap(2013, 1, 1, h) for h in range(24)]
  table = {"name": "resample", "time": "1D", "reduce": "sum"}
  assert needs_for(steps(table), hourly(times)).multiple == {"time": 24}


def test_regrid_needs_whole_horizontal_dimensions():
  table = {
    "name": "regrid",
    "grid": {
      "south": 0,
      "north": 2,
      "west": 0,
      "east": 3,
      "resolution_lat": 1,
      "resolution_lon": 1,
    },
    "latitude_dim": "latitude",
  }
  assert needs_for(steps(table), hourly(HOURS)).whole == {"latitude", "longitude"}


def test_get_bottom_gathers_depth():
  table = {"name": "get_bottom", "depth_dim": "depth"}
  assert needs_for(steps(table), hourly(HOURS)).gathered == {"depth"}


def test_needs_of_every_step_are_merged():
  """Steps without needs (here a Dataset method) add none."""
  tables = (
    {"name": "rename", "name_dict": {"t2m": "2t"}},
    {"name": "resample", "time": "1D", "reduce": "mean"},
    {"name": "get_bottom", "depth_dim": "depth"},
  )
  assert needs_for(steps(*tables), hourly(HOURS)) == Needs(
    gathered=frozenset({"depth"}), multiple={"time": 24}
  )


def test_needs_on_renamed_dimensions_are_ignored_by_the_plan():
  """After `rename_coordinates`, regrid names dimensions the opened part lacks."""
  table = {
    "name": "regrid",
    "grid": {
      "south": 0,
      "north": 2,
      "west": 0,
      "east": 3,
      "resolution_lat": 1,
      "resolution_lon": 1,
    },
    "latitude_dim": "lat",
    "longitude_dim": "lon",
  }
  ds = hourly(HOURS)
  needs = needs_for(steps(table), ds)
  assert needs.whole == {"lat", "lon"}
  result = plan(shapes(ds), Budget(memory_limit=10**9, threads=1), needs)
  assert set(result.chunks) == {"time", "latitude", "longitude"}


def test_a_dataset_resampled_to_days_is_chunked_in_days():
  """The needs and the plan together: hourly data resampled to days, read in whole days."""
  ds = hourly(pd.date_range("2013-01-01", periods=24 * 10, freq="h"))
  needs = needs_for(steps({"name": "resample", "time": "1D", "reduce": "mean"}), ds)
  budget = Budget(memory_limit=10**4 * 32 * 4, threads=32)  # about 10 KB a chunk: 48 hours fit
  chunks = plan(shapes(ds), budget, needs).chunks
  assert chunks["time"] % 24 == 0
  assert chunks["time"] == 24 * (budget.target // (24 * 3 * 4 * 4))


@pytest.mark.parametrize(
  ("extra", "times", "multiple"),
  [
    ({"closed": "left"}, HOURS, {"time": 24}),
    ({"closed": "right"}, HOURS, {}),  # (D-1 00:00, D 00:00]: bins end where chunks start
    ({"offset": "12h"}, HOURS, {}),
    ({"origin": "epoch"}, HOURS, {}),
    ({}, HOURS[1:], {}),  # a block starting at 01:00: chunks from there straddle the days
    (
      {},
      xr.date_range("2013-01-01T01:00", periods=47, freq="h", calendar="noleap", use_cftime=True),
      {},
    ),
  ],
)
def test_resampling_needs_bins_starting_at_the_first_stamp(extra, times, multiple):
  """Chunks start at the block's first stamp: whole bins only if a bin starts there too."""
  table = {"name": "resample", "time": "1D", "reduce": "mean", **extra}
  assert needs_for(steps(table), hourly(times)) == Needs(multiple=multiple)


def test_sub_daily_bins_may_start_inside_the_day():
  """Six-hour bins from midnight: a block starting at 06:00 starts on a bin edge, at 01:00 not."""
  table = {"name": "resample", "time": "6h", "reduce": "mean"}
  assert needs_for(steps(table), hourly(HOURS[6:])).multiple == {"time": 6}
  assert needs_for(steps(table), hourly(HOURS[1:])).multiple == {}
