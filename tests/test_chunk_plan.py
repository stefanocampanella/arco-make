# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Chunks are planned, not set by recipes (6.1): from shapes, the needs of later steps and a
memory budget. The planner is a pure function, tested here on the shapes of real sources."""

import dask
import dask.array
import numpy as np
import pytest
import xarray as xr

from arcomake.chunk_plan import (
  COPIES,
  USABLE,
  Budget,
  Needs,
  PlanError,
  Shape,
  plan,
  shapes,
  stored_warnings,
)

GB = 10**9
# The project VM: 110 GB and 32 threads, so tasks may hold 687 MB; chunks aim at Dask's 128 MiB.
VM = Budget(memory_limit=110 * GB, threads=32)
HORIZONTAL = frozenset({"latitude", "longitude"})


@pytest.fixture(autouse=True)
def default_chunk_size():
  with dask.config.set({"array.chunk-size": "128MiB"}):
    yield


def era5(hours: int) -> dict:
  return {"t2m": Shape(("time", "latitude", "longitude"), (hours, 721, 1440), 4)}


def test_budget_shares_the_memory_among_threads_and_blocks():
  """Each task may hold its share of the usable memory, `COPIES` times its chunk."""
  assert VM.per_block == int(110 * GB * USABLE)
  assert VM.per_task == VM.per_block // (32 * COPIES)
  assert VM.target == 128 * 2**20
  two = Budget(memory_limit=110 * GB, threads=32, blocks=2)
  assert two.per_task == pytest.approx(VM.per_task / 2, abs=1)
  small = Budget(memory_limit=GB, threads=32)
  assert small.target == small.per_task < 128 * 2**20
  assert "32 threads x 2 block(s)" in two.describe()


def test_space_stays_whole_and_time_grows_to_whole_bins():
  """ERA5 hourly at 0.25 deg, resampled to days: a day of 24 steps (100 MB) per chunk, the
  value the recipe set by hand; 48 steps would exceed the target."""
  result = plan(era5(8 * 24), VM, Needs(multiple={"time": 24}))
  assert result.chunks == {"time": 24, "latitude": 721, "longitude": 1440}
  assert result.largest == (24 * 721 * 1440 * 4, "t2m")
  assert result.warnings == ()


def test_time_grows_by_whole_bins_up_to_the_target():
  """WAVERYS three-hourly at 0.2 deg (6.5 MB a step), resampled to days: 16 steps, two days."""
  variables = {"VHM0": Shape(("time", "latitude", "longitude"), (64, 900, 1800), 4)}
  assert plan(variables, VM, Needs(multiple={"time": 8})).chunks["time"] == 16


def test_time_is_capped_at_the_block():
  """A short block is one chunk along time, even if more would fit."""
  assert plan(era5(24), Budget(memory_limit=10**13, threads=1)).chunks["time"] == 24


def test_a_bin_longer_than_the_block_is_the_whole_block():
  assert plan(era5(12), VM, Needs(multiple={"time": 24})).chunks["time"] == 12


def test_a_step_above_the_target_splits_the_outer_dimension():
  """GLORYS at 1/12 deg: one day of 50 levels is 1.76 GB; regrid keeps latitude and longitude
  whole, so depth is split, to 3 levels (106 MB); time stays at one day."""
  variables = {
    name: Shape(("time", "depth", "latitude", "longitude"), (8, 50, 2041, 4320), 4)
    for name in ("thetao", "so", "uo", "vo")
  }
  result = plan(variables, VM, Needs(whole=HORIZONTAL))
  assert result.chunks == {"time": 1, "depth": 3, "latitude": 2041, "longitude": 4320}
  assert result.largest[0] == 3 * 2041 * 4320 * 4 <= VM.target


def test_dimensions_split_in_multiples_of_their_unit():
  variables = {"v": Shape(("time", "depth", "x"), (4, 12, 10**6), 8)}
  result = plan(variables, Budget(memory_limit=10**9, threads=1), Needs(multiple={"depth": 4}))
  assert result.chunks["depth"] % 4 == 0


def test_gathered_dimensions_count_whole_for_the_tasks_of_their_step():
  """`get_bottom` joins the 50 levels of a split variable: its tasks (1.76 GB) are reported
  above the 687 MB a task may hold; the dataset keeps depth split."""
  variables = {"thetao": Shape(("time", "depth", "latitude", "longitude"), (8, 50, 2041, 4320), 4)}
  result = plan(variables, VM, Needs(whole=HORIZONTAL, gathered=frozenset({"depth"})))
  assert result.chunks["depth"] == 3
  assert result.gathered == (50 * 2041 * 4320 * 4, "thetao")
  assert len(result.warnings) == 1
  assert "thetao: task gathering depth of 1.64 GiB" in result.warnings[0]


def test_static_variables_are_one_chunk():
  variables = {"deptho": Shape(("latitude", "longitude"), (2041, 4320), 4)}
  assert plan(variables, VM).chunks == {"latitude": 2041, "longitude": 4320}


def test_needs_on_missing_dimensions_do_not_apply():
  result = plan(era5(48), VM, Needs(whole=frozenset({"depth"}), multiple={"step": 3}))
  assert set(result.chunks) == {"time", "latitude", "longitude"}


def test_no_room_for_one_task_fails_with_a_message():
  """A chunk that cannot be split further, `COPIES` times over the block's memory."""
  with pytest.raises(PlanError, match=r"t2m: a chunk of .*\(--memory-limit\)"):
    # 8 MB for the block: one hour (4.2 MB) needs 16.6 MB.
    plan(era5(24), Budget(memory_limit=10**7, threads=1), Needs(whole=HORIZONTAL))


def test_a_chunk_above_the_task_share_is_a_warning():
  """One hour (4.2 MB) is the smallest chunk; with 3.75 MB per task it is reported, and runs:
  four copies fit the block's 240 MB."""
  budget = Budget(memory_limit=2 * GB, threads=4)  # 100 MB per task: 24 hours fit
  result = plan(era5(24), budget, Needs(whole=HORIZONTAL))
  assert (result.chunks["time"], result.warnings) == (24, ())
  budget = Budget(memory_limit=3 * 10**8, threads=16)
  result = plan(era5(24), budget, Needs(whole=HORIZONTAL))
  assert result.chunks["time"] == 1
  assert result.warnings[0].startswith("t2m: chunk of 3.96 MiB, above 3.58 MiB per task")


def test_needs_merge():
  merged = Needs(whole=frozenset({"x"}), multiple={"time": 24}).merge(
    Needs(gathered=frozenset({"depth"}), multiple={"time": 8})
  )
  assert merged == Needs(frozenset({"x"}), frozenset({"depth"}), {"time": 24})


def test_shapes_of_a_dataset():
  ds = xr.Dataset({"v": (("time", "x"), np.zeros((3, 4), dtype=np.float32))})
  assert shapes(ds) == {"v": Shape(("time", "x"), (3, 4), 4)}


def test_stored_chunks_above_a_task_share_are_reported():
  """Writing a stored chunk (or shard) needs it whole in one task, whatever the planned chunks:
  whole-depth stored chunks of a 1/12 deg variable (1.76 GB) are above 687 MB a task."""
  data = dask.array.zeros((2, 50, 2041, 4320), chunks=(1, 3, 2041, 4320), dtype=np.float32)
  ds = xr.Dataset({"thetao": (("time", "depth", "lat", "lon"), data)})
  ds["thetao"].encoding["chunks"] = (1, 50, 2041, 4320)
  [warning] = stored_warnings(ds, VM)
  assert warning.startswith("thetao: stored chunks of 1.64 GiB, above")
  ds["thetao"].encoding["chunks"] = (1, 1, 2041, 4320)
  assert stored_warnings(ds, VM) == []
  ds["thetao"].encoding["shards"] = (1, 25, 2041, 4320)
  assert "thetao: stored shards of" in stored_warnings(ds, VM)[0]


def test_gathered_tasks_split_the_other_dimensions_first():
  """`get_bottom` without regrid: its tasks hold the 50 levels, so splitting depth would not bound
  them; latitude is split until a column task fits a task's share, then depth for the target."""
  variables = {"thetao": Shape(("time", "depth", "latitude", "longitude"), (8, 50, 2041, 4320), 4)}
  result = plan(variables, VM, Needs(gathered=frozenset({"depth"})))
  assert result.chunks["latitude"] < 2041
  assert result.gathered is not None and result.gathered[0] <= VM.per_task
  assert result.largest[0] <= VM.target
  assert result.warnings == ()


def test_splits_prefer_multiples_of_the_source_chunks():
  """16 levels fit the target; with source chunks of 5 levels, 15 (each read by one task); with
  source chunks larger than what fits, or on another dimension, the plain fit."""
  variables = {"v": Shape(("time", "depth", "x"), (4, 50, 10**6), 8)}
  budget = Budget(memory_limit=10**9, threads=1)
  assert plan(variables, budget).chunks["depth"] == 16
  assert plan(variables, budget, preferred={"depth": 5}).chunks["depth"] == 15
  assert plan(variables, budget, preferred={"depth": 20}).chunks["depth"] == 16
  assert plan(variables, budget, preferred={"elevation": 5}).chunks["depth"] == 16
