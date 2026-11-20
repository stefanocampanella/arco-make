# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`time_offset` of a part: values stamped at the end of their period (ERA5 hourly accumulations)
moved to the start of it before each block's time selection, so that day D holds its own hours."""

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arcomake import dataset_utils
from arcomake.chunk_plan import Budget
from arcomake.dataset_utils import (
  CopernicusMarinePartConfig,
  DatasetConfig,
  ZarrPartConfig,
  download_and_process,
)

DAY = datetime.timedelta(days=1)
START = datetime.datetime(2020, 1, 1)

BUDGET = Budget(memory_limit=10**9, threads=1)  # parts chunked as download plans them


@pytest.fixture
def accumulations(tmp_path: pathlib.Path) -> tuple[str, xr.Dataset]:
  """Hourly accumulations as ERA5 stamps them: the value stamped t covers (t - 1 h, t]. Each
  value is the number of the hour it covers since the start, so sums tell which hours a day got."""
  time = pd.date_range(START - datetime.timedelta(hours=2), periods=24 * 4 + 4, freq="1h")
  covered_hour = (time - pd.Timestamp(START)) / pd.Timedelta("1h") - 1  # hour ending at t
  ds = xr.Dataset({"tp": ("time", covered_hour.to_numpy())}, coords={"time": time})
  path = tmp_path / "era5.zarr"
  ds.chunk(time=24).to_zarr(path)
  return str(path), ds


def daily_sums(path: str, start: datetime.datetime, end: datetime.datetime, step, offset):
  part = {"filename_or_obj": path}
  if offset is not None:
    part["time_offset"] = offset
  config = DatasetConfig.model_validate(
    {
      "parts": [ZarrPartConfig.model_validate(part)],
      "postprocess": [
        {"name": "resample", "time": "1D", "label": "left", "closed": "left", "reduce": "sum"}
      ],
    }
  )
  blocks = []
  block_start = start
  while block_start < end:
    block_end = min(block_start + step, end)
    blocks.append(download_and_process(config, block_start, block_end, budget=BUDGET).load())
    block_start = block_end
  return xr.concat(blocks, dim="time")


def test_a_day_sums_its_own_hours_with_the_offset(accumulations):
  """With `time_offset = "-1h"`, day D sums the hours 00-23 of D (the values stamped 01:00 of D
  to 00:00 of D+1); without it, the hour before midnight of D-1 and not the last hour of D."""
  path, _ = accumulations
  end = START + 3 * DAY
  shifted = daily_sums(path, START, end, 3 * DAY, "-1h")
  plain = daily_sums(path, START, end, 3 * DAY, None)
  for day in range(3):
    hours = np.arange(24 * day, 24 * day + 24)
    assert shifted.tp.values[day] == hours.sum()
    assert plain.tp.values[day] == (hours - 1).sum()
  assert list(shifted.time.values) == list(plain.time.values)


@pytest.mark.parametrize("step", [DAY, 2 * DAY])
def test_blocks_equal_one_block_with_the_offset(accumulations, step):
  """Block edges lose or double no hour: each block is read over its own shifted range."""
  path, _ = accumulations
  end = START + 3 * DAY
  xr.testing.assert_identical(
    daily_sums(path, START, end, step, "-1h"), daily_sums(path, START, end, 3 * DAY, "-1h")
  )


def test_the_offset_keeps_the_time_coordinate_attributes(accumulations):
  """Only the values of the time coordinate move: its attributes and encoding stay."""
  path, _ = accumulations
  part = ZarrPartConfig.model_validate({"filename_or_obj": path, "time_offset": "-1h"})
  out = download_and_process(DatasetConfig(parts=[part]), START, START + DAY)
  assert str(out.time.values[0])[:16] == "2020-01-01T00:00" and out.time.size == 24
  assert out.tp.values[0] == 0  # the hour 00:00-01:00, stamped 01:00 in the source
  assert "units" in out.time.encoding


def test_backends_with_a_time_range_are_asked_for_the_shifted_range(
  monkeypatch: pytest.MonkeyPatch,
):
  """Copernicus Marine and EWDS parts get the block's range moved by the opposite amount, so the
  shifted values cover the block."""
  received = {}
  time = pd.date_range(START + datetime.timedelta(hours=1), periods=24, freq="1h")

  def fake_open_input(**kwargs):
    received.update(kwargs)
    return xr.Dataset({"tp": ("time", np.ones(24))}, coords={"time": time})

  monkeypatch.setattr(dataset_utils, "open_input", fake_open_input)
  part = CopernicusMarinePartConfig.model_validate(
    {"filename_or_obj": "cm://some-dataset", "time_offset": "-1h"}
  )
  out = download_and_process(DatasetConfig(parts=[part]), START, START + DAY)
  assert received["start_datetime"] == START + datetime.timedelta(hours=1)
  assert received["end_datetime"] == START + DAY + datetime.timedelta(hours=1)
  assert "time_offset" not in received
  assert out.time.size == 24 and str(out.time.values[0])[:16] == "2020-01-01T00:00"


def test_an_offset_needs_a_time_coordinate(tmp_path: pathlib.Path):
  """A part without the time coordinate cannot be shifted: an error, not a silent no-op."""
  xr.Dataset({"deptho": ("x", np.zeros(3))}).to_zarr(tmp_path / "static.zarr")
  part = ZarrPartConfig.model_validate(
    {"filename_or_obj": str(tmp_path / "static.zarr"), "time_offset": "-1h"}
  )
  with pytest.raises(ValueError, match="time_offset needs a 'time' coordinate"):
    download_and_process(DatasetConfig(parts=[part]), START, START + DAY)
