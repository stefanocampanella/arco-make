# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`static_time` of a part: a field constant in time that a source stores as a time series (ERA5's
geopotential at the surface, land-sea mask) read once, at a given instant, without the time
dimension, the same in every block, whatever glitches other steps have (findings 3, 4)."""

import datetime
import pathlib

import numpy as np
import pandas as pd
import pydantic
import pytest
import xarray as xr

from arcomake.chunk_plan import Budget
from arcomake.dataset_utils import (
  CopernicusMarinePartConfig,
  DatasetConfig,
  ZarrPartConfig,
  download_and_process,
)

START = datetime.datetime(2020, 1, 1)
BUDGET = Budget(memory_limit=10**9, threads=1)


@pytest.fixture
def source(tmp_path: pathlib.Path) -> str:
  """Hourly fields over four days: `t2m` varies, `lsm` is constant but for one glitched hour on
  the third day."""
  time = pd.date_range(START, periods=24 * 4, freq="1h")
  lsm = np.ones((time.size, 3))
  lsm[50] = 0.5  # 2020-01-03 02:00
  ds = xr.Dataset(
    {
      "t2m": (("time", "x"), np.arange(time.size * 3.0).reshape(-1, 3)),
      "lsm": (("time", "x"), lsm),
    },
    coords={"time": time, "x": [0, 1, 2]},
  )
  path = tmp_path / "era5.zarr"
  ds.chunk(time=24).to_zarr(path)
  return str(path)


def config(path: str, instant: str = "2020-01-01T00:00") -> DatasetConfig:
  return DatasetConfig.model_validate(
    {
      "parts": [
        ZarrPartConfig.model_validate({"filename_or_obj": path, "drop_variables": ["lsm"]}),
        ZarrPartConfig.model_validate(
          {"filename_or_obj": path, "drop_variables": ["t2m"], "static_time": instant}
        ),
      ]
    }
  )


def test_a_static_part_has_no_time_and_the_same_values_in_every_block(source: str):
  blocks = [
    download_and_process(
      config(source),
      START + datetime.timedelta(days=d),
      START + datetime.timedelta(days=d + 1),
      budget=BUDGET,
    ).load()
    for d in range(4)
  ]
  for block in blocks:
    assert block.lsm.dims == ("x",) and block.lsm.values.tolist() == [1.0, 1.0, 1.0]
    assert block.t2m.sizes["time"] == 24
  # The block holding the glitched hour does not see it.
  assert blocks[2].lsm.values.tolist() == blocks[0].lsm.values.tolist()


def test_static_time_must_be_a_stamp_of_the_source(source: str):
  with pytest.raises(ValueError, match="not a stamp"):
    download_and_process(
      config(source, "2020-01-01T00:30"), START, START + datetime.timedelta(days=1)
    )


def test_static_time_has_no_zone():
  """The stamps of the sources have no zone: a zoned time is refused, not converted."""
  with pytest.raises(pydantic.ValidationError, match="without a time zone"):
    ZarrPartConfig.model_validate(
      {"filename_or_obj": "x.zarr", "static_time": "2000-01-01T00:00+01:00"}
    )


def test_sources_with_a_time_range_do_not_take_static_time():
  """Copernicus Marine and EWDS parts are read over the block's range: no `static_time`."""
  with pytest.raises(pydantic.ValidationError):
    CopernicusMarinePartConfig.model_validate(
      {"filename_or_obj": "cm://some-dataset", "static_time": "2020-01-01T00:00"}
    )
