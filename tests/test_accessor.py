# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The `Dataset.arcomake` accessor: running recipe steps and selecting time intervals."""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arcomake.processing_utils import ProcessingStepConfig


def steps(*tables: dict) -> list[ProcessingStepConfig]:
  return [ProcessingStepConfig.model_validate(table) for table in tables]


def test_process_applies_steps_in_order(ocean_ds: xr.Dataset):
  """Steps run in recipe order; arguments come from the extra keys of each table."""
  out = ocean_ds.arcomake.process(
    steps(
      {"name": "select_variables", "variables": ["zos"]},
      {"name": "rescale", "zos": 2.0},
      {"name": "rename", "name_dict": {"zos": "ssh"}},
    )
  )
  xr.testing.assert_allclose(out.ssh, 2.0 * ocean_ds.zos.rename("ssh"))


def test_process_falls_back_to_dataset_methods(ocean_ds: xr.Dataset):
  """Names that are not arco-make steps are `xarray.Dataset` methods (e.g. `chunk`, `mean`)."""
  out = ocean_ds.arcomake.process(steps({"name": "chunk", "time": 2}, {"name": "drop_attrs"}))
  assert out.chunks["time"] == (2, 2, 2)
  assert out.attrs == {}


def test_process_prefers_arcomake_steps_to_dataset_methods(ocean_ds: xr.Dataset):
  """`sel` with a `{start, stop}` table is the arco-make step, not `Dataset.sel`."""
  out = ocean_ds.arcomake.process(steps({"name": "sel", "latitude": {"start": 0.0, "stop": 1.0}}))
  np.testing.assert_array_equal(out.latitude.values, [0.25, 0.75])


@pytest.mark.xfail(reason="B3: an unknown step name only warns", strict=True)
def test_process_rejects_unknown_steps(ocean_ds: xr.Dataset):
  """A misspelled step fails instead of being skipped (intent 4)."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with pytest.raises((ValueError, KeyError)):
      ocean_ds.arcomake.process(steps({"name": "selct_variables", "variables": ["zos"]}))


@pytest.mark.xfail(reason="B3: any name in the processing_utils namespace is callable", strict=True)
@pytest.mark.parametrize("name", ["np", "logger", "gaussian_filter", "xarray_regrid"])
def test_process_rejects_names_that_are_not_steps(ocean_ds: xr.Dataset, name: str):
  """Imports and private helpers of the steps module are not steps."""
  with pytest.raises((ValueError, KeyError)):
    ocean_ds.arcomake.process(steps({"name": name}))


# time_sel: right-open selection [start, end)


def test_time_sel_is_right_open(ocean_ds: xr.Dataset):
  """The end of the interval is excluded (intent 5)."""
  out = ocean_ds.arcomake.time_sel(datetime(2020, 2, 27), datetime(2020, 3, 1))
  assert list(pd.DatetimeIndex(out.time.values).day) == [27, 28, 29]


def test_time_sel_end_after_data(ocean_ds: xr.Dataset):
  """An end past the last time step keeps the last step."""
  out = ocean_ds.arcomake.time_sel(datetime(2020, 3, 1), datetime(2020, 4, 1))
  assert out.sizes["time"] == 2  # Mar 1 and Mar 2


def test_time_sel_without_time_returns_dataset(static_ds: xr.Dataset):
  """Static datasets have no time to select."""
  assert static_ds.arcomake.time_sel(datetime(2020, 1, 1), datetime(2020, 1, 2)) is static_ds


@pytest.mark.xfail(reason="B5: time_sel uses the literal keyword `time=`", strict=True)
def test_time_sel_other_time_dim(ocean_ds: xr.Dataset):
  """The time dimension may have another name."""
  ds = ocean_ds.rename(time="valid_time")
  out = ds.arcomake.time_sel(datetime(2020, 2, 27), datetime(2020, 3, 1), time_dim="valid_time")
  assert out.sizes["valid_time"] == 3


@pytest.mark.xfail(reason="B5: time_sel indexes [-1] and [-2] of the selection", strict=True)
@pytest.mark.parametrize(
  ("start", "end", "size"),
  [
    (datetime(2021, 1, 1), datetime(2021, 1, 2), 0),  # empty selection
    (datetime(2020, 2, 26, 12), datetime(2020, 2, 27), 0),  # the only stamp is the (excluded) end
  ],
)
def test_time_sel_short_intervals(ocean_ds: xr.Dataset, start, end, size):
  """Empty and one-step selections work."""
  assert ocean_ds.arcomake.time_sel(start, end).sizes["time"] == size


@pytest.mark.xfail(reason="B5: time_sel calls to_pydatetime, which cftime lacks", strict=True)
def test_time_sel_cftime():
  """Non-standard calendars (cftime) are selected the same way."""
  times = xr.date_range("2001-01-01", periods=5, freq="1D", calendar="noleap", use_cftime=True)
  ds = xr.Dataset({"a": ("time", np.arange(5.0))}, coords={"time": times})
  out = ds.arcomake.time_sel(times[1], times[3])
  assert out.sizes["time"] == 2
