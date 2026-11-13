# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The `Dataset.arcomake` accessor: running recipe steps and selecting time intervals."""

from datetime import datetime

import numpy as np
import pandas as pd
import pydantic
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


def test_process_rejects_unknown_steps(ocean_ds: xr.Dataset):
  """A misspelled step fails instead of being skipped (intent 4), even without a recipe model."""
  step = ProcessingStepConfig.model_construct(name="selct_variables", variables=["zos"])
  with pytest.raises(ValueError, match="Unknown processing step 'selct_variables'"):
    ocean_ds.arcomake.process([step])


@pytest.mark.parametrize(
  "name",
  ["np", "logger", "gaussian_filter", "xarray_regrid", "_get_selection", "thetao", "to_zarr"],
)
def test_names_that_are_not_steps_are_rejected_at_load_time(name: str):
  """Imports, private helpers, variable names and unlisted Dataset methods are not steps."""
  with pytest.raises(pydantic.ValidationError, match="Unknown processing step"):
    ProcessingStepConfig.model_validate({"name": name})


@pytest.mark.parametrize(
  ("table", "message"),
  [
    (
      {"name": "select_variables", "variable": ["zos"]},
      "Unknown argument 'variable' of step 'select_variables' \(did you mean 'variables'\?\)",
    ),
    ({"name": "select_variables"}, "missing a required argument: 'variables'"),
    ({"name": "flip", "dim": 3}, "Invalid argument 'dim' of step 'flip'"),
    ({"name": "regrid", "grid": {}, "method": "bilinear"}, "Invalid argument 'method'"),
    (
      {"name": "regrid", "grid": {}, "kwarg": {"skipna": True}},
      "Unknown argument 'kwarg' of step 'regrid'",
    ),
    ({"name": "drop_attrs", "deeep": False}, "Unknown argument 'deeep' of step 'drop_attrs'"),
  ],
)
def test_step_arguments_are_checked_at_load_time(table: dict, message: str):
  """Misspelled, missing or mistyped arguments fail when the recipe is read, before any download."""
  with pytest.raises(pydantic.ValidationError, match=message):
    ProcessingStepConfig.model_validate(table)


def test_recipe_errors_name_the_step_location(tmp_path):
  """The error message points at the faulty table of the recipe."""
  from arcomake.download import DownloadConfig

  recipe = {
    "start": "2020-01-01",
    "end": "2020-01-02",
    "step": "1D",
    "save": {},
    "datasets": {"ocean": {"postprocess": [{"name": "flip", "dim": "x"}, {"name": "flipp"}]}},
  }
  with pytest.raises(pydantic.ValidationError, match=r"datasets\.ocean\.postprocess\.1"):
    DownloadConfig.model_validate(recipe)


# time_sel: right-open selection [start, end)


def test_time_sel_is_right_open(ocean_ds: xr.Dataset):
  """The end of the interval is excluded (intent 5)."""
  out = ocean_ds.arcomake.time_sel(datetime(2020, 2, 27), datetime(2020, 3, 1))
  assert list(pd.DatetimeIndex(out.time.values).day) == [27, 28, 29]


def test_time_sel_keeps_dask_chunks(atmosphere_ds: xr.Dataset):
  """Selecting whole days keeps day-long chunks, as `Dataset.sel` does.

  Rechunking here changed the summation order of the daily resample downstream: the outputs of
  a real download differed from the previous version by a few float32 units in the last place.
  """
  ds = atmosphere_ds.chunk(time=24)
  out = ds.arcomake.time_sel(datetime(2020, 2, 26), datetime(2020, 2, 28))
  assert out.chunks["time"] == (24, 24)


def test_time_sel_end_after_data(ocean_ds: xr.Dataset):
  """An end past the last time step keeps the last step."""
  out = ocean_ds.arcomake.time_sel(datetime(2020, 3, 1), datetime(2020, 4, 1))
  assert out.sizes["time"] == 2  # Mar 1 and Mar 2


def test_time_sel_without_time_returns_dataset(static_ds: xr.Dataset):
  """Static datasets have no time to select."""
  assert static_ds.arcomake.time_sel(datetime(2020, 1, 1), datetime(2020, 1, 2)) is static_ds


def test_time_sel_other_time_dim(ocean_ds: xr.Dataset):
  """The time dimension may have another name."""
  ds = ocean_ds.rename(time="valid_time")
  out = ds.arcomake.time_sel(datetime(2020, 2, 27), datetime(2020, 3, 1), time_dim="valid_time")
  assert out.sizes["valid_time"] == 3


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


def test_time_sel_cftime():
  """Non-standard calendars (cftime) are selected the same way."""
  times = xr.date_range("2001-01-01", periods=5, freq="1D", calendar="noleap", use_cftime=True)
  ds = xr.Dataset({"a": ("time", np.arange(5.0))}, coords={"time": times})
  out = ds.arcomake.time_sel(times[1], times[3])
  assert out.sizes["time"] == 2
