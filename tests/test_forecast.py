# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Forecast downloads: runs read along valid time."""

import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arcomake import dataset_utils
from arcomake.dataset_utils import (
  CopernicusMarinePartConfig,
  DatasetConfig,
  EarlyWarningDataStorePartConfig,
  download_and_process,
)

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
  config = DatasetConfig.model_validate({"parts": [part | {"engine": "ecmwfopendata"}]})
  download_and_process(config, RUN, RUN + DAY, run_datetime=RUN)
  assert (received["run_datetime"], received["start_datetime"]) == (RUN, RUN)
  assert received["end_datetime"] == RUN + DAY and received["param"] == ["2t"]


def test_engines_without_runs_are_refused():
  """A part whose engine reads no forecast runs cannot be read as one."""
  part = CopernicusMarinePartConfig.model_validate({"filename_or_obj": "cm://some-dataset"})
  with pytest.raises(ValueError, match="reads no forecast runs"):
    download_and_process(DatasetConfig(parts=[part]), RUN, RUN + DAY, run_datetime=RUN)
