# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# The following implementation assumes that mask is unchunked, and that block is unchunked along mask dimensions.
def check_values(
  ds: xr.Dataset,
  variables: None | Sequence[str] = None,
  mask: None | xr.DataArray = None,
) -> xr.Dataset:
  logger.info("Checking for Nans.")
  allowed_variables = variables or ds.data_vars
  for var, da in ds.data_vars.items():
    if var in allowed_variables:
      if mask is not None:
        mask_ = mask.isel({dim: 0 for dim in mask.dims if dim not in da.dims}, drop=True)
        masked_data = da.where(mask_, 0.0)
        nonvalid_values = masked_data.isnull()
      else:
        nonvalid_values = da.isnull()
      if nonvalid_values.any():
        raise ValueError(
          f"{nonvalid_values.sum().values} NaN values found "
          f"for variable {da.name or 'unknown'}"
        )
  return ds


def check_dates(
  ds: xr.Dataset,
  start_date: datetime,
  end_date: datetime,
  freq: str = "D",
) -> xr.Dataset:
  """
  Check that all dates in a dataset are within the specified range
  and that each day is included in the interval.

  Args:
    ds: The dataset to check
    start_date: Start date (required)
    end_date: End date (required)
    dataset_name: Optional name of the dataset for the warning message
  """

  logger.info("Checking for missing dates.")
  if "time" in ds.dims:
    # Check that each day is included in the interval
    expected_dates = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive="left")
    time_values: np.ndarray = ds.time.values
    dataset_dates: pd.DatetimeIndex = pd.to_datetime(time_values)

    # Find missing dates within the expected range
    missing_dates = []
    for expected_date in expected_dates:
      if expected_date not in dataset_dates:
        missing_dates.append(expected_date)

    if missing_dates:
      missing_str = [date.strftime("%Y-%m-%d") for date in missing_dates[:10]]  # Show first 10
      if len(missing_dates) > 10:
        missing_str.append(f"... and {len(missing_dates) - 10} more")
      raise ValueError(
        f"Dataset is missing {len(missing_dates)} dates "
        f"from the expected interval [{start_date}, {end_date}]: {missing_str}"
      )

  return ds


def check_global_ecmwf(ds: xr.Dataset) -> xr.Dataset:
  logger.info("Checking that coordinates use ECMWF global convention.")

  # Check that each dataset uses the [0, 360) convention for longitude
  if "longitude" in ds.dims or "lon" in ds.dims:
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"
    if ds[lon_dim].min().item() < 0 or ds[lon_dim].max().item() >= 360:
      raise ValueError(
        "Dataset does not use the [0, 360) convention for longitude"
      )

  # Check that each dataset contains all latitudes in [-90, 90], and use the [-90, 90] convention
  if "latitude" in ds.dims or "lat" in ds.dims:
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    if ds[lat_dim].min().item() < -90 or ds[lat_dim].max().item() > 90:
      raise ValueError("Dataset has latitudes outside the [-90, 90] range")
    if ds[lat_dim][0].item() > ds[lat_dim][-1].item():
      raise ValueError(
        "Dataset does not use the [90, -90] convention for latitude"
      )

  return ds
