# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import functools
import logging
import pathlib
import warnings
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# The following implementation assumes that mask is unchunked, and that block is unchunked along mask dimensions.
def check_values(variables: None | Sequence[str] = None, mask: None | xr.DataArray = None):
  def checker(block, dataset_info=None, block_info=None):
    try:
      if mask is not None:
        mask_ = mask.isel({dim: 0 for dim in mask.dims if dim not in block.dims}, drop=True)
        masked_block = block.where(mask_, 0.0)
        nonvalid_values = masked_block.isnull()
      else:
        nonvalid_values = block.isnull()
      if nonvalid_values.any():
        warnings.warn(
          f"{nonvalid_values.sum().values} NaN values found "
          f"in dataset {dataset_info['dataset'] if dataset_info else 'unknown'} "
          f"for variable {dataset_info['variable'] if dataset_info else 'unknown'}"
        )
    except UserWarning as w:
      # FIXME: block_info here is None, find a way to forward information about the location of the nans
      w.add_note(f"{block_info=}")

    return block

  def decorator(reader):
    @functools.wraps(reader)
    def decorated(pathlike: str | pathlib.Path, *args, **kwargs) -> xr.Dataset:
      ds = reader(pathlike, *args, **kwargs)
      logger.info("Checking for Nans")
      allowed_variables = variables or ds.data_vars
      for var, da in ds.data_vars.items():
        if var in allowed_variables:
          checker_kwargs = {"dataset_info": {"dataset": pathlike, "variable": var}}
          ds[var] = da.map_blocks(checker, kwargs=checker_kwargs, template=da)
      return ds

    return decorated

  return decorator


# FIXME: this is probaby broken, and should be integrated by
def check_date_range(start_date: datetime, end_date: datetime):
  """
  Decorator to check that all dates in a dataset are within the specified range
  and that each day is included in the interval.

  Args:
    start_date: Start date in YYYY-MM-DD format (required)
    end_date: End date in YYYY-MM-DD format (required)
  """

  def decorator(reader):
    @functools.wraps(reader)
    def decorated(pathlike: str | pathlib.Path, *args, **kwargs) -> xr.Dataset:
      ds = reader(pathlike, *args, **kwargs)

      logger.info("Checking for mismatching dates")
      if "time" in ds.dims:
        # Check that each day is included in the interval
        expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
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
          warnings.warn(
            f"Dataset at {pathlike} is missing {len(missing_dates)} dates "
            f"from the expected interval [{start_date}, {end_date}]: {missing_str}"
          )

      return ds

    return decorated

  return decorator


def check_coordinates(reader):
  @functools.wraps(reader)
  def decorated(pathlike: str | pathlib.Path, *args, **kwargs) -> xr.Dataset:
    ds = reader(pathlike, *args, **kwargs)

    logger.info("Checking for mismatching coordinates")
    # 1. Check that each dataset contains all the days between beginning and end
    if "time" in ds.dims:
      time_range = pd.date_range(start=ds.time.min().item(), end=ds.time.max().item(), freq="D")
      if not all(date in ds.time.values for date in time_range):
        missing_dates = [date for date in time_range if date not in ds.time.values]
        warnings.warn(f"Dataset at {pathlike} is missing dates: {missing_dates}")

    # 2. Check that each dataset uses the [0, 360) convention for longitude
    if "longitude" in ds.dims or "lon" in ds.dims:
      lon_dim = "longitude" if "longitude" in ds.dims else "lon"
      if ds[lon_dim].min().item() < 0 or ds[lon_dim].max().item() >= 360:
        warnings.warn(f"Dataset at {pathlike} does not use the [0, 360) convention for longitude")

    # 3. Check that each dataset contains all latitudes in [-90, 90], and use the [-90, 90] convention
    if "latitude" in ds.dims or "lat" in ds.dims:
      lat_dim = "latitude" if "latitude" in ds.dims else "lat"
      if ds[lat_dim].min().item() < -90 or ds[lat_dim].max().item() > 90:
        warnings.warn("Dataset has latitudes outside the [-90, 90] range")
      if ds[lat_dim][0].item() > ds[lat_dim][-1].item():
        warnings.warn(f"Dataset at {pathlike} does not use the [90, -90] convention for latitude")

    return ds

  return decorated
