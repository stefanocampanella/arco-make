# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import sys
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
checks_module = sys.modules[__name__]


class ValidationError(Exception):
  """Raised when validation fails."""


def validate(
  dataset: xr.Dataset,
  checks: dict[str, Any],
  start_datetime: datetime,
  end_datetime: datetime,
  should_raise: bool = False,
) -> None:
  """
  Performs checks on a xarray.Dataset and raise an exception if any check fails.

  The checks are provided as a dictionary containing step configurations.
  All methods defined in this module can be used as checks.

  Args:
    dataset (xr.Dataset): The input dataset to process.
    checks (dict[str, Any]): Configuration for each processing step.
  Returns:
    None
  """
  logger.info("Validating dataset with the following checks: " + ", ".join(checks.keys()) + ". ")
  for name, config in checks.items():
    if not hasattr(checks_module, name):
      warnings.warn(f"Unrecognized validation check {name} with configuration {config}")
      continue
    if name == "valid_time_coordinate":
      config = config | {"start_datetime": start_datetime, "end_datetime": end_datetime}
    check_fn: Callable[..., None] = getattr(checks_module, name)
    try:
      check_fn(dataset, **config)
    except ValidationError as exc:
      failure_message = f"Validation step {name} failed: {exc}"
      if should_raise:
        raise ValidationError(failure_message) from exc
      else:
        warnings.warn(failure_message)


def ensure_no_nans(
  dataset: xr.Dataset,
  time_dim: str = "time",
  **kwargs,
) -> None:
  variable_mask_mapping = {}
  for mask_name, variable_names in kwargs.items():
    for variable_name in variable_names:
      if variable_name in dataset.data_vars:
        variable_mask_mapping[variable_name] = mask_name
      else:
        warnings.warn(f"Variable {variable_name} not found in dataset")

  def _get_nan_errors(ds: xr.Dataset):
    errors = []
    for variable_name, da in ds.data_vars.items():
      if variable_name in variable_mask_mapping:
        mask_name = variable_mask_mapping[variable_name]
        mask_da = ds[mask_name]
        # ensure_no_nans assumes that if data does not have a mask dimension, then the 0th component of the mask along
        # that dimension is the proper mask (e.g., surface data should be masked using the first depth level)
        mask_da = mask_da.isel({dim: 0 for dim in mask_da.dims if dim not in da.dims}, drop=True)
        masked_data = da.where(mask_da, 0.0)
        invalid_values = masked_data.isnull()
      else:
        invalid_values = da.isnull()
      if invalid_values.any():
        errors.append(
          f"{invalid_values.sum().values} NaN values found for variable {variable_name}"
        )
    return errors

  logger.info("Checking for Nans.")
  dataset_errors = []
  datetimes = dataset[time_dim].to_index()
  for d in datetimes:
    errors_on_datetime = _get_nan_errors(dataset.sel({time_dim: d}))
    if errors_on_datetime:
      message = f"{len(errors_on_datetime)} errors found for time {d}: " + ", ".join(
        errors_on_datetime
      )
      dataset_errors.append(message)
  if dataset_errors:
    raise ValidationError("\n".join(dataset_errors))


def valid_global_ecmwf_coordinates(
  ds: xr.Dataset, latitude_dim: str = "latitude", longitude_dim: str = "longitude"
) -> None:
  logger.info("Checking that coordinates use ECMWF global convention.")

  # Check that each dataset contains all latitudes in [-90, 90], and use the [-90, 90] convention
  if latitude_dim not in ds.dims:
    raise ValidationError(f"Latitude dimension {latitude_dim} not found in dataset")
  if ds[latitude_dim].min().item() < -90 or ds[latitude_dim].max().item() > 90:
    raise ValueError("Dataset has latitudes outside the [-90, 90] range")
  if ds[latitude_dim][0].item() > ds[latitude_dim][-1].item():
    raise ValueError("Dataset does not use the [90, -90] convention for latitude")

  # Check that each dataset uses the [0, 360) convention for longitude
  if longitude_dim not in ds.dims:
    raise ValidationError(f"Longitude dimension {longitude_dim} not found in dataset")
  if ds[longitude_dim].min().item() < 0 or ds[longitude_dim].max().item() >= 360:
    raise ValueError("Dataset does not use the [0, 360) convention for longitude")


def valid_time_coordinate(
  dataset: xr.Dataset,
  start_datetime: datetime,
  end_datetime: datetime,
  freq: str = "1D",
  time_dim="time",
) -> None:
  """
  Check that time coordinate is sorted, that all dates are within the specified range, and that there are no duplicates
  or missing dates.

  Args:
    dataset: The dataset to check
    start_datetime: Start date (required)
    end_datetime: End date (required)
    dataset_name: Optional name of the dataset for the warning message
  """
  assert end_datetime >= start_datetime, "End date must be after start date"
  logger.info(
    f"Checking that time coordinate contains all dates between {start_datetime} and {end_datetime}, with frequency {freq}."
  )
  idx = dataset[time_dim].to_index()
  if isinstance(idx, pd.DatetimeIndex):
    valid_datetime_index(idx, start_datetime, end_datetime, freq)
  elif isinstance(idx, xr.CFTimeIndex):
    valid_cftime_index(idx, start_datetime, end_datetime, freq)
  else:
    raise ValueError(f"Unexpected index type: {type(idx).__name__}")


# TODO: Check that the following implementation works for both DateTimeIndex and CFTimeIndex
def valid_datetime_index(
  idx: pd.DatetimeIndex, start_datetime: datetime, end_datetime: datetime, freq: str = "1D"
) -> None:

  # Check that the time coordinate is sorted
  if not idx.is_monotonic_increasing:
    raise ValidationError("Time coordinate is not sorted")

  # Check for duplicates
  if idx.has_duplicates:
    dups = idx[idx.duplicated()]
    dup_values = pd.DatetimeIndex(dups.unique())
    preview = ", ".join(str(ts) for ts in dup_values[:5])
    more = "" if len(dup_values) <= 5 else f" and {len(dup_values) - 5} more"
    raise ValidationError(f"Duplicate timestamps found in time coordinate: {preview}{more}")

  # Check for missings
  expected = pd.date_range(
    start=start_datetime, end=end_datetime, freq=freq, inclusive="left", tz=getattr(idx, "tz", None)
  )
  if not idx.equals(expected):
    # Report missing or irregular timestamps for easier debugging
    # Compute missing by comparing against the sorted unique expected sequence
    missing = expected.difference(idx)
    preview = ", ".join(str(ts) for ts in missing[:5])
    more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
    raise ValidationError(f"Missing or irregular dates detected: {preview}{more}")


def valid_cftime_index(
  idx: xr.CFTimeIndex, start_date: datetime, end_date: datetime, freq: str = "1D"
) -> None:
  # Equivalent checks for xarray.CFTimeIndex (cftime-based calendars)

  # Check that the time coordinate is sorted
  if not idx.is_monotonic_increasing:
    raise ValidationError("Time coordinate is not sorted")

  # Duplicates check
  if idx.has_duplicates:
    dups = idx[idx.duplicated()]
    dup_values = dups.unique()  # CFTimeIndex of unique duplicate timestamps
    preview = ", ".join(str(ts) for ts in dup_values[:5])
    more = "" if len(dup_values) <= 5 else f" and {len(dup_values) - 5} more"
    raise ValidationError(f"Duplicate timestamps found in time coordinate: {preview}{more}")

  # Regularity check (daily frequency across CF calendars)
  calendar = getattr(idx, "calendar", None)
  # Build expected daily sequence with same calendar
  expected = xr.cftime_range(
    start=start_date, end=end_date, freq=freq, inclusive="left", calendar=calendar
  )
  if not idx.equals(expected):
    # Compute missing days between min and max dates for helpful diagnostics
    missing = expected.difference(idx)
    preview = ", ".join(str(ts) for ts in missing[:5])
    more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
    raise ValidationError(f"Missing or irregular dates detected: {preview}{more}")
