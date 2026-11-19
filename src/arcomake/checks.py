# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import warnings
from collections.abc import Callable, Hashable
from datetime import datetime
from typing import Literal, Self, cast

import dask
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
  """Raised when validation fails."""


class ValidGlobalECMWFCoordinatesCheck(BaseModel):
  model_config = ConfigDict(extra="forbid")

  latitude_dim: str = "lat"
  longitude_dim: str = "lon"


class ValidTimeCoordinateCheck(BaseModel):
  model_config = ConfigDict(extra="forbid")
  start: datetime
  end: datetime
  freq: str = "1D"
  time_dim: str = "time"

  @model_validator(mode="after")
  def validate_date_range(self) -> Self:
    if self.start > self.end:
      raise ValueError("start datetime must be before or equal to end datetime")
    return self


class EnsureNoNansCheck(BaseModel):
  """`time_dim`, and one key per mask: `MASK = ["variable", ...]` checks those variables inside it."""

  model_config = ConfigDict(extra="allow")

  time_dim: str | None = None

  @model_validator(mode="after")
  def _masks_map_to_variable_lists(self) -> Self:
    # The extra keys are mask names, each with the list of variables it applies to.
    for mask_name, variables in (self.model_extra or {}).items():
      if not (isinstance(variables, list) and all(isinstance(v, str) for v in variables)):
        raise ValueError(f"mask {mask_name!r} must map to a list of variable names")
    return self


class FractionDomainCheck(BaseModel):
  """`time_dim`, `threshold`, `inclusive`, and one key per fraction:
  `FRACTION = ["variable", ...]` checks those variables against it."""

  model_config = ConfigDict(extra="allow")

  time_dim: str = "time"
  threshold: float = 0.0
  inclusive: bool = False

  @model_validator(mode="after")
  def _fractions_map_to_variable_lists(self) -> Self:
    for fraction, variables in (self.model_extra or {}).items():
      if not (isinstance(variables, list) and all(isinstance(v, str) for v in variables)):
        raise ValueError(f"fraction {fraction!r} must map to a list of variable names")
    return self


class ChecksConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  valid_global_ecmwf_coordinates: ValidGlobalECMWFCoordinatesCheck | None = None
  valid_time_coordinate: ValidTimeCoordinateCheck | None = None
  ensure_no_nans: EnsureNoNansCheck | None = None
  fraction_domain: FractionDomainCheck | None = None


def ensure_no_nans(
  dataset: xr.Dataset,
  time_dim: str = "time",
  **kwargs: list[str],
) -> None:
  """Check that variables have no missing value, inside their mask if one is given.

  Each extra argument maps a mask to the variables it applies to, e.g.
  `glorys_mask = ["thetao", "so"]`: those are checked only where the mask is True (the first
  level of a 3D mask for 2D variables). Other variables are checked everywhere.

  Raises:
    ValidationError: listing, per time step, the variables with missing values.
  """
  variable_mask_mapping: dict[Hashable, str] = {}
  for mask_name, variable_names in kwargs.items():
    for variable_name in variable_names:
      if variable_name in dataset.data_vars:
        variable_mask_mapping[variable_name] = mask_name
      else:
        warnings.warn(f"Variable {variable_name} not found in dataset")

  # Missing values inside the masks, counted per time step in one computation, not two per day
  # and variable: a check over decades looped over thousands of days.
  counts: dict[Hashable, xr.DataArray] = {}
  for variable_name, da in dataset.data_vars.items():
    invalid = da.isnull()
    if variable_name in variable_mask_mapping:
      mask_da = dataset[variable_mask_mapping[variable_name]]
      # A variable without a dimension of its mask (surface data, 3D mask) is checked against the
      # mask's first entry along it (the first depth level).
      mask_da = mask_da.isel({dim: 0 for dim in mask_da.dims if dim not in da.dims}, drop=True)
      invalid = invalid & mask_da.astype(bool)
    counts[variable_name] = invalid.sum([dim for dim in invalid.dims if dim != time_dim])

  logger.info("Checking for Nans.")
  computed = cast(dict[Hashable, xr.DataArray], dask.compute(counts)[0])
  per_time: dict[int, list[str]] = {}
  static_errors = []
  for variable_name, count in computed.items():
    if time_dim not in count.dims:
      if int(count) > 0:
        static_errors.append(f"{int(count)} NaN values found for variable {variable_name}")
      continue
    for index in np.flatnonzero(count.values):
      per_time.setdefault(int(index), []).append(
        f"{int(count.values[index])} NaN values found for variable {variable_name}"
      )
  dataset_errors = []
  if static_errors:
    dataset_errors.append(
      f"{len(static_errors)} errors found without time: " + ", ".join(static_errors)
    )
  datetimes = dataset[time_dim].to_index()
  for index in sorted(per_time):
    errors_on_datetime = per_time[index]
    dataset_errors.append(
      f"{len(errors_on_datetime)} errors found for time {datetimes[index]}: "
      + ", ".join(errors_on_datetime)
    )
  if dataset_errors:
    raise ValidationError("\n".join(dataset_errors))


def fraction_domain(
  dataset: xr.Dataset,
  time_dim: str = "time",
  threshold: float = 0.0,
  inclusive: bool = False,
  **kwargs: list[str],
) -> None:
  """Check that variables have values exactly where a static fraction says the domain is.

  Each extra argument maps a fraction (e.g. of sea, without the time dimension) to the variables
  it decides: inside (fraction above `threshold`, or at least with `inclusive`) no value may be
  missing, outside every value must be. Variables without one of the fraction's dimensions are
  checked against its first index along it (the surface level, for 2D fields). A fraction with
  the time dimension fails: the domain is static.

  Raises:
    ValidationError: listing, per time step, the variables with missing values inside or values
      outside.
  """
  errors: list[str] = []
  counts: dict[tuple[str, str], xr.DataArray] = {}
  for fraction_name, variable_names in kwargs.items():
    fraction = dataset[fraction_name]
    if time_dim in fraction.dims:
      errors.append(f"fraction {fraction_name!r} has the time dimension: it must be static")
      continue
    inside_all = fraction >= threshold if inclusive else fraction > threshold
    for name in variable_names:
      if name not in dataset.data_vars:
        errors.append(f"variable {name!r} of fraction {fraction_name!r} not found")
        continue
      da = dataset[name]
      inside = inside_all.isel({d: 0 for d in inside_all.dims if d not in da.dims}, drop=True)
      other = [d for d in da.dims if d != time_dim]
      counts[(name, "missing inside")] = (da.isnull() & inside).sum(other)
      counts[(name, "present outside")] = (da.notnull() & ~inside).sum(other)

  logger.info("Checking variables against their fractions' domains.")
  computed = cast(dict[tuple[str, str], xr.DataArray], dask.compute(counts)[0])
  per_time: dict[int, list[str]] = {}
  for (name, kind), count in computed.items():
    if time_dim not in count.dims:
      if int(count) > 0:
        errors.append(f"{name}: {int(count)} values {kind}")
      continue
    for index in np.flatnonzero(count.values):
      per_time.setdefault(int(index), []).append(
        f"{name}: {int(count.values[index])} values {kind}"
      )
  if per_time:
    stamps = dataset[time_dim].to_index()
    errors += [f"{stamps[i]}: " + ", ".join(per_time[i]) for i in sorted(per_time)]
  if errors:
    raise ValidationError("\n".join(errors))


def valid_global_ecmwf_coordinates(
  ds: xr.Dataset, latitude_dim: str = "lat", longitude_dim: str = "lon"
) -> None:
  """Check a global grid: latitudes within [-90, 90] and increasing, longitudes within [0, 360).

  Raises:
    ValidationError: describing the first violation.
  """
  logger.info("Checking that coordinates use ECMWF global convention.")

  if latitude_dim not in ds.dims:
    raise ValidationError(f"Latitude dimension {latitude_dim} not found in dataset")
  if ds[latitude_dim].min().item() < -90 or ds[latitude_dim].max().item() > 90:
    raise ValidationError("Dataset has latitudes outside the [-90, 90] range")
  if ds[latitude_dim][0].item() > ds[latitude_dim][-1].item():
    raise ValidationError("Latitudes are decreasing; they must increase from south to north")

  if longitude_dim not in ds.dims:
    raise ValidationError(f"Longitude dimension {longitude_dim} not found in dataset")
  if ds[longitude_dim].min().item() < 0 or ds[longitude_dim].max().item() >= 360:
    raise ValidationError("Dataset does not use the [0, 360) convention for longitude")


def valid_time_coordinate(
  dataset: xr.Dataset,
  start: datetime,
  end: datetime,
  freq: str = "1D",
  inclusive: Literal["left", "right", "both", "neither"] = "left",
  time_dim="time",
) -> None:
  """Check that the time coordinate is exactly the regular axis from `start` to `end`.

  The axis must be sorted, without duplicates, and equal to the stamps every `freq` from `start`
  to `end` (`end` excluded by default), in the dataset's calendar (numpy or cftime).

  Args:
    dataset: The dataset to check.
    start, end: Bounds of the expected axis.
    freq: Spacing of the expected axis (a pandas frequency, e.g. "1D").
    inclusive: Which bounds are part of the axis (default: `start` only).
    time_dim: Name of the time coordinate.

  Raises:
    ValidationError: on disorder, duplicates, or missing or irregular stamps.
  """
  logger.info(
    f"Checking that time coordinate contains all dates between {start} and {end}, with frequency {freq}."
  )
  idx = dataset[time_dim].to_index()
  if isinstance(idx, pd.DatetimeIndex):
    valid_datetime_index(idx, start, end, freq=freq, inclusive=inclusive)
  elif isinstance(idx, xr.CFTimeIndex):
    valid_cftime_index(idx, start, end, freq=freq, inclusive=inclusive)
  else:
    raise ValueError(f"Unexpected index type: {type(idx).__name__}")


# TODO: Check that the following implementation works for both DateTimeIndex and CFTimeIndex
def valid_datetime_index(
  idx: pd.DatetimeIndex,
  start_datetime: datetime,
  end_datetime: datetime,
  freq: str = "1D",
  inclusive: Literal["left", "right", "both", "neither"] = "left",
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
    start=start_datetime,
    end=end_datetime,
    freq=freq,
    inclusive=inclusive,
    tz=getattr(idx, "tz", None),
  )
  if not idx.equals(expected):
    # Report missing or irregular timestamps for easier debugging
    # Compute missing by comparing against the sorted unique expected sequence
    missing = expected.difference(idx)
    preview = ", ".join(str(ts) for ts in missing[:5])
    more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
    raise ValidationError(f"Missing or irregular dates detected: {preview}{more}")


def valid_cftime_index(
  idx: xr.CFTimeIndex,
  start_date: datetime,
  end_date: datetime,
  freq: str = "1D",
  inclusive: Literal["left", "right", "both", "neither"] = "left",
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
  calendar = getattr(idx, "calendar", None) or "standard"
  # Build expected daily sequence with same calendar
  # Bounds as ISO strings: given as `datetime` objects, the calendar and `inclusive` are ignored.
  expected = xr.date_range(
    start=start_date.isoformat(),
    end=end_date.isoformat(),
    freq=freq,
    inclusive=inclusive,
    calendar=calendar,
    use_cftime=True,
  )
  if not idx.equals(expected):
    # Compute missing days between min and max dates for helpful diagnostics
    missing = expected.difference(idx)
    preview = ", ".join(str(ts) for ts in missing[:5])
    more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
    raise ValidationError(f"Missing or irregular dates detected: {preview}{more}")


# The checks a recipe may name: each field of `ChecksConfig` maps to its function.
CHECKS: dict[str, Callable[..., None]] = {
  "valid_global_ecmwf_coordinates": valid_global_ecmwf_coordinates,
  "valid_time_coordinate": valid_time_coordinate,
  "ensure_no_nans": ensure_no_nans,
  "fraction_domain": fraction_domain,
}
assert set(CHECKS) == set(ChecksConfig.model_fields)
