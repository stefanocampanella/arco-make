# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from collections.abc import Callable, Hashable, Mapping
from datetime import datetime
from typing import Literal, Self, cast

import dask
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, model_validator

from arcomake.processing_utils import check_fraction_threshold

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
  """`time_dim`, `threshold`, `inclusive`, `exact` (all masks, or a list of them), and one key
  per mask: `MASK = ["variable", ...]` checks those variables inside it."""

  model_config = ConfigDict(extra="allow")

  time_dim: str | None = None
  threshold: float = 0.0
  inclusive: bool = False
  exact: bool | list[str] = False

  @model_validator(mode="after")
  def _threshold_leaves_cells_in_and_out(self) -> Self:
    check_fraction_threshold(self.threshold, self.inclusive)  # as the `fraction_mask` step
    return self

  @model_validator(mode="after")
  def _masks_map_to_variable_lists(self) -> Self:
    # The extra keys are mask names, each with the list of variables it applies to.
    for mask_name, variables in (self.model_extra or {}).items():
      if not (isinstance(variables, list) and all(isinstance(v, str) for v in variables)):
        raise ValueError(f"mask {mask_name!r} must map to a list of variable names")
    return self


class ChecksConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  valid_global_ecmwf_coordinates: ValidGlobalECMWFCoordinatesCheck | None = None
  valid_time_coordinate: ValidTimeCoordinateCheck | None = None
  ensure_no_nans: EnsureNoNansCheck | None = None


def _report_by_time[K: Hashable](
  dataset: xr.Dataset,
  flags: Mapping[K, xr.DataArray],
  time_dim: str,
  describe: Callable[[K, int], str],
) -> tuple[list[str], list[tuple[object, list[str]]]]:
  """Count the flagged cells of each boolean array per time step, all in one computation, not
  one per array and time step (a check over decades looped over thousands of days).

  Returns `describe(key, count)` for the non-zero counts of arrays without `time_dim`, and per time
  step with any, in time order, its stamp and the descriptions of its non-zero counts; keys keep
  the order of `flags`.
  """
  sums = {key: da.sum([d for d in da.dims if d != time_dim]) for key, da in flags.items()}
  computed = cast(dict[K, xr.DataArray], dask.compute(sums)[0])
  static: list[str] = []
  per_time: dict[int, list[str]] = {}
  for key, count in computed.items():
    if time_dim not in count.dims:
      if int(count) > 0:
        static.append(describe(key, int(count)))
      continue
    values = count.values
    for index in np.flatnonzero(values):
      per_time.setdefault(int(index), []).append(describe(key, int(values[index])))
  if not per_time:
    return static, []
  stamps = dataset[time_dim].to_index()
  return static, [(stamps[index], per_time[index]) for index in sorted(per_time)]


type _Key = tuple[Hashable, Hashable | None, str]


def _wrong(
  dataset: xr.Dataset,
  time_dim: str,
  threshold: float,
  inclusive: bool,
  exact: bool | list[str],
  masks: Mapping[str, list[str]],
) -> tuple[list[str], dict[_Key, xr.DataArray]]:
  """The rule of `ensure_no_nans`: errors found without computing (missing masks or variables),
  and per (variable, mask, kind) the lazy flags of its wrong cells."""
  dataset_errors: list[str] = []
  exact_masks = set(masks) if exact is True else set(exact or [])
  for mask_name in sorted(exact_masks - set(masks)):
    dataset_errors.append(f"exact: {mask_name!r} is not one of the masks")
  masks_of: dict[Hashable, list[Hashable]] = {}
  unchecked: set[Hashable] = set()
  for mask_name, variable_names in masks.items():
    if mask_name not in dataset.variables:
      # Its variables are not checked: everywhere, they would be reported missing on land.
      dataset_errors.append(f"mask {mask_name!r} not found")
      unchecked.update(variable_names)
      continue
    for variable_name in variable_names:
      if variable_name in dataset.data_vars:
        masks_of.setdefault(variable_name, []).append(mask_name)
      else:
        dataset_errors.append(f"variable {variable_name!r} of mask {mask_name!r} not found")

  # Per variable and mask: missing values inside it and, with `exact`, values outside.
  wrong: dict[_Key, xr.DataArray] = {}
  # Each mask is compared with the threshold once, then reduced to the dimensions of each variable.
  inside_all = {
    name: dataset[name] >= threshold if inclusive else dataset[name] > threshold
    for name in {m for masks in masks_of.values() for m in masks}
  }
  for variable_name, da in dataset.data_vars.items():
    if variable_name in unchecked and variable_name not in masks_of:
      continue
    missing = da.isnull()
    if variable_name not in masks_of:
      wrong[(variable_name, None, "missing")] = missing
    for mask_name in masks_of.get(variable_name, []):
      inside = inside_all[mask_name]
      if time_dim in inside.dims and time_dim not in da.dims:
        dataset_errors.append(
          f"variable {variable_name!r} has no {time_dim!r} but its mask {mask_name!r} has"
        )
        continue
      inside = inside.isel({dim: 0 for dim in inside.dims if dim not in da.dims}, drop=True)
      wrong[(variable_name, mask_name, "missing")] = missing & inside
      if mask_name in exact_masks:
        wrong[(variable_name, mask_name, "outside")] = da.notnull() & ~inside

  return dataset_errors, wrong


def _describe(key: _Key, n: int) -> str:
  variable, mask_name, kind = key
  if kind == "outside":
    return f"{n} values found outside mask {mask_name} for variable {variable}"
  inside = "" if mask_name is None else f" inside mask {mask_name}"
  return f"{n} NaN values found for variable {variable}{inside}"


def ensure_no_nans(
  dataset: xr.Dataset,
  time_dim: str = "time",
  threshold: float = 0.0,
  inclusive: bool = False,
  exact: bool | list[str] = False,
  **kwargs: list[str],
) -> None:
  """Check that variables have no missing value: inside their masks, or everywhere.

  Each extra argument maps a mask to the variables it applies to, e.g.
  `seafrac = ["thetao", "so"]`: those are checked only inside it, where it is above `threshold`
  (at least `threshold` with `inclusive`): by default where a boolean mask is True or a fraction
  positive. Missing values of a mask count as outside; the mask itself is checked like any other
  variable. Variables without one of the mask's dimensions are checked against its first index
  along it (the surface level, for 2D fields); a mask with the time dimension needs it in its
  variables. With `exact` (true, or a list of masks), values outside those masks are failures
  too: their variables have values exactly inside. A variable under several masks is checked
  against each. Other variables are checked everywhere. A mask or a listed variable missing from
  the dataset is a failure; the variables of a missing mask are not checked.

  Raises:
    ValidationError: listing, per time step, the variables with missing values inside their
      masks (or anywhere) and, with `exact`, with values outside.
  """
  dataset_errors, wrong = _wrong(dataset, time_dim, threshold, inclusive, exact, kwargs)

  logger.info("Checking for Nans.")
  static, per_time = _report_by_time(dataset, wrong, time_dim, _describe)
  if static:
    dataset_errors.append(f"{len(static)} errors found without time: " + ", ".join(static))
  dataset_errors += [
    f"{len(found)} errors found for time {stamp}: " + ", ".join(found) for stamp, found in per_time
  ]
  if dataset_errors:
    raise ValidationError("\n".join(dataset_errors))


def guard_no_nans(
  dataset: xr.Dataset,
  time_dim: str = "time",
  threshold: float = 0.0,
  inclusive: bool = False,
  exact: bool | list[str] = False,
  **kwargs: list[str],
) -> xr.Dataset:
  """`dataset` whose variables fail when computed if they break the rule of `ensure_no_nans`
  (same arguments): checked while the dataset is computed (written, say), without reading it
  again. Each variable passes through unchanged, chunk by chunk; the first chunk breaking the rule
  raises, naming the variable, the mask and its first time step wrong. Missing masks or variables
  fail at once.

  Raises:
    ValidationError: at once, for missing masks or variables; when computed, for wrong values.
  """
  errors, wrong = _wrong(dataset, time_dim, threshold, inclusive, exact, kwargs)
  if errors:
    raise ValidationError("\n".join(errors))
  flags_of: dict[Hashable, list[tuple[_Key, xr.DataArray]]] = {}
  for key, flags in wrong.items():
    flags_of.setdefault(key[0], []).append((key, flags))
  data_vars = dict(dataset.data_vars)
  for name, keyed in flags_of.items():
    da = dataset[name]
    timed = time_dim in da.dims
    inputs = [da, *(flags for _, flags in keyed)] + ([da[time_dim]] if timed else [])
    guarded = xr.apply_ufunc(
      _pass_if_right,
      *inputs,
      dask="parallelized",
      output_dtypes=[da.dtype],
      join="exact",
      keep_attrs="drop_conflicts",
      kwargs={"keys": [key for key, _ in keyed], "timed": timed},
    ).transpose(*da.dims)
    guarded.attrs = dict(da.attrs)
    guarded.encoding = dict(da.encoding)
    data_vars[name] = guarded
  return xr.Dataset(data_vars=data_vars, coords=dataset.coords, attrs=dataset.attrs)


def _pass_if_right(
  values: np.ndarray, *flags_and_stamps: np.ndarray, keys: list[_Key], timed: bool
) -> np.ndarray:
  """`values`, unchanged, if none of the flags (one per key, broadcast to `values`) is set; else
  ValidationError naming the key, its count in this chunk and its first time stamp wrong."""
  flags = flags_and_stamps[:-1] if timed else flags_and_stamps
  for key, flag in zip(keys, flags, strict=True):
    flag = np.broadcast_to(flag, values.shape)
    if not flag.any():
      continue
    message = _describe(key, int(flag.sum())) + " in a block"
    if timed:
      stamps = np.broadcast_to(flags_and_stamps[-1], values.shape)
      message += f", first at {pd.Timestamp(stamps[flag].min())}"
    raise ValidationError(message)
  return values


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
}
assert set(CHECKS) == set(ChecksConfig.model_fields)
