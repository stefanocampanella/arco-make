# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import datetime
from typing import Literal, Self, cast

import dask
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arcomake.processing_utils import check_fraction_threshold

logger = logging.getLogger(__name__)


class ValidationError(Exception):
  """Raised when validation fails."""


class ValidGlobalECMWFCoordinatesCheck(BaseModel):
  model_config = ConfigDict(extra="forbid")

  latitude_dim: str = "lat"
  longitude_dim: str = "lon"


class ValidTimeCoordinateCheck(BaseModel):
  """`start` and `end` may come from the command (`validate --start/--end`); a download checks
  each block over the block, without them."""

  model_config = ConfigDict(extra="forbid")
  start: datetime | None = None
  end: datetime | None = None
  freq: str = "1D"
  time_dim: str = "time"

  @model_validator(mode="after")
  def validate_date_range(self) -> Self:
    if self.start is not None and self.end is not None and self.start > self.end:
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


class AttributesCheck(BaseModel):
  """`required`: attributes every data variable must have (not empty); `distinct`: attributes
  whose values must differ between data variables."""

  model_config = ConfigDict(extra="forbid")

  required: list[str] = Field(default_factory=lambda: ["long_name", "units"])
  distinct: list[str] = Field(default_factory=lambda: ["long_name"])


class ChecksConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  valid_global_ecmwf_coordinates: ValidGlobalECMWFCoordinatesCheck | None = None
  valid_time_coordinate: ValidTimeCoordinateCheck | None = None
  ensure_no_nans: EnsureNoNansCheck | None = None
  attributes: AttributesCheck | None = None


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
# Per variable: (mask name, where it is inside, reduced to the variable's dimensions, whether
# values outside are failures); (None, None, False) for a variable checked everywhere.
type _Rule = tuple[Hashable | None, xr.DataArray | None, bool]


def _rules(
  dataset: xr.Dataset,
  time_dim: str,
  threshold: float,
  inclusive: bool,
  exact: bool | list[str],
  masks: Mapping[str, list[str]],
) -> tuple[list[str], dict[Hashable, list[_Rule]]]:
  """The rule of `ensure_no_nans`: errors found without computing (missing masks or variables, a
  static variable under a time-varying mask), and the rules each variable is checked with."""
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

  # Each mask is compared with the threshold once, then reduced to the dimensions of each variable.
  inside_all = {
    name: dataset[name] >= threshold if inclusive else dataset[name] > threshold
    for name in {m for masks in masks_of.values() for m in masks}
  }
  rules: dict[Hashable, list[_Rule]] = {}
  for variable_name, da in dataset.data_vars.items():
    if variable_name in unchecked and variable_name not in masks_of:
      continue
    if variable_name not in masks_of:
      rules[variable_name] = [(None, None, False)]
    for mask_name in masks_of.get(variable_name, []):
      inside = inside_all[mask_name]
      if time_dim in inside.dims and time_dim not in da.dims:
        dataset_errors.append(
          f"variable {variable_name!r} has no {time_dim!r} but its mask {mask_name!r} has"
        )
        continue
      inside = inside.isel({dim: 0 for dim in inside.dims if dim not in da.dims}, drop=True)
      rule = (mask_name, inside, mask_name in exact_masks)
      rules.setdefault(variable_name, []).append(rule)
  return dataset_errors, rules


def _wrong(dataset: xr.Dataset, rules: dict[Hashable, list[_Rule]]) -> dict[_Key, xr.DataArray]:
  """Per (variable, mask, kind), the lazy flags of the cells breaking `rules`."""
  wrong: dict[_Key, xr.DataArray] = {}
  for variable_name, variable_rules in rules.items():
    da = dataset[variable_name]
    missing = da.isnull()
    for mask_name, inside, exact in variable_rules:
      if inside is None:
        wrong[(variable_name, None, "missing")] = missing
        continue
      wrong[(variable_name, mask_name, "missing")] = missing & inside
      if exact:
        wrong[(variable_name, mask_name, "outside")] = da.notnull() & ~inside
  return wrong


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
  dataset_errors, rules = _rules(dataset, time_dim, threshold, inclusive, exact, kwargs)

  logger.info("Checking for Nans.")
  static, per_time = _report_by_time(dataset, _wrong(dataset, rules), time_dim, _describe)
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
  again. Each variable passes through unchanged, chunk by chunk, with its own chunks (masks are
  aligned to them); the flags are computed inside each chunk's task. The first chunk computed
  that breaks the rule raises, naming the variable, the mask and the first day wrong in that
  chunk (not necessarily the first in the dataset). Missing masks or variables fail at once.

  Raises:
    ValidationError: at once, for missing masks or variables; when computed, for wrong values.
  """
  errors, rules = _rules(dataset, time_dim, threshold, inclusive, exact, kwargs)
  if errors:
    raise ValidationError("ensure_no_nans: " + "\n".join(errors))
  guarded: dict[Hashable, xr.DataArray] = {}
  for name, variable_rules in rules.items():
    da = dataset[name]
    insides = [inside for _, inside, _ in variable_rules if inside is not None]
    if da.chunks is not None:
      # Masks take the variable's chunks, never the reverse: the chunk plan decides the write.
      insides = [
        inside.chunk({dim: da.chunksizes[dim] for dim in inside.dims}) for inside in insides
      ]
    timed = time_dim in da.dims
    inputs = [da, *insides]
    if timed:
      # Day positions, not the time coordinate (cftime dates are objects Dask cannot chunk).
      positions = xr.DataArray(np.arange(da.sizes[time_dim]), dims=time_dim)
      if da.chunks is not None:
        positions = positions.chunk({time_dim: da.chunksizes[time_dim]})
      inputs.append(positions)
    checked = xr.apply_ufunc(
      _pass_if_right,
      *inputs,
      dask="parallelized",
      output_dtypes=[da.dtype],
      join="exact",
      keep_attrs="drop_conflicts",
      kwargs={
        "rules": [(mask, exact) for mask, _, exact in variable_rules],
        "name": name,
        "stamps": [str(stamp) for stamp in da[time_dim].values] if timed else None,
      },
    ).transpose(*da.dims)
    checked.attrs = dict(da.attrs)
    checked.encoding = dict(da.encoding)
    guarded[name] = checked
  return dataset.assign(guarded)


def _pass_if_right(
  values: np.ndarray,
  *insides_and_positions: np.ndarray,
  rules: list[tuple[Hashable | None, bool]],
  name: Hashable,
  stamps: list[str] | None,
) -> np.ndarray:
  """`values`, unchanged, if they follow the rules (one per mask, with its inside array broadcast
  to `values`; a rule without mask checks everywhere); else ValidationError naming the variable,
  the mask, the count in this chunk and its first day wrong."""
  insides = iter(insides_and_positions[:-1] if stamps is not None else insides_and_positions)
  missing = pd.isnull(values)
  for mask_name, exact in rules:
    inside = None if mask_name is None else np.broadcast_to(next(insides), values.shape)
    checks = [("missing", missing if inside is None else missing & inside)]
    if exact and inside is not None:
      checks.append(("outside", ~missing & ~inside))
    for kind, flag in checks:
      if not flag.any():
        continue
      message = "ensure_no_nans: " + _describe((name, mask_name, kind), int(flag.sum()))
      message += " in a block"
      if stamps is not None:
        positions = np.broadcast_to(insides_and_positions[-1], values.shape)
        message += f", first in its chunk at {stamps[int(positions[flag].min())]}"
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
  start: datetime | None = None,
  end: datetime | None = None,
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
    ValidationError: on disorder, duplicates, or missing or irregular stamps, or without bounds.
  """
  if start is None or end is None:
    raise ValidationError(
      "valid_time_coordinate needs start and end: in its table, or validate --start/--end"
    )
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


def attributes(
  dataset: xr.Dataset,
  required: Sequence[str] | None = None,
  distinct: Sequence[str] | None = None,
) -> None:
  """Check the data variables' attributes: each has every `required` one, not empty, and the
  values of each `distinct` one differ between variables (two `long_name`s alike tell a reader
  nothing, as a mask named after the variable it was made from).

  Raises:
    ValidationError: naming the variables without an attribute, and those sharing a value.
  """
  defaults = AttributesCheck()
  required = defaults.required if required is None else required
  distinct = defaults.distinct if distinct is None else distinct
  errors = []
  for attribute in required:
    lacking = [
      str(name) for name, da in dataset.data_vars.items() if not str(da.attrs.get(attribute, ""))
    ]
    if lacking:
      errors.append(f"without {attribute}: {', '.join(lacking)}")
  for attribute in distinct:
    holders: dict[str, list[str]] = {}
    for name, da in dataset.data_vars.items():
      if (value := str(da.attrs.get(attribute, ""))) != "":
        holders.setdefault(value, []).append(str(name))
    for value, names in holders.items():
      if len(names) > 1:
        errors.append(f"same {attribute} {value!r}: {', '.join(names)}")
  if errors:
    raise ValidationError("\n".join(errors))


# The checks a recipe may name: each field of `ChecksConfig` maps to its function.
CHECKS: dict[str, Callable[..., None]] = {
  "valid_global_ecmwf_coordinates": valid_global_ecmwf_coordinates,
  "valid_time_coordinate": valid_time_coordinate,
  "ensure_no_nans": ensure_no_nans,
  "attributes": attributes,
}
assert set(CHECKS) == set(ChecksConfig.model_fields)

# How `download` checks each block before its commit: these read only coordinates and attributes
# and run before writing; the time axis is checked over the block; `ensure_no_nans` is checked
# while the block is written (`guard_no_nans`). A check reading data needs a guard of its own.
METADATA_CHECKS = frozenset({"valid_global_ecmwf_coordinates", "attributes"})
assert set(CHECKS) == METADATA_CHECKS | {"valid_time_coordinate", "ensure_no_nans"}
