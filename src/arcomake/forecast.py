# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Forecast downloads: one run per block, stored along an initialization dimension.

A recipe with a `[forecast]` table writes forecast runs instead of a series. Run R is a download
block over [R, R + horizon) of valid time: its datasets are read, processed and checked as any
block's (`download._processed_block`), then `as_run` turns its time axis into lead times and
adds R along `init_dim`, with `valid_coord` (init + lead) as a coordinate. Runs are appended
along `init_dim`, one commit each; `download --extend` adds the runs after the last stored one,
missed days included.

Each dataset says how its source gives runs (`DatasetConfig.forecast`):
- "runs": its parts read run R itself (engines of `dataset_utils.RUN_ENGINES`, which return it
  along valid time). A run the source does not have (`backends.RunUnavailable`) leaves the
  dataset out of that run: NaN, flagged.
- "series" (the default; static datasets too): its parts are read by valid time, as in a
  historical block, e.g. a best-estimate series overwritten by each release (GLO12). Run R is a
  real forecast only if the series has no stamp at or after R + horizon when it is read: a later
  release would have one. Otherwise its values come from later releases, which the flag tells.

With `window`, the output keeps the last `window` runs: a new run shifts the stored ones back
(`icechunk_utils.Run.write_rolled`, chunks moved, not copied) and takes the last slot; the runs
dropped stay in the snapshots that hold them until the recipe's retention expires those. The
stored layout then holds one run per object along `init_dim` (`[save] chunks` and `shards`).

A dataset's `flag` names a boolean variable (`init_dim` x lead): true where its values are run
R's own forecast. Datasets may share a flag (true where all of them are).
"""

import datetime
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from arcomake.datetime_utils import may_parse_timedelta


class ForecastConfig(BaseModel):
  """The `[forecast]` table of a download recipe."""

  model_config = ConfigDict(extra="forbid")

  # Lead times [0, horizon): run R's block covers [R, R + horizon) of valid time.
  horizon: datetime.timedelta
  # Between runs: the recipe's `step` (one block per run).
  every: datetime.timedelta = datetime.timedelta(days=1)
  # The runs kept: each new run drops the oldest beyond (a rolling window); all without. Out of
  # the recipe's fingerprint: an output's window may grow or shrink (`download --extend`).
  window: Annotated[int, Field(ge=1)] | None = None
  # A source without a run younger than this (from its initialization) has not published it
  # yet: the run fails, to be collected again later, instead of storing NaN for good. Older,
  # the source lacks it (not archived): NaN, flagged. None: NaN whatever the run's age.
  missing_after: datetime.timedelta | None = None
  init_dim: str = "init_time"
  lead_dim: str = "lead_time"
  valid_coord: str = "valid_time"

  @field_validator("horizon", "every", "missing_after", mode="before")
  @classmethod
  def _parse(cls, value: str | datetime.timedelta | None) -> datetime.timedelta | None:
    return None if value is None else may_parse_timedelta(value)

  @field_validator("horizon", "every")
  @classmethod
  def _positive(cls, value: datetime.timedelta) -> datetime.timedelta:
    if value <= datetime.timedelta(0):
      raise ValueError("must be positive")
    return value

  def not_published_yet(self, run: datetime.datetime) -> bool:
    """Whether a source without run `run` may still publish it (`missing_after`)."""
    return self.missing_after is not None and now() - run < self.missing_after

  def interval(self, run: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime]:
    """The valid times of run `run`'s block: [run, run + horizon)."""
    return run, run + self.horizon


# The stored units of initialization and valid times: any run of any spacing, to the second.
TIME_UNITS = "seconds since 1970-01-01"


def now() -> datetime.datetime:
  """The current time, UTC, without zone (as the runs' stamps)."""
  return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


@dataclass
class RunReport:
  """What a run's datasets tell about it, gathered while its block is read: per dataset, whether
  its values are the run's own forecast; the datasets whose source had no such run, and the
  variables standing in for them (NaN); the variables each dataset gave (after its own steps),
  recorded in the commit (`VARIABLES_KEY`) for the stand-ins of later runs."""

  real: dict[str, bool] = field(default_factory=dict)
  unavailable: list[str] = field(default_factory=list)
  standing_in: list[str] = field(default_factory=list)
  variables: dict[str, list[str]] = field(default_factory=dict)


class RunNotPublished(RuntimeError):
  """A source has not published a run yet, or not all of it (younger than `missing_after`): the
  download stops before it, the runs before stored, to collect it later."""


# Commit metadata: the variables of each dataset of the forecast runs, {dataset: [variable]}.
VARIABLES_KEY = "arcomake_dataset_variables"


def missing_dataset(
  stored: xr.Dataset,
  names: Iterable[str],
  run: datetime.datetime,
  config: ForecastConfig,
  time_dim: str,
) -> xr.Dataset:
  """The stand-in for a dataset whose source has no run `run`: its variables `names` (as its own
  steps gave them in earlier runs, `VARIABLES_KEY`) taken from the output, those along runs NaN
  along `time_dim` at the run's valid times (lazily: nothing is read), the static ones as stored.
  Without attributes: the caller gives the coordinates those of the run's other datasets (they
  are merged with identical attributes); the recipe's later steps set the variables' again.

  It needs the variables to keep their names through the recipe's later steps, and those along
  runs to be floating point (NaN).

  Raises:
    ValueError: for a variable not in the output (renamed by the later steps?) or not floating
      point.
  """
  names = list(names)
  if missing := [name for name in names if name not in stored.data_vars]:
    raise ValueError(
      f"{missing} are not in the output: the recipe's later steps rename them? A dataset without"
      " its run is stood in for by its variables as stored"
    )
  timed = [name for name in names if config.init_dim in stored[name].dims]
  static = [name for name in names if name not in timed]
  if not_float := [name for name in timed if stored[name].dtype.kind != "f"]:
    raise ValueError(f"{not_float} are not floating point: NaN cannot stand in for them")
  template = stored[timed].isel({config.init_dim: -1}, drop=True)
  template = template.drop_vars([n for n in template.coords if n not in template.dims])
  valid = np.datetime64(run, "ns") + template[config.lead_dim].values
  template = template.assign_coords({config.lead_dim: valid}).rename({config.lead_dim: time_dim})
  filled = xr.full_like(template, np.nan)
  if static:
    filled = filled.assign({name: stored[name] for name in static})
  return filled.drop_attrs()


def series_is_the_run(part_ends: Iterable[np.datetime64], run_end: datetime.datetime) -> bool:
  """Whether a series read for a run (parts' last stamps, as opened) is that run's forecast:
  no stamp at or after the run's end (a later release would reach it)."""
  ends = list(part_ends)
  return bool(ends) and bool(max(ends) < np.datetime64(run_end, "ns"))


def flags(
  report: RunReport, flag_of: Mapping[str, str | None], leads: int, init_dim: str, lead_dim: str
) -> dict[str, xr.DataArray]:
  """The flag variables of a run: per flag name, true if every dataset sharing it is the run's
  own forecast (a dataset left out is not)."""
  values: dict[str, bool] = {}
  for dataset, name in flag_of.items():
    if name is None:
      continue
    real = report.real.get(dataset, False) and dataset not in report.unavailable
    values[name] = values.get(name, True) and real
  return {
    name: xr.DataArray(
      np.full((1, leads), value),
      dims=(init_dim, lead_dim),
      attrs={
        "long_name": f"{name}: values of the run's own forecast",
        "comment": (
          "true where the variables of this component are the forecast of this run; false"
          " where they come from later data (a best-estimate series overwritten by later"
          " releases) or are missing (the source had no such run)"
        ),
      },
    )
    for name, value in values.items()
  }


def as_run(
  block: xr.Dataset,
  run: datetime.datetime,
  config: ForecastConfig,
  time_dim: str,
  run_flags: Mapping[str, xr.DataArray],
) -> xr.Dataset:
  """A processed block of valid times [run, run + horizon) as one run: `time_dim` becomes the
  lead dimension (valid time - run), variables along it get `init_dim` (length one, the run),
  `valid_coord` (init + lead) is added; static variables (without `time_dim`) are kept as they
  are; the flags are added."""
  if time_dim not in block.dims:
    raise ValueError(f"the run of {run} has no {time_dim!r} dimension")
  for name in run_flags:
    if name in block.variables:
      raise ValueError(f"flag {name!r} is also a variable of the run: rename one")
  init = np.datetime64(run, "ns")
  leads = block[time_dim].values.astype("datetime64[ns]") - init
  timed = [name for name, var in block.data_vars.items() if time_dim in var.dims]
  static = [name for name in block.data_vars if name not in timed]
  renamed = block.assign_coords({time_dim: leads}).rename({time_dim: config.lead_dim})
  out = renamed[timed].expand_dims({config.init_dim: [init]})
  out = out.merge(renamed[static], compat="identical", combine_attrs="override")
  # Coordinates without time on dimensions of no variable (cell bounds, on `bnds`): selecting the
  # variables left them out.
  static_coords = {
    name: renamed[name]
    for name in renamed.coords
    if config.lead_dim not in renamed[name].dims and name not in out.coords
  }
  out = out.assign_coords(static_coords)
  out = out.assign(dict(run_flags))
  valid = out[config.init_dim] + out[config.lead_dim]
  out = out.assign_coords({config.valid_coord: valid.transpose(config.init_dim, config.lead_dim)})
  out[config.lead_dim].attrs = {"long_name": "lead time: valid time minus initialization time"}
  out[config.init_dim].attrs = {"long_name": "initialization time of the forecast run"}
  out[config.valid_coord].attrs = {"long_name": "valid time: initialization time plus lead time"}
  # Explicit units: xarray infers them from the first run's one value ("days since" it), which
  # cannot hold the later runs of a recipe with runs less than a day apart.
  for name in (config.init_dim, config.valid_coord):
    out[name].encoding = {"units": TIME_UNITS, "calendar": "proleptic_gregorian", "dtype": "int64"}
  out[config.lead_dim].encoding = {"units": "seconds", "dtype": "int64"}
  return out.assign_attrs(block.attrs)


__all__ = [
  "VARIABLES_KEY",
  "ForecastConfig",
  "RunNotPublished",
  "RunReport",
  "as_run",
  "flags",
  "missing_dataset",
  "series_is_the_run",
]
