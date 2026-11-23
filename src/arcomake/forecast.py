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

A dataset's `flag` names a boolean variable (`init_dim` x lead): true where its values are run
R's own forecast. Datasets may share a flag (true where all of them are).
"""

import datetime
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, field_validator

from arcomake.datetime_utils import may_parse_timedelta


class ForecastConfig(BaseModel):
  """The `[forecast]` table of a download recipe."""

  model_config = ConfigDict(extra="forbid")

  # Lead times [0, horizon): run R's block covers [R, R + horizon) of valid time.
  horizon: datetime.timedelta
  # Between runs: the recipe's `step` (one block per run).
  every: datetime.timedelta = datetime.timedelta(days=1)
  init_dim: str = "init_time"
  lead_dim: str = "lead_time"
  valid_coord: str = "valid_time"

  @field_validator("horizon", "every", mode="before")
  @classmethod
  def _parse(cls, value: str | datetime.timedelta) -> datetime.timedelta:
    return may_parse_timedelta(value)

  @field_validator("horizon", "every")
  @classmethod
  def _positive(cls, value: datetime.timedelta) -> datetime.timedelta:
    if value <= datetime.timedelta(0):
      raise ValueError("must be positive")
    return value

  def interval(self, run: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime]:
    """The valid times of run `run`'s block: [run, run + horizon)."""
    return run, run + self.horizon


@dataclass
class RunReport:
  """What a run's datasets tell about it, gathered while its block is read: per dataset, whether
  its values are the run's own forecast; the datasets whose source had no such run."""

  real: dict[str, bool] = field(default_factory=dict)
  unavailable: list[str] = field(default_factory=list)


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
  out = out.assign(dict(run_flags))
  valid = out[config.init_dim] + out[config.lead_dim]
  out = out.assign_coords({config.valid_coord: valid.transpose(config.init_dim, config.lead_dim)})
  out[config.lead_dim].attrs = {"long_name": "lead time: valid time minus initialization time"}
  out[config.init_dim].attrs = {"long_name": "initialization time of the forecast run"}
  out[config.valid_coord].attrs = {"long_name": "valid time: initialization time plus lead time"}
  return out.assign_attrs(block.attrs)


def with_missing(
  run: xr.Dataset, stored: xr.Dataset, init_dim: str, unavailable: Iterable[str]
) -> xr.Dataset:
  """`run` with the stored variables along `init_dim` it lacks (datasets whose source had no
  such run) as NaN, shaped and typed as stored (lazily: nothing is read)."""
  missing = [
    name
    for name, var in stored.data_vars.items()
    if init_dim in var.dims and name not in run.variables
  ]
  if not missing:
    return run
  template = stored[missing].isel({init_dim: [-1]}).drop_vars(init_dim)
  filled = xr.full_like(template, np.nan).assign_coords({init_dim: run[init_dim].values})
  filled = filled.drop_vars([n for n in filled.coords if n not in filled.dims], errors="ignore")
  for name in missing:
    filled[name].attrs = stored[name].attrs
  return run.merge(filled, join="left", compat="override")


__all__ = [
  "ForecastConfig",
  "RunReport",
  "as_run",
  "flags",
  "series_is_the_run",
  "with_missing",
]
