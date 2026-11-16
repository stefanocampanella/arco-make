# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from arcomake.checks import CHECKS, ChecksConfig, ValidationError
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.steps import resolve as resolve_step

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("arcomake")
class ArcoMakeSelector:
  def __init__(self, dataset: xr.Dataset):
    self._dataset = dataset

  def time_sel(self, start_datetime: Any, end_datetime: Any, time_dim: str = "time") -> xr.Dataset:
    """Select the right-open interval `[start, end)` along `time_dim`.

    `Dataset.sel` with a slice includes its end; this selection excludes it, so that consecutive
    intervals do not overlap. Works for numpy datetimes and cftime calendars, and for empty or
    one-step selections. A dataset without `time_dim` (e.g. static fields) is returned unchanged.

    Raises:
      ValueError: if the time coordinate is not sorted.
    """
    if time_dim not in self._dataset.coords:
      return self._dataset
    index = self._dataset.indexes[time_dim]
    if not index.is_monotonic_increasing:
      raise ValueError(f"Time coordinate is not sorted: {self._dataset[time_dim]}")
    if isinstance(index, xr.CFTimeIndex):
      # cftime dates do not compare with `datetime` or strings: convert the bounds.
      start_datetime = _as_cftime(start_datetime, index.date_type)
      end_datetime = _as_cftime(end_datetime, index.date_type)
    selected = self._dataset.sel({time_dim: slice(start_datetime, end_datetime)})
    # The index is sorted, so the stamps before `end` are a prefix. Select them with a slice:
    # an index array would rechunk Dask arrays (and change the order of later reductions).
    n_before_end = int(np.count_nonzero(np.asarray(selected.indexes[time_dim] < end_datetime)))
    return selected.isel({time_dim: slice(0, n_before_end)})

  def process(
    self,
    steps: Sequence[ProcessingStepConfig],
  ) -> xr.Dataset:
    """
    Apply recipe steps, in order, and return the resulting dataset.

    A step name is either a registered arco-make step or one of the allowed `xarray.Dataset`
    methods (see `arcomake.steps`); arco-make steps take precedence. Steps never modify the
    dataset they receive.

    Args:
      steps (Sequence[ProcessingStepConfig]): Configuration for each processing step.
    Returns:
      xr.Dataset: The processed dataset.
    Raises:
      ValueError: if a step name is unknown.
    """
    dataset = self._dataset
    step_names = [step.name for step in steps]
    logger.info("Processing dataset following steps: " + ", ".join(step_names) + ". ")
    for raw_step in steps:
      config = raw_step.model_dump(exclude_unset=True)
      name = config.pop("name")
      step_fn = resolve_step(name)
      logger.info(f"Applying {name} with configuration {config}")
      dataset = step_fn(dataset, **config)
    assert isinstance(dataset, xr.Dataset)
    return dataset

  def validate(
    self,
    checks: ChecksConfig,
    fail=False,
  ) -> None:
    """
    Performs checks on a xarray.Dataset and raise an exception if any check fails.

    The checks are the tables of a `ChecksConfig`; each table name is a function of
    `arcomake.checks` (see `checks.CHECKS`).

    Args:
      checks (ChecksConfig): Configuration for each validation step.
      fail (bool): Raise on the first failed check instead of warning.
    Returns:
      None
    """
    dataset = self._dataset
    checks_dict = checks.model_dump(exclude_none=True)
    logger.info(
      "Validating dataset with the following checks: " + ", ".join(checks_dict.keys()) + ". "
    )
    for name, config in checks_dict.items():
      check_fn = CHECKS[name]
      try:
        check_fn(dataset, **config)
      except ValidationError as exc:
        failure_message = f"Validation step {name} failed: {exc}"
        if fail:
          raise ValidationError(failure_message) from exc
        else:
          warnings.warn(failure_message)


def _as_cftime(value: Any, date_type: type) -> Any:
  """`value` (a cftime date, `datetime`, numpy or pandas time, or string) as `date_type`."""
  if isinstance(value, date_type):
    return value
  # A cftime date of another calendar is taken field by field, as any other time.
  t = value if isinstance(value, cftime.datetime) else pd.Timestamp(value)
  return date_type(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)
