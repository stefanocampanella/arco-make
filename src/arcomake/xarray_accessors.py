# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import warnings
from collections.abc import Sequence
from datetime import datetime

import xarray as xr

from arcomake import steps as step_registry
from arcomake.checks import CHECKS, ChecksConfig, ValidationError
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("arcomake")
class ArcoMakeSelector:
  def __init__(self, dataset: xr.Dataset):
    self._dataset = dataset

  def time_sel(self, start_datetime: datetime, end_datetime: datetime, time_dim: str = "time"):
    """XArray DataArray and Dataset sel method is right inclusive, this accessor provides right exclusive selection."""
    if time_dim not in self._dataset.coords:
      return self._dataset
    time_coordinate = self._dataset[time_dim]
    time_coordinate = time_coordinate.sel(time=slice(start_datetime, end_datetime))
    time_coordinate = time_coordinate.to_index()
    if not time_coordinate.is_monotonic_increasing:
      raise ValueError(f"Time coordinate is not sorted: {self._dataset[time_dim]}")
    time_coordinate_last_value = time_coordinate[-1].to_pydatetime()
    assert isinstance(time_coordinate_last_value, datetime)
    if time_coordinate_last_value == end_datetime:
      end_val = time_coordinate[-2].to_pydatetime()
      assert isinstance(end_val, datetime)
      end_datetime = end_val
    return self._dataset.sel(time=slice(start_datetime, end_datetime))

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
      step_fn = step_registry.resolve(name)
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
