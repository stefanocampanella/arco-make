import logging
import sys
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime

import xarray as xr

from arcomake.checks import ChecksConfig, ValidationError
from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("arcomake")
class ArcoMakeSelector:
  _processing_module = sys.modules["arcomake.processing_utils"]
  _checks_module = sys.modules["arcomake.checks"]

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
    Applies a sequence of postprocessing steps to a xarray.Dataset.

    The steps are provided as a list of ProcessingStepConfig configurations.
    All methods defined on a xarray.Dataset can be used as processing steps, in addition to the
    methods defined in this module (which have precedence).

    Args:
      steps (Sequence[ProcessingStepConfig]): Configuration for each processing step.
    Returns:
      xr.Dataset: The processed dataset.
    """
    dataset = self._dataset
    step_names = [step.name for step in steps]
    logger.info("Processing dataset following steps: " + ", ".join(step_names) + ". ")
    for raw_step in steps:
      config = raw_step.model_dump(exclude_unset=True)
      name = config.pop("name")
      logger.info(f"Applying {name} with configuration {config}")
      step_fn: Callable[..., xr.Dataset]
      if name in dir(self._processing_module):
        step_fn = getattr(self._processing_module, name)
        dataset = step_fn(dataset, **config)
      elif name in dir(dataset):
        step_fn = getattr(dataset, name)
        dataset = step_fn(**config)
      else:
        warnings.warn(f"Unrecognized processing step {name} with configuration {config}")
    assert isinstance(dataset, xr.Dataset)
    return dataset

  def validate(
    self,
    checks: ChecksConfig,
    fail=False,
  ) -> None:
    """
    Performs checks on a xarray.Dataset and raise an exception if any check fails.

    The checks are provided as a ChecksConfig model containing step configurations.
    All methods defined in this module can be used as checks.

    Args:
      checks (ChecksConfig): Configuration for each validation step.
      should_raise (bool): Whether to raise an exception if validation fails.
    Returns:
      None
    """
    dataset = self._dataset
    checks_dict = checks.model_dump(exclude_none=True)
    logger.info(
      "Validating dataset with the following checks: " + ", ".join(checks_dict.keys()) + ". "
    )
    for name, config in checks_dict.items():
      if not hasattr(self._checks_module, name):
        warnings.warn(f"Unrecognized validation check {name} with configuration {config}")
        continue
      check_fn: Callable[..., None] = getattr(self._checks_module, name)
      if not isinstance(config, dict):
        config = {}
      try:
        check_fn(dataset, **config)
      except ValidationError as exc:
        failure_message = f"Validation step {name} failed: {exc}"
        if fail:
          raise ValidationError(failure_message) from exc
        else:
          warnings.warn(failure_message)
