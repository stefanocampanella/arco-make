import logging
import sys
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime

import xarray as xr

from arcomake.processing_utils import ProcessingStepConfig

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("arcomake")
class ArcoMakeSelector:
  processing_module = sys.modules["arcomake.processing_utils"]

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
    time_coordinate_last_value: datetime = time_coordinate[-1].to_pydatetime()
    if time_coordinate_last_value == end_datetime:
      end_datetime = time_coordinate[-2].to_pydatetime()
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
      dataset (xr.Dataset): The input dataset to process.
      steps (Sequence[ProcessingStepConfig]): Configuration for each processing step.
    Returns:
      xr.Dataset: The processed dataset.
    """
    dataset = self._dataset
    step_names = [step.name for step in steps]
    logger.info("Postprocessing dataset following steps: " + ", ".join(step_names) + ". ")
    for raw_step in steps:
      config = raw_step.model_dump(exclude_unset=True)
      name = config.pop("name")
      logger.info(f"Applying {name} with configuration {config}")
      step_fn: Callable[..., xr.Dataset]
      if name in dir(self.processing_module):
        step_fn = getattr(self.processing_module, name)
        dataset = step_fn(dataset, **config)
      elif name in dir(dataset):
        step_fn = getattr(dataset, name)
        dataset = step_fn(**config)
      else:
        warnings.warn(f"Unrecognized processing step {name} with configuration {config}")
    return dataset
