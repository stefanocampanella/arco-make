# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from contextlib import nullcontext
from datetime import datetime
from typing import get_args

import click
import xarray as xr
from dask.diagnostics import ProgressBar

from arcomake.checks import validate
from arcomake.cli_utils import (
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import (
  maybe_checkpointing_open_dataset,
  save_to_zarr,
)
from arcomake.processing import process

logger = logging.getLogger(__name__)


def bar(progress):
  if progress:
    return ProgressBar()
  else:
    return nullcontext()


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, file_okay=True, readable=True),
)
@click.argument(
  "output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, dir_okay=True, writable=True),
)
@click.option(
  "--start",
  "start_datetime",
  default=None,
  help="Override start datetime of the timeseries.",
  type=click.DateTime(),
)
@click.option(
  "--end",
  "end_datetime",
  default=None,
  help="Override end datetime of the timeseries.",
  type=click.DateTime(),
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--raise/--no-raise",
  "should_raise",
  help="Whether to raise an exception if validation fails",
  default=True,
  is_flag=True,
)
@click.option("--time-dim", help="Time dimension name used in the input dataset", default="time")
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
@click.option(
  "--scheduler-type",
  default="threads",
  type=click.Choice(
    get_args(SchedulerOptionType),
    case_sensitive=False,
  ),
  help="Type of Dask scheduler to use, for remote downloads the choice is between 'synchronous' or 'threads'.",
)
@click.option(
  "--progress/--no-progress",
  "progress",
  help="Whether to display a progress bar",
  default=False,
  is_flag=True,
)
def download(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  overwrite: bool = False,
  should_raise: bool = True,
  time_dim: str = "time",
  log_level: str = "info",
  scheduler_type: SchedulerOptionType = "threads",
  progress: bool = False,
):
  """
  Download and process multiple datasets into a single ARCO dataset.

  The function reads dataset configurations, applies necessary postprocessing steps,
  and saves the merged dataset to a Zarr store.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up the Dask client. Notice: distributed dask clusters are not available due to serialization issues.
  client = get_client(scheduler_type=scheduler_type)
  if scheduler_type in ("processes", "mpi", "localcluster"):
    logger.warning(
      f"Using {scheduler_type} scheduler, which is not compatible with remote arco-make xarray backends. "
      "Please use 'threads' or 'synchronous' instead."
    )

  # Open the configuration file and load the TOML configs.
  configs = read_configs(config_path)

  # Update start_datetime and end_datetime based on CLI arguments
  start_datetime = start_datetime or configs["start"]
  end_datetime = end_datetime or configs["end"]
  if (
    not isinstance(start_datetime, datetime)
    or not isinstance(end_datetime, datetime)
    or start_datetime > end_datetime
  ):
    raise ValueError(
      "start_datetime and end_datetime must be datetime objects, and end_datetime must be after start_datetime"
    )
  logger.info(f"Downloading data from {start_datetime} to {end_datetime}")

  # Check if the output path exists.
  check_output_path(output_path, overwrite=overwrite)

  # Download and postprocess each dataset, possibly using checkpointing to disk
  datasets = []
  try:
    for dataset_name, dataset_conf in configs.get("datasets", {}).items():
      if dataset_conf.get("skip", False) is True:
        logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
        continue
      logger.info(f"Downloading {dataset_name}")
      datasets.append(
        maybe_checkpointing_open_dataset(
          dataset_conf, start_datetime, end_datetime, time_dim=time_dim
        )
      )
    dataset: xr.Dataset = xr.merge(
      datasets, join="exact", compat="no_conflicts", combine_attrs="identical"
    )

    # Postprocess the merged dataset (e.g., apply masks)
    if postprocess_conf := configs.get("postprocess", []):
      dataset = process(dataset=dataset, steps=postprocess_conf)

    # Save the dataset in a Zarr using sensible chunking and compression
    with bar(progress):
      store = save_to_zarr(
        dataset=dataset,
        path=output_path,
        configs=configs.get("save", {}),
        compute=True,
      )

    # Clean up
    store.close()
    dataset.close()
  finally:
    # Clean up temporary files
    for source_dataset in datasets:
      source_dataset.close()

  # Validate the dataset
  if checks := configs.get("checks", {}):
    with xr.open_dataset(output_path, engine="zarr") as dataset:
      validate(
        dataset=dataset,
        checks=checks,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        should_raise=should_raise,
      )

  client.close()
