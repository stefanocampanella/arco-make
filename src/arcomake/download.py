# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import Self, get_args

import click
import xarray as xr
from dask.diagnostics import ProgressBar
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arcomake.cli_utils import (
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import (
  DatasetConfig,
  SaveConfig,
  maybe_checkpointing_open_dataset,
  save_to_zarr,
)
from arcomake.processing import ProcessingStepConfig, process

logger = logging.getLogger(__name__)


class DownloadConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  time_dim: str = "time"
  start: datetime
  end: datetime
  datasets: dict[str, DatasetConfig]
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig

  @model_validator(mode="after")
  def validate_date_range(self) -> Self:
    if self.start > self.end:
      raise ValueError("start datetime must be before or equal to end datetime")
    return self


def bar(progress):
  if progress:
    return ProgressBar()
  else:
    return nullcontext()


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, exists=True, dir_okay=False),
)
@click.argument(
  "output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, writable=True),
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

  # Read configs and inject start and end time.
  update_time_interval = {}
  if start_datetime is not None:
    update_time_interval["start"] = start_datetime
  if end_datetime is not None:
    update_time_interval["end"] = end_datetime
  configs = read_configs(config_path, inject=update_time_interval, schema=DownloadConfig)

  # Check if the output path exists.
  check_output_path(output_path, overwrite=overwrite)

  # Set up the Dask client. Notice: distributed dask clusters are not available due to serialization issues.
  with get_client(scheduler_type=scheduler_type):
    if scheduler_type in ("processes", "mpi", "localcluster"):
      logger.warning(
        f"Using {scheduler_type} scheduler, which is not compatible with remote arco-make xarray backends. "
        "Please use 'threads' or 'synchronous' instead."
      )

    logger.info(f"Downloading data from {configs.start} to {configs.end}")
    datasets: list[xr.Dataset] = []
    store = None
    try:
      # Download and postprocess each dataset, possibly using checkpointing to disk.
      for dataset_name, dataset_conf in configs.datasets.items():
        if dataset_conf.skip:
          logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
          continue
        logger.info(f"Downloading {dataset_name}")
        datasets.append(
          maybe_checkpointing_open_dataset(
            dataset_conf,
            configs.start,
            configs.end,
            time_dim=configs.time_dim,
          )
        )
      # Merge the datasets
      with xr.merge(
        datasets, join="exact", compat="no_conflicts", combine_attrs="identical"
      ) as dataset:
        # During postprocessing computation, which may even happen during `save_to_zarr`, if each operation does not
        # trigger dask computations (e.g., no calls to persist or compute) some RuntimeWarnings may be issued.
        # This happens frequently with certain algorithms when regridding masked data (containing NaNs).
        # We filter them to avoid cluttering the log.
        with warnings.catch_warnings():
          warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
          )
          # Postprocess the merged dataset
          if configs.postprocess:
            dataset = process(
              dataset=dataset,
              steps=configs.postprocess,
            )
          # Save the dataset in a Zarr using sensible chunking and compression
          with bar(progress):
            store = save_to_zarr(
              dataset=dataset,
              path=output_path,
              configs=configs.save,
              compute=True,
            )
        # Close the store, see: https://github.com/pydata/xarray/issues/4076
        store.close()
    except Exception as exc:
      logger.exception("An error occurred during download")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
    finally:
      # Clean up temporary files
      for source_dataset in datasets:
        source_dataset.close()
