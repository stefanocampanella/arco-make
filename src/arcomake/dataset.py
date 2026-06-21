# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from datetime import datetime, timedelta

import click
import dask
import xarray as xr

from arcomake.cli_utils import (
  DictParamType,
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dataset_utils import (
  maybe_checkpointing_open_dataset,
  open_mfdataset,
  save_to_zarr,
  valid_time_coordinate,
)
from arcomake.datetime_utils import may_parse_timedelta

# TODO:
#   1. Documentation is missing, fix it.
#   2. Should download should use Dask MPI?


logger = logging.getLogger(__name__)


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
@click.option("--array-id", help="ID of the SLURM array to download", default=None, type=int)
@click.option("--array-step", default=None, type=str)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--progress/--no-progress",
  "progress",
  help="Whether to display a progress bar",
  default=False,
  is_flag=True,
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
@click.option(
  "--debug/--no-debug",
  "debug",
  help="Use synchronous Dask scheduler",
  default=False,
  is_flag=True,
)
def download(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  array_id: int | None = None,
  array_step: str | timedelta | None = None,
  progress: bool = False,
  log_level: str = "info",
  overwrite: bool = False,
  debug: bool = False,
):
  """
  Download an process multiple datasets into a single ARCO dataset.

  The function reads dataset configurations, applies necessary postprocessing steps,
  and saves the merged dataset to a Zarr store.
  """

  if debug:
    dask.config.set(scheduler="synchronous")

  # Set up logging.
  set_default_logger(log_level)

  # Open the configuration file and load the TOML configs.
  configs = read_configs(config_path)

  # Update start_datetime and end_datetime based CLI arguments
  if start_datetime is not None:
    logger.info(f"Overriding start datetime with {start_datetime}")
  if end_datetime is not None:
    logger.info(f"Overriding end datetime with {end_datetime}")
  start_datetime = start_datetime or configs["start"]
  end_datetime = end_datetime or configs["end"]

  if array_id is not None and array_step is not None:
    logger.info(f"Array ID {array_id} with step {array_step}")
    array_step = may_parse_timedelta(array_step)
    start_datetime = start_datetime + array_id * array_step
    end_datetime = min(end_datetime + (array_id + 1) * array_step, configs["end"])
    output_path = output_path / f"{start_datetime.strftime('%Y%m%d')}-{end_datetime.strftime('%Y%m%d')}"

  # Check if the output path exists.
  check_output_path(output_path, overwrite=overwrite)

  # Download and postprocess each dataset, possibly using checkpointing to disk
  dataset = xr.Dataset()
  for dataset_name, dataset_conf in configs.get("datasets", {}).items():
    if dataset_conf.get("skip", False) is True:
      logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
      continue
    logger.info(f"Downloading {dataset_name}")
    with maybe_checkpointing_open_dataset(dataset_conf, start_datetime, end_datetime) as source_dataset:
      dataset = xr.merge([dataset, source_dataset], join="exact")

  # Save the dataset in a Zarr using sensible chunking and compression
  save_to_zarr(
    dataset=dataset, path=output_path, configs=configs.get("save", {}), progress=progress
  )


@click.command()
@click.argument(
  "input_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, dir_okay=True, readable=True),
)
@click.argument(
  "output_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, resolve_path=True, dir_okay=True, writable=True),
)
@click.option("--time-dim", help="Time dimension name used in the input dataset", default="time")
@click.option(
  "--start",
  "start_date",
  help="Override start of the date interval",
  default=None,
  type=click.DateTime(),
)
@click.option(
  "--end",
  "end_date",
  help="Override end of the date interval",
  default=None,
  type=click.DateTime(),
)
@click.option(
  "--chunks",
  default=None,
  show_default=True,
  type=DictParamType(),
  help="String containing chunking specs used when reading.",
)
@click.option(
  "--compressor-name",
  "cname",
  default="lz4",
  show_default=True,
  help="Name of the compressor to use.",
)
@click.option(
  "--compressor-level", "clevel", default=1, show_default=True, help="Compressor level to use."
)
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--debug/--no-debug", default=False, help="Use synchronous Dask scheduler", is_flag=True
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
def unpack_zips(
  input_path: pathlib.Path,
  output_path: pathlib.Path,
  time_dim: str = "time",
  start_date: datetime | None = None,
  end_date: datetime | None = None,
  chunks: dict | None = None,
  cname: str = "lz4",
  clevel: int = 1,
  overwrite: bool = False,
  debug: bool = False,
  log_level: str = "info",
):
  """
  Unpack a collection of zipped Zarr datasets into a single directory Zarr store.

  The command reads zipped Zarr fragments with xarray.open_mfdataset(engine='zarr'),
  optionally slices the time range, rechunks, and saves to a DirectoryStore at the
  given output path.
  """

  if debug:
    dask.config.set(scheduler="synchronous")

  # Set up logging.
  set_default_logger(log_level)

  # Check if the output path exists.
  check_output_path(output_path, overwrite=overwrite)

  def build_paths(path: pathlib.Path):
    if path.is_dir():
      return sorted(path.glob("*.zip"))
    else:
      return [path.with_suffix(".zip")]

  paths = build_paths(input_path)
  logger.info(f"Reading {len(paths)} zipped Zarrs from {input_path}")

  dataset = open_mfdataset(paths, chunks=chunks)
  if not valid_time_coordinate(dataset, time_dim=time_dim):
    raise ValueError(
      f"Time coordinate {time_dim} is not valid (contains duplicates or missing dates)"
    )
  dataset = dataset.sel(time=slice(start_date, end_date))
  dataset = dataset.chunk({dim: (1 if dim == "time" else -1) for dim in dataset.dims})

  save_to_zarr(
    dataset=dataset,
    path=output_path,
    configs={"compressor": {"cname": cname, "clevel": clevel}},
  )
