# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
from datetime import datetime

import click
import dask
import xarray as xr

from arcomake.checks import valid_time_coordinate, validate
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
)
from arcomake.processing import process

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
@click.option(
  "--overwrite/--no-overwrite",
  help="Whether to overwrite existing outputs",
  default=False,
  is_flag=True,
)
@click.option(
  "--should-raise/--no-should-raise",
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
  "--debug/--no-debug",
  "debug",
  help="Use synchronous Dask scheduler",
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
def download(
  config_path: pathlib.Path,
  output_path: pathlib.Path,
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  overwrite: bool = False,
  should_raise: bool = True,
  time_dim: str = "time",
  log_level: str = "info",
  debug: bool = False,
  progress: bool = False,
):
  """
  Download and process multiple datasets into a single ARCO dataset.

  The function reads dataset configurations, applies necessary postprocessing steps,
  and saves the merged dataset to a Zarr store.
  """

  if debug:
    dask.config.set(scheduler="synchronous")

  # Set up logging.
  set_default_logger(log_level)

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
  dataset = xr.Dataset()
  for dataset_name, dataset_conf in configs.get("datasets", {}).items():
    if dataset_conf.get("skip", False) is True:
      logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
      continue
    logger.info(f"Downloading {dataset_name}")
    with maybe_checkpointing_open_dataset(
      dataset_conf, start_datetime, end_datetime, time_dim=time_dim
    ) as source_dataset:
      dataset = xr.merge([dataset, source_dataset], join="exact", compat="no_conflicts")

  # Postprocess the merged dataset (e.g., apply masks)
  if postprocess_conf := configs.pop("postprocess", []):
    dataset = process(dataset=dataset, steps=postprocess_conf)

  # Save the dataset in a Zarr using sensible chunking and compression
  save_to_zarr(
    dataset=dataset,
    path=output_path,
    configs=configs.get("save", {}),
    progress=progress,
  )

  # Validate the dataset
  if checks := configs.pop("checks", {}):
    with xr.open_dataset(output_path, engine="zarr") as dataset:
      validate(
        dataset=dataset,
        checks=checks,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        should_raise=should_raise,
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
@click.option("--freq", help="Time frequency of the timeseries", default="1D")
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
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  freq: str = "1D",
  chunks: dict[str, int] | None = None,
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
  valid_time_coordinate(
    dataset,
    start_datetime=start_datetime
    if start_datetime is not None
    else dataset[time_dim].to_index().min().to_pydatetime(),
    end_datetime=end_datetime
    if end_datetime is not None
    else dataset[time_dim].to_index().max().to_pydatetime(),
    freq=freq,
    time_dim=time_dim,
  )
  dataset = dataset.arcomake.time_sel(start_datetime=start_datetime, end_datetime=end_datetime)
  dataset = dataset.chunk({dim: (1 if dim == time_dim else -1) for dim in dataset.dims})

  save_to_zarr(
    dataset=dataset,
    path=output_path,
    configs={"compressor": {"cname": cname, "clevel": clevel}},
  )
