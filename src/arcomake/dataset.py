# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import pathlib
import pprint
from datetime import datetime
from typing import Any

import click
import dask
import xarray as xr

from arcomake.checks import check_dates, check_global_ecmwf, check_values
from arcomake.cli_utils import (
  DictParamType,
  check_output_path,
  read_configs,
  set_default_logger,
)
from arcomake.dataset_utils import (
  get_dataset,
  open_mfdataset,
  parse_timeseries_arguments,
  save_to_zarr,
  valid_time_coordinate,
)
from arcomake.datetime_utils import DateInterval, IterableDateInterval, may_parse_timedelta
from arcomake.providers import ProvidersRegistry

# TODO:
#   1. Documentation is missing, fix it.
#   2. Should download should use Dask MPI?


logger = logging.getLogger(__name__)


@click.command()
@click.argument(
  "config_path",
  required=True,
  type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True),
)
@click.option(
  "--start", help="Start of the date interval to download", default=None, type=click.DateTime()
)
@click.option(
  "--end", help="End of the date interval to download", default=None, type=click.DateTime()
)
@click.option("--sbatch-flag/--no-sbatch-flag", "sbatch_flag", default=False, is_flag=True)
def array_range(
  config_path: pathlib.Path,
  start: datetime | None = None,
  end: datetime | None = None,
  sbatch_flag: bool = False,
):
  """
  Generate an array of date intervals for downloading data from a configuration file.

  This command allows you to specify a configuration file and optionally a date range to generate
  an array of date intervals for downloading data. The output can be formatted for use with
  Slurm's --array option or printed in a compact format.
  """

  # Open the configuration file and load the TOML configs.
  configs = read_configs(config_path)

  array_id = 0
  array = []
  while True:
    try:
      date_interval, _ = parse_timeseries_arguments(
        configs, pathlib.Path(), start=start, end=end, array_id=array_id
      )
      array.append(date_interval)
      array_id += 1
    except IndexError:
      break

  if sbatch_flag:
    print(f"--array=0-{len(array) - 1}")
  else:
    pprint.pprint(array, compact=True)


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
@click.option("--array-id", help="ID of the SLURM array to download", default=None, type=int)
@click.option(
  "--start",
  "start_date",
  help="Start of the date interval to download",
  default=None,
  type=click.DateTime(),
)
@click.option(
  "--end",
  "end_date",
  help="End of the date interval to download",
  default=None,
  type=click.DateTime(),
)
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
  array_id: int | None = None,
  start_date: datetime | None = None,
  end_date: datetime | None = None,
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

  date_interval, output_path = parse_timeseries_arguments(
    configs, output_path, start=start_date, end=end_date, array_id=array_id
  )

  # Check if the output path exists.
  check_output_path(output_path, overwrite=overwrite)

  def _download_dataset(
      configs: dict[str, Any],
      date_interval: DateInterval | None,
      mask: xr.DataArray | None = None,
  ) -> xr.Dataset:
    provider_name = configs.get("provider")
    if provider_name is None:
      raise ValueError("The 'provider' key must be specified in the config file.")
    elif provider_name not in ProvidersRegistry:
      raise ValueError(f"The 'provider' key value must be one of {ProvidersRegistry.keys()}.")
    else:
      provider = ProvidersRegistry[provider_name](
        progress=progress, log_level=logging.getLevelName(logger.getEffectiveLevel()), client_logger=logger
      )

    dataset_type = configs.get("type")
    if dataset_type is None:
      raise ValueError("The 'type' key must be specified in the config file.")
    if dataset_type not in ["timeseries", "static"]:
      raise ValueError("The 'type' key value must be one of 'static' or 'timeseries'.")
    if date_interval is not None and dataset_type == "timeseries":
      # Compute the IterableDateInterval which will actually be used to download temporary datasets.
      tmp_step = configs.get("tmp_step")
      if tmp_step is None:
        raise ValueError("`tmp_step` must be specified in the top table of the config file.")
      tmp_step = may_parse_timedelta(tmp_step)
      iterable_date_interval = IterableDateInterval(interval=date_interval, step=tmp_step)
    else:
      iterable_date_interval = None
    logger.info(f"Downloading {dataset_type} dataset from {provider_name}")
    dataset = get_dataset(
      provider=provider,
      configs=configs,
      date_intervals=iterable_date_interval,
      mask=mask,
      progress=progress,
    )
    return dataset

  datasets = []
  for dataset_conf in configs.get("datasets", []):
    dataset_name = dataset_conf.get("name")
    if dataset_conf.get("skip") is True:
      logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
      continue
    if mask_conf := dataset_conf.get("mask"):
      mask_ds = _download_dataset(configs=mask_conf, date_interval=None, mask=None)
      mask_name: str = mask_conf["variable"]
      mask = mask_ds[mask_name]
    else:
      mask_ds = None
      mask = None
    ds = _download_dataset(configs=dataset_conf, date_interval=date_interval, mask=mask)
    if mask_ds is not None:
      ds = xr.merge([ds, mask_ds])
    checks = dataset_conf.get("checks", [])
    if "global_ecmwf_coords" in checks:
      ds = check_global_ecmwf(ds)
    if "dates" in checks:
      if freq := dataset_conf.get("freq"):
        ds = check_dates(
          ds,
          start_date=date_interval.start,
          end_date=date_interval.end,
          freq=freq,
        )
      else:
        raise ValueError("`freq` must be specified in the top table of the config file.")
    if "values" in checks:
      ds = check_values(ds, mask=mask)
    datasets.append(ds)
  dataset = xr.merge(datasets, join="inner")

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
