# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import pathlib
import warnings
from datetime import datetime
from typing import Literal, get_args

import click

from arcomake.checks import ValidationError, valid_time_coordinate
from arcomake.cli_utils import (
  DictParamType,
  ListParamType,
  check_output_path,
  set_default_logger,
)
from arcomake.dask_distributed_utils import SchedulerOptionType, get_client
from arcomake.dataset_utils import (
  open_archive,
  save_to_zarr,
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
  "start_datetime",
  help="Override start of the date interval",
  default=None,
  type=click.DateTime(),
)
@click.option(
  "--end",
  "end_datetime",
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
  "--attrs-to-drop",
  default=None,
  show_default=True,
  type=ListParamType(),
  help="List of attributes to drop from the dataset.",
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
  "--scheduler-type",
  default="mpi",
  type=click.Choice(
    get_args(SchedulerOptionType),
    case_sensitive=False,
  ),
  help="Type of Dask scheduler to use.",
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
def unpack(
  input_path: pathlib.Path,
  output_path: pathlib.Path,
  time_dim: str = "time",
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  freq: str = "1D",
  chunks: dict[str, int | Literal["auto"]] | None = None,
  attrs_to_drop: list[str] | None = None,
  cname: str = "lz4",
  clevel: int = 1,
  overwrite: bool = False,
  scheduler_type: SchedulerOptionType = "mpi",
  should_raise: bool = False,
  log_level: str = "info",
):
  """
  Unpack a collection of zip files containing Zarr datasets into a single Zarr dataset.

  The command reads zipped Zarr fragments with xarray.open_mfdataset(engine='zarr'),
  optionally slices the time range, rechunks, and saves to a DirectoryStore at the
  given output path.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  client = get_client(scheduler_type=scheduler_type)

  # Check input and ouput paths
  if not input_path.exists() or not input_path.is_dir():
    raise ValueError(f"Invalid input path: {input_path} does not exist or is not a directory")
  check_output_path(output_path, overwrite=overwrite)

  # Validate chunks argument
  if chunks is not None and not all(
    isinstance(value, int) or value == "auto" for value in chunks.values()
  ):
    raise ValueError("Chunk option value must be a dictionary with integer or 'auto' values")

  dataset = open_archive(input_path, chunks=chunks, attrs_to_drop=attrs_to_drop)

  try:
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
  except ValidationError as exc:
    if should_raise:
      raise exc from None
    else:
      warnings.warn(f"Datetimes validation failed: {exc}")

  dataset = dataset.chunk({dim: (1 if dim == time_dim else -1) for dim in dataset.dims})

  store = save_to_zarr(
    dataset=dataset,
    path=output_path,
    configs=dict(compressor={"cname": cname, "clevel": clevel}, consolidated=True),
    compute=True,
  )

  # Clean up
  store.close()
  dataset.close()
  client.close()
