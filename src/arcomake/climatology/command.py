# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The `compute-climatology` command: reads the input, then computes and writes the
statistics in one Dask graph (`graph`) or series by series (`streamed`)."""

import logging

import click
import xarray as xr

from arcomake import icechunk_utils
from arcomake.cli_utils import (
  check_aws_profile,
  check_destination,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.climatology import binning, graph, streamed
from arcomake.climatology.config import ClimatologyConfig
from arcomake.dask_utils import (
  DaskSetupType,
  dask_setup_option,
  get_client,
)
from arcomake.dataset_utils import (
  open_input,
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
  "configs_path",
  required=True,
  type=str,
  callback=check_path(dir_okay=False),
)
@click.argument(
  "input_path",
  required=True,
  type=str,
  callback=check_path(),
)
@click.argument(
  "output_path",
  required=True,
  type=str,
)
@click.option(
  "--replace/--no-replace",
  help=(
    "Write a new version of an existing output (an Icechunk repository): it replaces what `main`"
    " shows once the run succeeds; earlier versions stay in its history until the recipe's"
    " [save] keep_history or keep_versions expires them"
  ),
  default=False,
  is_flag=True,
)
@click.option(
  "--streaming/--graph",
  default=False,
  help=(
    "Compute each series apart (each variable, and each index of the recipe's `split_dims`),"
    " streaming over time, as many at a time as fit in memory, instead of everything in one Dask"
    " graph (the default)"
  ),
)
@dask_setup_option
@click.option(
  "--frisky",
  "use_frisky",
  default=False,
  is_flag=True,
  help="Hijack Dask client with frisky, this is an experimental feature.",
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
  show_default=True,
)
def compute_climatology(
  configs_path: str,
  input_path: str,
  output_path: str,
  replace: bool = False,
  streaming: bool = False,
  dask_setup: DaskSetupType = "threads",
  use_frisky: bool = False,
  log_level: str = "info",
):
  """Compute a periodic climatology and the standard deviation of the anomalies.

  After the [[preprocess]] steps, and within `reference_period` if set, each time step falls in
  one of `period` bins: by position (`bins = "position"`: step k of every run of `period` steps
  is in bin k) or by day of the year (`bins = "day_of_year"`: daily data, 365 bins, Feb 29 in
  Feb 28's bin, later days of a leap year in the bin of their date). For each bin, the
  climatology is the weighted mean of the values of a centred `window` of bins (wrapping around
  the period), and the anomaly standard deviation is the unbiased weighted standard deviation of
  the same values. Missing values are skipped: each bin counts the values it has.

  The output is one Icechunk repository with two groups, `climatology` and `anomaly_std`
  (xarray: `open_datatree`, or `open_zarr` with `group=`), written in one commit. Assumed, not
  checked: the time axis is sorted and every data variable has the time dimension. Both groups
  have `period` steps along time, the bin index (0 is Jan 1 for day-of-year bins).

  Example::

    period = 365
    bins = "day_of_year"
    reference_period = {start = 1991-01-01, end = 2021-01-01}
    window = 15
    weighting = "gaussian"
    weighting_scale = 3.0
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up Dask client.
  with get_client(dask_setup=dask_setup, use_frisky=use_frisky):
    try:
      # Read configs, then check the output paths (with the recipe's storage options).
      configs = read_configs(configs_path, schema=ClimatologyConfig)
      _warn_ignored_split_dims(configs, streaming)
      check_aws_profile(input_path, output_path)
      check_destination(output_path, replace=replace, save_configs=configs.save)
      # The run's label (its branch, commit metadata and message).
      label = icechunk_utils.run_label("compute-climatology")
      # Open input dataset
      logger.info(f"Opening input dataset from {input_path} with configs {configs.read}")
      with open_input(input_path, **configs.read.model_dump(exclude_unset=True)) as dataset:
        # Preprocess input dataset
        if configs.preprocess:
          dataset = dataset.arcomake.process(steps=configs.preprocess)
        # Statistics over the reference period only, if one is set.
        if configs.reference_period is not None:
          dataset = dataset.arcomake.time_sel(
            configs.reference_period.start, configs.reference_period.end, configs.time_dim
          )
          if dataset.sizes[configs.time_dim] == 0:
            raise ValueError(f"No data in the reference period {configs.reference_period}")
        if not dataset.data_vars:
          raise ValueError("Input dataset contains no data variables.")
        # The bins checked before the output is opened (streaming finds them in every series).
        binning.time_bins(dataset[configs.time_dim], configs.period, configs.bins)
        _compute_and_write(dataset, configs, output_path, label=label, streaming=streaming)
      logger.info("Input dataset closed; closing Dask client.")
    except click.ClickException:
      raise
    except icechunk_utils.PromotionError as exc:
      raise click.ClickException(str(exc)) from exc
    except Exception as exc:
      logger.exception("An error occurred while processing the climatology and anomaly std.")
      raise click.ClickException(f"An error occurred ({type(exc).__name__}). Aborting.") from exc
  logger.info("Climatology complete; Dask client closed.")


def _compute_and_write(
  dataset: xr.Dataset,
  configs: ClimatologyConfig,
  output_path: str,
  *,
  label: str,
  streaming: bool,
) -> None:
  """Compute the climatology and anomaly std of `dataset` and write them into the groups of
  `output_path` (`GROUPS`), series by series (`streamed.compute`) or in one Dask graph
  (`graph.compute`)."""
  if streaming:
    streamed.compute(dataset, configs, output_path, label=label)
  else:
    graph.compute(dataset, configs, output_path, label=label)


def _warn_ignored_split_dims(configs: ClimatologyConfig, streaming: bool) -> None:
  """`split_dims` without `--streaming` (`[]` too: it selected streaming before the option)."""
  if configs.split_dims is not None and not streaming:
    logger.warning(
      f"split_dims {configs.split_dims} ignored: they apply to --streaming only (without it,"
      " everything is one Dask graph)"
    )
