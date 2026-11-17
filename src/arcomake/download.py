# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import functools
import hashlib
import logging
import warnings
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta
from typing import Self, get_args

import click
import xarray as xr
from dask.diagnostics import ProgressBar
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arcomake import icechunk_utils
from arcomake.cli_utils import (
  check_aws_profile,
  check_destination,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_utils import DaskSetupType, get_client
from arcomake.dataset_utils import (
  DatasetConfig,
  SaveConfig,
  maybe_checkpointing_download_and_process,
  write_output,
)
from arcomake.datetime_utils import DateInterval, IterableDateInterval, may_parse_timedelta
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.retries import RetryConfig, with_retries
from arcomake.temporary_utils import TempDirectoryRegistry

logger = logging.getLogger(__name__)


class DownloadConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  time_dim: str = "time"
  step: timedelta
  start: datetime
  end: datetime
  datasets: dict[str, DatasetConfig]
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig
  # How often a block is redone after a transient error (see `retries.RetryConfig`).
  retries: RetryConfig = Field(default_factory=RetryConfig)

  @field_validator("step", mode="before")
  @classmethod
  def parse_step(cls, value: str | timedelta) -> timedelta:
    return may_parse_timedelta(value)

  @model_validator(mode="after")
  def validate_date_range(self) -> Self:
    if self.start > self.end:
      raise ValueError("start datetime must be before or equal to end datetime")
    return self


@click.command()
@click.argument(
  "configs_path",
  required=True,
  type=str,
  callback=check_path(dir_okay=False),
)
@click.argument(
  "output_path",
  required=True,
  type=str,
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
  "--resume/--no-resume",
  help=(
    "Continue a download that failed on this output: its committed blocks are kept, the others"
    " written (same recipe, --start and --end)"
  ),
  default=False,
  is_flag=True,
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
@click.option(
  "--dask-setup",
  default="threads",
  type=click.Choice(
    get_args(DaskSetupType),
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
  configs_path: str,
  output_path: str,
  start_datetime: datetime | None = None,
  end_datetime: datetime | None = None,
  replace: bool = False,
  resume: bool = False,
  log_level: str = "info",
  dask_setup: DaskSetupType = "threads",
  progress: bool = False,
):
  """Download, process and merge the datasets of a recipe into one Zarr dataset.

  The interval [start, end) of the recipe (or of --start/--end) is processed in blocks of
  `step`: for each block, every dataset is read from its parts, time-sliced and processed by
  its [[postprocess]] steps (optionally in sub-blocks checkpointed to temporary Zarr, per
  `checkpointing.step`); the datasets are merged, the global [[postprocess]] steps run, and the
  block is written (first block) or appended along time, one commit per block. OUTPUT_PATH is an
  Icechunk repository: a directory, an s3:// or a gs:// URL. The blocks are committed on a branch
  of their own; `main` moves to them only when the last one is written, so a failed run leaves
  the repository's current version as it was. An existing repository is only written with
  --replace; the version it replaces stays in the history until the recipe's retention expires
  it.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up the Dask client and TempDirectoryRegistry.
  # Notice: distributed dask clusters are not available due to serialization issues.
  if dask_setup in ("processes", "mpi", "localcluster"):
    logger.warning(
      f"Using {dask_setup} scheduler, which is not compatible with remote arco-make xarray backends. "
      "Please use 'threads' or 'synchronous' instead."
    )
  with get_client(dask_setup=dask_setup), TempDirectoryRegistry():
    # Read configs and inject start and end time.
    update_time_interval = {}
    if start_datetime is not None:
      update_time_interval["start"] = start_datetime
    if end_datetime is not None:
      update_time_interval["end"] = end_datetime
    configs = read_configs(configs_path, inject=update_time_interval, schema=DownloadConfig)
    parts = [part for dataset in configs.datasets.values() for part in dataset.parts]
    check_aws_profile(
      output_path,
      *(part.filename_or_obj for part in parts),
      copernicus_marine=any(part.engine == "copernicusmarine" for part in parts),
    )
    # Check the output destination (a failed run waiting there, or an existing version), then
    # start a run on a branch of its own, or continue the failed one.
    recipe = recipe_fingerprint(configs)
    waiting = _waiting_run(output_path, configs, resume=resume, replace=replace, recipe=recipe)
    check_destination(output_path, replace=replace or resume, save_configs=configs.save)
    try:
      run = icechunk_utils.Run(
        output_path,
        configs.save.storage_options,
        icechunk_utils.run_label("download"),
        waiting=waiting if resume else None,
      )
    except Exception as exc:
      raise click.ClickException(f"Could not start writing {output_path}: {exc}") from exc
    blocks = list(IterableDateInterval(configs.start, configs.end, configs.step))
    written = list(waiting.blocks) if resume and waiting is not None else []
    done = set(written)
    if written != [block_key(b) for b in blocks[: len(written)]]:
      raise click.ClickException(
        f"The blocks on {run.branch} are not the first ones of this recipe: start over with --replace"
      )
    if done:
      logger.info(f"Resuming {run.branch}: {len(done)} of {len(blocks)} blocks are written")
    # One block at a time, each committed on the run's branch; a block that fails with a
    # transient error is redone (`retries`), others stop the run.
    for number, date_interval in enumerate(blocks):
      if block_key(date_interval) in done:
        continue
      what = f"Block [{date_interval.start:%Y-%m-%dT%H:%M}, {date_interval.end:%Y-%m-%dT%H:%M})"
      logger.info(f"Downloading data from {date_interval.start} to {date_interval.end}")
      try:
        with_retries(
          functools.partial(
            _write_block,
            configs,
            date_interval,
            run,
            first=number == 0,
            progress=progress and dask_setup in ("processes", "threads"),
            metadata={
              icechunk_utils.BLOCK_KEY: block_key(date_interval),
              icechunk_utils.RECIPE_KEY: recipe,
            },
          ),
          configs.retries,
          what,
        )
      except click.ClickException:
        raise
      except Exception as exc:
        raise click.ClickException(
          f"An error occurred ({type(exc).__name__}). Aborting: the blocks written are on branch"
          f" {run.branch}, `main` is unchanged; --resume continues from there."
        ) from exc
    # Every block is written: `main` moves to them, then the recipe's retention applies.
    try:
      run.finish(configs.save.retention())
    except icechunk_utils.PromotionError as exc:
      raise click.ClickException(str(exc)) from exc


def _write_block(
  configs: DownloadConfig,
  date_interval: DateInterval,
  run: icechunk_utils.Run,
  *,
  first: bool,
  progress: bool,
  metadata: dict[str, str],
) -> None:
  """Read, process and merge the datasets of one block, and commit it to the run's branch
  (written by the first block, appended along time by the others)."""
  # Download and postprocess each dataset, possibly using checkpointing to disk.
  with ExitStack() as stack, warnings.catch_warnings():
    # During postprocessing computation, which may even happen while writing, if each operation
    # does not trigger dask computations (e.g., no calls to persist or compute) some
    # RuntimeWarnings may be issued. This happens frequently with certain algorithms when
    # regridding masked data (containing NaNs). We filter them to avoid cluttering the log.
    warnings.filterwarnings(
      "ignore",
      message="invalid value encountered in divide",
      category=RuntimeWarning,
    )
    datasets: list[xr.Dataset] = []
    for dataset_name, dataset_conf in configs.datasets.items():
      if dataset_conf.skip:
        logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
        continue
      datasets.append(
        stack.enter_context(
          maybe_checkpointing_download_and_process(
            dataset_conf,
            date_interval.start,
            date_interval.end,
            time_dim=configs.time_dim,
            progress=progress,
          )
        )
      )
    with xr.merge(
      datasets, join="exact", compat="no_conflicts", combine_attrs="identical"
    ) as dataset:
      if configs.postprocess:
        dataset = dataset.arcomake.process(steps=configs.postprocess)
      # One commit per block, named by its interval: the repository's history tells which
      # blocks are in.
      block = f"download [{date_interval.start:%Y-%m-%dT%H:%M}, {date_interval.end:%Y-%m-%dT%H:%M})"
      with ProgressBar() if progress else nullcontext():
        if first:
          write_output(dataset, run, configs.save, message=block, mode="w", metadata=metadata)
        else:
          write_output(
            dataset,
            run,
            configs.save,
            message=block,
            mode="a-",
            append_dim=configs.time_dim,
            metadata=metadata,
          )


def block_key(interval: DateInterval) -> str:
  """How a block is named in the commit metadata (`icechunk_utils.BLOCK_KEY`)."""
  return f"{interval.start:%Y-%m-%dT%H:%M}/{interval.end:%Y-%m-%dT%H:%M}"


def recipe_fingerprint(configs: DownloadConfig) -> str:
  """A fingerprint of everything that decides what a download writes (the recipe, its start and
  end; not the retries): a run is resumed only with the same."""
  text = configs.model_dump_json(exclude={"retries"})
  return hashlib.sha256(text.encode()).hexdigest()[:16]


def _waiting_run(
  output_path: str, configs: DownloadConfig, *, resume: bool, replace: bool, recipe: str
) -> icechunk_utils.WaitingRun | None:
  """The failed download run at `output_path`, checked against the options: --resume needs one
  that can continue with this recipe, a plain run refuses to start beside one."""
  try:
    waiting = icechunk_utils.waiting_run(output_path, configs.save.storage_options, "download")
  except Exception as exc:
    raise click.ClickException(f"Could not check {output_path} for a failed run: {exc}") from exc
  if resume:
    if waiting is None:
      raise click.ClickException(f"Nothing to resume: no failed download run at {output_path}")
    if not waiting.current:
      raise click.ClickException(
        f"main changed since {waiting.branch} started, which can no longer be promoted: start"
        " over with --replace"
      )
    if waiting.recipe is not None and waiting.recipe != recipe:
      raise click.ClickException(
        f"{waiting.branch} was written with another recipe (or --start/--end): resume it with"
        " the same, or start over with --replace"
      )
  elif waiting is not None and waiting.current and not replace:
    raise click.ClickException(
      f"A download run failed at {output_path} ({waiting.branch}, {len(waiting.blocks)} blocks"
      " written): --resume continues it, --replace starts over"
    )
  return waiting
