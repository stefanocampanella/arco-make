# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import functools
import logging
import math
import warnings
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Self, override

import click
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.utils import format_bytes
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arcomake import checks, icechunk_utils, provenance
from arcomake.backends import RunUnavailable
from arcomake.checks import ChecksConfig, ValidationError
from arcomake.chunk_plan import Budget, PlanError, detect_budget, stored_warnings
from arcomake.cli_utils import (
  check_aws_profile,
  check_destination,
  check_path,
  read_configs,
  set_default_logger,
)
from arcomake.dask_utils import (
  DaskSetupType,
  WorkerSetup,
  check_workers_reach,
  dask_setup_option,
  get_client,
)
from arcomake.dataset_utils import (
  DatasetConfig,
  SaveConfig,
  download_and_process,
  inline_grids,
  maybe_checkpointing_download_and_process,
  prepare_for_writing,
  refuse_chunk_steps,
  write_output,
)
from arcomake.datetime_utils import (
  DateInterval,
  IterableDateInterval,
  may_parse_timedelta,
  naive_utc,
)
from arcomake.forecast import (
  ForecastConfig,
  RunReport,
  as_run,
  flags,
  series_is_the_run,
  with_missing,
)
from arcomake.processing_utils import ProcessingStepConfig, time_coverage
from arcomake.retries import RetryConfig, with_retries
from arcomake.settings import parse_memory
from arcomake.temporary_utils import TempDirectoryRegistry

logger = logging.getLogger(__name__)


class Memory(click.ParamType[int]):
  """An amount of memory, as Dask writes them: "100GB", "80GiB", "2e9"."""

  name = "memory"

  @override
  def convert(self, value, param, ctx) -> int:
    if isinstance(value, int):
      return value
    try:
      return parse_memory(value)
    except ValueError as error:
      self.fail(f"{value!r} is {error}", param, ctx)


class DownloadConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  time_dim: str = "time"
  # The length of a block (a forecast recipe gives `[forecast] every` instead: a run per block).
  step: timedelta | None = None
  # Dates may be left to the command (--start, --end; --extend sets the start): a recipe is then
  # the same for every period, and so is its fingerprint.
  start: datetime | None = None
  end: datetime | None = None
  datasets: dict[str, DatasetConfig]
  postprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig
  # How often a block is redone after a transient error (see `retries.RetryConfig`).
  retries: RetryConfig = Field(default_factory=RetryConfig)
  # Target grids of `regrid` steps, by name (see `dataset_utils.inline_grids`).
  grids: dict[str, dict[str, Any]] = Field(default_factory=dict)
  # Checks run on each block before its commit (see `_checked_block`).
  checks: ChecksConfig | None = None
  # Forecast runs instead of a series (see `forecast`).
  forecast: ForecastConfig | None = None

  @model_validator(mode="before")
  @classmethod
  def _inline_grids(cls, data: Any) -> Any:
    return inline_grids(data)

  @model_validator(mode="before")
  @classmethod
  def _time_check_over_the_block(cls, data: Any) -> Any:
    # Each block's time axis is checked over the block: dates in the table would go unused.
    if isinstance(data, dict) and isinstance(data.get("checks"), dict):
      table = data["checks"].get("valid_time_coordinate")
      if isinstance(table, dict) and ("start" in table or "end" in table):
        raise ValueError(
          "checks.valid_time_coordinate: a download checks each block's time axis over the"
          " block: remove start and end"
        )
    return data

  @field_validator("start", "end")
  @classmethod
  def _without_zone(cls, value: datetime | None) -> datetime | None:
    # Compared with the stored stamps and the start of an extension, which have no zone: an
    # offset made `download --extend` fail comparing them (TypeError).
    return naive_utc(value, "start and end are stamps of the time axis")

  @field_validator("step", mode="before")
  @classmethod
  def parse_step(cls, value: str | timedelta | None) -> timedelta | None:
    return None if value is None else may_parse_timedelta(value)

  @model_validator(mode="after")
  def _blocks_or_runs(self) -> Self:
    if self.forecast is None:
      if self.step is None:
        raise ValueError('step: give the length of a block (e.g. step = "32D")')
      for name, dataset in self.datasets.items():
        if dataset.forecast != "series" or dataset.flag is not None:
          raise ValueError(
            f"datasets.{name}: `forecast` and `flag` are for forecast recipes (a [forecast] table)"
          )
    elif self.step is not None:
      raise ValueError(
        "step: a forecast recipe writes a run per block: give [forecast] every instead"
      )
    return self

  @property
  def block_step(self) -> timedelta:
    """The length of a block: `step`, or the time between forecast runs."""
    if self.forecast is not None:
      return self.forecast.every
    assert self.step is not None  # checked at load
    return self.step

  @field_validator("postprocess")
  @classmethod
  def _no_chunk_steps(cls, steps: list[ProcessingStepConfig]) -> list[ProcessingStepConfig]:
    return refuse_chunk_steps(steps)

  @model_validator(mode="after")
  def validate_date_range(self) -> Self:
    if self.start is not None and self.end is not None and self.start > self.end:
      raise ValueError("start datetime must be before or equal to end datetime")
    return self

  @model_validator(mode="after")
  def _stored_layout(self) -> Self:
    # Without it, the stored chunks would be the chunks planned for reading: they depend on the
    # machine's memory and its threads, and so would the output.
    if self.save.chunks is None:
      raise ValueError(
        "[save] chunks: give the stored layout (e.g. {time = 1}; dimensions not listed are"
        " whole): chunks for reading are planned for each machine"
      )
    return self


def ignore_nan_warnings() -> None:
  """Ignore NumPy's warnings on NaN met while computing (masked data regridded, means of empty
  cells), which would clutter the log: they are expected, and the checks count missing values."""
  warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", category=RuntimeWarning
  )


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
  "--extend/--no-extend",
  help=(
    "Append the days after the last one stored in OUTPUT_PATH, up to --end (or the recipe's end),"
    " as a new version: same recipe (its dates aside), static variables compared with the stored"
    " ones. --start is the day after the last stored; with --resume, continues a failed extension"
  ),
  default=False,
  is_flag=True,
)
@click.option(
  "--memory-limit",
  type=Memory(),
  default=None,
  help=(
    "Memory the chunks of each block are planned for, e.g. 100GB (default: ARCOMAKE_MEMORY_LIMIT,"
    " else what this process may use: physical memory, or its cgroup's or Slurm job's limit),"
    " shared among Dask's threads"
  ),
)
@click.option(
  "--log-level",
  default="info",
  type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
)
@dask_setup_option
@click.option(
  "--progress/--no-progress",
  "progress",
  help="Whether to display a progress bar",
  default=False,
  is_flag=True,
)
@click.option(
  "--plan",
  "plan_only",
  help=(
    "Print the chunks planned for the first block and exit, without reading data or writing"
    " OUTPUT_PATH. Sources are opened (EWDS parts download their files to do so)"
  ),
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
  extend: bool = False,
  memory_limit: int | None = None,
  log_level: str = "info",
  dask_setup: DaskSetupType = "threads",
  progress: bool = False,
  plan_only: bool = False,
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

  A recipe with a [forecast] table writes forecast runs instead: each block is one run, of the
  initialization times of [start, end) every `every`, read over its valid times [run, run +
  horizon) and stored along an initialization dimension, with lead times and valid times
  (see the `forecast` module). --extend adds the runs after the last stored one.
  """

  # Set up logging.
  set_default_logger(log_level)

  # Set up the Dask client and TempDirectoryRegistry.
  with (
    get_client(dask_setup=dask_setup) as client,
    TempDirectoryRegistry() as registry,
    warnings.catch_warnings(),
  ):
    # Set once for the command: `catch_warnings` swaps the process-wide filters, and is not
    # thread-safe (Dask computes in threads).
    ignore_nan_warnings()
    if client is not None:
      # Workers write checkpoints and read the files fetched once per run, in the run's
      # temporary directory; they warn as this process does.
      check_workers_reach(registry.job_dir, "the run's temporary directory")
      client.register_plugin(WorkerSetup(ignore_nan_warnings), name="arcomake-warnings")
    # Read configs and inject start and end time.
    update_time_interval = {}
    if start_datetime is not None:
      update_time_interval["start"] = start_datetime
    if end_datetime is not None:
      update_time_interval["end"] = end_datetime
    configs = read_configs(configs_path, inject=update_time_interval, schema=DownloadConfig)
    tip: icechunk_utils.Tip | None = None
    if extend:
      if replace or start_datetime is not None:
        raise click.ClickException(
          "--extend starts where the output's last block ended: not with --replace nor --start"
        )
      # `main` read once: the extension's start and recipe, and the version it extends.
      tip = _main_tip(output_path, configs)
      configs = configs.model_copy(update={"start": _extension_start(output_path, tip)})
    if configs.start is None or configs.end is None:
      raise click.ClickException(
        "The recipe has no end: give --end"
        if extend
        else "The recipe has no dates: give --start and --end"
      )
    if extend and configs.start >= configs.end:
      logger.info(f"Nothing to extend: {output_path} reaches {configs.end} already")
      return
    budget = detect_budget(memory_limit)
    logger.info(f"Chunks planned for a budget of {budget.describe()}")
    parts = [part for dataset in configs.datasets.values() for part in dataset.parts]
    check_aws_profile(
      output_path,
      *(part.filename_or_obj for part in parts),
      copernicus_marine=any(part.engine == "copernicusmarine" for part in parts),
    )
    blocks = list(IterableDateInterval(configs.start, configs.end, configs.block_step))
    if plan_only:
      if blocks:
        _print_plan(configs, blocks[0], budget)
      return
    # Check the output destination (a failed run waiting there, or an existing version), then
    # start a run on a branch of its own, or continue the failed one.
    recipe = recipe_fingerprint(configs)
    waiting = _waiting_run(
      output_path, configs, resume=resume, replace=replace, extend=extend, recipe=recipe
    )
    if tip is not None:
      _check_extended_recipe(output_path, tip, recipe)
    check_destination(output_path, replace=replace or resume or extend, save_configs=configs.save)
    try:
      run = icechunk_utils.Run(
        output_path,
        configs.save.storage_options,
        icechunk_utils.run_label(EXTEND_COMMAND if extend else "download"),
        waiting=waiting if resume else None,
        recipe=recipe,
      )
    except Exception as exc:
      raise click.ClickException(f"Could not start writing {output_path}: {exc}") from exc
    if tip is not None and run.base != tip.snapshot:
      raise click.ClickException(
        f"main of {output_path} moved while the extension started (another run was promoted):"
        " run it again"
      )
    again = "--extend --resume" if extend else "--resume"
    written = list(waiting.blocks) if resume and waiting is not None else []
    done = set(written)
    if written != [block_key(b) for b in blocks[: len(written)]]:
      raise click.ClickException(
        f"The blocks on {run.branch} are not the first ones of this recipe: start over with --replace"
      )
    if done:
      logger.info(f"Resuming {run.branch}: {len(done)} of {len(blocks)} blocks are written")
    # Each block is committed on the run's branch, in order; a block that fails with a transient
    # error is redone (`retries`), others stop the run.
    todo = [(n, b) for n, b in enumerate(blocks) if block_key(b) not in done]
    progress = progress and dask_setup in ("processes", "threads")
    static = RunStatic()
    try:
      for number, date_interval in todo:
        logger.info(f"Downloading data from {date_interval.start} to {date_interval.end}")
        with_retries(
          functools.partial(
            _write_block,
            configs,
            date_interval,
            run,
            first=number == 0 and not extend,
            progress=progress,
            budget=budget,
            static=static,
          ),
          configs.retries,
          _describe(date_interval),
        )
    except click.ClickException:
      raise
    except ValidationError as exc:
      raise click.ClickException(
        f"A check of [checks] failed, so the block was not committed: {exc}. The blocks written"
        f" are on branch {run.branch}, `main` is unchanged; once the cause (or the check) is"
        f" fixed, {again} continues from there."
      ) from exc
    except PlanError as exc:
      raise click.ClickException(
        f"{exc}. The blocks written are on branch {run.branch}, `main` is unchanged;"
        f" {again} continues from there."
      ) from exc
    except Exception as exc:
      logger.exception("An error occurred during download")
      raise click.ClickException(
        f"An error occurred ({type(exc).__name__}). Aborting: the blocks written are on branch"
        f" {run.branch}, `main` is unchanged; {again} continues from there."
      ) from exc
    # Every block is written: `main` moves to them, then the recipe's retention applies.
    try:
      run.finish(configs.save.retention())
    except icechunk_utils.PromotionError as exc:
      raise click.ClickException(str(exc)) from exc


def _print_plan(configs: DownloadConfig, interval: DateInterval, budget: Budget) -> None:
  """Print the chunks of the first block's merged dataset, as it would be written, with the
  stored chunks (the plan of each dataset is logged as it is opened). Nothing is computed:
  checkpointed datasets are planned over the whole block (a run plans each sub-block, so its
  time chunks may be shorter).

  A forecast recipe's first block is its first run: its datasets are read over the run's valid
  times, and the run is printed as stored (lead and initialization dimensions, no flags).

  Raises:
    click.ClickException: if the budget leaves no room for a task (`PlanError`), or a source has
      no such run (`RunUnavailable`).
  """
  click.echo(f"{_describe(interval)}, budget: {budget.describe()}")
  fc = configs.forecast
  run = interval.start if fc is not None else None
  start, end = fc.interval(interval.start) if fc is not None else (interval.start, interval.end)
  try:
    with ExitStack() as stack:
      datasets = []
      for name, conf in configs.datasets.items():
        if conf.skip:
          continue
        if conf.checkpointing is not None:
          click.echo(f"  ({name} is checkpointed: a run plans each sub-block of it)")
        datasets.append(
          stack.enter_context(
            download_and_process(
              conf,
              start,
              end,
              time_dim=configs.time_dim,
              budget=budget,
              name=name,
              later_steps=configs.postprocess,
              run_datetime=run if conf.forecast == "runs" else None,
            )
          )
        )
      dataset = xr.merge(datasets, join="exact", compat="no_conflicts", combine_attrs="identical")
      if configs.postprocess:
        dataset = dataset.arcomake.process(steps=configs.postprocess)
      if fc is not None and run is not None:
        dataset = as_run(dataset, run, fc, configs.time_dim, {})
      prepared = prepare_for_writing(dataset, configs.save)
  except PlanError as error:
    raise click.ClickException(str(error)) from error
  except RunUnavailable as error:
    raise click.ClickException(f"{error}: plan from another run (--start)") from error
  for name, var in sorted(prepared.data_vars.items(), key=lambda item: str(item[0])):
    dims = ", ".join(map(str, var.dims))
    stored = var.encoding.get("shards") or var.encoding.get("chunks")
    where = f"; stored {tuple(stored)}" if stored is not None else ""
    if var.chunks is None:
      click.echo(f"  {name} ({dims}) {var.dtype}: in memory, {format_bytes(var.nbytes)}{where}")
      continue
    shape = tuple(max(sizes) for sizes in var.chunks)
    size = math.prod(shape) * var.dtype.itemsize
    count = math.prod(len(sizes) for sizes in var.chunks)
    click.echo(
      f"  {name} ({dims}) {var.dtype}: chunks {shape}, {format_bytes(size)} each, {count} chunks"
      f"{where}"
    )
  for warning in stored_warnings(prepared, budget):
    click.echo(f"  warning: {warning}")


def _describe(interval: DateInterval) -> str:
  return f"Block [{interval.start:%Y-%m-%dT%H:%M}, {interval.end:%Y-%m-%dT%H:%M})"


@dataclass
class RunStatic:
  """The static datasets of a run (no time dimension once processed: masks, fractions,
  bathymetry), computed once, by the first block, and reused by the others; `checked` once the
  static variables of a resumed run were compared with the stored ones."""

  datasets: dict[str, xr.Dataset] = field(default_factory=dict)
  checked: bool = False


@contextmanager
def _processed_block(
  configs: DownloadConfig,
  date_interval: DateInterval,
  *,
  progress: bool,
  budget: Budget | None,
  static: RunStatic | None = None,
  run: datetime | None = None,
  report: RunReport | None = None,
) -> Generator[xr.Dataset]:
  """The datasets of one block, read, processed and merged, with the global postprocessing
  (lazy: computed when written); with a `budget`, the parts are chunked as planned for it.

  With `static`, a dataset without the time dimension once processed is computed by the first
  block (in memory) and reused by the others, instead of being read and processed for each.

  With `run` (a forecast run; `date_interval` its valid times), "runs" datasets read that run; a
  source without it leaves its dataset out (`report.unavailable`); `report.real` tells, per
  dataset, whether its values are the run's own forecast (`forecast` module).
  """
  # Download and postprocess each dataset, possibly using checkpointing to disk.
  with ExitStack() as stack:
    datasets: list[xr.Dataset] = []
    valid_times: dict[str, np.ndarray] = {}  # of a forecast run's datasets with time
    for dataset_name, dataset_conf in configs.datasets.items():
      if dataset_conf.skip:
        logger.info(f"Skipping dataset {dataset_name} due to 'skip' flag")
        continue
      if static is not None and dataset_name in static.datasets:
        datasets.append(static.datasets[dataset_name])
        continue
      runs = run is not None and dataset_conf.forecast == "runs"
      part_ends: list[np.datetime64] | None = [] if run is not None else None
      try:
        dataset = stack.enter_context(
          maybe_checkpointing_download_and_process(
            dataset_conf,
            date_interval.start,
            date_interval.end,
            time_dim=configs.time_dim,
            progress=progress,
            budget=budget,
            name=dataset_name,
            later_steps=configs.postprocess,
            run_datetime=run if runs else None,
            part_ends=part_ends,
          )
        )
      except RunUnavailable as error:
        if not runs or report is None:
          raise
        logger.warning(f"{dataset_name}: {error}: NaN in this run, its flag false")
        report.unavailable.append(dataset_name)
        continue
      if report is not None and part_ends is not None:
        report.real[dataset_name] = runs or series_is_the_run(part_ends, date_interval.end)
      if static is not None and configs.time_dim not in dataset.dims:
        logger.info(f"{dataset_name} is static: computed once for the run")
        dataset = dataset.load()
        static.datasets[dataset_name] = dataset
      datasets.append(dataset)
      if run is not None and configs.time_dim in dataset.dims:
        valid_times[dataset_name] = dataset[configs.time_dim].values
    if run is not None:
      _same_valid_times(valid_times, run)
    with xr.merge(
      datasets, join="exact", compat="no_conflicts", combine_attrs="identical"
    ) as dataset:
      if configs.postprocess:
        dataset = dataset.arcomake.process(steps=configs.postprocess)
      assert isinstance(dataset, xr.Dataset)
      if budget is not None:
        for warning in stored_warnings(prepare_for_writing(dataset, configs.save), budget):
          logger.warning(warning)
      yield dataset


def _same_valid_times(valid_times: dict[str, np.ndarray], run: datetime) -> None:
  """A forecast run's datasets cover the same valid times: a source ending before the run's
  horizon (a release not out yet) fails the run, named, before the merge would."""
  if not valid_times:
    return
  longest = max(valid_times.values(), key=len)
  short = {name: times for name, times in valid_times.items() if not np.array_equal(times, longest)}
  if short:
    ends = ", ".join(
      f"{name} to {np.datetime_as_string(times.max(), unit='m') if times.size else 'nothing'}"
      for name, times in short.items()
    )
    last = np.datetime_as_string(longest.max(), unit="m")
    raise ValueError(
      f"The datasets of the run of {run:%Y-%m-%d %H:%M} have different valid times: {ends}, the"
      f" others to {last} (a source not published to the run's horizon yet?)"
    )


def _commit_message(interval: DateInterval) -> str:
  """One commit per block, named by its interval: the repository's history tells which blocks
  are in."""
  return f"download [{interval.start:%Y-%m-%dT%H:%M}, {interval.end:%Y-%m-%dT%H:%M})"


def _block_metadata(interval: DateInterval) -> dict[str, str]:
  """The block of a commit; the run adds the recipe's fingerprint (`Run(recipe=...)`)."""
  return {icechunk_utils.BLOCK_KEY: block_key(interval)}


def _write_block(
  configs: DownloadConfig,
  date_interval: DateInterval,
  run: icechunk_utils.Run,
  *,
  first: bool,
  progress: bool,
  budget: Budget | None = None,
  static: RunStatic | None = None,
) -> None:
  """Read, process and merge the datasets of one block, and commit it to the run's branch
  (written by the first block, appended along time by the others).

  Appends write only variables with the time dimension: the static ones are those of the first
  block. A run that does not write the first block (`--resume`) compares its static variables
  with the stored ones, once: they must be identical.
  """
  if first and static is not None:
    static.checked = True  # this run writes them (or wrote them, if its answer was lost)
  if _committed(run, date_interval):
    return
  fc = configs.forecast
  # A forecast block is run `date_interval.start`, read over its valid times.
  read_interval = date_interval if fc is None else DateInterval(*fc.interval(date_interval.start))
  report = RunReport() if fc is not None else None
  with (
    _processed_block(
      configs,
      read_interval,
      progress=progress,
      budget=budget,
      static=static,
      run=date_interval.start if fc is not None else None,
      report=report,
    ) as dataset,
    ProgressBar() if progress else nullcontext(),
  ):
    group, time_dim = configs.save.group, configs.time_dim
    if not first and static is not None and not static.checked:
      _check_static(dataset, run, group, time_dim)
      static.checked = True
    stamps_of = time_dim if fc is None else fc.valid_coord
    dataset = _with_output_coverage(dataset, run, group, time_dim, first, stored=stamps_of)
    if configs.checks is not None:
      dataset = _checked_block(dataset, configs.checks, read_interval, time_dim)
    if fc is not None and report is not None:
      dataset = _as_stored_run(configs, fc, dataset, date_interval.start, report, run, first)
    append_dim = time_dim if fc is None else fc.init_dim
    write_output(
      dataset,
      run,
      configs.save,
      message=_commit_message(date_interval),
      mode="w" if first else "a-",
      append_dim=None if first else append_dim,
      metadata=_block_metadata(date_interval),
      # Chunks are planned for reading; the stored layout is the recipe's.
      align_chunks=True,
    )


def _as_stored_run(
  configs: DownloadConfig,
  fc: ForecastConfig,
  block: xr.Dataset,
  start: datetime,
  report: RunReport,
  run: icechunk_utils.Run,
  first: bool,
) -> xr.Dataset:
  """A forecast block as the run it is (`forecast.as_run`, with its flags), with NaN for the
  datasets whose source had no such run (as stored before). The first run needs every dataset
  (it creates the arrays); later runs need the stored lead times."""
  if first and report.unavailable:
    raise click.ClickException(
      f"The first run ({start:%Y-%m-%d %H:%M}) lacks {', '.join(report.unavailable)} (no such run"
      " at the source): start from a run every source has (--start)"
    )
  flag_of = {name: conf.flag for name, conf in configs.datasets.items() if not conf.skip}
  run_flags = flags(report, flag_of, block.sizes[configs.time_dim], fc.init_dim, fc.lead_dim)
  out = as_run(block, start, fc, configs.time_dim, run_flags)
  if first:
    return out
  session = run.repo.readonly_session(branch=run.branch)
  with xr.open_zarr(session.store, group=configs.save.group, consolidated=False) as stored:
    if not np.array_equal(stored[fc.lead_dim].values, out[fc.lead_dim].values):
      raise click.ClickException(
        f"The run of {start:%Y-%m-%d %H:%M} has other lead times than those stored (a source"
        f" ends before the run's horizon?): {out[fc.lead_dim].values.astype('timedelta64[h]')}"
      )
    return with_missing(out, stored, fc.init_dim, report.unavailable)


def _checked_block(
  dataset: xr.Dataset, config: ChecksConfig, date_interval: DateInterval, time_dim: str
) -> xr.Dataset:
  """`dataset` checked by the recipe's `[checks]`, like pre-commit hooks: the time axis over the
  block, and the checks reading only coordinates and attributes (`checks.METADATA_CHECKS`), now;
  missing values while the block is written (`guard_no_nans`, no second reading). Tables without
  a `time_dim` take the recipe's. A failure raises `ValidationError` naming the check, so the
  block is not committed.
  """
  if (time_check := config.valid_time_coordinate) is not None:
    _named(
      "valid_time_coordinate",
      checks.valid_time_coordinate,
      dataset,
      date_interval.start,
      date_interval.end,
      freq=time_check.freq,
      time_dim=time_check.time_dim if "time_dim" in time_check.model_fields_set else time_dim,
    )
  for name in sorted(checks.METADATA_CHECKS):
    if (table := getattr(config, name)) is not None:
      _named(name, checks.CHECKS[name], dataset, **table.model_dump())
  if (no_nans := config.ensure_no_nans) is not None:
    table = no_nans.model_dump(exclude_none=True)
    table.setdefault("time_dim", time_dim)
    dataset = checks.guard_no_nans(dataset, **table)
  return dataset


def _named(name: str, check: Callable[..., None], *args: Any, **kwargs: Any) -> None:
  """Run `check`; its `ValidationError` names it."""
  try:
    check(*args, **kwargs)
  except ValidationError as exc:
    raise ValidationError(f"{name}: {exc}") from exc


def _check_static(
  dataset: xr.Dataset, run: icechunk_utils.Run, group: str | None, time_dim: str
) -> None:
  """Fail if a static variable of `dataset` differs from the one stored on the run's branch
  (compared as stored: packed with the stored variable's encoding, NaN equal to NaN): appends do
  not rewrite them, so the output would keep the old ones. Static coordinates other than indexes
  (cell bounds, say) are compared too. Units are not read as durations: "seconds" stays a number,
  whose missing values equal each other."""
  names = [
    name
    for name, var in dataset.variables.items()
    if time_dim not in var.dims and name not in dataset.indexes
  ]
  if not names:
    return
  store = run.repo.readonly_session(branch=run.branch).store
  with xr.open_zarr(store, group=group, consolidated=False, decode_timedelta=False) as stored:
    for name in names:
      if name not in stored:
        raise click.ClickException(
          f"Static variable {name!r} is not in the output: start over with --replace"
        )
      expected = stored[name].values
      computed = np.asarray(_as_stored(dataset[name], stored[name]).values).astype(expected.dtype)
      equal_nan = expected.dtype.kind in "fcmM"  # NaN, NaT
      if computed.shape != expected.shape or not np.array_equal(
        computed, expected, equal_nan=equal_nan
      ):
        raise click.ClickException(
          f"Static variable {name!r} differs from the stored one (a source or the recipe"
          " changed): start over with --replace"
        )
  logger.info(f"Static variables identical to the stored ones: {', '.join(map(str, names))}")


# Encoding keys that change stored values: packing into another type, with scale and offset, and
# the units and calendar of times.
_VALUE_ENCODING = ("dtype", "scale_factor", "add_offset", "_FillValue", "units", "calendar")


def _as_stored(computed: xr.DataArray, stored: xr.DataArray) -> xr.DataArray:
  """`computed` encoded as `stored` was (e.g. floats packed into int16) and decoded again."""
  variable = computed.variable.copy(deep=False)
  variable.encoding = {k: v for k, v in stored.encoding.items() if k in _VALUE_ENCODING}
  encoded = xr.conventions.encode_cf_variable(variable, name=str(computed.name))
  decoded = xr.conventions.decode_cf_variable(computed.name, encoded, decode_timedelta=False)
  return computed.copy(data=decoded.values)


# The attributes `processing_utils.time_coverage` computes (a one-stamp output has no resolution).
TIME_COVERAGE = ("time_coverage_start", "time_coverage_end", "time_coverage_resolution")


def _with_output_coverage(
  dataset: xr.Dataset,
  run: icechunk_utils.Run,
  group: str | None,
  time_dim: str,
  first: bool,
  stored: str | None = None,
) -> xr.Dataset:
  """`dataset` with the time coverage of the whole output, if it has one (`coverage_attrs`).

  xarray replaces a store's attributes with each appended block's: the coverage is computed here
  from the output's stamps (those stored on the run's branch, variable `stored`, by default
  `time_dim`; a forecast's valid times), then the block's, for every block.
  """
  if "time_coverage_start" not in dataset.attrs:
    return dataset
  stamps = dataset[time_dim].values
  if not first:
    store = run.repo.readonly_session(branch=run.branch).store
    with xr.open_zarr(store, group=group, consolidated=False) as output:
      kept = output[stored or time_dim].values.ravel()
      # A forecast's runs share valid times: each counted once, so the resolution is found.
      stamps = np.unique(np.concatenate([kept, stamps]))
  computed = time_coverage(stamps)
  attrs = {k: v for k, v in dataset.attrs.items() if k not in TIME_COVERAGE}
  return dataset.drop_attrs(deep=False).assign_attrs(attrs | computed)


def _committed(run: icechunk_utils.Run, interval: DateInterval) -> bool:
  """Whether the block is already the run's last commit: an attempt whose commit landed but
  whose answer was lost (a transient error) must not append it again."""
  if run.last_block() != block_key(interval):
    return False
  logger.warning(f"{_describe(interval)}: already committed by the previous attempt")
  return True


def block_key(interval: DateInterval) -> str:
  """How a block is named in the commit metadata (`icechunk_utils.BLOCK_KEY`)."""
  return f"{interval.start:%Y-%m-%dT%H:%M}/{interval.end:%Y-%m-%dT%H:%M}"


def recipe_fingerprint(configs: DownloadConfig) -> str:
  """A fingerprint of what decides what a download writes: a run is resumed, and an output
  extended, only with the same. Settings that only change how blocks are read are left out
  (`[retries]`, parts' `timeout`, `checkpointing`): a run that failed on timeouts can resume with
  longer ones. Options at their default are left out too: an option a later version adds, with a
  default keeping the earlier behaviour, does not change the fingerprint of recipes without it.
  `[grids]` is left out: the regrid steps carry the grids they name. `[checks]` is left out: a
  run stopped by a check may resume once the check is changed. The dates are left out: a recipe
  is the same over any period (an extension keeps it), and a resumed run's blocks are checked
  against those stored. `[save]` access and retention (`storage_options`, `keep_history`,
  `keep_versions`) are left out: a renamed profile or a new retention must not block extensions."""
  recipe = configs.model_dump(
    mode="json", exclude={"retries", "grids", "checks", "start", "end"}, exclude_defaults=True
  )
  for key in ("storage_options", "keep_history", "keep_versions"):
    recipe.get("save", {}).pop(key, None)
  for dataset in recipe["datasets"].values():
    dataset.pop("checkpointing", None)
    for part in dataset["parts"]:
      part.pop("timeout", None)
  return provenance.hash_recipe(recipe)


# The command of extension runs (their label and branch): a failed one resumes only as one.
EXTEND_COMMAND = "download-extend"


def _main_tip(output_path: str, configs: DownloadConfig) -> icechunk_utils.Tip:
  try:
    return icechunk_utils.main_tip(output_path, configs.save.storage_options)
  except Exception as exc:
    raise click.ClickException(f"Nothing to extend at {output_path}: {exc}") from exc


def _extension_start(output_path: str, tip: icechunk_utils.Tip) -> datetime:
  """Where an extension starts: the end of the last block `main`'s tip commits (exact, whatever
  the time axis: calendars, labels, resampling)."""
  block = tip.metadata.get(icechunk_utils.BLOCK_KEY)
  if not isinstance(block, str) or "/" not in block:
    raise click.ClickException(
      f"{output_path} records no download block in its last commit: if its first download"
      " failed, resume it (--resume, with its dates); if it is another kind of output, rewrite it"
      " with --replace"
    )
  return datetime.strptime(block.split("/")[1], "%Y-%m-%dT%H:%M")


def _check_extended_recipe(output_path: str, tip: icechunk_utils.Tip, recipe: str) -> None:
  """An extension uses the recipe the output was written with (its dates aside)."""
  stored = tip.metadata.get(icechunk_utils.RECIPE_KEY)
  if stored != recipe:
    raise click.ClickException(
      f"{output_path} was written with another recipe (fingerprint {stored}, this one {recipe}):"
      " extend it with the same, or rewrite it with --replace"
    )


def _waiting_run(
  output_path: str,
  configs: DownloadConfig,
  *,
  resume: bool,
  replace: bool,
  extend: bool,
  recipe: str,
) -> icechunk_utils.WaitingRun | None:
  """The failed download run at `output_path`, checked against the options: --resume needs one
  that can continue with this recipe, of the same kind (an extension resumes only with
  --extend: resumed as a download, its first block would replace the output), a plain run
  refuses to start beside one."""
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
    extension = waiting.label.startswith(f"{EXTEND_COMMAND}-")
    if extension != extend:
      raise click.ClickException(
        f"{waiting.branch} is a failed extension: --extend --resume continues it"
        if extension
        else f"{waiting.branch} is a failed download, not an extension: --resume continues it"
      )
    if waiting.recipe is not None and waiting.recipe != recipe:
      raise click.ClickException(
        f"{waiting.branch} was written with another recipe: resume it with the same, or start"
        " over with --replace"
      )
  elif waiting is not None and waiting.current and not replace:
    if waiting.label.startswith(f"{EXTEND_COMMAND}-"):
      how = "--extend --resume continues it; --replace (without --extend) rewrites the output"
    elif extend:
      how = "it is not an extension: --resume or --replace, without --extend"
    else:
      how = "--resume continues it, --replace starts over"
    raise click.ClickException(
      f"A download run failed at {output_path} ({waiting.branch}, {len(waiting.blocks)} blocks"
      f" written): {how}"
    )
  return waiting
