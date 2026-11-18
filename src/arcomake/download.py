# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import collections
import contextvars
import dataclasses
import functools
import hashlib
import json
import logging
import pathlib
import shutil
import threading
import warnings
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timedelta
from typing import Self, get_args, override

import click
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.utils import parse_bytes
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arcomake import icechunk_utils
from arcomake.chunk_plan import Budget, detect_budget
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
  _write_zarr,
  maybe_checkpointing_download_and_process,
  prepare_for_writing,
  refuse_chunk_steps,
  write_output,
)
from arcomake.datetime_utils import DateInterval, IterableDateInterval, may_parse_timedelta
from arcomake.processing_utils import ProcessingStepConfig
from arcomake.retries import RetryConfig, with_retries
from arcomake.temporary_utils import TempDirectoryRegistry, get_current_temp_registry

logger = logging.getLogger(__name__)


class Memory(click.ParamType[int]):
  """An amount of memory, as Dask writes them: "100GB", "80GiB", "2e9"."""

  name = "memory"

  @override
  def convert(self, value, param, ctx) -> int:
    if isinstance(value, int):
      return value
    try:
      amount = parse_bytes(value)
    except (ValueError, TypeError):
      amount = 0
    if amount <= 0:
      self.fail(f"{value!r} is not an amount of memory (e.g. 100GB, 80GiB)", param, ctx)
    return amount


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

  @field_validator("postprocess")
  @classmethod
  def _no_chunk_steps(cls, steps: list[ProcessingStepConfig]) -> list[ProcessingStepConfig]:
    return refuse_chunk_steps(steps)

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
  "--blocks",
  "blocks_at_once",
  default=1,
  type=click.IntRange(min=1),
  help=(
    "Blocks read and processed at once (default 1). Above 1, each block is staged on local disk"
    " (TMPDIR) and committed in order: memory and disk grow with it"
  ),
)
@click.option(
  "--memory-limit",
  type=Memory(),
  default=None,
  envvar="ARCOMAKE_MEMORY_LIMIT",
  show_envvar=True,
  help=(
    "Memory the chunks of each block are planned for, e.g. 100GB (default: what this process may"
    " use: physical memory, or its cgroup's or Slurm job's limit), shared among Dask's threads"
    " and the blocks at once"
  ),
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
  blocks_at_once: int = 1,
  memory_limit: int | None = None,
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
  with get_client(dask_setup=dask_setup), TempDirectoryRegistry(), warnings.catch_warnings():
    # During postprocessing computation, which may even happen while writing, if each operation
    # does not trigger dask computations (e.g., no calls to persist or compute) some
    # RuntimeWarnings may be issued. This happens frequently with certain algorithms when
    # regridding masked data (containing NaNs). We filter them to avoid cluttering the log.
    # Set once for the command: `catch_warnings` swaps the process-wide filters, and is not
    # thread-safe (blocks run in threads with --blocks).
    warnings.filterwarnings(
      "ignore",
      message="invalid value encountered in divide",
      category=RuntimeWarning,
    )
    # Read configs and inject start and end time.
    update_time_interval = {}
    if start_datetime is not None:
      update_time_interval["start"] = start_datetime
    if end_datetime is not None:
      update_time_interval["end"] = end_datetime
    configs = read_configs(configs_path, inject=update_time_interval, schema=DownloadConfig)
    budget = detect_budget(memory_limit, blocks_at_once)
    logger.info(f"Chunks planned for a budget of {budget.describe()}")
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
    # Each block is committed on the run's branch, in order; a block that fails with a transient
    # error is redone (`retries`), others stop the run.
    todo = [(n, b) for n, b in enumerate(blocks) if block_key(b) not in done]
    progress = progress and dask_setup in ("processes", "threads") and blocks_at_once == 1
    try:
      if blocks_at_once == 1:
        for number, date_interval in todo:
          logger.info(f"Downloading data from {date_interval.start} to {date_interval.end}")
          with_retries(
            functools.partial(
              _write_block,
              configs,
              date_interval,
              run,
              first=number == 0,
              progress=progress,
              recipe=recipe,
              budget=budget,
            ),
            configs.retries,
            _describe(date_interval),
          )
      else:
        _write_blocks_staged(
          configs, todo, run, blocks_at_once=blocks_at_once, recipe=recipe, budget=budget
        )
    except click.ClickException:
      raise
    except Exception as exc:
      logger.exception("An error occurred during download")
      raise click.ClickException(
        f"An error occurred ({type(exc).__name__}). Aborting: the blocks written are on branch"
        f" {run.branch}, `main` is unchanged; --resume continues from there."
      ) from exc
    # Every block is written: `main` moves to them, then the recipe's retention applies.
    try:
      run.finish(configs.save.retention())
    except icechunk_utils.PromotionError as exc:
      raise click.ClickException(str(exc)) from exc


def _describe(interval: DateInterval) -> str:
  return f"Block [{interval.start:%Y-%m-%dT%H:%M}, {interval.end:%Y-%m-%dT%H:%M})"


@contextmanager
def _processed_block(
  configs: DownloadConfig, date_interval: DateInterval, *, progress: bool, budget: Budget | None
) -> Generator[xr.Dataset]:
  """The datasets of one block, read, processed and merged, with the global postprocessing
  (lazy: computed when written); with a `budget`, the parts are chunked as planned for it."""
  # Download and postprocess each dataset, possibly using checkpointing to disk.
  with ExitStack() as stack:
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
            budget=budget,
            name=dataset_name,
          )
        )
      )
    with xr.merge(
      datasets, join="exact", compat="no_conflicts", combine_attrs="identical"
    ) as dataset:
      if configs.postprocess:
        dataset = dataset.arcomake.process(steps=configs.postprocess)
      assert isinstance(dataset, xr.Dataset)
      yield dataset


def _commit_message(interval: DateInterval) -> str:
  """One commit per block, named by its interval: the repository's history tells which blocks
  are in."""
  return f"download [{interval.start:%Y-%m-%dT%H:%M}, {interval.end:%Y-%m-%dT%H:%M})"


def _block_metadata(interval: DateInterval, recipe: str) -> dict[str, str]:
  return {icechunk_utils.BLOCK_KEY: block_key(interval), icechunk_utils.RECIPE_KEY: recipe}


def _write_block(
  configs: DownloadConfig,
  date_interval: DateInterval,
  run: icechunk_utils.Run,
  *,
  first: bool,
  progress: bool,
  recipe: str,
  budget: Budget | None = None,
) -> None:
  """Read, process and merge the datasets of one block, and commit it to the run's branch
  (written by the first block, appended along time by the others)."""
  if _committed(run, date_interval):
    return
  with (
    _processed_block(configs, date_interval, progress=progress, budget=budget) as dataset,
    ProgressBar() if progress else nullcontext(),
  ):
    write_output(
      dataset,
      run,
      configs.save,
      message=_commit_message(date_interval),
      mode="w" if first else "a-",
      append_dim=None if first else configs.time_dim,
      metadata=_block_metadata(date_interval, recipe),
      # Chunks are planned for reading; the stored layout is the recipe's.
      align_chunks=True,
    )


@dataclasses.dataclass
class _Staged:
  """A block computed into a local Zarr store, with what its direct write would have had: the
  prepared dataset (encodings, attributes, coordinates) and its Dask chunks."""

  path: pathlib.Path
  prepared: xr.Dataset


def _stage_block(
  configs: DownloadConfig, date_interval: DateInterval, budget: Budget | None = None
) -> _Staged:
  """Read, process and merge one block, and compute its values into a local Zarr store (a
  worker of `_write_blocks_staged`). Only values go there (no encoding but the chunks): the
  output's encodings are those `prepare_for_writing` gives the block, kept in memory."""
  registry = get_current_temp_registry()
  assert registry is not None
  with _processed_block(configs, date_interval, progress=False, budget=budget) as dataset:
    prepared = prepare_for_writing(dataset, configs.save)
    values = prepared.copy(deep=False)
    for variable in values.variables.values():
      chunks = variable.encoding.get("chunks")
      variable.encoding = {} if chunks is None else {"chunks": chunks}
    path = registry.create_temp_dir(suffix=".zarr")
    logger.info(f"{_describe(date_interval)}: staging to {path}")
    try:
      _write_zarr(values, str(path), mode="w", consolidated=False, align_chunks=True)
    except BaseException:
      shutil.rmtree(path, ignore_errors=True)  # a retry stages again, elsewhere
      raise
  # The prepared dataset is kept for its metadata only: every lazy variable is replaced by the
  # staged values before writing (`_from_stage`).
  return _Staged(path, prepared)


def _from_stage(staged: _Staged) -> xr.Dataset:
  """The block as its direct write would see it: the prepared dataset with the staged values in
  place of every variable not in memory: Dask arrays in the same chunks, lazily indexed arrays
  (variables not chunked) loaded, as their direct write would. The block's sources are
  closed by then: nothing may read them again."""
  values = xr.open_zarr(str(staged.path), consolidated=False)
  dataset = staged.prepared.copy(deep=False)
  for name, variable in dataset.variables.items():
    if variable._in_memory:  # index coordinates, loaded variables: kept as they are
      continue
    stored = values[name].variable
    chunks = variable.chunks
    variable.data = stored.values if chunks is None else stored.data.rechunk(chunks)
  return dataset


def _start(action: Callable[[], _Staged], name: str) -> Future[_Staged]:
  """Run `action` in a daemon thread, in a copy of the current context (the temporary registry
  is a context variable). Daemon: a failed run exits without waiting for the blocks in progress,
  whose work could not be committed anyway (blocks are appended in order)."""
  future: Future[_Staged] = Future()
  context = contextvars.copy_context()

  def target() -> None:
    if not future.set_running_or_notify_cancel():
      return
    try:
      future.set_result(context.run(action))
    except BaseException as error:
      future.set_exception(error)

  threading.Thread(target=target, name=name, daemon=True).start()
  return future


def _write_blocks_staged(
  configs: DownloadConfig,
  todo: list[tuple[int, DateInterval]],
  run: icechunk_utils.Run,
  *,
  blocks_at_once: int,
  recipe: str,
  budget: Budget | None = None,
) -> None:
  """Read and process up to `blocks_at_once` blocks at once, each staged on local disk by a
  worker thread, and commit them in order from this thread (appends along time need it). Each
  stage is removed once committed. The first error, in block order, stops the run: blocks
  before it are committed."""
  pending: collections.deque[tuple[int, DateInterval, Future[_Staged]]] = collections.deque()
  remaining = iter(todo)

  def start_next() -> None:
    item = next(remaining, None)
    if item is None:
      return
    number, interval = item
    logger.info(f"Downloading data from {interval.start} to {interval.end}")
    stage = functools.partial(_stage_block, configs, interval, budget)
    action = functools.partial(with_retries, stage, configs.retries, _describe(interval))
    pending.append((number, interval, _start(action, f"block-{block_key(interval)}")))

  for _ in range(blocks_at_once):
    start_next()
  while pending:
    number, interval, future = pending.popleft()
    staged = future.result()
    first = number == 0

    def commit(staged: _Staged = staged, interval: DateInterval = interval, first: bool = first):
      if _committed(run, interval):
        return
      run.write(
        _from_stage(staged),
        message=_commit_message(interval),
        mode="w" if first else "a-",
        append_dim=None if first else configs.time_dim,
        group=configs.save.group,
        metadata=_block_metadata(interval, recipe),
        align_chunks=True,
      )

    with_retries(commit, configs.retries, f"{_describe(interval)} (commit)")
    shutil.rmtree(staged.path, ignore_errors=True)
    # Only now: at most `blocks_at_once` blocks are staging or staged at any time.
    start_next()


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
  """A fingerprint of what decides what a download writes (the recipe, its start and end): a run
  is resumed only with the same. Settings that only change how blocks are read are left out
  (`[retries]`, parts' `timeout`, `checkpointing`): a run that failed on timeouts can resume with
  longer ones."""
  recipe = configs.model_dump(mode="json", exclude={"retries"})
  for dataset in recipe["datasets"].values():
    dataset.pop("checkpointing", None)
    for part in dataset["parts"]:
      part.pop("timeout", None)
  text = json.dumps(recipe, sort_keys=True)
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
