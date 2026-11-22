# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The Dask scheduler of a command (`--dask-setup`), for the command's duration (`get_client`).

- `synchronous`, `threads`, `processes`: Dask's local schedulers.
- `localcluster`: a `distributed.LocalCluster` on this machine (worker processes with memory
  limits, spilling to disk).
- `mpi`: a cluster over the ranks of an MPI job, with dask-mpi (the `mpi` extra). Rank 0 runs the
  scheduler, rank 1 the command, the others are workers, so a job needs at least 3 ranks (e.g.
  `srun -n 34 arco-make process --dask-setup mpi ...`). The scheduler and worker ranks never
  return from `get_client`: they exit once the command's rank closes the cluster.

The distributed setups take keyword arguments from Dask's configuration (its files, `DASK_*`
variables, and those of arco-make's settings file): `localcluster` those of `LocalCluster` and of
its workers (`DASK_LOCALCLUSTER__N_WORKERS=1`, `DASK_LOCALCLUSTER__MEMORY_LIMIT=24GiB`), `mpi`
those of `dask_mpi.initialize` (`DASK_MPI__NTHREADS=8`, `DASK_MPI__LOCAL_DIRECTORY=/scratch`).
"""

import atexit
import logging
import os
import pathlib
import sys
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from typing import Any, Literal, cast, get_args, override

import click
import dask
import dask.config
import distributed

logger = logging.getLogger(__name__)

DaskSetupType = Literal["synchronous", "threads", "processes", "mpi", "localcluster"]
LOCAL_SCHEDULERS = ("synchronous", "threads", "processes")
# Seconds the command's rank waits for every worker rank of an MPI job to register.
WORKERS_TIMEOUT = 300

dask_setup_option = click.option(
  "--dask-setup",
  default="threads",
  show_default=True,
  type=click.Choice(get_args(DaskSetupType), case_sensitive=False),
  help=(
    "Dask scheduler: one of Dask's local schedulers, a local cluster of worker processes"
    " (localcluster: memory limits, spilling; DASK_LOCALCLUSTER__* settings), or a cluster over the"
    " ranks of an MPI job (mpi: at least 3 ranks, the `mpi` extra; DASK_MPI__* settings)"
  ),
)


def config_options(section: str) -> dict[str, Any]:
  """Keyword arguments from the section `section` of Dask's configuration.

  Keys are given as Python names: Dask's configuration treats `-` and `_` alike, and its files
  usually spell keys with `-`.
  """
  options = dask.config.get(section, default=None) or {}
  if not isinstance(options, dict):
    raise ValueError(f"Dask configuration {section!r} must be a table of keyword arguments")
  return {str(key).replace("-", "_"): value for key, value in options.items()}


@contextmanager
def get_client(
  dask_setup: DaskSetupType = "threads", use_frisky: bool = False
) -> Generator[distributed.Client | None]:
  """The scheduler `dask_setup` while the context lasts: None for Dask's local schedulers, else a
  client of the cluster, closed (with the cluster) at the end.

  Raises:
    ValueError: for an unknown setup.
    click.ClickException: for `mpi` without the `mpi` extra.
  """
  if dask_setup in LOCAL_SCHEDULERS:
    logger.info(f"Using Dask's {dask_setup} scheduler")
    with dask.config.set(scheduler=dask_setup):
      yield None
    return
  with ExitStack() as stack:
    if dask_setup == "localcluster":
      cluster = stack.enter_context(distributed.LocalCluster(**config_options("localcluster")))
      client = distributed.Client(cluster)
      stack.enter_context(client)
      logger.info(f"Using a local Dask cluster, dashboard at {client.dashboard_link}")
    elif dask_setup == "mpi":
      client = _mpi_client()
      stack.callback(_shutdown, client)
      logger.info(f"Using dask-mpi, dashboard at {client.dashboard_link}")
    else:
      raise ValueError(f"Invalid Dask setup: {dask_setup}")
    if use_frisky:
      client = _frisky(client)
    yield client


def check_workers_reach(path: pathlib.Path, what: str) -> None:
  """On a distributed client, fail unless every worker sees the directory `path` (`what`, for the
  message): files a worker writes where this process does not look would be missing, and chunks
  of a Zarr store read as its fill value (NaN), silently."""
  try:
    client = distributed.default_client()
  except ValueError:  # no client: Dask's local schedulers share this process's files
    return
  blind = sorted(w for w, seen in client.run(os.path.isdir, str(path)).items() if not seen)
  if blind:
    raise RuntimeError(
      f"Dask workers {', '.join(blind)} do not see {what} {path}: set TMPDIR to a filesystem every"
      " worker reaches"
    )


class WorkerSetup(distributed.WorkerPlugin):
  """Runs `setup` (a module-level function: workers unpickle it) on every worker, those started
  later included."""

  def __init__(self, setup: Callable[[], None]):
    self._setup = setup

  @override
  def setup(self, worker: distributed.Worker) -> None:
    self._setup()


def _mpi_client() -> distributed.Client:
  """Start a cluster over the ranks of the MPI job (dask-mpi) and connect to it from rank 1."""
  try:
    import dask_mpi
  except ImportError as error:
    raise click.ClickException(
      "--dask-setup mpi needs the `mpi` extra (dask-mpi and mpi4py) and an MPI library:"
      " `uv sync --extra mpi`, or `pip install '.[mpi]'` in arco-make's source tree"
    ) from error
  from mpi4py import MPI

  options = config_options("mpi")
  # Ranks other than 1 run the scheduler or a worker here, and exit when it closes (dask-mpi
  # exits them; with `exit = false` they return False, and must not run the command).
  if not dask_mpi.initialize(**options):
    sys.exit(0)
  # dask-mpi closes the cluster at exit; `_shutdown` does it when the command ends instead.
  atexit.unregister(dask_mpi.send_close_signal)
  client = distributed.Client()
  # Every worker rank, before the command plans its work: they register one by one. A rank that
  # failed to start would leave the command waiting, silently.
  workers = (options.get("comm") or MPI.COMM_WORLD).Get_size() - 2
  try:
    client.wait_for_workers(workers, timeout=WORKERS_TIMEOUT)
  except TimeoutError as error:
    registered = len(client.scheduler_info()["workers"])
    client.shutdown()
    raise RuntimeError(
      f"{registered} of {workers} Dask worker ranks registered in {WORKERS_TIMEOUT} s"
    ) from error
  return client


def _shutdown(client: distributed.Client) -> None:
  """Close the workers, then the scheduler: closed first, the scheduler leaves each worker to log
  its lost connection with a traceback. Errors are logged: raised while the command's own error
  unwinds, or after it succeeded, they would replace its outcome."""
  try:
    client.retire_workers(close_workers=True)
  except Exception as error:
    logger.warning(f"Could not retire the Dask workers: {error!r}")
  try:
    client.shutdown()
  except Exception as error:
    logger.warning(f"Could not shut the Dask cluster down: {error!r}")


def _frisky(client: distributed.Client) -> distributed.Client:
  """The client wrapped by frisky (the `experimental` extra), or as is without it."""
  try:
    import frisky  # ty: ignore[unresolved-import]  # optional "experimental" extra
  except ImportError:
    logger.warning("frisky not found, falling back to default Dask client.")
    return client
  logger.info("Hijacking Dask client with frisky, this is an experimental feature.")
  return cast(distributed.Client, frisky.hijack(client))  # a wrapper, used as one
