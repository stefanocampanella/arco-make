# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The Dask setups of `--dask-setup` (`dask_utils.get_client`)."""

import logging
import sys

import click
import dask
import dask.array as da
import pytest

from arcomake import dask_utils


@pytest.mark.parametrize("setup", ["synchronous", "threads", "processes"])
def test_local_schedulers_last_as_long_as_the_context(setup: str):
  before = dask.config.get("scheduler", None)
  with dask_utils.get_client(setup) as client:
    assert client is None
    assert dask.config.get("scheduler") == setup
  assert dask.config.get("scheduler", None) == before


def test_config_options_take_python_names():
  """Dask's files spell keys with `-`; keyword arguments need `_`."""
  with dask.config.set({"localcluster": {"memory-limit": "1GiB", "n_workers": 2}}):
    assert dask_utils.config_options("localcluster") == {"memory_limit": "1GiB", "n_workers": 2}
  assert dask_utils.config_options("no-such-section") == {}


def test_config_options_refuse_a_value():
  with (
    dask.config.set({"localcluster": "big"}),
    pytest.raises(ValueError, match="table of keyword arguments"),
  ):
    dask_utils.config_options("localcluster")


def test_localcluster_takes_its_workers_keywords():
  """`DASK_LOCALCLUSTER__MEMORY_LIMIT` reaches the workers: only `LocalCluster`'s own parameters
  were passed, so worker keywords such as `memory_limit` were dropped, silently."""
  options = {"n_workers": 1, "threads_per_worker": 1, "processes": False, "memory_limit": "1GiB"}
  with (
    dask.config.set({"localcluster": options, "distributed.dashboard.enabled": False}),
    dask_utils.get_client("localcluster") as client,
  ):
    assert client is not None
    workers = client.scheduler_info()["workers"].values()
    assert [w["memory_limit"] for w in workers] == [2**30]
    assert da.ones(10, chunks=5).sum().compute() == 10
    cluster = client.cluster
  assert client.status == "closed"
  assert cluster is not None and cluster.status.name == "closed"


def test_frisky_is_optional(monkeypatch, caplog):
  monkeypatch.setitem(sys.modules, "frisky", None)  # importing it raises ImportError
  options = {"n_workers": 1, "threads_per_worker": 1, "processes": False}
  with (
    caplog.at_level(logging.WARNING),
    dask.config.set({"localcluster": options}),
    dask_utils.get_client("localcluster", use_frisky=True) as client,
  ):
    assert client is not None
  assert "frisky not found" in caplog.text


def test_mpi_without_the_extra_says_how_to_install_it(monkeypatch):
  monkeypatch.setitem(sys.modules, "dask_mpi", None)
  with (
    pytest.raises(click.ClickException, match=r"--extra mpi"),
    dask_utils.get_client("mpi"),
  ):
    pass


def test_an_unknown_setup_is_refused():
  with (
    pytest.raises(ValueError, match="Invalid Dask setup"),
    dask_utils.get_client("cloud"),  # ty: ignore[invalid-argument-type]
  ):
    pass


class _FakeDaskMpi:
  """dask_mpi's interface, for the command's rank (`initialize` returns `is_client`)."""

  def __init__(self, is_client: bool):
    self.is_client = is_client

  def initialize(self, **kwargs) -> bool:
    return self.is_client

  def send_close_signal(self) -> None:
    pass


def test_mpi_ranks_returned_from_dask_mpi_do_not_run_the_command(monkeypatch):
  """With `exit = false`, the scheduler's and workers' ranks return from `initialize` (False)
  once the cluster closes: they exit instead of running the command."""
  pytest.importorskip("mpi4py")
  monkeypatch.setitem(sys.modules, "dask_mpi", _FakeDaskMpi(is_client=False))
  with pytest.raises(SystemExit) as exit_info, dask_utils.get_client("mpi"):
    pass
  assert exit_info.value.code == 0


def test_mpi_workers_that_never_register_fail_the_command(monkeypatch):
  """A worker rank that failed to start left the command waiting forever."""
  pytest.importorskip("mpi4py")
  import distributed

  calls = []

  class Client:
    def wait_for_workers(self, n, timeout):
      calls.append((n, timeout))
      raise TimeoutError

    def scheduler_info(self):
      return {"workers": {}}

    def shutdown(self):
      calls.append("shutdown")

  monkeypatch.setitem(sys.modules, "dask_mpi", _FakeDaskMpi(is_client=True))
  monkeypatch.setattr(distributed, "Client", Client)
  monkeypatch.setattr(dask_utils, "WORKERS_TIMEOUT", 1)
  with pytest.raises(RuntimeError, match="worker ranks registered in 1 s"):
    dask_utils._mpi_client()
  assert calls[-1] == "shutdown"


def test_errors_closing_the_cluster_are_logged_not_raised(caplog):
  """Raised while the command's error unwinds, or after it succeeded, they replaced its outcome."""

  class Client:
    shut = False

    def retire_workers(self, close_workers):
      raise OSError("worker gone")

    def shutdown(self):
      self.shut = True

  client = Client()
  with caplog.at_level(logging.WARNING):
    dask_utils._shutdown(client)  # ty: ignore[invalid-argument-type]
  assert client.shut and "worker gone" in caplog.text


def test_a_directory_some_worker_does_not_see_is_refused(monkeypatch, tmp_path):
  """Files a worker writes where this process does not look would be missing: chunks of a Zarr
  store read as NaN."""
  import distributed

  class Client:
    def __init__(self, seen: dict[str, bool]):
      self.seen = seen

    def run(self, function, path):
      return {worker: seen and function(path) for worker, seen in self.seen.items()}

  dask_utils.check_workers_reach(tmp_path, "the store")  # no client: nothing to check
  monkeypatch.setattr(distributed, "default_client", lambda: Client({"tcp://a": True}))
  dask_utils.check_workers_reach(tmp_path, "the store")
  monkeypatch.setattr(
    distributed, "default_client", lambda: Client({"tcp://a": True, "tcp://b": False})
  )
  with pytest.raises(RuntimeError, match="tcp://b do not see the store"):
    dask_utils.check_workers_reach(tmp_path, "the store")


def _filters_nan_warnings() -> bool:
  import warnings

  return any("invalid value" in str(f[1]) for f in warnings.filters)


def test_worker_setup_runs_on_every_worker():
  """download's warning filters, on every worker (workers unpickle the function by reference:
  it must be importable there, as arcomake's functions are)."""
  from arcomake.download import ignore_nan_warnings

  options = {"n_workers": 2, "threads_per_worker": 1, "processes": True}
  with (
    dask.config.set({"localcluster": options, "distributed.dashboard.enabled": False}),
    dask_utils.get_client("localcluster") as client,
  ):
    assert client is not None
    assert not any(client.run(_filters_nan_warnings).values())
    client.register_plugin(dask_utils.WorkerSetup(ignore_nan_warnings), name="warnings")
    assert list(client.run(_filters_nan_warnings).values()) == [True, True]
