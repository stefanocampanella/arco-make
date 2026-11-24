# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""netCDF4 and HDF5 files are opened and read under one lock (`arcomake.hdf5_lock`):
`download --blocks 2` opened the GloFAS static file in two blocks at once and failed with
"NetCDF: Can't open HDF5 attribute" (17 Nov, real data). xarray locks reads of array values,
not what it reads while opening a file, and these libraries are not thread-safe."""

import ast
import pathlib
import pickle
import threading
import warnings
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from dask.base import tokenize
from xarray.backends import plugins
from xarray.backends.h5netcdf_ import H5NetCDFStore
from xarray.backends.netCDF4_ import NetCDF4BackendEntrypoint, NetCDF4DataStore

import arcomake
from arcomake import backends
from arcomake.backends import NetCDFOverHTTP
from arcomake.dataset_utils import open_input
from arcomake.hdf5_lock import HDF5_LOCK
from arcomake.temporary_utils import TempDirectoryRegistry, set_current_temp_registry
from tests.conftest import FakeResponse

URL = "https://example.org/static.nc"


def _dataset() -> xr.Dataset:
  """Many attributes: opening reads them all, which is what failed."""
  return xr.Dataset(
    {"uparea": (("y", "x"), np.arange(12.0).reshape(3, 4), {f"a{i}": i for i in range(50)})},
    coords={"y": np.arange(3.0), "x": np.arange(4.0)},
    attrs={f"g{i}": f"value {i}" for i in range(50)},
  )


@pytest.fixture
def served(tmp_path, monkeypatch: pytest.MonkeyPatch) -> xr.Dataset:
  """A NetCDF file at `URL` (a fake `requests.get`) and at `tmp_path / "static.nc"`."""
  expected = _dataset()
  source = tmp_path / "static.nc"
  expected.to_netcdf(source, engine="netcdf4")
  payload = source.read_bytes()
  monkeypatch.setattr(backends.requests, "get", lambda url, **kwargs: FakeResponse(payload))
  return expected


def _held_elsewhere() -> bool:
  """Whether another thread holds `HDF5_LOCK` (it is reentrant: try from a new thread)."""
  acquired = []

  def attempt() -> None:
    if HDF5_LOCK.acquire(blocking=False):
      acquired.append(True)
      HDF5_LOCK.release()

  thread = threading.Thread(target=attempt)
  thread.start()
  thread.join()
  return not acquired


@pytest.fixture
def attrs_read_unlocked(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
  """For each read of a file's attributes while opening: whether `HDF5_LOCK` was not held."""
  unlocked: list[bool] = []
  for store in (NetCDF4DataStore, H5NetCDFStore):
    original = store.get_attrs

    def get_attrs(self, original=original):
      unlocked.append(not _held_elsewhere())
      return original(self)

    monkeypatch.setattr(store, "get_attrs", get_attrs)
  return unlocked


def test_netcdf_over_http_opens_under_the_lock(tmp_path, served, attrs_read_unlocked):
  with TempDirectoryRegistry(base_dir=tmp_path / "tmp"):
    dataset = NetCDFOverHTTP().open_dataset(URL)
    xr.testing.assert_identical(dataset.load(), served)
  assert attrs_read_unlocked == [False]


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf", None, NetCDF4BackendEntrypoint])
def test_inputs_open_under_the_lock(tmp_path, served, attrs_read_unlocked, engine):
  """Generic parts and `[read]` tables too, also when xarray guesses the engine or when it is
  given as a backend class."""
  with open_input(tmp_path / "static.nc", engine=engine, chunks={"y": 1}) as dataset:
    xr.testing.assert_identical(dataset.load(), served)
  assert attrs_read_unlocked == [False]


def test_the_lock_tokenizes_and_pickles_as_itself(tmp_path, served):
  """xarray puts `lock` into the names of the Dask arrays it opens: they stay deterministic."""
  assert tokenize(HDF5_LOCK) == tokenize(HDF5_LOCK)
  assert pickle.loads(pickle.dumps(HDF5_LOCK)) is HDF5_LOCK
  names = []
  for _ in range(2):
    with open_input(tmp_path / "static.nc", engine="netcdf4", chunks={"y": 1}) as dataset:
      names.append(dataset.uparea.data.name)
  assert names[0] == names[1]


def test_the_lock_is_reentrant():
  """The open reads index coordinates through the lock it holds."""
  with HDF5_LOCK, HDF5_LOCK:
    assert _held_elsewhere()
  assert not _held_elsewhere()


def test_plugins_built_on_netcdf4_open_under_the_lock(
  tmp_path, served, attrs_read_unlocked, monkeypatch: pytest.MonkeyPatch
):
  """An engine named by a plugin whose backend extends netCDF4's is locked too."""

  class Plugin(NetCDF4BackendEntrypoint):
    pass

  engines = {**plugins.list_engines(), "nc_plugin": Plugin()}
  monkeypatch.setattr(plugins, "list_engines", lambda: engines)
  with open_input(tmp_path / "static.nc", engine="nc_plugin") as dataset:
    xr.testing.assert_identical(dataset.load(), served)
  assert attrs_read_unlocked == [False]


def test_guessing_the_engine_leaves_the_warning_filters_alone(tmp_path, served):
  """`catch_warnings` swaps the process-wide filters: from block threads, one block could leave
  "ignore" in place for the whole run. Guessing the engine adds no use of it."""
  original = warnings.catch_warnings
  uses = {}
  for engine in ("netcdf4", None):
    calls = []

    def spy(*args, calls=calls, **kwargs):
      calls.append(1)
      return original(*args, **kwargs)

    warnings.catch_warnings = spy  # only around the open: pytest uses it too
    try:
      dataset = open_input(tmp_path / "static.nc", engine=engine)
    finally:
      warnings.catch_warnings = original
    dataset.close()
    uses[engine] = len(calls)
  assert uses[None] == uses["netcdf4"]


def test_a_lock_given_is_left_to_xarray(tmp_path, served, attrs_read_unlocked):
  options = {"engine": "netcdf4", "backend_kwargs": {"lock": False}}
  with open_input(tmp_path / "static.nc", **options) as dataset:
    xr.testing.assert_identical(dataset.load(), served)
  assert attrs_read_unlocked == [True]


OPENERS = {"open_dataset", "open_mfdataset", "open_datatree", "open_groups", "load_dataset"}
HDF5_BACKENDS = {"NetCDF4BackendEntrypoint", "H5netcdfBackendEntrypoint", "NetCDF4DataStore"}


def _bypasses(tree: ast.AST) -> list[int]:
  """Lines that open datasets with xarray without `hdf5_lock`: `xr.open_*` calls without a
  literal engine other than netCDF4 or h5netcdf, xarray's openers imported by name, and xarray's
  netCDF4/h5netcdf backends used directly."""
  lines = []
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("xarray"):
      if any(alias.name in OPENERS | HDF5_BACKENDS for alias in node.names):
        lines.append(node.lineno)
    elif isinstance(node, ast.Attribute) and node.attr in HDF5_BACKENDS:
      lines.append(node.lineno)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      func = node.func
      if func.attr not in OPENERS or not isinstance(func.value, ast.Name):
        continue
      if func.value.id not in ("xr", "xarray"):
        continue
      engines = [k.value for k in node.keywords if k.arg == "engine"]
      if not (engines and isinstance(engines[0], ast.Constant)) or engines[0].value in (
        "netcdf4",
        "h5netcdf",
      ):
        lines.append(node.lineno)
  return lines


@pytest.mark.parametrize(
  "code",
  [
    "xr.open_dataset(path)",
    "xr.open_mfdataset(paths, engine='h5netcdf')",
    "from xarray import open_dataset",
    "from xarray.backends.netCDF4_ import NetCDF4BackendEntrypoint",
    "xarray.backends.netCDF4_.NetCDF4BackendEntrypoint().open_dataset(path)",
  ],
)
def test_the_bypass_check_finds(code: str):
  assert _bypasses(ast.parse(code))


def test_datasets_are_opened_through_the_lock():
  """No code opens a dataset with xarray directly, except through `hdf5_lock` or with an engine
  that is not netCDF4 or HDF5: the lock would be bypassed."""
  package = pathlib.Path(arcomake.__file__).parent
  direct = [
    f"{path.relative_to(package)}:{line}"
    for path in package.rglob("*.py")
    if path.name != "hdf5_lock.py"
    for line in _bypasses(ast.parse(path.read_text()))
  ]
  assert not direct
  assert not _bypasses(ast.parse("xr.open_mfdataset(paths, engine='cfgrib')"))


def _in_threads(action: Callable[[], None], threads: int = 8, rounds: int = 20) -> None:
  """`action` in `threads` threads at once, `rounds` times; fails with the first error."""
  barrier = threading.Barrier(threads, timeout=60)
  errors: list[BaseException] = []

  def run() -> None:
    try:
      for _ in range(rounds):
        barrier.wait()
        action()
    except BaseException as error:  # collected for the main thread
      errors.append(error)
      barrier.abort()  # the others stop waiting

  workers = [threading.Thread(target=run) for _ in range(threads)]
  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  assert not errors, errors[0]


def test_netcdf_over_http_opens_in_many_threads_at_once(tmp_path, served):
  """Blocks run in threads (`download --blocks N`) open the same file at the same moment.
  Without the lock, this crashed the interpreter."""
  with TempDirectoryRegistry(base_dir=tmp_path / "tmp") as registry:

    def open_and_read() -> None:
      set_current_temp_registry(registry)
      dataset = NetCDFOverHTTP().open_dataset(URL)
      xr.testing.assert_identical(dataset.compute(), served)

    _in_threads(open_and_read)


@pytest.mark.parametrize("engine", ["netcdf4", None])
def test_inputs_open_and_read_in_many_threads_at_once(tmp_path, served, engine):
  """Opens of one block overlap the lazy reads (Dask threads) of another."""

  def open_and_read() -> None:
    with open_input(tmp_path / "static.nc", engine=engine, chunks={"y": 1}) as dataset:
      xr.testing.assert_identical(dataset.compute(scheduler="threads"), served)

  _in_threads(open_and_read)
