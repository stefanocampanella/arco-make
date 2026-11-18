# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""One lock for arco-make's uses of netCDF4 and HDF5.

The netCDF-C and HDF5 libraries are not thread-safe. xarray locks the reads of array values,
not what it reads while opening a file (dimensions, variables, attributes): two download
blocks run at once (a `--blocks` option, since removed) opened the GloFAS static file at the
same moment and failed ("NetCDF: Can't open HDF5 attribute"), and eight threads opening files at
once crash the interpreter (`tests/test_native_locks.py`). Opens and Dask's reads in threads
still meet.

`open_dataset` holds `HDF5_LOCK` while it opens a file with an HDF5 engine and gives the same
lock to xarray for the reads that follow, so opens and reads never overlap. The lock is
reentrant (the open reads index coordinates through the lock it holds). It replaces xarray's
own netCDF locks for these files, so it excludes only netCDF work done through this module:
every netCDF4 or h5netcdf file arco-make reads is opened here (`dataset_utils.open_input` and
the `netcdfoverhttp` backend; a test checks that no other code opens datasets with xarray
directly). It does not take xarray's locks as well: xarray combines locks in an arbitrary
order, so taking them in ours could deadlock.

The lock covers what netCDF-C does during an open, network included for remote files: a stalled
remote open stops the netCDF reads of other blocks too, as a stalled read already did.
"""

import threading
from typing import Any, override

import xarray as xr
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netCDF4_ import NetCDF4BackendEntrypoint
from xarray.backends.plugins import get_backend, guess_engine

_HDF5_BACKENDS = (NetCDF4BackendEntrypoint, H5netcdfBackendEntrypoint)


class _ReentrantLock:
  """A process-wide reentrant lock that Dask tokenizes the same every time (xarray puts the
  `lock` argument into the names of the Dask arrays it opens) and that pickles as itself."""

  def __init__(self) -> None:
    self._lock = threading.RLock()

  def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
    return self._lock.acquire(blocking, timeout)

  def release(self) -> None:
    self._lock.release()

  def __enter__(self) -> bool:
    return self.acquire()

  def __exit__(self, *args: object) -> None:
    self.release()

  def __dask_tokenize__(self) -> str:
    return "arcomake.hdf5_lock.HDF5_LOCK"

  @override
  def __reduce__(self) -> str:
    return "HDF5_LOCK"


HDF5_LOCK = _ReentrantLock()


def _is_hdf5(engine: Any) -> bool:
  """Whether `engine` (a name or a backend class) reads with netCDF4 or h5netcdf, plugins built
  on them included."""
  if isinstance(engine, str):
    try:
      return isinstance(get_backend(engine), _HDF5_BACKENDS)
    except ValueError:  # not installed: xarray says so when opening
      return False
  return isinstance(engine, type) and issubclass(engine, _HDF5_BACKENDS)


def open_dataset(filename_or_obj: Any, **kwargs: Any) -> xr.Dataset:
  """`xarray.open_dataset`, with opens and reads of HDF5 files under `HDF5_LOCK`.

  Without `engine`, the one xarray would guess is passed on (guessed once). A `lock` given,
  also in `backend_kwargs`, is left to xarray.
  """
  engine = kwargs.get("engine")
  if engine is None:
    try:
      engine = guess_engine(filename_or_obj)  # may warn: catch_warnings is not thread-safe
    except ValueError:  # no engine can open it: xarray says so below
      return xr.open_dataset(filename_or_obj, **kwargs)
  options: dict[str, Any] = {**kwargs, "engine": engine}
  if not _is_hdf5(engine) or "lock" in kwargs or "lock" in (kwargs.get("backend_kwargs") or {}):
    return xr.open_dataset(filename_or_obj, **options)
  with HDF5_LOCK:
    return xr.open_dataset(filename_or_obj, **options, lock=HDF5_LOCK)
