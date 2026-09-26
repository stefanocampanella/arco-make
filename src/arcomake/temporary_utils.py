# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import atexit
import contextlib
import contextvars
import logging
import os
import pathlib
import shutil
import signal
import sys
import tempfile
import threading
import uuid
from typing import Any, Self

logger = logging.getLogger(__name__)


def _pid_exists(pid: int) -> bool:
  """Check if a process with given PID is running on the system."""
  if pid <= 0:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    # Process exists but is owned by another user
    return True
  except OSError:
    return False
  return True


_ACTIVE_REGISTRIES: set["TempDirectoryRegistry"] = set()
_ACTIVE_REGISTRIES_LOCK = threading.Lock()
_SIGNAL_HANDLERS_INSTALLED = False


def _cleanup_all_active_registries() -> None:
  """Clean up all currently active registries on process exit or termination."""
  with _ACTIVE_REGISTRIES_LOCK:
    registries = list(_ACTIVE_REGISTRIES)
  for reg in registries:
    with contextlib.suppress(Exception):
      reg.close()


atexit.register(_cleanup_all_active_registries)


def _install_signal_handlers() -> None:
  """Install signal handlers on the main thread for graceful cleanup."""
  global _SIGNAL_HANDLERS_INSTALLED
  if _SIGNAL_HANDLERS_INSTALLED:
    return
  if threading.current_thread() is not threading.main_thread():
    return

  signals_to_handle = [
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
  ]

  for sig in signals_to_handle:
    try:
      prev_handler = signal.getsignal(sig)
      if prev_handler in (signal.SIG_DFL, None):

        def _handler(signum: int, frame: Any, prev: Any = prev_handler) -> None:
          _cleanup_all_active_registries()
          if callable(prev):
            prev(signum, frame)
          else:
            sys.exit(128 + signum)

        signal.signal(sig, _handler)
    except (ValueError, OSError):
      pass

  _SIGNAL_HANDLERS_INSTALLED = True


_CURRENT_TEMP_REGISTRY: contextvars.ContextVar["TempDirectoryRegistry | None"] = (
  contextvars.ContextVar("current_temp_registry", default=None)
)


def get_current_temp_registry() -> "TempDirectoryRegistry | None":
  """Get the active temporary directory registry in the current context, if any."""
  return _CURRENT_TEMP_REGISTRY.get()


def set_current_temp_registry(registry: "TempDirectoryRegistry | None") -> None:
  """Set the active temporary directory registry in the current context."""
  _CURRENT_TEMP_REGISTRY.set(registry)


class TempDirectoryRegistry:
  """Thread-safe and context-scoped manager for temporary directories and resources.

  Ensures temporary directories remain intact while lazy computations (like Dask)
  stream data from disk, and are deterministically deleted upon exiting the context.
  Also manages process-scoped scratch paths and self-heals stale directories from killed runs.
  """

  BASE_DIR_NAME: str = "arcomake"

  @classmethod
  def get_base_scratch_dir(cls) -> pathlib.Path:
    """Return the root scratch directory for arcomake temporary files."""
    base = pathlib.Path(tempfile.gettempdir()) / cls.BASE_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base

  @classmethod
  def clean_stale_directories(cls, base_dir: str | pathlib.Path | None = None) -> None:
    """Scan and delete temporary directories left behind by dead or killed processes."""
    base = cls.get_base_scratch_dir() if base_dir is None else pathlib.Path(base_dir)
    if not base.exists():
      return
    current_pid = os.getpid()
    for item in base.glob("pid_*"):
      if not item.is_dir():
        continue
      try:
        parts = item.name.split("_")
        if len(parts) >= 2:
          pid = int(parts[1])
          if pid != current_pid and not _pid_exists(pid):
            shutil.rmtree(item, ignore_errors=True)
      except (ValueError, IndexError):
        continue

  def __init__(self, base_dir: str | pathlib.Path | None = None):
    self.clean_stale_directories(base_dir)
    _install_signal_handlers()

    self._stack = contextlib.ExitStack()
    self._lock = threading.Lock()
    self._token: contextvars.Token[TempDirectoryRegistry | None] | None = None
    self._pid = os.getpid()
    self._closed = False

    parent_base = pathlib.Path(base_dir) if base_dir is not None else self.get_base_scratch_dir()
    parent_base.mkdir(parents=True, exist_ok=True)
    job_dir_path = tempfile.mkdtemp(
      prefix=f"pid_{self._pid}_{uuid.uuid4().hex[:6]}_", dir=parent_base
    )
    self._job_dir = pathlib.Path(job_dir_path)

    with _ACTIVE_REGISTRIES_LOCK:
      _ACTIVE_REGISTRIES.add(self)

  @property
  def job_dir(self) -> pathlib.Path:
    """Return the process-scoped root directory for this registry instance."""
    return self._job_dir

  def create_temp_dir(
    self,
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | pathlib.Path | None = None,
    **kwargs: Any,
  ) -> pathlib.Path:
    """Create a temporary directory tracked by this registry."""
    with self._lock:
      target_dir = dir if dir is not None else self._job_dir
      tmpdir = self._stack.enter_context(
        tempfile.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=target_dir, **kwargs)
      )
      return pathlib.Path(tmpdir)

  def register[T: contextlib.AbstractContextManager[Any]](self, context_or_cleanup: T) -> T:
    """Register an existing context manager (e.g. TemporaryDirectory) for cleanup."""
    with self._lock:
      return self._stack.enter_context(context_or_cleanup)

  def close(self) -> None:
    """Clean up all registered temporary directories and the job directory."""
    with self._lock:
      if self._closed:
        return
      self._closed = True

      with _ACTIVE_REGISTRIES_LOCK:
        _ACTIVE_REGISTRIES.discard(self)

      try:
        self._stack.close()
      finally:
        if self._job_dir.exists():
          shutil.rmtree(self._job_dir, ignore_errors=True)

  def __enter__(self) -> Self:
    self._token = _CURRENT_TEMP_REGISTRY.set(self)
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    try:
      self.close()
    finally:
      if self._token is not None:
        _CURRENT_TEMP_REGISTRY.reset(self._token)
