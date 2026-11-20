# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""What made an output: recorded in the metadata of every commit (`icechunk_utils.Run.commit`).

`environment()` gives arco-make's version and commit, the Python and operating system, the
machine's size (CPUs, memory), the versions of the packages that decide the values written, and a
hash of every installed distribution. Nothing names the machine or its paths (no host name, no
user, no directory): outputs are published. `fingerprint()` hashes a recipe.
"""

import functools
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
from typing import Any

from pydantic import BaseModel

# The key of the provenance in a commit's metadata.
PROVENANCE_KEY = "arcomake_provenance"

# Written by the image build (CI) next to the package: the commit the image was built from.
BUILD_COMMIT_FILE = pathlib.Path(__file__).with_name("BUILD_COMMIT")

# Packages that decide the values written, by name in the environment.
KEY_PACKAGES = (
  "xarray",
  "zarr",
  "icechunk",
  "dask",
  "numpy",
  "pandas",
  "xarray-regrid",
  "numcodecs",
  "copernicusmarine",
  "cdsapi",
  "gcsfs",
  "cfgrib",
  "eccodes",
  "netcdf4",
  "h5netcdf",
)


@functools.cache
def environment() -> dict[str, Any]:
  """The provenance of this process's outputs (computed once): JSON-serializable."""
  distributions = sorted(
    f"{dist.metadata['Name']}=={dist.version}"
    for dist in importlib.metadata.distributions()
    if dist.metadata["Name"]
  )
  return {
    "arcomake_version": _version("arcomake"),
    "arcomake_commit": _commit(),
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "system": platform.system(),
    "release": platform.release(),
    "machine": platform.machine(),
    "cpus": os.cpu_count(),
    "memory_bytes": _memory_bytes(),
    "packages": {name: version for name in KEY_PACKAGES if (version := _version(name)) is not None},
    "distributions_sha256": hashlib.sha256("\n".join(distributions).encode()).hexdigest(),
  }


def fingerprint(model: BaseModel, *, exclude: set[str] | None = None) -> str:
  """A short hash of a recipe (its parsed form, options at their default left out: an option a
  later version adds, with a default keeping the earlier behaviour, does not change it)."""
  return hash_recipe(model.model_dump(mode="json", exclude=exclude or set(), exclude_defaults=True))


def hash_recipe(recipe: dict[str, Any]) -> str:
  """The short hash of a recipe's JSON form (keys sorted)."""
  return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def _version(name: str) -> str | None:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def _commit() -> str | None:
  """The commit of arco-make's code: `ARCOMAKE_COMMIT`, else the file the image build writes,
  else git, for a checkout ("-dirty" with uncommitted changes); None if none tells."""
  if (commit := os.environ.get("ARCOMAKE_COMMIT", "")) != "":
    return commit
  if BUILD_COMMIT_FILE.is_file():
    return BUILD_COMMIT_FILE.read_text().strip() or None
  source = pathlib.Path(__file__).resolve().parent
  try:
    # Only arco-make's own checkout: installed into another project's tree, git would name it.
    top = pathlib.Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    if source != top / "src" / "arcomake":
      return None
    commit = _git(source, "rev-parse", "HEAD")
    dirty = _git(source, "status", "--porcelain", "--untracked-files=no", "--", ".")
  except (OSError, subprocess.SubprocessError):
    return None
  return f"{commit}-dirty" if dirty else commit


def _git(directory: pathlib.Path, *args: str) -> str:
  result = subprocess.run(
    ["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=10, check=True
  )
  return result.stdout.strip()


def _memory_bytes() -> int | None:
  try:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  except (ValueError, OSError, AttributeError):
    return None
