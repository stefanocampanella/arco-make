# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Read errors on Zarr sources fail the run; they never become fill values (N8).

Zarr reads a chunk whose object is missing as the fill value (NaN): that is how sparse stores
work. zarr 2's default fsspec store did the same for `PermissionError` and any `OSError`, which
include timeouts and connection errors. A real download lost two hourly ERA5 fields this way,
without any message, and its daily means were computed from 23 hours. zarr 3's fsspec store
raises them; these tests keep checking it, on sources in both Zarr formats (the remote sources
are format 2).
"""

import pathlib
import uuid
from collections.abc import Iterator
from datetime import datetime

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner
from fsspec.implementations.memory import MemoryFileSystem

from arcomake import cli
from arcomake.chunk_plan import Budget
from arcomake.dataset_utils import DatasetConfig, ZarrPartConfig, download_and_process

TIMES = pd.date_range("2020-01-01", periods=24, freq="1h")


class FlakyFileSystem(MemoryFileSystem):
  """An in-memory filesystem (`flaky://`) whose reads of some objects time out."""

  protocol = "flaky"
  failing: set[str] = set()
  hour_3 = ""

  @classmethod
  def _strip_protocol(cls, path):
    return super()._strip_protocol(str(path).removeprefix("flaky://"))

  def cat_file(self, path, start=None, end=None, **kwargs):
    if self._strip_protocol(path) in self.failing:
      raise TimeoutError(f"simulated read timeout on {path}")
    return super().cat_file(path, start=start, end=end, **kwargs)


# The object of hour 3 of `t`, by Zarr format.
HOUR_3 = {2: "t/3.0", 3: "t/c/3/0"}


@pytest.fixture(params=[2, 3], ids=["zarr2", "zarr3"])
def source(request: pytest.FixtureRequest) -> Iterator[str]:
  """An hourly Zarr store, one hour per chunk, on `flaky://`; yields its URL."""
  fsspec.register_implementation("flaky", FlakyFileSystem, clobber=True)
  root = f"/arcomake-tests-{uuid.uuid4().hex}"
  ds = xr.Dataset(
    {"t": (("time", "latitude"), np.arange(48, dtype="f4").reshape(24, 2))},
    coords={"time": TIMES, "latitude": [0.0, 1.0]},
  )
  url = f"memory://{root}/src.zarr"
  ds.chunk(time=1).to_zarr(url, mode="w", consolidated=True, zarr_format=request.param)
  FlakyFileSystem.hour_3 = FlakyFileSystem._strip_protocol(f"{url}/{HOUR_3[request.param]}")
  yield f"flaky://{root}/src.zarr"
  FlakyFileSystem.failing = set()
  MemoryFileSystem().rm(root, recursive=True)


def fail_reads_of_hour_3() -> None:
  FlakyFileSystem.failing = {FlakyFileSystem.hour_3}


def read_part(url: str) -> xr.Dataset:
  part = ZarrPartConfig(filename_or_obj=url, consolidated=True)
  return download_and_process(
    DatasetConfig(parts=[part]),
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    budget=Budget(memory_limit=10**9, threads=1),  # parts chunked as download plans them
  ).load()


def test_download_part_reads_a_remote_zarr_store(source: str):
  """Without errors, a remote Zarr part reads every hour."""
  assert not np.isnan(read_part(source).t.values).any()


def test_missing_chunks_are_fill_values(source: str):
  """A chunk that does not exist is the fill value, as zarr defines (sparse stores)."""
  MemoryFileSystem().rm(FlakyFileSystem.hour_3)
  values = read_part(source).t.values
  assert np.isnan(values[3]).all() and not np.isnan(np.delete(values, 3, axis=0)).any()


def test_download_part_read_error_fails(source: str):
  """A chunk whose read times out fails the download instead of becoming NaN."""
  fail_reads_of_hour_3()
  with pytest.raises(TimeoutError):
    read_part(source)


@pytest.mark.parametrize(
  "read_table",
  [
    'engine = "zarr"',
    "",  # engine guessed by xarray from the `.zarr` suffix
    'engine = "zarr"\nstorage_options = {skip_instance_cache = true}',
  ],
)
def test_read_table_read_error_fails(read_table: str, source: str, tmp_path: pathlib.Path):
  """The `[read]` table of `process` (and `validate`, `compute-climatology`) follows the rule."""
  recipe = tmp_path / "process.toml"
  recipe.write_text(f'[read]\n{read_table}\n\n[[process]]\nname = "mean"\ndim = "time"\n')
  fail_reads_of_hour_3()
  args = ["process", "--dask-setup", "threads", str(recipe), source, str(tmp_path / "out.zarr")]
  result = CliRunner().invoke(cli, args)
  assert result.exit_code != 0 and "(TimeoutError)" in result.output
