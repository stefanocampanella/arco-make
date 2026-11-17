# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Read errors on Zarr sources fail the run; they never become fill values (N8).

zarr 2 reads a chunk whose object is missing as the fill value (NaN): that is how sparse stores
work. Its default fsspec store does the same for `PermissionError` and any `OSError`, which
include timeouts and connection errors. A real download lost two hourly ERA5 fields this way,
without any message, and its daily means were computed from 23 hours.
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
from arcomake.dataset_utils import DatasetConfig, ZarrPartConfig, download_and_process

TIMES = pd.date_range("2020-01-01", periods=24, freq="1h")


class FlakyFileSystem(MemoryFileSystem):
  """An in-memory filesystem (`flaky://`) whose reads of some objects time out."""

  protocol = "flaky"
  failing: set[str] = set()

  @classmethod
  def _strip_protocol(cls, path):
    return super()._strip_protocol(str(path).removeprefix("flaky://"))

  def cat_file(self, path, start=None, end=None, **kwargs):
    if self._strip_protocol(path) in self.failing:
      raise TimeoutError(f"simulated read timeout on {path}")
    return super().cat_file(path, start=start, end=end, **kwargs)


@pytest.fixture
def source() -> Iterator[str]:
  """An hourly Zarr store, one hour per chunk, on `flaky://`; yields its URL."""
  fsspec.register_implementation("flaky", FlakyFileSystem, clobber=True)
  root = f"/arcomake-tests-{uuid.uuid4().hex}"
  ds = xr.Dataset(
    {"t": (("time", "latitude"), np.arange(48, dtype="f4").reshape(24, 2))},
    coords={"time": TIMES, "latitude": [0.0, 1.0]},
  )
  ds.chunk(time=1).to_zarr(f"memory://{root}/src.zarr", mode="w", consolidated=True)
  yield f"flaky://{root}/src.zarr"
  FlakyFileSystem.failing = set()
  MemoryFileSystem().rm(root, recursive=True)


def fail_reads_of(url: str, key: str) -> None:
  FlakyFileSystem.failing = {FlakyFileSystem._strip_protocol(f"{url}/{key}")}


def read_part(url: str) -> xr.Dataset:
  part = ZarrPartConfig(filename_or_obj=url, consolidated=True, chunks={})
  return download_and_process(
    DatasetConfig(parts=[part]), datetime(2020, 1, 1), datetime(2020, 1, 2)
  ).load()


def test_download_part_reads_a_remote_zarr_store(source: str):
  """Without errors, a remote Zarr part reads every hour."""
  assert not np.isnan(read_part(source).t.values).any()


def test_missing_chunks_are_fill_values(source: str):
  """A chunk that does not exist is the fill value, as zarr defines (sparse stores)."""
  MemoryFileSystem().rm(FlakyFileSystem._strip_protocol(f"{source}/t/3.0"))
  values = read_part(source).t.values
  assert np.isnan(values[3]).all() and not np.isnan(np.delete(values, 3, axis=0)).any()


def test_download_part_read_error_fails(source: str):
  """A chunk whose read times out fails the download instead of becoming NaN."""
  fail_reads_of(source, "t/3.0")
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
  fail_reads_of(source, "t/3.0")
  args = ["process", "--dask-setup", "threads", str(recipe), source, str(tmp_path / "out.zarr")]
  result = CliRunner().invoke(cli, args)
  assert result.exit_code != 0 and "(TimeoutError)" in result.output
