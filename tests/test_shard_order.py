# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`[save] shard_order`: the physical order of the chunks inside shards, read back from the
shards' indexes, for arrays created by a write and appended to, written into sessions, and
written region by region (the climatology's series)."""

import asyncio
import pathlib
import pickle
import subprocess
import sys

import dask
import icechunk
import numpy as np
import pandas as pd
import pydantic
import pytest
import xarray as xr
import zarr
from click.testing import CliRunner
from zarr.core.buffer import default_buffer_prototype

from arcomake import cli, icechunk_utils, sharding
from arcomake.dataset_utils import SaveConfig, write_into_session, write_output

LAYOUT = {"chunks": {"time": 1, "depth": 1}, "shards": {"time": 4, "depth": -1}}


def physical_order(path: pathlib.Path, name: str, shard: tuple[int, ...]) -> np.ndarray:
  """Where each chunk of a shard lies in its object: its rank by offset, over the chunk grid."""
  store = icechunk_utils.readonly_store(path)
  array = zarr.open_array(store, path=name, mode="r")
  per_shard = tuple(s // c for s, c in zip(array.shards, array.chunks, strict=True))
  key = f"{name}/c/" + "/".join(str(i) for i in shard)
  raw = asyncio.run(store.get(key, prototype=default_buffer_prototype()))
  assert raw is not None
  count = int(np.prod(per_shard))
  # The index, at the end of the shard: (offset, length) per chunk, then a CRC32C checksum.
  index = np.frombuffer(raw.to_bytes()[-(count * 16 + 4) : -4], dtype="<u8").reshape(count, 2)
  return np.argsort(np.argsort(index[:, 0])).reshape(per_shard)


def expected(order: str, shape: tuple[int, ...]) -> np.ndarray:
  return np.arange(int(np.prod(shape))).reshape(shape, order="F" if order[0] == "c" else "C")


def ocean(days: int, start: str = "2020-01-01") -> xr.Dataset:
  values = np.random.default_rng(len(start) + days).random((days, 5, 3, 4)).astype("float32")
  return xr.Dataset(
    {"thetao": (("time", "depth", "lat", "lon"), values)},
    coords={"time": pd.date_range(start, periods=days, freq="1D"), "depth": np.arange(5.0)},
  ).chunk(time=4)


@pytest.mark.parametrize("order", ["lexicographic", "colexicographic"])
def test_shards_of_created_and_appended_arrays_follow_the_order(tmp_path: pathlib.Path, order):
  """As in a download: the first block creates the arrays, the next ones are appended (zarr
  builds their codec from the metadata, where the order is not stored), and Dask chunks that
  straddle shards are aligned to them (the planner's chunks need not fit the stored layout).
  Values and metadata are those of zarr's default order."""
  first, then = ocean(8).chunk(time=3), ocean(8, "2020-01-09").chunk(time=3)
  outputs = {}
  for chosen in ("morton", order):
    configs = SaveConfig.model_validate({**LAYOUT, "shard_order": chosen})
    path = tmp_path / f"{chosen}.zarr"
    run = icechunk_utils.Run(path, None, "test")
    write_output(first, run, configs, message="first", align_chunks=True)
    write_output(then, run, configs, message="then", mode="a", append_dim="time", align_chunks=True)
    run.finish()
    outputs[chosen] = path
  for shard in range(4):  # two shards per block
    assert np.array_equal(
      physical_order(outputs[order], "thetao", (shard, 0, 0, 0)), expected(order, (4, 5, 1, 1))
    )
  assert not np.array_equal(
    physical_order(outputs["morton"], "thetao", (0, 0, 0, 0)), expected(order, (4, 5, 1, 1))
  )
  read = {
    k: xr.open_zarr(icechunk_utils.readonly_store(p), consolidated=False)
    for k, p in outputs.items()
  }
  xr.testing.assert_identical(read["morton"].load(), read[order].load())
  metadata = {
    k: zarr.open_array(icechunk_utils.readonly_store(p), path="thetao").metadata.to_dict()
    for k, p in outputs.items()
  }
  assert metadata["morton"] == metadata[order]


def test_writes_into_sessions_follow_the_order(tmp_path: pathlib.Path):
  """As the climatology writes its outputs: metadata now, values computed later."""
  configs = SaveConfig.model_validate({**LAYOUT, "shard_order": "lexicographic"})
  path = tmp_path / "out.zarr"
  run = icechunk_utils.Run(path, None, "test")
  session = run.session()
  write = write_into_session(ocean(8), session, configs, compute=False)
  with dask.config.set(scheduler="threads"):
    dask.compute(write)
  run.commit(session, "test")
  run.finish()
  assert np.array_equal(
    physical_order(path, "thetao", (1, 0, 0, 0)), expected("lexicographic", (4, 5, 1, 1))
  )
  xr.testing.assert_equal(
    xr.open_zarr(icechunk_utils.readonly_store(path), consolidated=False).thetao.load(),
    ocean(8).thetao.load(),
  )


@pytest.mark.parametrize("split", ["", 'split_dims = ["lat"]\n'], ids=["graph", "series"])
def test_climatology_shards_follow_the_order(tmp_path: pathlib.Path, split: str):
  """Both ways of `compute-climatology` (one graph, or series written region by region)."""
  time = pd.date_range("2019-01-01", periods=730, freq="1D")
  values = np.random.default_rng(0).random((time.size, 1, 2))
  ds = xr.Dataset(
    {"x": (("time", "lat", "lon"), values)},
    coords={"time": time, "lat": [0.0], "lon": [0.0, 1.0]},
  )
  write_output(ds, tmp_path / "in.zarr", message="input")
  recipe = tmp_path / "clim.toml"
  recipe.write_text(
    f'period = 365\nwindow = 1\nweighting = "constant"\n{split}[read]\nengine = "icechunk"\n\n'
    "[save]\nchunks = {time = 73, lat = 1, lon = 1}\nshards = {time = -1, lat = 1, lon = -1}\n"
    'shard_order = "lexicographic"\n'
  )
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  args = [str(recipe), str(tmp_path / "in.zarr"), str(clim), str(std)]
  path = ["--streaming"] if split else []
  result = CliRunner().invoke(cli, ["compute-climatology", *path, "--dask-setup", "threads", *args])
  assert result.exit_code == 0, result.output
  for path in (clim, std):
    assert icechunk.Repository.exists(icechunk_utils.storage(path))
    assert np.array_equal(
      physical_order(path, "x", (0, 0, 0)), expected("lexicographic", (5, 1, 2))
    )


def test_the_order_travels_to_other_processes(tmp_path: pathlib.Path):
  """Distributed workers get the arrays pickled: they write in the order without zarr's
  configuration (a fresh process has only the default)."""
  path = tmp_path / "a.zarr"
  zarr.create_array(path, shape=(8, 4), chunks=(1, 1), shards=(8, 4), dtype="f4")
  with sharding.write_order("lexicographic"):
    array = zarr.open_array(path, mode="r+")
  code = (
    "import pickle, sys; c = pickle.loads(sys.stdin.buffer.read()).metadata.codecs[0]; "
    "print(c.subchunk_write_order)"
  )
  out = subprocess.run(
    [sys.executable, "-c", code], input=pickle.dumps(array), capture_output=True, check=True
  )
  assert out.stdout.decode().split() == ["lexicographic"]


def test_an_order_needs_shards():
  with pytest.raises(pydantic.ValidationError, match="give \\[save\\] shards"):
    SaveConfig.model_validate({"chunks": {"time": 1}, "shard_order": "lexicographic"})
  with pytest.raises(pydantic.ValidationError):
    SaveConfig.model_validate({**LAYOUT, "shard_order": "hilbert"})
