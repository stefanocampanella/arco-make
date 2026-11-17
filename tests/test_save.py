# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Writing outputs: Icechunk repositories with the layout of `[save]` (`write_output`)."""

import pathlib

import icechunk
import numcodecs
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from arcomake import dataset_utils, icechunk_utils
from arcomake.dataset_utils import SaveConfig, write_output, write_outputs_together
from tests.conftest import open_output, output_group


def save(dataset, path, configs=None, **kwargs):
  """`write_output` with a test commit message."""
  return write_output(dataset, path, configs, message="test", **kwargs)


LZ4 = {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle"}}
SAVE = SaveConfig.model_validate(
  {"chunks": {"time": 1, "depth": 1, "latitude": -1, "longitude": -1}, "compressors": [LZ4]}
)


def test_safe_to_zarr_directory_layout(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save] chunks` sets the stored chunks per dimension (-1: whole dimension) and `compressors`
  the compressors, in Zarr format 3."""
  save(ocean_ds, tmp_path / "out.zarr", configs=SAVE)
  group = output_group(tmp_path / "out.zarr")
  assert group.metadata.zarr_format == 3
  assert group["thetao"].chunks == (1, 1, 8, 8)
  assert group["zos"].chunks == (1, 8, 8)
  (codec,) = group["thetao"].compressors
  assert (codec.cname, codec.clevel, codec.shuffle) == ("lz4", 1, "shuffle")
  xr.testing.assert_allclose(open_output(tmp_path / "out.zarr").load(), ocean_ds)


def test_safe_to_zarr_whole_dimension_with_split_dask_chunks(tmp_path: pathlib.Path):
  """`-1` with Dask chunks smaller than the dimension fails instead of losing data.

  xarray's check that Dask chunks do not straddle Zarr chunks passed for -1 (`n % -1 == 0`), so
  several Dask chunks wrote into the same Zarr chunk concurrently: about 95 % of the values were
  wrong, with no error.
  """
  data = np.random.default_rng(0).random((2, 40, 40)).astype("float32")
  ds = xr.Dataset({"v": (("time", "lat", "lon"), data)}).chunk({"time": 1, "lat": 10, "lon": 10})
  conf = SaveConfig(chunks={"time": 1, "lat": -1, "lon": -1})
  with pytest.raises(ValueError, match="overlap multiple Dask chunks"):
    save(ds, tmp_path / "split.zarr", configs=conf)
  save(ds.chunk({"lat": -1, "lon": -1}), tmp_path / "whole.zarr", configs=conf)
  out = open_output(tmp_path / "whole.zarr")["v"]
  assert out.encoding["chunks"] == (1, 40, 40)
  np.testing.assert_array_equal(out.values, data)


@pytest.mark.parametrize("key", ["_FillValue", "fill_value"])
def test_safe_to_zarr_sets_the_fill_value(key: str, tmp_path: pathlib.Path):
  """`[save.encoding.<var>] _FillValue` (or `fill_value`) becomes the CF fill value: missing
  values are stored as it, and the `_FillValue` attribute names it (in Zarr format 3, xarray keeps
  it apart from the array's own fill value, used for chunks never written).

  The model forbade `_FillValue` and passed `fill_value`, which xarray rejects: the write failed
  after the first interval was downloaded.
  """
  ds = xr.Dataset({"v": ("x", np.array([1.0, np.nan, 3.0]))})
  conf = SaveConfig.model_validate({"encoding": {"v": {key: -9.0}}})
  save(ds, tmp_path / "out.zarr", configs=conf)
  stored = output_group(tmp_path / "out.zarr")["v"]
  assert stored[:].tolist() == [1.0, -9.0, 3.0]
  out = open_output(tmp_path / "out.zarr")
  assert out["v"].encoding["_FillValue"] == -9.0
  xr.testing.assert_identical(out.load(), ds)


def test_safe_to_zarr_packs_with_scale_factor(tmp_path: pathlib.Path):
  """`scale_factor`, `add_offset` and an integer `dtype` pack floats; they decode on reading."""
  ds = xr.Dataset({"v": ("x", np.array([10.0, 10.5, 11.0]))})
  encoding = {"v": {"dtype": "int16", "scale_factor": 0.5, "add_offset": 10.0}}
  save(ds, tmp_path / "out.zarr", configs=SaveConfig.model_validate({"encoding": encoding}))
  assert output_group(tmp_path / "out.zarr")["v"][:].tolist() == [0, 1, 2]
  np.testing.assert_array_equal(open_output(tmp_path / "out.zarr")["v"].values, ds["v"].values)


def test_safe_to_zarr_append(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Appending along time gives the same data as one write."""
  path = tmp_path / "out.zarr"
  save(ocean_ds.isel(time=slice(0, 3)), path, configs=SAVE, mode="w")
  save(ocean_ds.isel(time=slice(3, None)), path, configs=SAVE, mode="a-", append_dim="time")
  xr.testing.assert_allclose(open_output(path).load(), ocean_ds)


def test_outputs_written_together_share_one_computation(ocean_ds, tmp_path, monkeypatch):
  """`write_outputs_together` computes datasets that share inputs once, then commits each
  repository (with a local scheduler; see its docstring for distributed ones)."""
  calls = []
  chunked = ocean_ds.chunk(time=1)
  shared = chunked.map_blocks(lambda block: calls.append(1) or block, template=chunked)
  snapshots = write_outputs_together(
    [(shared, tmp_path / "a.zarr"), (shared * 2, tmp_path / "b.zarr")], SAVE, message="both"
  )
  assert len(snapshots) == 2
  xr.testing.assert_allclose(open_output(tmp_path / "a.zarr").load(), ocean_ds)
  xr.testing.assert_allclose(open_output(tmp_path / "b.zarr").load(), ocean_ds * 2)
  assert len(calls) == ocean_ds.sizes["time"]  # each block once, not once per output


def test_outputs_are_repositories_with_one_commit_per_write(ocean_ds, tmp_path):
  """Each write is a commit on `main` with its message; overwriting keeps the earlier versions."""
  path = tmp_path / "out.zarr"
  first = write_output(ocean_ds.isel(time=slice(0, 3)), path, SAVE, message="first")
  write_output(
    ocean_ds.isel(time=slice(3, None)), path, SAVE, message="second", mode="a-", append_dim="time"
  )
  write_output(ocean_ds * 0, path, SAVE, message="overwrite")
  repo = icechunk.Repository.open(icechunk_utils.storage(path))
  assert [s.message for s in repo.ancestry(branch="main")][:3] == ["overwrite", "second", "first"]
  assert float(open_output(path).thetao.max()) == 0
  old = xr.open_zarr(icechunk_utils.readonly_store(path, snapshot_id=first), consolidated=False)
  xr.testing.assert_allclose(old.load(), ocean_ds.isel(time=slice(0, 3)))


def test_safe_to_zarr_does_not_modify_input_encoding(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Saving a dataset leaves its encoding unchanged (intent 1), so saving twice is consistent."""
  before = {name: dict(da.encoding) for name, da in ocean_ds.data_vars.items()}
  save(ocean_ds, tmp_path / "out.zarr", configs=SAVE)
  assert {name: dict(da.encoding) for name, da in ocean_ds.data_vars.items()} == before
  # A second save with another layout is not affected by the first one.
  other = SaveConfig(chunks={"time": 2})
  save(ocean_ds, tmp_path / "other.zarr", configs=other)
  assert output_group(tmp_path / "other.zarr")["thetao"].chunks == (2, 3, 8, 8)


def test_safe_to_zarr_default_compression(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Without compressors in the recipe, zarr's default compressor is used, not none; an empty
  list stores the data uncompressed."""
  save(ocean_ds, tmp_path / "out.zarr", configs=SaveConfig())
  assert output_group(tmp_path / "out.zarr")["thetao"].compressors != ()
  save(ocean_ds, tmp_path / "raw.zarr", configs=SaveConfig(compressors=[]))
  assert output_group(tmp_path / "raw.zarr")["thetao"].compressors == ()


@pytest.mark.parametrize("compressors", [[LZ4], None])
def test_codecs_of_zarr2_sources_are_replaced(
  compressors, ocean_ds: xr.Dataset, tmp_path: pathlib.Path
):
  """Data read from a Zarr v2 store (numcodecs codecs in their encoding, as the remote sources
  have) are written in format 3, with the recipe's compressors or zarr's default."""
  source = tmp_path / "v2.zarr"
  ocean_ds.to_zarr(
    source, zarr_format=2, encoding={"thetao": {"compressors": [numcodecs.Blosc("zstd", 3)]}}
  )
  read = xr.open_zarr(source)
  assert isinstance(read.thetao.encoding["compressors"][0], numcodecs.Blosc)
  save(read, tmp_path / "out.zarr", configs=SaveConfig(compressors=compressors))
  out = output_group(tmp_path / "out.zarr")
  expected = "BloscCodec" if compressors else "ZstdCodec"
  assert [type(c).__name__ for c in out["thetao"].compressors] == [expected]
  assert read.thetao.encoding["compressors"][0] == numcodecs.Blosc("zstd", 3)  # input unchanged
  xr.testing.assert_identical(open_output(tmp_path / "out.zarr").load(), read.load())


def test_per_variable_compressors(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save.encoding.<var>] compressors` replace those of `[save]` for that variable."""
  zstd = {"name": "zstd", "configuration": {"level": 5}}
  conf = SaveConfig.model_validate(
    {"compressors": [LZ4], "encoding": {"zos": {"compressors": [zstd]}}}
  )
  save(ocean_ds, tmp_path / "out.zarr", configs=conf)
  group = output_group(tmp_path / "out.zarr")
  assert type(group["thetao"].compressors[0]).__name__ == "BloscCodec"
  assert type(group["zos"].compressors[0]).__name__ == "ZstdCodec"


@pytest.mark.parametrize(
  "codec, message",
  [
    ({"name": "lz5"}, "invalid codec 'lz5'"),
    ({"name": "blosc", "configuration": {"cname": "lz4", "clevl": 1}}, "clevl"),
    ({"name": "blosc", "configuration": {"cname": "lz5"}}, "cname"),
    ({"name": "bytes"}, "not a compressor"),
    ({"name": "zstd", "level": 3}, "Extra inputs"),
  ],
)
def test_invalid_compressors_fail_when_loaded(codec: dict, message: str):
  """Unknown codecs, unknown or invalid settings and codecs that are not compressors are errors
  when the recipe is loaded, before any download."""
  with pytest.raises(ValidationError, match=message):
    SaveConfig.model_validate({"compressors": [codec]})


def test_safe_to_zarr_append_mode_on_a_new_store(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Mode "a" on a store that does not exist yet writes it, with the configured encodings."""
  path = tmp_path / "out.zarr"
  configs = SaveConfig(encoding={"time": {"units": "days since 2020-01-01"}})
  save(ocean_ds, path, configs=configs, mode="a")
  assert output_group(path)["time"].attrs["units"] == "days since 2020-01-01"


def test_appending_with_per_variable_encodings(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save.encoding]` applies when a variable is created; appended blocks reuse the stored
  encoding (time units included), so every block gets the same recipe and the data match one
  write. The existing store is not read beforehand: errors reading it come from xarray itself."""
  path = tmp_path / "out.zarr"
  configs = SaveConfig(encoding={"time": {"units": "days since 2020-01-01"}})
  save(ocean_ds.isel(time=slice(0, 3)), path, configs=configs)
  save(ocean_ds.isel(time=slice(3, None)), path, configs=configs, mode="a-", append_dim="time")
  assert output_group(path)["time"].attrs["units"] == "days since 2020-01-01"
  xr.testing.assert_allclose(open_output(path).load(), ocean_ds)


def test_per_variable_encoding_keeps_the_save_layout(ocean_ds: xr.Dataset, tmp_path):
  """`[save.encoding.VAR]` is merged over `[save]` key by key: a variable listed only for its
  dtype keeps the chunks, shards and compressors of `[save]` (xarray's `encoding=` argument
  would replace them all)."""
  conf = SaveConfig.model_validate(
    {
      "chunks": {"time": 1, "depth": 1, "latitude": 4, "longitude": -1},
      "shards": {"time": 2},
      "compressors": [LZ4],
      "encoding": {"zos": {"dtype": "float32"}, "thetao": {"chunks": [2, 3, 8, 8]}},
    }
  )
  save(ocean_ds, tmp_path / "out.zarr", configs=conf)
  group = output_group(tmp_path / "out.zarr")
  zos, thetao = group["zos"], group["thetao"]
  assert zos.dtype == np.float32 and zos.chunks == (1, 4, 8) and zos.shards == (2, 8, 8)
  assert type(zos.compressors[0]).__name__ == "BloscCodec"
  assert thetao.chunks == (2, 3, 8, 8) and thetao.shards == (2, 3, 8, 8)


def test_per_variable_encoding_of_a_missing_variable_fails(ocean_ds, tmp_path):
  """A name in `[save.encoding]` that the dataset lacks is an error, as with xarray."""
  conf = SaveConfig.model_validate({"encoding": {"nope": {"dtype": "float32"}}})
  with pytest.raises(ValueError, match="'nope'"):
    save(ocean_ds, tmp_path / "out.zarr", configs=conf)


def test_shards_of_a_sharded_source_do_not_outlive_new_chunks(ocean_ds, tmp_path):
  """Rewriting a sharded output with other chunks and no shards writes it unsharded (the
  source's shards were sized for its own chunks)."""
  save(ocean_ds, tmp_path / "sharded.zarr", configs=SHARDED)
  read = open_output(tmp_path / "sharded.zarr")
  assert read.thetao.encoding["shards"] == (2, 3, 8, 8)
  # Dask chunks of one time step: the stale shards (2 time steps) would straddle them.
  conf = SaveConfig.model_validate({"chunks": {"time": 1, "depth": 1, "latitude": 4}})
  save(read, tmp_path / "out.zarr", configs=conf)
  out = output_group(tmp_path / "out.zarr")["thetao"]
  assert out.chunks == (1, 1, 4, 8) and out.shards is None


def test_uncompressed_sources_give_compressed_outputs(ocean_ds, tmp_path):
  """A source stored uncompressed (in either format) does not make the output uncompressed: the
  empty list of compressors it reports is dropped, and zarr's default applies (N1)."""
  for zarr_format in (2, 3):
    source = tmp_path / f"raw{zarr_format}.zarr"
    ocean_ds.to_zarr(source, zarr_format=zarr_format, encoding={"thetao": {"compressors": None}})
    read = xr.open_zarr(source)
    assert tuple(read.thetao.encoding["compressors"]) == ()
    save(read, tmp_path / f"out{zarr_format}.zarr", configs=SaveConfig())
    out = output_group(tmp_path / f"out{zarr_format}.zarr")["thetao"]
    assert out.compressors != ()
  save(read, tmp_path / "raw.zarr", configs=SaveConfig(compressors=[]))
  assert output_group(tmp_path / "raw.zarr")["thetao"].compressors == ()


@pytest.mark.parametrize("destination", ["local", "file_url"])
def test_storage_options_are_ignored_for_local_paths(ocean_ds, tmp_path, destination: str):
  """Recipes may give `storage_options` for remote runs: they do not apply to local outputs
  (plain paths and `file://` URLs), nor to local Zarr inputs (zarr 3 refuses them there)."""

  def url(path: pathlib.Path) -> str:
    return str(path) if destination == "local" else path.as_uri()

  options = {"anon": True}
  save(ocean_ds, url(tmp_path / "out.zarr"), configs=SaveConfig(storage_options=options))
  read = dataset_utils.open_input(
    url(tmp_path / "out.zarr"), engine="icechunk", storage_options=options
  )
  xr.testing.assert_allclose(read.load(), ocean_ds)
  ocean_ds.to_zarr(tmp_path / "plain.zarr")
  plain = dataset_utils.open_input(
    url(tmp_path / "plain.zarr"), engine="zarr", storage_options=options
  )
  xr.testing.assert_allclose(plain.load(), ocean_ds)


def test_zip_destinations_are_refused(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Zip outputs were removed with Zarr v3: a `.zip` destination is refused instead of
  becoming a directory with that name."""
  with pytest.raises(ValueError, match="zip outputs are not supported"):
    save(ocean_ds, tmp_path / "out.zip")
  assert not (tmp_path / "out.zip").exists()


@pytest.mark.parametrize(
  "save",
  [
    {"chunks": 3, "shards": 4},
    {"chunks": (1, 3), "shards": (2, 4)},
    {"chunks": {"lat": 3}, "shards": 4},
    {"chunks": 3, "shards": {"lat": 4}},
    {"chunks": (1, 3), "shards": 4},
  ],
)
def test_shard_sizes_known_at_load_are_checked_at_load(save: dict):
  """Every pair of explicit sizes is checked when the recipe is loaded, whatever form the
  sizes take; whole dimensions are checked before writing."""
  with pytest.raises(ValidationError, match="must be a multiple of chunks"):
    SaveConfig.model_validate(save)
  SaveConfig.model_validate({"chunks": {"lat": 100}, "shards": {"time": 10}})  # lat: known later


SHARDED = SaveConfig.model_validate(
  {
    "chunks": {"time": 1, "depth": 1, "latitude": 4, "longitude": -1},
    "shards": {"time": 2, "depth": -1},
    "compressors": [LZ4],
  }
)


def test_shards_group_whole_chunks(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save] shards` (per dimension; -1 or not listed: whole dimension) store several chunks per
  object; without them, every chunk is an object."""
  save(ocean_ds, tmp_path / "out.zarr", configs=SHARDED)
  group = output_group(tmp_path / "out.zarr")
  assert group["thetao"].chunks == (1, 1, 4, 8) and group["thetao"].shards == (2, 3, 8, 8)
  assert group["zos"].chunks == (1, 4, 8) and group["zos"].shards == (2, 8, 8)
  xr.testing.assert_allclose(open_output(tmp_path / "out.zarr").load(), ocean_ds)
  save(ocean_ds, tmp_path / "plain.zarr", configs=SAVE)
  assert output_group(tmp_path / "plain.zarr")["thetao"].shards is None


def test_appending_into_a_partial_shard(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Blocks need not fill whole shards: appending completes the last shard."""
  path = tmp_path / "out.zarr"
  save(ocean_ds.isel(time=slice(0, 3)), path, configs=SHARDED)
  save(ocean_ds.isel(time=slice(3, None)), path, configs=SHARDED, mode="a-", append_dim="time")
  xr.testing.assert_allclose(open_output(path).load(), ocean_ds)


def test_dask_chunks_may_not_straddle_shards(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Two Dask chunks writing into one shard would race: xarray refuses."""
  with pytest.raises(ValueError):
    save(ocean_ds.chunk(time=1), tmp_path / "out.zarr", configs=SHARDED)
  save(ocean_ds.chunk(time=2), tmp_path / "out.zarr", configs=SHARDED)


@pytest.mark.parametrize(
  "save, message",
  [
    ({"shards": {"time": 2}}, "shards need chunks"),
    ({"chunks": {"time": 2}, "shards": {"time": 3}}, "must be a multiple of chunks"),
    ({"chunks": {"time": 1}, "shards": {"time": 0}}, "sizes are positive"),
  ],
)
def test_invalid_shards_fail_when_loaded(save: dict, message: str):
  """Shards hold whole chunks: impossible layouts fail when the recipe is loaded."""
  with pytest.raises(ValidationError, match=message):
    SaveConfig.model_validate(save)


def test_shards_of_whole_dimensions_hold_whole_chunks(ocean_ds: xr.Dataset, tmp_path):
  """A layout that fails only once -1 is resolved (8 latitudes, chunks of 3) fails before writing."""
  conf = SaveConfig.model_validate({"chunks": {"latitude": 3}, "shards": {"latitude": -1}})
  with pytest.raises(ValueError, match="must be a multiple of chunks"):
    save(ocean_ds, tmp_path / "out.zarr", configs=conf)
  assert not (tmp_path / "out.zarr").exists()
