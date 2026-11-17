# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Writing outputs (`safe_to_zarr`)."""

import pathlib
import warnings

import dask
import numcodecs
import numpy as np
import pytest
import xarray as xr
import zarr
import zarr.errors
from pydantic import ValidationError

from arcomake import dataset_utils
from arcomake.dataset_utils import SaveConfig, safe_to_zarr

LZ4 = {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle"}}
SAVE = SaveConfig.model_validate(
  {"chunks": {"time": 1, "depth": 1, "latitude": -1, "longitude": -1}, "compressors": [LZ4]}
)


def test_safe_to_zarr_directory_layout(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save] chunks` sets the stored chunks per dimension (-1: whole dimension) and `compressors`
  the compressors, in Zarr format 3."""
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SAVE)
  group = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
  assert group.metadata.zarr_format == 3
  assert group["thetao"].chunks == (1, 1, 8, 8)
  assert group["zos"].chunks == (1, 8, 8)
  (codec,) = group["thetao"].compressors
  assert (codec.cname, codec.clevel, codec.shuffle) == ("lz4", 1, "shuffle")
  xr.testing.assert_allclose(xr.open_zarr(tmp_path / "out.zarr").load(), ocean_ds)


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
    safe_to_zarr(ds, tmp_path / "split.zarr", configs=conf)
  safe_to_zarr(ds.chunk({"lat": -1, "lon": -1}), tmp_path / "whole.zarr", configs=conf)
  out = xr.open_zarr(tmp_path / "whole.zarr")["v"]
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
  safe_to_zarr(ds, tmp_path / "out.zarr", configs=conf)
  stored = zarr.open_array(str(tmp_path / "out.zarr" / "v"), mode="r")
  assert stored[:].tolist() == [1.0, -9.0, 3.0]
  out = xr.open_zarr(tmp_path / "out.zarr")
  assert out["v"].encoding["_FillValue"] == -9.0
  xr.testing.assert_identical(out.load(), ds)


def test_safe_to_zarr_packs_with_scale_factor(tmp_path: pathlib.Path):
  """`scale_factor`, `add_offset` and an integer `dtype` pack floats; they decode on reading."""
  ds = xr.Dataset({"v": ("x", np.array([10.0, 10.5, 11.0]))})
  encoding = {"v": {"dtype": "int16", "scale_factor": 0.5, "add_offset": 10.0}}
  safe_to_zarr(ds, tmp_path / "out.zarr", configs=SaveConfig.model_validate({"encoding": encoding}))
  assert zarr.open_array(str(tmp_path / "out.zarr" / "v"), mode="r")[:].tolist() == [0, 1, 2]
  np.testing.assert_array_equal(xr.open_zarr(tmp_path / "out.zarr")["v"].values, ds["v"].values)


def test_safe_to_zarr_append(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Appending along time gives the same data as one write."""
  path = tmp_path / "out.zarr"
  safe_to_zarr(ocean_ds.isel(time=slice(0, 3)), path, configs=SAVE, mode="w")
  safe_to_zarr(ocean_ds.isel(time=slice(3, None)), path, configs=SAVE, mode="a-", append_dim="time")
  xr.testing.assert_allclose(xr.open_zarr(path).load(), ocean_ds)


def test_safe_to_zarr_delayed(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """With `compute=False`, the data are written when the returned task is computed."""
  delayed = safe_to_zarr(ocean_ds, tmp_path / "out.zarr", compute=False)
  dask.compute(delayed)
  xr.testing.assert_allclose(xr.open_zarr(tmp_path / "out.zarr").load(), ocean_ds)


def test_safe_to_zarr_does_not_modify_input_encoding(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Saving a dataset leaves its encoding unchanged (intent 1), so saving twice is consistent."""
  before = {name: dict(da.encoding) for name, da in ocean_ds.data_vars.items()}
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SAVE)
  assert {name: dict(da.encoding) for name, da in ocean_ds.data_vars.items()} == before
  # A second save with another layout is not affected by the first one.
  other = SaveConfig(chunks={"time": 2})
  safe_to_zarr(ocean_ds, tmp_path / "other.zarr", configs=other)
  assert zarr.open_group(str(tmp_path / "other.zarr"), mode="r")["thetao"].chunks == (2, 3, 8, 8)


def test_safe_to_zarr_default_compression(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Without compressors in the recipe, zarr's default compressor is used, not none; an empty
  list stores the data uncompressed."""
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SaveConfig())
  assert zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["thetao"].compressors != ()
  safe_to_zarr(ocean_ds, tmp_path / "raw.zarr", configs=SaveConfig(compressors=[]))
  assert zarr.open_group(str(tmp_path / "raw.zarr"), mode="r")["thetao"].compressors == ()


def test_consolidated_metadata_without_warning(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Consolidated metadata are written by default (maintainer's choice), without zarr's warning
  that they are not part of the format 3 specification."""
  with warnings.catch_warnings():
    warnings.simplefilter("error", zarr.errors.ZarrUserWarning)
    safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SAVE)
  assert zarr.open_group(str(tmp_path / "out.zarr"), mode="r").metadata.consolidated_metadata


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
  safe_to_zarr(read, tmp_path / "out.zarr", configs=SaveConfig(compressors=compressors))
  out = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
  expected = "BloscCodec" if compressors else "ZstdCodec"
  assert [type(c).__name__ for c in out["thetao"].compressors] == [expected]
  assert read.thetao.encoding["compressors"][0] == numcodecs.Blosc("zstd", 3)  # input unchanged
  xr.testing.assert_identical(xr.open_zarr(tmp_path / "out.zarr").load(), read.load())


def test_per_variable_compressors(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save.encoding.<var>] compressors` replace those of `[save]` for that variable."""
  zstd = {"name": "zstd", "configuration": {"level": 5}}
  conf = SaveConfig.model_validate(
    {"compressors": [LZ4], "encoding": {"zos": {"compressors": [zstd]}}}
  )
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=conf)
  group = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
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
  safe_to_zarr(ocean_ds, path, configs=configs, mode="a")
  assert zarr.open_group(str(path), mode="r")["time"].attrs["units"] == "days since 2020-01-01"


def test_safe_to_zarr_append_does_not_hide_store_errors(
  ocean_ds: xr.Dataset, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  """An unexpected error reading the existing store is raised, not ignored."""
  path = tmp_path / "out.zarr"
  configs = SaveConfig(encoding={"time": {"units": "days since 2020-01-01"}})
  safe_to_zarr(ocean_ds.isel(time=slice(0, 3)), path, configs=configs)

  open_group = zarr.open_group

  def broken_read_only_open_group(*args, **kwargs):
    # Only the read-only inspection of the existing store fails; xarray's own writes still work.
    if kwargs.get("mode") == "r":
      raise PermissionError("store not readable")
    return open_group(*args, **kwargs)

  monkeypatch.setattr(dataset_utils.zarr, "open_group", broken_read_only_open_group)
  with pytest.raises(PermissionError):
    safe_to_zarr(
      ocean_ds.isel(time=slice(3, None)), path, configs=configs, mode="a-", append_dim="time"
    )
