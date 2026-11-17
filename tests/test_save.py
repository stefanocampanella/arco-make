# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Writing outputs (`safe_to_zarr`)."""

import pathlib

import dask
import numpy as np
import pytest
import xarray as xr
import zarr

from arcomake import dataset_utils
from arcomake.dataset_utils import CompressorConfig, SaveConfig, safe_to_zarr

SAVE = SaveConfig(
  chunks={"time": 1, "depth": 1, "latitude": -1, "longitude": -1},
  compressor=CompressorConfig(cname="lz4", clevel=1),
)


def test_safe_to_zarr_directory_layout(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """`[save] chunks` sets the stored chunks per dimension (-1: whole dimension) and the compressor."""
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SAVE)
  group = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
  assert group["thetao"].chunks == (1, 1, 8, 8)
  assert group["zos"].chunks == (1, 8, 8)
  assert group["thetao"].compressor.cname == "lz4"
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
  """`[save.encoding.<var>] _FillValue` (or `fill_value`) becomes the stored fill value.

  The model forbade `_FillValue` and passed `fill_value`, which xarray rejects: the write failed
  after the first interval was downloaded.
  """
  ds = xr.Dataset({"v": ("x", np.array([1.0, np.nan, 3.0]))})
  conf = SaveConfig.model_validate({"encoding": {"v": {key: -9.0}}})
  safe_to_zarr(ds, tmp_path / "out.zarr", configs=conf)
  assert zarr.open_array(str(tmp_path / "out.zarr" / "v"), mode="r").fill_value == -9.0
  xr.testing.assert_identical(xr.open_zarr(tmp_path / "out.zarr").load(), ds)


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
  """Without a compressor in the recipe, zarr's default compressor is used, not none."""
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SaveConfig())
  assert zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["thetao"].compressor is not None


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
