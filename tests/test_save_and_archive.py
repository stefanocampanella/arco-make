# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Writing outputs (`safe_to_zarr`) and joining pieces (`open_archive`)."""

import pathlib
from contextlib import ExitStack

import dask
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import ZipStore

from arcomake import dataset_utils
from arcomake.dataset_utils import CompressorConfig, SaveConfig, open_archive, safe_to_zarr

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


def test_safe_to_zarr_zip_and_append(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """A `.zip` destination is a ZipStore; appending along time gives the same data as one write."""
  path = tmp_path / "out.zip"
  safe_to_zarr(ocean_ds.isel(time=slice(0, 3)), path, configs=SAVE, mode="w")
  safe_to_zarr(ocean_ds.isel(time=slice(3, None)), path, configs=SAVE, mode="a-", append_dim="time")
  with ZipStore(str(path), mode="r") as store, xr.open_zarr(store) as out:
    xr.testing.assert_allclose(out.load(), ocean_ds)


def test_safe_to_zarr_delayed_requires_stack(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Delayed writes need an ExitStack that keeps the store open until computed."""
  with pytest.raises(ValueError):
    safe_to_zarr(ocean_ds, tmp_path / "out.zarr", compute=False)
  with ExitStack() as stack:
    delayed = safe_to_zarr(ocean_ds, tmp_path / "out.zarr", compute=False, store_stack=stack)
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


@pytest.mark.xfail(
  reason="N1: without [save] compressor the output is written uncompressed", strict=True
)
def test_safe_to_zarr_default_compression(ocean_ds: xr.Dataset, tmp_path: pathlib.Path):
  """Without a compressor in the recipe, zarr's default compressor is used, not none."""
  safe_to_zarr(ocean_ds, tmp_path / "out.zarr", configs=SaveConfig())
  assert zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["thetao"].compressor is not None


@pytest.mark.xfail(
  reason="B13: errors while pruning encodings on append are swallowed", strict=True
)
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


def test_open_archive_joins_pieces_and_keeps_static_once(ocean_ds, static_ds, tmp_path):
  """Pieces are concatenated along time; static variables, identical in every piece, appear once."""
  full = xr.merge([ocean_ds, static_ds])
  for i, piece in enumerate([slice(0, 2), slice(2, 4), slice(4, None)]):
    full.isel(time=piece).to_zarr(tmp_path / f"piece_{i}.zarr", mode="w")
  out = open_archive(str(tmp_path), glob="*.zarr", engine="zarr")
  xr.testing.assert_allclose(out.load(), full)
  assert out.deptho.dims == ("latitude", "longitude")


def test_open_archive_rejects_different_static_variables(ocean_ds, static_ds, tmp_path):
  """Static variables that differ between pieces are an error (intent 11)."""
  xr.merge([ocean_ds.isel(time=[0]), static_ds]).to_zarr(tmp_path / "a.zarr", mode="w")
  other = static_ds.assign(deptho=static_ds.deptho + 1.0)
  xr.merge([ocean_ds.isel(time=[1]), other]).to_zarr(tmp_path / "b.zarr", mode="w")
  with pytest.raises(ValueError, match="differs across inputs"):
    open_archive(str(tmp_path), glob="*.zarr", engine="zarr")


def test_open_archive_requires_matches(tmp_path: pathlib.Path):
  """An empty glob is an error, not an empty dataset."""
  with pytest.raises(ValueError, match="does not contain"):
    open_archive(str(tmp_path), glob="*.zarr", engine="zarr")


def test_open_archive_drops_conflicting_attrs(ocean_ds, tmp_path):
  """Attributes listed in `attrs_to_drop` (e.g. a creation date) do not block the join."""
  for i in range(2):
    ocean_ds.isel(time=[i]).assign_attrs(date_created=f"day {i}").to_zarr(
      tmp_path / f"{i}.zarr", mode="w"
    )
  out = open_archive(str(tmp_path), glob="*.zarr", engine="zarr", attrs_to_drop=["date_created"])
  assert "date_created" not in out.attrs
  np.testing.assert_array_equal(out.time.values, ocean_ds.time.values[:2])
