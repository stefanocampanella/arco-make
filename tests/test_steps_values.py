# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Steps that change values, types or attributes (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu


def test_astype_whole_dataset(ocean_ds: xr.Dataset):
  """`dtype` with `casting` converts every data variable (the form used by the recipes)."""
  out = pu.astype(ocean_ds, dtype="float32", casting="same_kind")
  assert all(da.dtype == np.float32 for da in out.data_vars.values())


@pytest.mark.xfail(reason="N2: astype ignores `dtype` unless `casting` is also given", strict=True)
def test_astype_dtype_alone(ocean_ds: xr.Dataset):
  """`dtype` alone converts too; it must not be a silent no-op."""
  out = pu.astype(ocean_ds, dtype="float32")
  assert out.thetao.dtype == np.float32


def test_astype_per_variable(ocean_ds: xr.Dataset):
  """A table per variable converts that variable only."""
  out = pu.astype(ocean_ds, zos={"dtype": "float32", "casting": "same_kind"})
  assert out.zos.dtype == np.float32


@pytest.mark.xfail(
  reason="N2: the per-variable form of astype drops unlisted variables", strict=True
)
def test_astype_per_variable_keeps_other_variables(ocean_ds: xr.Dataset):
  """Variables not listed are kept unchanged."""
  out = pu.astype(ocean_ds, zos={"dtype": "float32", "casting": "same_kind"})
  assert set(out.data_vars) == {"thetao", "zos"}


def test_clip_negative(atmosphere_ds: xr.Dataset):
  """Listed variables are clipped at 0 (e.g. precipitation); others are untouched; NaN stay NaN."""
  ds = atmosphere_ds.assign(tp=atmosphere_ds.tp.where(atmosphere_ds.tp > -1e-3))
  out = pu.clip_negative(ds, ["tp"])
  assert float(out.tp.min()) >= 0.0
  assert int(out.tp.isnull().sum()) == int(ds.tp.isnull().sum())
  xr.testing.assert_identical(out.t2m, ds.t2m)


def test_rescale(ocean_ds: xr.Dataset):
  """Variables are multiplied by the given factors."""
  out = pu.rescale(ocean_ds, zos=100.0)
  xr.testing.assert_allclose(out.zos, 100.0 * ocean_ds.zos)
  xr.testing.assert_identical(out.thetao, ocean_ds.thetao)


def test_assign_and_drop_var_attrs(ocean_ds: xr.Dataset):
  """Attributes of variables and coordinates are replaced or dropped, without touching the input."""
  out = pu.assign_var_attrs(ocean_ds, latitude={"units": "degrees_north"})
  assert out.latitude.attrs == {"units": "degrees_north"}
  assert ocean_ds.latitude.attrs == {}
  out = pu.drop_var_attrs(ocean_ds, ["thetao"])
  assert out.thetao.attrs == {}
  assert ocean_ds.thetao.attrs == {"units": "degC"}
