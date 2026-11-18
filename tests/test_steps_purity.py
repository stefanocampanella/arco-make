# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Intent 1: a step returns a new dataset and never changes its input.

Every step of `processing_utils` is applied to a deep copy of a dataset with attributes and
encodings; the input must be identical afterwards, including attributes and per-variable encoding.
"""

import copy

import pytest
import xarray as xr

from arcomake import processing_utils as pu
from arcomake.steps import registered_steps
from tests.conftest import make_ocean_dataset, make_static_dataset

GRID = {
  "south": -1.5,
  "north": 1.5,
  "west": 0.5,
  "east": 3.5,
  "resolution_lat": 1.0,
  "resolution_lon": 1.0,
}

STEPS = {
  "apply_mask": {"mask_name": "m", "variables": ["zos"]},
  "assign_var_attrs": {"zos": {"units": "cm"}},
  "astype": {"dtype": "float32", "casting": "same_kind"},
  "clip_negative": {"variables": ["zos"]},
  "coverage_attrs": {"latitude_dim": "latitude", "longitude_dim": "longitude"},
  "drop_static_vars": {},
  "drop_var_attrs": {"variables": ["thetao"]},
  "flip": {"dim": "latitude"},
  "get_bottom": {"depth_dim": "depth", "variables": {"thetao": "tb"}, "replace": False},
  "get_notnull_mask": {"variable": "deptho", "mask_name": "m2"},
  "get_sea_mask": {"bathymetry": "deptho", "depth_dim": "depth", "mask_name": "m3"},
  "isel": {"time": {"start": 0, "stop": 2}},
  "is_positive_mask": {"variable": "deptho", "mask_name": "m2"},
  "masked_fill": {"variables": ["thetao"], "fill_value": 0.0, "mask_name": "m"},
  "regrid": {"grid": GRID, "method": "conservative", "kwargs": {"skipna": True}},
  "rename_coordinates": {"name_dict": {"latitude": "lat"}},
  "resample": {"reduce": "mean", "time": "2D"},
  "rescale": {"zos": 2.0},
  "sel": {"latitude": {"start": 0.0, "stop": 1.0}},
  "select_variables": {"variables": ["zos"]},
  "sin_cos": {"variable": "zos", "sin_name": "zs", "cos_name": "zc"},
  "transpose": {"dims": ["longitude", "latitude", "depth", "time"]},
}


def dataset() -> xr.Dataset:
  ds = xr.merge([make_ocean_dataset(), make_static_dataset()], combine_attrs="override")
  ds["m"] = ds.deptho.notnull()
  ds["zos"].encoding = {"dtype": "float32", "chunks": (1, 8, 8)}
  ds["zos"].attrs = {"units": "m"}
  return ds


def snapshot(ds: xr.Dataset) -> dict:
  return {
    "attrs": copy.deepcopy(ds.attrs),
    "variables": {
      name: (copy.deepcopy(v.attrs), copy.deepcopy(v.encoding)) for name, v in ds.variables.items()
    },
  }


def test_every_step_is_covered():
  """Each public step of processing_utils has an entry here (new steps must be added)."""
  registered = set(registered_steps())
  assert registered == set(STEPS)


@pytest.mark.parametrize("name", sorted(STEPS))
def test_step_does_not_modify_its_input(name: str):
  """The input dataset (values, attributes, encoding) and the step's arguments are unchanged."""
  ds = dataset()
  original, before = ds.copy(deep=True), snapshot(ds)
  arguments = copy.deepcopy(STEPS[name])
  getattr(pu, name)(ds, **arguments)
  xr.testing.assert_identical(ds, original)
  assert snapshot(ds) == before
  assert arguments == STEPS[name]
