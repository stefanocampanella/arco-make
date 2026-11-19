# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Steps for coarsened ocean datasets: fractions and masks, fills along depth, products and
ratios, values kept where a condition holds, quantities per unit area (processing_utils)."""

import numpy as np
import pytest
import xarray as xr

from arcomake import processing_utils as pu


def fraction_ds() -> xr.Dataset:
  values = np.array([0.0, 1e-8, 2e-6, 0.25, 0.5, 0.75, 1.0], dtype="float32")
  return xr.Dataset({"f": ("x", values, {"units": "1"})})


def test_fraction_mask_cleans_round_off_and_adds_any_water_mask():
  """Below the tolerance a fraction is round-off: it becomes 0; the default mask is `f > 0`."""
  out = pu.fraction_mask(fraction_ds(), fraction="f", mask_name="m")
  expected = np.array([0.0, 0.0, 2e-6, 0.25, 0.5, 0.75, 1.0], dtype="float32")
  np.testing.assert_array_equal(out.f.values, expected)
  assert out.f.dtype == np.float32 and out.f.attrs == {"units": "1"}
  np.testing.assert_array_equal(out.m.values, [False, False, True, True, True, True, True])


@pytest.mark.parametrize(
  ("threshold", "inclusive", "expected"),
  [
    (0.5, False, [False, False, False, False, False, True, True]),
    (0.5, True, [False, False, False, False, True, True, True]),
  ],
)
def test_fraction_mask_threshold_and_inclusive(threshold, inclusive, expected):
  out = pu.fraction_mask(
    fraction_ds(), fraction="f", mask_name="m", threshold=threshold, inclusive=inclusive
  )
  np.testing.assert_array_equal(out.m.values, expected)


def test_fraction_mask_settable_tolerance():
  """`tolerance = 0` keeps every value: the round-off cell is then in the mask."""
  out = pu.fraction_mask(fraction_ds(), fraction="f", mask_name="m", tolerance=0.0)
  assert out.f.values[1] == np.float32(1e-8) and out.m.values[1]


def test_fraction_mask_without_mask_name_only_cleans():
  out = pu.fraction_mask(fraction_ds(), fraction="f")
  assert set(out.data_vars) == {"f"}


@pytest.mark.parametrize(
  "kwargs", [{"threshold": 1.0}, {"threshold": 0.0, "inclusive": True}, {"tolerance": -1.0}]
)
def test_fraction_mask_rejects_meaningless_arguments(kwargs):
  """A threshold of 1 or an inclusive 0 make every cell sea or none; a negative tolerance is
  meaningless."""
  with pytest.raises(ValueError):
    pu.fraction_mask(fraction_ds(), fraction="f", mask_name="m", **kwargs)


def column_ds(chunked: bool = False) -> xr.Dataset:
  """Two columns of five levels: gaps under valid values, a column starting with a gap."""
  t = np.array([[1.0, np.nan, 3.0, np.nan, np.nan], [np.nan, 2.0, np.nan, np.nan, np.nan]])
  mask = np.array([[True, True, True, True, False], [True, True, True, False, False]])
  ds = xr.Dataset(
    {"t": (("x", "depth"), t, {"units": "degC"}), "m": (("x", "depth"), mask)},
    coords={"depth": [1.0, 2.0, 3.0, 4.0, 5.0]},
  )
  return ds.chunk(depth=2) if chunked else ds


@pytest.mark.parametrize("chunked", [False, True])
def test_fill_from_above_inside_the_mask(chunked: bool):
  """Each gap inside the mask takes the nearest value above; a gap with nothing above stays."""
  out = pu.fill_from_above(column_ds(chunked), variables=["t"], depth_dim="depth", mask_name="m")
  expected = [[1.0, 1.0, 3.0, 3.0, np.nan], [np.nan, 2.0, 2.0, np.nan, np.nan]]
  np.testing.assert_array_equal(out.t.values, expected)
  assert out.t.attrs == {"units": "degC"} and out.t.dims == ("x", "depth")


def test_fill_from_above_without_mask_fills_every_gap_below_a_value():
  out = pu.fill_from_above(column_ds(), variables=["t"], depth_dim="depth")
  expected = [[1.0, 1.0, 3.0, 3.0, 3.0], [np.nan, 2.0, 2.0, 2.0, 2.0]]
  np.testing.assert_array_equal(out.t.values, expected)


def test_fill_from_above_then_zero_for_columns_without_values():
  """The recipe's pair: from above, then `masked_fill` with 0 for what has nothing above."""
  ds = pu.fill_from_above(column_ds(), variables=["t"], depth_dim="depth", mask_name="m")
  out = pu.masked_fill(ds, variables=["t"], fill_value=0.0, mask_name="m")
  assert out.t.values[1, 0] == 0.0 and np.isnan(out.t.values[0, 4])


# A 1-degree grid whose cells hold 2 x 2 cells of the synthetic grid (its first column is land).
GRID = {
  "south": -1.5,
  "north": 1.5,
  "west": 0.5,
  "east": 3.5,
  "resolution_lat": 1.0,
  "resolution_lon": 1.0,
}


def ice_ds(static_ds: xr.Dataset) -> xr.Dataset:
  """Sea ice as GLORYS gives it: concentration and thickness NaN where there is no ice (and on
  land), velocity 0 where there is no ice; a sea mask."""
  rng = np.random.default_rng(3)
  sea = static_ds.deptho.notnull()
  shape = sea.shape
  has_ice = sea.values & (rng.random(shape) < 0.6)
  c = np.where(has_ice, rng.uniform(0.05, 1.0, shape), np.nan)
  h = np.where(has_ice, rng.uniform(0.1, 3.0, shape), np.nan)
  u = np.where(sea.values, np.where(has_ice, rng.normal(0, 0.1, shape), 0.0), np.nan)
  dims = ("latitude", "longitude")
  return xr.Dataset(
    {"siconc": (dims, c), "sithick": (dims, h), "usi": (dims, u), "sea": (dims, sea.values * 1.0)},
    coords=static_ds.drop_vars("depth").coords,
  )


def test_products_multiply_and_propagate_missing_values():
  ds = xr.Dataset({"a": ("x", [2.0, np.nan, 3.0], {"units": "1"}), "b": ("x", [4.0, 5.0, 0.5])})
  out = pu.products(ds, ab=["a", "b"], aab=["a", "a", "b"])
  np.testing.assert_array_equal(out.ab.values, [8.0, np.nan, 1.5])
  np.testing.assert_array_equal(out.aab.values, [16.0, np.nan, 4.5])
  assert out.ab.attrs == {} and out.a.attrs == {"units": "1"}
  with pytest.raises(ValueError):
    pu.products(ds, empty=[])


def test_divide_by_is_nan_where_the_divisor_is_not_positive():
  ds = xr.Dataset({"v": ("x", [1.0, 2.0, 3.0], {"units": "m"}), "f": ("x", [0.5, 0.0, np.nan])})
  out = pu.divide_by(ds, variables=["v"], by="f")
  np.testing.assert_array_equal(out.v.values, [2.0, np.nan, np.nan])
  assert out.v.attrs == {"units": "m"}


def test_divide_by_takes_the_first_level_of_a_3d_divisor(static_ds: xr.Dataset):
  """2D fields are divided by the surface level of a 3D fraction."""
  frac = xr.DataArray(np.array([0.5, 0.25, 1.0]), dims="depth")
  ds = xr.Dataset({"z": ("x", [1.0, 2.0]), "frac": frac})
  out = pu.divide_by(ds, variables=["z"], by="frac")
  np.testing.assert_array_equal(out.z.values, [2.0, 4.0])


def test_cell_means_over_the_sea_part_from_whole_cell_means(static_ds: xr.Dataset):
  """Concentration over the sea part of coarse cells = (regridded with missing as 0) / (sea
  fraction), exactly as regridding with land missing and ice-free sea as 0 gives it; thickness
  and velocity over the ice cover = regridded c*h (c*u) / regridded c: ice volume (c h, per unit
  sea area) is recovered from the published products."""
  ice = ice_ds(static_ds)
  native = pu.products(ice, sivol=["siconc", "sithick"], usic=["siconc", "usi"])
  coarse = pu.regrid(native, grid=GRID, method="conservative", kwargs={"skipna": False})
  coarse = pu.divide_by(coarse, variables=["siconc", "sivol", "usic"], by="sea")
  # Reference: missing as land only, 0 on ice-free sea, the mean over the sea part.
  sea = ice.sea > 0
  c0 = ice.siconc.fillna(0).where(sea)
  ref = pu.regrid(
    xr.Dataset({"c": c0, "ch": (c0 * ice.sithick.fillna(0)).where(sea)}),
    grid=GRID,
    method="conservative",
    kwargs={"skipna": True, "nan_threshold": 1.0},
  )
  np.testing.assert_allclose(coarse.siconc.values, ref.c.values, rtol=1e-12)
  np.testing.assert_allclose(coarse.sivol.values, ref.ch.values, rtol=1e-12)
  # Thickness over the ice cover, as GLORYS defines it, back from the published fields.
  thickness = (coarse.sivol / coarse.siconc).values
  lat = np.deg2rad(ice.latitude)
  area = np.sin(lat + np.deg2rad(0.25)) - np.sin(lat - np.deg2rad(0.25))  # cell areas, relative
  weights = (area * c0).coarsen(latitude=2, longitude=2).sum()
  direct = (area * c0 * ice.sithick.fillna(0)).coarsen(latitude=2, longitude=2).sum() / weights
  ok = weights.values > 0
  np.testing.assert_allclose(thickness[ok], direct.values[ok], rtol=1e-6)


def with_coordinate_attrs(ds: xr.Dataset) -> xr.Dataset:
  for name in ds.coords:
    ds[name].attrs = {"long_name": f"the {name}"}
  return ds


@pytest.mark.parametrize(
  ("step", "kwargs"),
  [
    (pu.fraction_mask, {"fraction": "sea", "mask_name": "m"}),
    (pu.products, {"ch": ["siconc", "sithick"]}),
    (pu.divide_by, {"variables": ["siconc"], "by": "sea"}),
    (pu.keep_where, {"variables": ["siconc"], "condition": "sea", "equals": 1}),
    (pu.per_unit_area, {"variables": ["siconc"]}),
  ],
)
def test_steps_keep_the_coordinates_attributes(static_ds: xr.Dataset, step, kwargs):
  """A new variable assigned with its coordinates' attributes dropped took them from the
  dataset's coordinates too (`products`, `fraction_mask`)."""
  ds = with_coordinate_attrs(ice_ds(static_ds))
  out = step(ds, **kwargs)
  for name in ds.coords:
    assert out[name].attrs == ds[name].attrs, name


def test_fill_from_above_keeps_the_coordinates_attributes():
  ds = with_coordinate_attrs(column_ds().assign_coords(x=[10, 20]))
  out = pu.fill_from_above(ds, variables=["t"], depth_dim="depth", mask_name="m")
  for name in ds.coords:
    assert out[name].attrs == ds[name].attrs, name


def test_keep_where_keeps_values_at_the_condition_only():
  ds = xr.Dataset(
    {
      "q": ("x", [10.0, 20.0, np.nan, 40.0], {"units": "m3 s-1"}),
      "ldd": ("x", [5.0, 2.0, 5.0, np.nan]),
    }
  )
  out = pu.keep_where(ds, variables=["q"], condition="ldd", equals=5)
  np.testing.assert_array_equal(out.q.values, [10.0, 0.0, np.nan, 0.0])
  assert out.q.attrs == {"units": "m3 s-1"}
  other = pu.keep_where(ds, variables=["q"], condition="ldd", equals=5, other=np.nan)
  np.testing.assert_array_equal(other.q.values, [10.0, np.nan, np.nan, np.nan])


def test_cell_areas_of_a_global_grid_sum_to_the_sphere():
  """Bounds midway between centres, clipped at the poles: a global grid covers the sphere."""
  areas = pu.cell_areas(np.arange(-90.0, 90.5, 1.0), np.arange(0.0, 360.0, 1.0), radius=1.0)
  assert areas.shape == (181, 360)
  np.testing.assert_allclose(areas.sum(), 4 * np.pi, rtol=1e-12)
  np.testing.assert_allclose(areas[0, 0], np.pi / 180 * (1 - np.cos(np.deg2rad(0.5))), rtol=1e-9)


def outlets_ds(static_ds: xr.Dataset) -> xr.Dataset:
  """Point discharges (m3/s) in interior cells of the synthetic grid, 0 elsewhere."""
  q = np.zeros(static_ds.deptho.shape)
  q[2, 3], q[4, 4], q[5, 2] = 1000.0, 250.0, 40.0
  return xr.Dataset(
    {"q": (("latitude", "longitude"), q)}, coords=static_ds.drop_vars("depth").coords
  )


@pytest.mark.parametrize("offset", [0.0, 0.25])
def test_densities_regridded_conservatively_keep_the_total(static_ds: xr.Dataset, offset: float):
  """Discharge per unit area, regridded conservatively (missing as 0), times the coarse cells'
  areas: the total is kept, whether source cells nest in the coarse ones or straddle their edges
  (`offset`: the coarse grid moved by half a source cell, each point then split by overlap).

  Split cells keep the total to about 1e-5 here: xarray-regrid splits a source cell by its
  overlap in degrees of latitude, not in area (0.5-degree cells; 1e-7 for GloFAS's 0.05-degree
  cells on a 0.25-degree grid)."""
  ds = outlets_ds(static_ds)
  grid = {
    "south": -1.5 + offset,
    "north": 1.5 + offset,
    "west": 0.5 + offset,
    "east": 3.5 + offset,
    "resolution_lat": 1.0,
    "resolution_lon": 1.0,
  }
  density = pu.per_unit_area(ds, variables=["q"])
  coarse = pu.regrid(density, grid=grid, method="conservative", kwargs={"skipna": False})
  areas = pu.cell_areas(coarse.latitude.values, coarse.longitude.values)
  total = float(np.nansum(coarse.q.values * areas))
  np.testing.assert_allclose(total, float(ds.q.sum()), rtol=1e-5 if offset else 1e-12)
  if offset:
    assert int((coarse.q.values > 0).sum()) > 3, "points on edges are split between cells"


def test_per_unit_area_factor_gives_metres_per_day(static_ds: xr.Dataset):
  ds = outlets_ds(static_ds)
  out = pu.per_unit_area(ds, variables=["q"], factor=86400.0)
  areas = pu.cell_areas(ds.latitude.values, ds.longitude.values)
  np.testing.assert_allclose(out.q.values, ds.q.values * 86400.0 / areas)


CONSERVATIVE = {"method": "conservative", "kwargs": {"skipna": False}}


@pytest.mark.parametrize("chunked", [False, True])
def test_conservation_check_passes_when_totals_are_kept(static_ds: xr.Dataset, chunked: bool):
  density = pu.per_unit_area(outlets_ds(static_ds), variables=["q"])
  density = density.expand_dims(time=3).chunk(time=1) if chunked else density
  out = pu.regrid(density, grid=GRID, conservation_tolerance=1e-9, **CONSERVATIVE)
  xr.testing.assert_allclose(
    out.load(), pu.regrid(density, grid=GRID, **CONSERVATIVE).load(), rtol=0, atol=0
  )


@pytest.mark.parametrize("chunked", [False, True])
def test_conservation_check_fails_when_points_are_lost(static_ds: xr.Dataset, chunked: bool):
  """A point in a source cell whose target cell is centred beyond the source's centres is lost
  (xarray-regrid blanks that cell): the check names the variable."""
  ds = outlets_ds(static_ds)
  ds["q"][0, 3] = 500.0  # latitude -1.75, in the target row centred at -2
  grid = {"south": -2.0, "north": 2.0, "west": 0.0, "east": 4.0}
  grid |= {"resolution_lat": 1.0, "resolution_lon": 1.0}
  density = pu.per_unit_area(ds, variables=["q"])
  density = density.expand_dims(time=2).chunk(time=1) if chunked else density
  with pytest.raises(ValueError, match="total of 'q' changed"):
    pu.regrid(density, grid=grid, conservation_tolerance=1e-6, **CONSERVATIVE).load()


def test_conservation_check_on_an_output_in_several_horizontal_chunks(static_ds: xr.Dataset):
  """`process` recipes may split the regridded output along latitude and longitude
  (`output_chunks`): the check needs no whole horizontal dimensions, and still fails."""
  density = pu.per_unit_area(outlets_ds(static_ds), variables=["q"]).expand_dims(time=2)
  split = {"skipna": False, "output_chunks": {"latitude": 2, "longitude": 2}}
  out = pu.regrid(
    density.chunk(time=1),
    grid=GRID,
    method="conservative",
    kwargs=split,
    conservation_tolerance=1e-9,
  )
  assert len(out.q.chunks[1]) > 1 and len(out.q.chunks[2]) > 1
  expected = pu.regrid(density, grid=GRID, **CONSERVATIVE)
  xr.testing.assert_allclose(out.load(), expected, rtol=0, atol=0)
  ds = outlets_ds(static_ds)
  ds["q"][0, 3] = 500.0  # latitude -1.75, in the target row centred at -2, blanked
  lost = pu.per_unit_area(ds, variables=["q"]).expand_dims(time=2)
  grid = {"south": -2.0, "north": 2.0, "west": 0.0, "east": 4.0}
  grid |= {"resolution_lat": 1.0, "resolution_lon": 1.0}
  with pytest.raises(ValueError, match="total of 'q' changed"):
    pu.regrid(
      lost.chunk(time=1),
      grid=grid,
      method="conservative",
      kwargs=split,
      conservation_tolerance=1e-6,
    ).load()


@pytest.mark.parametrize(
  ("method", "kwargs"), [("conservative", {"skipna": True}), ("conservative", {}), ("nearest", {})]
)
def test_conservation_check_needs_conservative_without_skipna(static_ds, method, kwargs):
  """With skipna the conservative method averages the valid part of cells: totals change."""
  with pytest.raises(ValueError, match="conservation_tolerance needs"):
    pu.regrid(
      outlets_ds(static_ds), grid=GRID, method=method, kwargs=kwargs, conservation_tolerance=1e-6
    )
