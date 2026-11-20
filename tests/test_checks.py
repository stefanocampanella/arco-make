# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Validation checks and the `validate` accessor."""

import warnings
from datetime import datetime

import dask
import numpy as np
import pydantic
import pytest
import xarray as xr

from arcomake import checks
from arcomake.checks import ChecksConfig, ValidationError


def global_grid(
  lat_name: str = "lat", lon_name: str = "lon", decreasing: bool = False
) -> xr.Dataset:
  lat = np.arange(-90.0, 90.1, 45.0)
  return xr.Dataset(
    {"a": ((lat_name, lon_name), np.zeros((5, 4)))},
    coords={lat_name: lat[::-1] if decreasing else lat, lon_name: np.arange(0.0, 360.0, 90.0)},
  )


def test_valid_time_coordinate_accepts_complete_daily_axis(ocean_ds: xr.Dataset):
  """A daily axis from start to end (excluded) passes, Feb 29 included."""
  checks.valid_time_coordinate(ocean_ds, datetime(2020, 2, 26), datetime(2020, 3, 3))


@pytest.mark.parametrize(
  ("selection", "message"),
  [
    ([0, 1, 3, 4, 5], "Missing"),
    ([0, 1, 1, 2, 3, 4, 5], "Duplicate"),
    ([1, 0, 2, 3, 4, 5], "not sorted"),
  ],
)
def test_valid_time_coordinate_reports_problems(ocean_ds: xr.Dataset, selection, message):
  """Gaps, duplicates and disorder are reported as validation failures."""
  with pytest.raises(ValidationError, match=message):
    checks.valid_time_coordinate(
      ocean_ds.isel(time=selection), datetime(2020, 2, 26), datetime(2020, 3, 3)
    )


def test_valid_time_coordinate_cftime_without_deprecation():
  """cftime axes are checked with the non-deprecated API."""
  times = xr.date_range("2001-01-01", periods=3, freq="1D", calendar="noleap", use_cftime=True)
  ds = xr.Dataset(coords={"time": times})
  with warnings.catch_warnings():
    warnings.simplefilter("error", FutureWarning)
    warnings.simplefilter("error", DeprecationWarning)
    checks.valid_time_coordinate(ds, datetime(2001, 1, 1), datetime(2001, 1, 4))


def test_valid_time_coordinate_cftime_calendar():
  """A complete `noleap` axis passes: Feb 29 is not expected in that calendar."""
  times = xr.date_range(
    "2020-02-27", periods=3, freq="1D", calendar="noleap", use_cftime=True
  )  # Feb 27, Feb 28, Mar 1
  ds = xr.Dataset(coords={"time": times})
  checks.valid_time_coordinate(ds, datetime(2020, 2, 27), datetime(2020, 3, 2))
  with pytest.raises(ValidationError, match="Missing"):
    checks.valid_time_coordinate(ds.isel(time=[0, 2]), datetime(2020, 2, 27), datetime(2020, 3, 2))


def test_valid_global_ecmwf_coordinates_accepts_increasing_latitudes():
  """Increasing latitudes in [-90, 90] and longitudes in [0, 360) pass."""
  checks.valid_global_ecmwf_coordinates(global_grid(), latitude_dim="lat", longitude_dim="lon")


@pytest.mark.parametrize(
  ("grid", "message"),
  [
    (global_grid(decreasing=True), "decreasing"),
    (global_grid().assign_coords(lat=np.linspace(-95, 90, 5)), r"outside the \[-90, 90\]"),
    (global_grid().assign_coords(lon=np.arange(-180.0, 180.0, 90.0)), r"\[0, 360\)"),
  ],
)
def test_valid_global_ecmwf_coordinates_failure_is_a_validation_error(grid, message):
  """Rejected coordinates are validation failures, so `--no-fail` reports them instead of aborting."""
  with pytest.raises(ValidationError, match=message):
    checks.valid_global_ecmwf_coordinates(grid, latitude_dim="lat", longitude_dim="lon")


def test_validate_without_fail_warns_on_bad_coordinates():
  """With `fail=False` the accessor turns the failure into a warning (B11 aborted here)."""
  config = ChecksConfig.model_validate({"valid_global_ecmwf_coordinates": {}})
  with pytest.warns(UserWarning, match="decreasing"):
    global_grid(decreasing=True).arcomake.validate(config)


def test_ecmwf_check_defaults_agree():
  """The defaults of the recipe table and of the function are the same."""
  model_defaults = checks.ValidGlobalECMWFCoordinatesCheck().model_dump()
  ds = global_grid(
    lat_name=model_defaults["latitude_dim"], lon_name=model_defaults["longitude_dim"]
  )
  checks.valid_global_ecmwf_coordinates(ds)


def test_ensure_no_nans_respects_masks(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """NaN outside a variable's mask are allowed; NaN inside it are reported, per day."""
  ds = xr.merge([ocean_ds, static_ds])
  ds["m"] = ds.deptho.notnull()
  ds = ds.drop_vars("deptho").assign(zos=ds.zos.fillna(0.0).where(ds.m))
  checks.ensure_no_nans(ds[["zos", "m"]], m=["zos"])
  broken = ds.assign(zos=ds.zos.where(ds.time != ds.time[2]))
  with pytest.raises(ValidationError, match="2020-02-28"):
    checks.ensure_no_nans(broken[["zos", "m"]], m=["zos"])


def test_ensure_no_nans_computes_once(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """One computation for every variable and day, not two per day and variable (11,322 days in
  the published dataset); the report is the same: per day, the variables with NaN inside
  their mask (a 3D mask's first level for 2D variables), and the static variables once."""
  ds = xr.merge([ocean_ds, static_ds])
  ds["m"] = ds.deptho.notnull()
  ds["m3"] = ds.thetao.isel(time=0, drop=True).notnull()
  ds = ds.assign(
    zos=ds.zos.fillna(0.0).where(ds.m).where(ds.time != ds.time[1]),
    thetao=ds.thetao.where(ds.time != ds.time[3]),
    deptho=ds.deptho.fillna(1.0).where(ds.latitude != ds.latitude[0]),
  ).chunk(time=1)
  calls = []

  def counting(dsk, keys, **kwargs):
    calls.append(1)
    return dask.get(dsk, keys, **kwargs)

  with dask.config.set(scheduler=counting), pytest.raises(ValidationError) as raised:
    checks.ensure_no_nans(ds[["zos", "thetao", "deptho", "m", "m3"]], m=["zos"], m3=["thetao"])
  assert len(calls) == 1
  lines = str(raised.value).splitlines()
  n_lat = ds.sizes["longitude"]
  assert lines[0] == f"1 errors found without time: {n_lat} NaN values found for variable deptho"
  times = ds.time.to_index()
  in_mask = int(ds.m.sum())
  assert lines[1] == (
    f"1 errors found for time {times[1]}: {in_mask} NaN values found for variable zos inside mask m"
  )
  assert lines[2].startswith(f"1 errors found for time {times[3]}: ")
  assert lines[2].endswith(
    f"{int(ds.m3.sum())} NaN values found for variable thetao inside mask m3"
  )
  assert len(lines) == 3


def test_validate_accessor_warns_or_raises(ocean_ds: xr.Dataset):
  """Failed checks warn by default and raise with `fail=True`."""
  config = ChecksConfig.model_validate(
    {
      "valid_time_coordinate": {
        "start": datetime(2020, 2, 26),
        "end": datetime(2020, 3, 10),
        "freq": "1D",
      }
    }
  )
  with pytest.warns(UserWarning, match="valid_time_coordinate failed"):
    ocean_ds.arcomake.validate(config)
  with pytest.raises(ValidationError):
    ocean_ds.arcomake.validate(config, fail=True)


def with_seafrac(ocean_ds: xr.Dataset) -> xr.Dataset:
  """The synthetic ocean with a static sea fraction: 1 where it has values, 0 below the sea
  floor and on land (`thetao` is NaN exactly there)."""
  frac = ocean_ds.thetao.isel(time=0, drop=True).notnull().astype("float32")
  return ocean_ds.assign(seafrac=frac)


def no_nans_config(**arguments) -> ChecksConfig:
  return ChecksConfig.model_validate({"ensure_no_nans": arguments})


def test_exact_accepts_values_exactly_inside(ocean_ds):
  ds = with_seafrac(ocean_ds)
  ds.arcomake.validate(no_nans_config(exact=True, seafrac=["thetao", "zos"]), fail=True)


@pytest.mark.parametrize(
  ("kind", "message"),
  [
    ("missing inside", "1 NaN values found for variable thetao inside mask seafrac"),
    ("present outside", "1 values found outside mask seafrac for variable thetao"),
  ],
)
def test_exact_reports_both_directions(ocean_ds, kind: str, message: str):
  ds = with_seafrac(ocean_ds)
  thetao = ds.thetao.values.copy()
  if kind == "missing inside":
    thetao[2, 0, 3, 4] = np.nan  # a sea cell
  else:
    thetao[2, 2, 3, 1] = 1.0  # below the sea floor
  ds["thetao"] = (ds.thetao.dims, thetao)
  with pytest.raises(ValidationError, match=f"2020-02-28.*{message}"):
    ds.arcomake.validate(no_nans_config(exact=True, seafrac=["thetao", "zos"]), fail=True)
  if kind == "present outside":  # without `exact`, values outside are allowed
    ds.arcomake.validate(no_nans_config(seafrac=["thetao", "zos"]), fail=True)


def test_exact_uses_the_surface_for_2d_fields(ocean_ds):
  ds = with_seafrac(ocean_ds)
  ds["zos"] = ds.zos.fillna(0.0)  # values on land
  with pytest.raises(ValidationError, match="8 values found outside mask seafrac for variable zos"):
    ds.arcomake.validate(no_nans_config(exact=True, seafrac=["thetao", "zos"]), fail=True)


def test_masks_are_read_with_the_threshold(ocean_ds):
  """A fraction is inside where above `threshold` (at least, with `inclusive`): with 0.5, cells
  half sea are outside, so their values are reported with `exact`."""
  ds = with_seafrac(ocean_ds)
  ds["seafrac"] = ds.seafrac * 0.5
  config = {"exact": True, "seafrac": ["thetao", "zos"]}
  with pytest.raises(ValidationError, match="outside mask seafrac"):
    ds.arcomake.validate(no_nans_config(threshold=0.5, **config), fail=True)
  ds.arcomake.validate(no_nans_config(threshold=0.5, inclusive=True, **config), fail=True)


@pytest.mark.parametrize(
  "values", [[True, True, False, False], [1.0, 0.3, 0.0, 0.0], [1.0, 0.3, -1e-9, 0.0]]
)
def test_boolean_masks_and_fractions_alike(values):
  """A boolean mask is inside where True, a fraction where positive (negative round-off is
  outside); the variable is missing on the last two cells."""
  ds = xr.Dataset({"v": ("x", [1.0, 2.0, np.nan, np.nan]), "m": ("x", values)})
  checks.ensure_no_nans(ds, time_dim="t", exact=True, m=["v"])


def test_missing_values_of_a_mask_are_outside_and_reported():
  """The mask's NaN are outside (no false alarm for its variables), and the mask itself, checked
  everywhere like any other variable, is reported."""
  ds = xr.Dataset({"v": ("x", [1.0, 2.0, np.nan, np.nan]), "m": ("x", [1.0, 1.0, np.nan, np.nan])})
  with pytest.raises(ValidationError) as raised:
    checks.ensure_no_nans(ds, time_dim="t", m=["v"])
  assert str(raised.value) == "1 errors found without time: 2 NaN values found for variable m"


def test_a_mask_stored_as_a_coordinate(ocean_ds):
  ds = with_seafrac(ocean_ds).set_coords("seafrac")
  ds.arcomake.validate(no_nans_config(exact=True, seafrac=["thetao", "zos"]), fail=True)
  ds["zos"] = ds.zos.fillna(0.0)
  with pytest.raises(ValidationError, match="8 values found outside mask seafrac for variable zos"):
    ds.arcomake.validate(no_nans_config(exact=True, seafrac=["thetao", "zos"]), fail=True)


def test_a_variable_under_two_masks_is_checked_against_each(ocean_ds):
  """The second mask does not replace the first: thetao is missing below the sea floor, inside
  `everywhere`, whichever order the masks come in."""
  ds = with_seafrac(ocean_ds)
  ds["everywhere"] = xr.ones_like(ds.seafrac)
  for masks in (
    {"seafrac": ["thetao"], "everywhere": ["thetao"]},
    {"everywhere": ["thetao"], "seafrac": ["thetao"]},
  ):
    with pytest.raises(ValidationError, match="NaN values found for variable thetao"):
      checks.ensure_no_nans(ds[["thetao", "seafrac", "everywhere"]], **masks)
  checks.ensure_no_nans(ds[["thetao", "seafrac"]], seafrac=["thetao"])


def test_inclusive_at_zero_is_refused():
  """Every cell, land included, would be inside."""
  with pytest.raises(pydantic.ValidationError, match="not inclusive at 0"):
    no_nans_config(seafrac=["thetao"], inclusive=True)


def test_a_mask_with_time_is_read_at_each_time_step(ocean_ds):
  ds = with_seafrac(ocean_ds)
  ds["seafrac"] = ds.seafrac.expand_dims(time=ds.time).copy()
  ds["seafrac"][3] = 0.0  # nothing inside on the fourth day: its values are outside
  with pytest.raises(ValidationError) as raised:
    ds.arcomake.validate(no_nans_config(exact=True, seafrac=["thetao", "zos"]), fail=True)
  assert str(ds.time.to_index()[3]) in str(raised.value)
  assert len(str(raised.value).splitlines()) == 1


def test_exact_computes_once(ocean_ds, static_ds):
  """One computation for every variable and time step; the report lists the static variables,
  then per time step the variables wrong at it."""
  ds = with_seafrac(ocean_ds)
  ds["deptho"] = static_ds.deptho.fillna(0.0)  # values on land
  thetao = ds.thetao.values.copy()
  thetao[1, 0, 3, 4] = np.nan  # a sea cell
  thetao[3, 2, 3, 1] = 1.0  # below the sea floor
  ds["thetao"] = (ds.thetao.dims, thetao)
  ds["zos"] = ds.zos.where(ds.time != ds.time[3])
  calls = []

  def counting(dsk, keys, **kwargs):
    calls.append(1)
    return dask.get(dsk, keys, **kwargs)

  with dask.config.set(scheduler=counting), pytest.raises(ValidationError) as raised:
    checks.ensure_no_nans(ds.chunk(time=1), exact=True, seafrac=["deptho", "thetao", "zos"])
  assert len(calls) == 1
  land = int((ds.seafrac.isel(depth=0) == 0).sum())
  sea = int(ds.seafrac.isel(depth=0).sum())
  times = ds.time.to_index()
  assert str(raised.value).splitlines() == [
    f"1 errors found without time: {land} values found outside mask seafrac for variable deptho",
    f"1 errors found for time {times[1]}: 1 NaN values found for variable thetao inside mask seafrac",
    f"2 errors found for time {times[3]}: 1 values found outside mask seafrac for variable"
    f" thetao, {sea} NaN values found for variable zos inside mask seafrac",
  ]


def test_ensure_no_nans_reports_a_missing_mask(ocean_ds: xr.Dataset):
  """A failure to report, not a KeyError that stops `validate` and the other checks; the
  variables of the missing mask are not checked (without it, NaN on land would be reported)."""
  config = ChecksConfig.model_validate({"ensure_no_nans": {"m": ["zos"]}})
  with pytest.warns(UserWarning, match="mask 'm' not found") as record:
    ocean_ds[["zos"]].arcomake.validate(config)
  assert "NaN values" not in str(record[0].message)


def test_ensure_no_nans_fails_on_a_missing_variable(ocean_ds: xr.Dataset, static_ds: xr.Dataset):
  """A listed variable missing from the dataset (a typo, a variable dropped upstream) fails: with
  only a warning, the check passed without checking it."""
  ds = xr.merge([ocean_ds, static_ds])
  ds["m"] = ds.deptho.notnull()
  ds = ds.assign(zos=ds.zos.fillna(0.0).where(ds.m))[["zos", "m"]]
  checks.ensure_no_nans(ds, m=["zos"])
  with pytest.raises(ValidationError, match="variable 'zoss' of mask 'm' not found"):
    checks.ensure_no_nans(ds, m=["zos", "zoss"])


def test_exact_for_some_masks_only(ocean_ds):
  """`exact` lists the masks whose outside must be empty: values outside the others are allowed
  (a river mask, say, next to the sea fraction)."""
  ds = with_seafrac(ocean_ds)
  ds["zos"] = ds.zos.fillna(0.0)  # values on land
  ds["wet"] = ds.seafrac.isel(depth=0, drop=True)
  ds = ds.assign(zos2=ds.zos)
  config = {"seafrac": ["thetao"], "wet": ["zos", "zos2"]}
  ds.arcomake.validate(no_nans_config(exact=["seafrac"], **config), fail=True)
  with pytest.raises(ValidationError, match="outside mask wet for variable zos"):
    ds.arcomake.validate(no_nans_config(exact=["seafrac", "wet"], **config), fail=True)
  with pytest.raises(ValidationError, match="exact: 'sea' is not one of the masks"):
    ds.arcomake.validate(no_nans_config(exact=["sea"], **config), fail=True)


def test_a_static_variable_under_a_time_varying_mask_is_refused(ocean_ds, static_ds):
  """Checked against the mask's first time step only, it would look checked at every step."""
  ds = with_seafrac(ocean_ds).assign(deptho=static_ds.deptho)
  ds["ice"] = xr.ones_like(ds.zos)
  with pytest.raises(ValidationError, match="'deptho' has no 'time' but its mask 'ice' has"):
    checks.ensure_no_nans(ds[["deptho", "ice"]], ice=["deptho"])


def test_messages_name_the_mask_of_each_count(ocean_ds):
  """A variable under two masks: each count says which mask it belongs to."""
  ds = with_seafrac(ocean_ds)
  ds["everywhere"] = xr.ones_like(ds.seafrac)
  with pytest.raises(ValidationError) as raised:
    checks.ensure_no_nans(
      ds[["thetao", "seafrac", "everywhere"]], seafrac=["thetao"], everywhere=["thetao"]
    )
  assert "inside mask everywhere" in str(raised.value)
  assert "inside mask seafrac" not in str(raised.value)


def test_guard_no_nans_passes_values_through_lazily(ocean_ds):
  """Nothing is computed until the dataset is; values, attributes and encodings are unchanged."""
  ds = with_seafrac(ocean_ds).chunk(time=1)
  calls = []

  def counting(dsk, keys, **kwargs):
    calls.append(1)
    return dask.get(dsk, keys, **kwargs)

  with dask.config.set(scheduler=counting):
    guarded = checks.guard_no_nans(ds, exact=True, seafrac=["thetao", "zos"])
  assert calls == []
  xr.testing.assert_identical(guarded.compute(), ds.compute())
  assert guarded.thetao.encoding == ds.thetao.encoding


@pytest.mark.parametrize(
  ("kind", "message"),
  [
    (
      "missing",
      "NaN values found for variable thetao inside mask seafrac in a block, first at 2020-02-28",
    ),
    (
      "outside",
      "values found outside mask seafrac for variable thetao in a block, first at 2020-02-28",
    ),
  ],
)
def test_guard_no_nans_fails_when_computed(ocean_ds, kind: str, message: str):
  """The rule of `ensure_no_nans`, checked chunk by chunk: the error names the first time step
  wrong."""
  ds = with_seafrac(ocean_ds)
  thetao = ds.thetao.values.copy()
  if kind == "missing":
    thetao[2, 0, 3, 4] = np.nan
    thetao[4, 0, 3, 4] = np.nan
  else:
    thetao[2, 2, 3, 1] = 1.0
  ds["thetao"] = (ds.thetao.dims, thetao)
  # One chunk over all days: the first day wrong in it is named (chunks fail in any order).
  guarded = checks.guard_no_nans(ds.chunk(time=-1), exact=True, seafrac=["thetao", "zos"])
  with pytest.raises(ValidationError, match=message):
    guarded.compute()


def test_guard_no_nans_refuses_missing_masks_at_once(ocean_ds):
  with pytest.raises(ValidationError, match="mask 'sea' not found"):
    checks.guard_no_nans(ocean_ds.chunk(time=1), sea=["thetao"])


def test_attributes_required_and_distinct(ocean_ds):
  """The Moby-Dick notes (item 18): masks carried the long_name of the field they came from, and
  bottom values the long_name of the full field."""
  ds = ocean_ds.assign(
    thetao_bottom=ocean_ds.thetao.isel(depth=-1, drop=True).assign_attrs(
      long_name="Temperature", units="degC"
    )
  )
  ds["thetao"].attrs.update(long_name="Temperature")
  ds["zos"].attrs.update(long_name="Sea surface height")
  with pytest.raises(ValidationError) as raised:
    checks.attributes(ds)
  assert str(raised.value).splitlines() == ["same long_name 'Temperature': thetao, thetao_bottom"]
  ds["thetao_bottom"].attrs["long_name"] = "Temperature at the sea floor"
  checks.attributes(ds)
  del ds["zos"].attrs["units"]
  with pytest.raises(ValidationError, match="without units: zos"):
    checks.attributes(ds)
  checks.attributes(ds, required=["long_name"])


def test_attributes_check_in_a_recipe(ocean_ds):
  config = ChecksConfig.model_validate({"attributes": {"required": ["long_name"]}})
  with pytest.raises(ValidationError, match="without long_name: thetao, zos"):
    ocean_ds.arcomake.validate(config, fail=True)
