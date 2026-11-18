# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The processing steps a recipe can name (`[[...postprocess]]` tables and similar).

Every step takes a dataset and returns a new one; it never modifies its input (values,
attributes or encodings) nor its arguments. Steps are lazy: they build Dask graphs and leave
the computation to the final write, except `get_sea_mask`, which reads the (static) bathymetry.

In a recipe, `name` selects the step and every other key is an argument:

    [[datasets.ocean.postprocess]]
    name = "select_variables"
    variables = ["thetao", "so"]

`docs/steps.md` (or `arco-make steps`) lists every step with its arguments.
"""

import datetime
import logging
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, model_validator

from arcomake.chunk_plan import Needs
from arcomake.steps import check_arguments, step

logger = logging.getLogger(__name__)

Number = int | float


class ProcessingStepConfig(BaseModel):
  """One step of a recipe: `name` selects the step, every other key is one of its arguments.

  The step and its arguments are checked when the recipe is loaded (see `arcomake.steps`).
  """

  model_config = ConfigDict(extra="allow")

  name: str

  @model_validator(mode="after")
  def _check_step(self):
    check_arguments(self.name, self.model_extra or {})
    return self


@step
def apply_mask(
  ds: xr.Dataset,
  mask_name: Hashable,
  variables: Iterable[Hashable] | None = None,
) -> xr.Dataset:
  """Set variables to NaN outside a mask.

  Args:
    mask_name: Boolean variable, True where values are kept.
    variables: Variables to mask; by default every data variable except the mask.
  """
  mask = ds[mask_name]
  if variables is None:
    variables = [name for name in ds.data_vars if name != mask_name]
  data_vars = {}
  for var in ds.data_vars:
    if var in variables:
      data_vars[var] = ds[var].where(mask)
    else:
      data_vars[var] = ds[var]
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


@step
def assign_var_attrs(ds: xr.Dataset, **kwargs: Mapping[str, Any]) -> xr.Dataset:
  """Replace the attributes of variables or coordinates, one table per name::

  name = "assign_var_attrs"
  lat = {standard_name = "latitude", units = "degrees_north"}
  """
  ds = ds.copy(deep=False)
  for key, attrs in kwargs.items():
    ds[key].attrs = attrs
  return ds


CastingType = Literal["no", "equiv", "safe", "same_kind", "unsafe"]


@step
def astype(
  ds: xr.Dataset,
  dtype: str | None = None,
  casting: CastingType | None = None,
  **kwargs: Mapping[str, Any],
) -> xr.Dataset:
  """Convert the data type of the data variables, all at once or per variable.

  Either `dtype` (and optionally `casting`, as in NumPy) for every data variable::

    name = "astype"
    dtype = "float32"
    casting = "same_kind"

  or one table per variable, with the arguments of `DataArray.astype`; the variables not listed
  are kept unchanged::

    name = "astype"
    uparea = {dtype = "float32", casting = "same_kind"}
  """
  if dtype is not None:
    if kwargs:
      raise ValueError("astype takes either `dtype` for all variables or one table per variable")
    return ds.astype(dtype=dtype, **({"casting": casting} if casting is not None else {}))
  if casting is not None:
    raise ValueError("astype: `casting` without `dtype`")
  converted = {}
  for variable, astype_kwargs in kwargs.items():
    if variable not in ds.data_vars:
      raise ValueError(f"astype: {variable!r} is not a data variable")
    converted[variable] = ds[variable].astype(**astype_kwargs)
  return ds.assign(converted)


@step
def clip_negative(ds: xr.Dataset, variables: Iterable[Hashable]) -> xr.Dataset:
  """Set negative values of the listed variables to 0 (e.g. accumulated precipitation); NaN stay."""
  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in variables:
      data_vars[var] = da.clip(min=0.0)
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def _regular_step(values: np.ndarray) -> Any:
  """The spacing of coordinate values if it is constant, else None: exactly for times, within a
  relative 1e-3 for numbers (a 0.1 or float32 1/12 degree grid is not exact in binary)."""
  if values.size < 2:
    return None
  ordered = np.sort(values)
  diffs = np.diff(ordered)
  if not np.issubdtype(diffs.dtype, np.floating):
    steps = np.unique(diffs)
    return steps[0] if steps.size == 1 else None
  step = (ordered[-1] - ordered[0]) / (values.size - 1)
  return step if step > 0 and np.allclose(diffs, step, rtol=1e-3, atol=0.0) else None


def time_coverage(stamps: Any) -> dict[str, str]:
  """ACDD `time_coverage_start`, `_end` (first and last stamps) and `_resolution` (an ISO 8601
  duration, if the stamps are regular)."""
  index = pd.DatetimeIndex(np.asarray(stamps))
  attrs = {
    "time_coverage_start": f"{index.min():%Y-%m-%dT%H:%M:%SZ}",
    "time_coverage_end": f"{index.max():%Y-%m-%dT%H:%M:%SZ}",
  }
  if (spacing := _regular_step(index.values)) is not None:
    attrs["time_coverage_resolution"] = _iso_duration(spacing)
  return attrs


def _iso_duration(step: np.timedelta64) -> str:
  """An ISO 8601 duration: P1D, PT3H, or pandas' form for others."""
  delta = pd.Timedelta(step)
  if delta % pd.Timedelta("1D") == pd.Timedelta(0):
    return f"P{delta // pd.Timedelta('1D')}D"
  if delta % pd.Timedelta("1h") == pd.Timedelta(0):
    return f"PT{delta // pd.Timedelta('1h')}H"
  return delta.isoformat()


@step
def coverage_attrs(
  ds: xr.Dataset,
  latitude_dim: Hashable = "lat",
  longitude_dim: Hashable = "lon",
  vertical_dim: Hashable | None = None,
  time_dim: Hashable | None = "time",
) -> xr.Dataset:
  """Set the ACDD attributes of the dataset's extent from its coordinates.

  `geospatial_lat_*` and `geospatial_lon_*` (min, max, units, and resolution if the spacing is
  regular), `geospatial_vertical_*` (min, max, and the coordinate's `units` and `positive`) if
  `vertical_dim` is given, and `time_coverage_start`, `_end` (the first and last stamps) and
  `_resolution` (an ISO 8601 duration). Attributes typed by hand drift from the data, e.g.
  longitudes -180 to 180 over coordinates 0 to 359.75. In `download`, the time coverage is that
  of the whole output (the stored stamps and the block's), not of the block::

    name = "coverage_attrs"
    vertical_dim = "depth"
  """
  attrs: dict[str, Any] = {}
  for dim, axis, units in (
    (latitude_dim, "lat", "degrees_north"),
    (longitude_dim, "lon", "degrees_east"),
  ):
    values = np.asarray(ds[dim].values, dtype=np.float64)
    attrs[f"geospatial_{axis}_min"] = round(float(values.min()), 6)
    attrs[f"geospatial_{axis}_max"] = round(float(values.max()), 6)
    attrs[f"geospatial_{axis}_units"] = units
    if (spacing := _regular_step(values)) is not None:
      attrs[f"geospatial_{axis}_resolution"] = f"{float(spacing):.6g} degree"
  if vertical_dim is not None:
    vertical = ds[vertical_dim]
    values = np.asarray(vertical.values, dtype=np.float64)
    attrs["geospatial_vertical_min"] = round(float(values.min()), 6)
    attrs["geospatial_vertical_max"] = round(float(values.max()), 6)
    for key in ("units", "positive"):
      if key in vertical.attrs:
        attrs[f"geospatial_vertical_{key}"] = vertical.attrs[key]
  if time_dim is not None:
    attrs |= time_coverage(ds[time_dim].values)
  return ds.assign_attrs(attrs)


@step
def drop_static_vars(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
  """Drop the data variables without the time dimension (masks, bathymetry...)."""
  ds = ds.drop_vars([name for name, var in ds.data_vars.items() if time_dim not in var.dims])
  return ds


@step
def drop_var_attrs(ds: xr.Dataset, variables: Iterable[Hashable]) -> xr.Dataset:
  """Remove all attributes of the listed variables or coordinates."""
  ds = ds.copy(deep=False)
  for key in variables:
    ds[key] = ds[key].drop_attrs()
  return ds


@step
def flip(ds: xr.Dataset, dim: str) -> xr.Dataset:
  """Reverse the order along a dimension (e.g. make ERA5 latitudes increase)."""
  return ds.isel({dim: slice(None, None, -1)})


def _get_bottom_needs(ds: xr.Dataset, depth_dim: str, **_: Any) -> Needs:
  """`get_bottom` joins the whole depth of a variable in each of its tasks."""
  return Needs(gathered=frozenset({depth_dim}))


@step(needs=_get_bottom_needs)
def get_bottom(
  ds: xr.Dataset,
  depth_dim: str,
  variables: Iterable[Hashable] | Mapping[Hashable, Hashable] | None = None,
  replace: bool = True,
) -> xr.Dataset:
  """The deepest valid (non-NaN) value of each column along `depth_dim`.

  The depth dimension is rechunked to a single chunk. Columns without valid values stay NaN.

  Args:
    depth_dim: Vertical dimension; its coordinate must increase downwards.
    variables: Variables to reduce (default: all). With `replace = false`, a table from each
      variable to the name of its new bottom variable, e.g. `{thetao = "thetao_bottom"}`.
    replace: Replace the variables by their bottom values instead of adding new ones.
  """
  ds = ds.copy()
  if variables is None:
    variables = ds.data_vars.keys()
  if replace and isinstance(variables, Mapping):
    logger.warning(
      "Replacing variables in place with bottom values; mapping target names are ignored because 'replace=True'."
    )
  if not (replace or isinstance(variables, Mapping)):
    raise ValueError(
      "When not replacing, variables needs to be a mapping from source variable names to bottom variable names."
    )
  for name in variables:
    bottom_da = _get_bottom_values(ds[name], depth_dim)
    if replace:
      ds[name] = bottom_da
    else:
      assert isinstance(variables, Mapping)
      bottom_name = variables[name]
      ds[bottom_name] = bottom_da
  return ds


@step
def get_notnull_mask(
  ds: xr.Dataset,
  variable: Hashable,
  mask_name: Hashable,
) -> xr.Dataset:
  """Add a boolean mask, True where `variable` is not null."""
  return ds.assign({mask_name: ds[variable].notnull()})


@step
def get_sea_mask(
  ds: xr.Dataset,
  bathymetry: Hashable = "deptho",
  depth_coordinate: Hashable = "depth",
  depth_dim: Hashable | None = None,
  threshold: float = 0.5,
  mask_name: Hashable = "sea_land_mask",
) -> xr.Dataset:
  """Add a 3D sea mask from a bathymetry: a cell is sea where the sea floor is below it.

  Cell faces are rebuilt from the level centres, assuming each centre is mid-cell and the top
  face of the first level is at 0 m. A cell is sea where the bathymetry is deeper than
  `threshold * top + (1 - threshold) * bottom` of the cell; with `threshold = 0.5` this is the
  level centre. Land (NaN bathymetry) is never sea. Eager: the bathymetry and the depth
  coordinate are loaded.

  Args:
    bathymetry: Sea-floor depth, positive downwards.
    depth_coordinate: Level centres, positive downwards.
    depth_dim: Vertical dimension of the mask (default: the depth coordinate's dimension).
    threshold: Fraction of the cell, between 0 (bottom face) and 1 (top face).
    mask_name: Name of the new mask.
  """
  if depth_dim is None:
    # The mask's vertical dimension is the one of the depth coordinate.
    (depth_dim,) = ds[depth_coordinate].dims
  bathymetry_values = ds[bathymetry].values
  depth = ds[depth_coordinate].values
  if threshold < 0.0 or threshold > 1.0:
    raise ValueError(f"depth_threshold must be between 0.0 and 1.0, got {threshold}")

  def get_mask(bathy: np.ndarray, centroid_depths: np.ndarray, dtype="bool") -> np.ndarray:
    assert centroid_depths.ndim == 1
    # The top face depth of the first level is zero meters below the geoid
    boundary_depths = [0.0]
    for centroid_depth in centroid_depths:
      last_boundary_depth = boundary_depths[-1]
      cell_height = 2 * (centroid_depth - last_boundary_depth)
      boundary_depths.append(boundary_depths[-1] + cell_height)
    boundary_depths = np.asarray(boundary_depths)
    top_face_depths = boundary_depths[:-1]
    bottom_face_depths = boundary_depths[1:]
    minimum_depths = threshold * top_face_depths + (1 - threshold) * bottom_face_depths
    mask = np.stack([bathy > d for d in minimum_depths], axis=0)
    return mask.astype(dtype)

  mask = get_mask(bathymetry_values, depth)
  dims = (depth_dim,) + ds[bathymetry].dims
  return ds.assign({mask_name: (dims, mask)})


@step
def isel(ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
  """Select by position along dimensions: an index, a list, or `{start, stop, step}`::

  name = "isel"
  time = {start = 0, stop = 365}
  """
  return ds.isel({dim: _get_selection(values) for dim, values in kwargs.items()})


@step
def is_positive_mask(ds: xr.Dataset, variable: Hashable, mask_name: Hashable) -> xr.Dataset:
  """Add a boolean mask, True where `variable` is strictly positive (NaN gives False)."""
  return ds.assign({mask_name: ds[variable] > 0.0})


@step
def masked_fill(
  ds: xr.Dataset,
  variables: Iterable[Hashable],
  fill_value: Number | str | Mapping[Hashable, Number | Hashable],
  mask_name: Hashable,
) -> xr.Dataset:
  """Fill missing values inside a mask; values outside the mask are left as they are.

  The fill value is a number, the name of another variable (e.g. the bottom value of the
  column), or a table with one of those per variable::

    name = "masked_fill"
    mask_name = "glorys_mask"
    variables = ["thetao", "so"]
    fill_value = {thetao = "thetao_bottom", so = "so_bottom"}

  For variables without one of the mask's dimensions (e.g. 2D fields and a 3D mask), the
  first index of that dimension is used (e.g. the surface level).
  """
  if mask_name not in ds.data_vars:
    raise ValueError(f"Mask {mask_name} does not exist")
  mask = ds[mask_name]
  data_vars = {}
  for var, da in ds.data_vars.items():
    _mask = mask.isel({dim: 0 for dim in mask.dims if dim not in da.dims}, drop=True)
    if var in variables:
      # Check if the fill value is per variable
      _fill_name_or_value = fill_value[var] if isinstance(fill_value, dict) else fill_value
      #  Check if the fill value is constant or a dataset variable
      _fill_value = (
        ds[_fill_name_or_value] if isinstance(_fill_name_or_value, str) else _fill_name_or_value
      )
      data_vars[var] = xr.where(da.notnull() | ~_mask, da, _fill_value, keep_attrs="override")
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def _regrid_needs(
  ds: xr.Dataset, latitude_dim: str = "latitude", longitude_dim: str = "longitude", **_: Any
) -> Needs:
  """Whole horizontal dimensions: xarray-regrid keeps their chunks, and one chunk is one task
  per time and level instead of a graph of partial products."""
  return Needs(whole=frozenset({latitude_dim, longitude_dim}))


@step(needs=_regrid_needs)
def regrid(
  ds: xr.Dataset,
  grid: Mapping[str, Any],
  latitude_dim: str = "latitude",
  longitude_dim: str = "longitude",
  method: Literal[
    "nearest", "linear", "cubic", "conservative", "most_common", "least_common", "stat"
  ] = "nearest",
  kwargs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
  """Regrid to a regular latitude-longitude grid with xarray-regrid.

  The target coordinates are float32. Example::

    name = "regrid"
    method = "conservative"
    grid = {south = -90.0, north = 90.0, west = 0.0, east = 359.75, resolution_lat = 0.25, resolution_lon = 0.25}
    kwargs = {skipna = true, nan_threshold = 1.0}

  With `method = "conservative"`, `skipna = false` counts missing source cells as 0 (e.g. land
  as depth 0), while `skipna = true` averages the valid part of each target cell, and leaves NaN
  where its valid fraction is below `1 - nan_threshold`.

  Args:
    grid: Arguments of `xarray_regrid.Grid` (`north`, `east`, `south`, `west`,
      `resolution_lat`, `resolution_lon`). In a download or process recipe, also the name of a
      grid of its `[grids]` table, to write a grid once for several steps.
    latitude_dim, longitude_dim: Names of the horizontal dimensions.
    method: The xarray-regrid method (default "nearest").
    kwargs: Further arguments of that method (e.g. `skipna`, `nan_threshold`; `output_chunks`
      only outside download recipes, whose chunks are planned).
  """
  new_grid = xarray_regrid.Grid(**grid)
  target_dataset = new_grid.create_regridding_dataset(lat_name=latitude_dim, lon_name=longitude_dim)
  target_dataset = target_dataset.assign_coords(
    {
      latitude_dim: target_dataset[latitude_dim].astype(np.float32),
      longitude_dim: target_dataset[longitude_dim].astype(np.float32),
    }
  )
  regrid_conf = dict(kwargs or {})
  if method == "conservative":
    regrid_conf["latitude_coord"] = latitude_dim
  regridded = getattr(ds.regrid, method)(target_dataset, **regrid_conf)
  assert isinstance(regridded, xr.Dataset)
  return regridded


@step
def rename_coordinates(
  ds: xr.Dataset,
  name_dict: Mapping[Hashable, Hashable],
  set_new_coordinate: Mapping[Hashable, Literal["auto"] | Iterable[Number]] | None = None,
) -> xr.Dataset:
  """Rename coordinates (and their dimensions); names that are not coordinates are skipped.

  Skipping lets one table serve several sources. With `set_new_coordinate = {NEW = "auto"}`
  (or a list of values), the renamed dimension gets a new index 0..n-1 (or those values) and
  the old values are kept as a non-index coordinate under the old name.
  """
  name_dict = {
    old_name: new_name for old_name, new_name in name_dict.items() if old_name in ds.coords
  }
  ds = ds.rename(name_dict=name_dict)
  for old_name, new_name in name_dict.items():
    if (set_new_coordinate is not None) and (new_name in set_new_coordinate):
      # Keep old coordinate values as a non-index coordinate (with new name)
      old_coordinate = ds.coords[new_name]
      new_coordinate = set_new_coordinate[new_name]
      if isinstance(new_coordinate, str):
        if new_coordinate == "auto":
          new_coordinate = list(range(len(old_coordinate.data)))
        else:
          raise ValueError(f"Invalid value for new coordinate: {new_coordinate}")
      ds = ds.drop_indexes(new_name)
      ds = ds.drop_vars([new_name])
      ds = ds.assign_coords({new_name: (new_name, new_coordinate)})
      ds = ds.assign_coords({old_name: (new_name, old_coordinate.data)})
  return ds


def _fixed_period(frequency: Any) -> pd.Timedelta | None:
  """The length of a resampling frequency if fixed ("1D", "3h"), else None ("1MS", "1W")."""
  if isinstance(frequency, datetime.timedelta):
    return pd.Timedelta(frequency)
  try:
    offset = to_offset(frequency)
  except (TypeError, ValueError):
    return None
  if isinstance(offset, pd.offsets.Tick):
    return pd.Timedelta(offset)
  if isinstance(offset, pd.offsets.Day):  # calendar days: 24 h in UTC, as the data are
    return pd.Timedelta(days=offset.n)
  return None


def _regular_spacing(index: pd.Index) -> pd.Timedelta | None:
  """The step between consecutive values of a time index, if they are evenly spaced."""
  if len(index) < 2:
    return None
  steps = np.unique(np.diff(np.asarray(index)))
  return pd.Timedelta(steps[0]) if len(steps) == 1 else None


def _since_midnight(stamp: Any) -> pd.Timedelta:
  """Time from the midnight of its day to `stamp` (numpy or cftime)."""
  if isinstance(stamp, np.datetime64):
    timestamp = pd.Timestamp(stamp)
    return timestamp - timestamp.normalize()
  return pd.Timedelta(stamp - stamp.replace(hour=0, minute=0, second=0, microsecond=0))


# Arguments of `Dataset.resample` that move the bins away from the day's midnight.
_SHIFTED_BINS = ("offset", "origin", "base", "loffset")


def _resample_needs(
  ds: xr.Dataset, indexer: Mapping[Hashable, Any] | None = None, **kwargs: Any
) -> Needs:
  """Chunks of whole bins: a fixed frequency over evenly spaced stamps is a number of steps per
  bin, and chunk sizes along the dimension are multiples of it, so that no bin straddles chunks.
  Only for bins starting at the first stamp (chunks start there): left-closed, from midnight
  (no offset or origin), the first stamp on a bin edge."""
  if kwargs.get("closed") == "right" or any(key in kwargs for key in _SHIFTED_BINS):
    return Needs()
  multiple: dict[Hashable, int] = {}
  for dim, frequency in {**(indexer or {}), **kwargs}.items():
    if dim not in ds.indexes:  # `reduce`, `label`, `closed`...
      continue
    index = ds.indexes[dim]
    period = _fixed_period(frequency)
    spacing = _regular_spacing(index)
    if period is None or spacing is None or spacing <= pd.Timedelta(0):
      logger.debug(f"resample along {dim!r}: bins of no fixed number of steps")
      continue
    if _since_midnight(np.asarray(index)[0]) % period != pd.Timedelta(0):
      logger.debug(f"resample along {dim!r}: the first stamp is not on a bin edge")
      continue
    if period % spacing == pd.Timedelta(0) and period >= spacing:
      multiple[dim] = int(period // spacing)
  return Needs(multiple=multiple)


@step(needs=_resample_needs)
def resample(
  ds: xr.Dataset,
  reduce: Literal["mean", "sum", "min", "max", "median", "std", "var", "first", "last", "count"],
  **kwargs: Any,
) -> xr.Dataset:
  """Resample along time and reduce each bin; the other arguments go to `Dataset.resample`::

    name = "resample"
    time = "1D"
    label = "left"
    closed = "left"
    reduce = "mean"

  With `label` and `closed` "left", day D is the reduction of the stamps of D (00:00 to 23:00
  for hourly data). Each Dask chunk should hold whole bins, or the reduction rechunks.
  """
  ds_resample = ds.resample(**kwargs)
  ds_reduced = getattr(ds_resample, reduce)()
  assert isinstance(ds_reduced, xr.Dataset)
  return ds_reduced


@step
def rescale(ds: xr.Dataset, **kwargs: Number) -> xr.Dataset:
  """Multiply variables by constant factors, one per variable (e.g. `tp = 1000.0`)."""
  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in kwargs:
      data_vars[var] = kwargs[var] * da  # type: ignore
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


@step
def sel(ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
  """Select by label along dimensions: a label, a list, or `{start, stop, step}` (inclusive)::

  name = "sel"
  latitude = {start = -76.9167, stop = 89.9167}
  """
  return ds.sel({dim: _get_selection(values) for dim, values in kwargs.items()})


@step
def select_variables(ds: xr.Dataset, variables: Iterable[Hashable]) -> xr.Dataset:
  """Keep only the listed data variables (and the coordinates of their dimensions)."""
  return ds[list(variables)]


@step
def sin_cos(
  ds: xr.Dataset,
  variable: Hashable,
  sin_name: Hashable,
  cos_name: Hashable,
  unit: Literal["degrees", "radians"] = "degrees",
  keep: bool = False,
) -> xr.Dataset:
  """Replace an angle by its sine and cosine, two ordinary fields to regrid and average.

  An angle cannot be averaged as a number: 350 and 10 degrees give 180. Its sine and cosine can,
  and the averaged pair gives the mean direction (`atan2(sin, cos)`) and, by its length (0 to
  1), how steady the direction was. Apply it before any regridding or time mean, e.g. to a mean
  wave direction::

    name = "sin_cos"
    variable = "VMDR"
    sin_name = "mwd_sin"
    cos_name = "mwd_cos"

  Arguments:
    variable: The angle, in `unit`.
    sin_name, cos_name: Names of the new variables (units "1"); missing values stay missing.
    unit: "degrees" or "radians".
    keep: Keep the angle too (dropped by default).
  """
  angle = ds[variable]
  radians = angle * (np.pi / 180.0) if unit == "degrees" else angle
  described = angle.attrs.get("long_name", str(variable))
  components = {
    sin_name: xr.apply_ufunc(np.sin, radians, dask="allowed").assign_attrs(
      units="1", long_name=f"sine of {described}"
    ),
    cos_name: xr.apply_ufunc(np.cos, radians, dask="allowed").assign_attrs(
      units="1", long_name=f"cosine of {described}"
    ),
  }
  for component in components.values():
    component.encoding = {}
  return (ds if keep else ds.drop_vars([variable])).assign(components)


@step
def transpose(ds: xr.Dataset, dims: Iterable[Hashable], **kwargs: Any) -> xr.Dataset:
  """Reorder dimensions; other arguments go to `Dataset.transpose` (e.g. `missing_dims`)."""
  return ds.transpose(*dims, **kwargs)


def _get_bottom_values(da: xr.DataArray, depth_dim: str = "depth") -> xr.DataArray:
  """
  Extract the deepest valid (non-NaN) value along `depth_dim` for every
  (time, latitude, longitude) point.

  Assumes:
      - `da` has dims (time, depth, latitude, longitude) (order doesn't matter).
      - `depth_dim` coordinate is sorted.
      - `da` may be dask-backed and chunked along `time` and `depth_dim`.

  Returns
  -------
  xr.DataArray
      Same dims as `da` minus `depth_dim`, i.e. (time, latitude, longitude).
  """

  # apply_ufunc needs the full core dimension (depth) in a single chunk.
  # We only rechunk depth (usually small), keeping time/lat/lon chunking
  # untouched so we don't lose parallelism there.
  if da.chunks is not None:
    da = da.chunk({depth_dim: -1})

  def _last_valid_along_last_axis(block: np.ndarray) -> np.ndarray:
    # `block` has `depth` as its last axis (apply_ufunc moves core dims
    # to the end automatically).
    valid = ~np.isnan(block)
    has_valid = valid.any(axis=-1)

    # Index of the last True (deepest valid) along the depth axis.
    # argmax on the reversed mask finds the first True from the end,
    # in O(depth) with no Python-level loop.
    reversed_valid = valid[..., ::-1]
    idx_from_end = np.argmax(reversed_valid, axis=-1)
    last_idx = valid.shape[-1] - 1 - idx_from_end

    result = np.take_along_axis(block, last_idx[..., np.newaxis], axis=-1)[..., 0]

    # Profiles with no valid value at all should stay NaN.
    return np.where(has_valid, result, np.nan)

  bottom = xr.apply_ufunc(
    _last_valid_along_last_axis,
    da,
    input_core_dims=[[depth_dim]],
    output_core_dims=[[]],
    dask="parallelized",
    output_dtypes=[da.dtype],
  )
  assert isinstance(bottom, xr.DataArray)
  return bottom


def _get_selection[T: int | float](values: Mapping[str, T] | list[T] | T) -> slice | list[T] | T:
  if isinstance(values, Mapping):
    if not set(values.keys()).issubset({"start", "stop", "step"}):
      raise ValueError(f"Invalid selection: {values}")
    return slice(values.get("start"), values.get("stop"), values.get("step"))
  else:
    return values
