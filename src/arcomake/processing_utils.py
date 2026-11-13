# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal

import numpy as np
import xarray as xr
import xarray_regrid
from pydantic import BaseModel, ConfigDict, model_validator

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
  mask = ds[mask_name]
  if variables is None:
    variables = ds.data_vars.keys()
  data_vars = {}
  for var in ds.data_vars:
    if var in variables:
      data_vars[var] = ds[var].where(mask)
    else:
      data_vars[var] = ds[var]
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


@step
def assign_var_attrs(ds: xr.Dataset, **kwargs) -> xr.Dataset:
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
def clip_negative(ds: xr.Dataset, variables: Iterable[Hashable]):
  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in variables:
      data_vars[var] = da.clip(min=0.0)
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


@step
def drop_static_vars(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
  ds = ds.drop_vars([name for name, var in ds.data_vars.items() if time_dim not in var.dims])
  return ds


@step
def drop_var_attrs(ds: xr.Dataset, variables: Iterable[Hashable]) -> xr.Dataset:
  ds = ds.copy(deep=False)
  for key in variables:
    ds[key] = ds[key].drop_attrs()
  return ds


@step
def flip(ds: xr.Dataset, dim: str) -> xr.Dataset:
  return ds.isel({dim: slice(None, None, -1)})


@step
def get_bottom(
  ds: xr.Dataset,
  depth_dim: str,
  variables: Iterable[Hashable] | Mapping[Hashable, Hashable] | None = None,
  replace=True,
) -> xr.Dataset:
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
def isel(ds: xr.Dataset, **kwargs) -> xr.Dataset:
  return ds.isel({dim: _get_selection(values) for dim, values in kwargs.items()})


@step
def is_positive_mask(ds: xr.Dataset, variable: Hashable, mask_name: Hashable) -> xr.Dataset:
  return ds.assign({mask_name: ds[variable] > 0.0})


@step
def masked_fill(
  ds: xr.Dataset,
  variables: Iterable[Hashable],
  fill_value: Number | str | Mapping[Hashable, Number | Hashable],
  mask_name: Hashable,
) -> xr.Dataset:
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


@step
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
  """
  Regrids the dataset to a new grid using xarray-regrid (default using nearest neighbor algorithm).

  Args:
    ds (xr.Dataset): The input dataset.
    grid (dict): Grid specification for the target grid (`xarray_regrid.Grid` arguments).
    method: The xarray-regrid method.
    kwargs: Additional arguments for the regridding method.

  Returns:
    xr.Dataset: The regridded dataset.
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


@step
def resample(ds: xr.Dataset, reduce: str, **kwargs) -> xr.Dataset:
  """
  Resamples the dataset along a specified dimension and reduces it using a given method.
  The implementation checks that the 'reduce' method is specified.

  Args:
    ds (xr.Dataset): The input dataset.
    **kwargs: Arguments for resampling, it must include 'reduce' specifying the reduction method.

  Returns:
    xr.Dataset: The resampled dataset.

  Raises:
    ValueError: If 'reduce' method is not specified.
  """
  ds_resample = ds.resample(**kwargs)
  ds_reduced = getattr(ds_resample, reduce)()
  assert isinstance(ds_reduced, xr.Dataset)
  return ds_reduced


@step
def rescale(ds: xr.Dataset, **kwargs) -> xr.Dataset:
  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in kwargs:
      data_vars[var] = kwargs[var] * da  # type: ignore
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


@step
def sel(ds: xr.Dataset, **kwargs) -> xr.Dataset:
  return ds.sel({dim: _get_selection(values) for dim, values in kwargs.items()})


@step
def select_variables(ds: xr.Dataset, variables: Iterable[Hashable]) -> xr.Dataset:
  return ds[list(variables)]


@step
def transpose(ds: xr.Dataset, dims: Iterable[Hashable], **kwargs) -> xr.Dataset:
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
