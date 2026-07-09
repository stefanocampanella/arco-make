# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import copy
import logging
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import xarray as xr
import xarray_regrid
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)
processing_module = sys.modules[__name__]

Number = int | float


def process(dataset: xr.Dataset, steps: Sequence[dict[str, Any]]) -> xr.Dataset:
  """
  Applies a sequence of postprocessing steps to a xarray.Dataset.

  The steps are provided as a list of dictionaries containing step configurations.
  All methods defined on a xarray.Dataset can be used as processing steps, in addition to the
  methods defined in this module (which have precedence).

  Args:
    dataset (xr.Dataset): The input dataset to process.
    steps (Sequence[dict[str, Any]]): Configuration for each processing step.
  Returns:
    xr.Dataset: The processed dataset.
  """

  logger.info(
    "Postprocessing dataset following steps: " + ", ".join(step["name"] for step in steps) + ". "
  )
  for config in steps:
    config = copy.deepcopy(config)
    name = config.pop("name")
    logger.info(f"Applying {name} with configuration {config}")
    step_fn: Callable[..., xr.Dataset]
    if name in dir(processing_module):
      step_fn = getattr(processing_module, name)
      dataset = step_fn(dataset, **config)
    elif name in dir(dataset):
      step_fn = getattr(dataset, name)
      dataset = step_fn(**config)
    else:
      warnings.warn(f"Unrecognized processing step {name} with configuration {config}")
  return dataset


def apply_mask(
  ds: xr.Dataset,
  mask_name: str,
  variables: Iterable[str] | None = None,
) -> xr.Dataset:
  mask = ds[mask_name]
  if variables is None:
    variables: str = ds.data_vars.keys()  # type: ignore
  data_vars = {}
  for var in ds.data_vars:
    if var in variables:
      data_vars[var] = ds[var].where(mask)
    else:
      data_vars[var] = ds[var]
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


# TODO: document astype behaviour
def astype(
  ds: xr.Dataset,
  dtype: str | None = None,
  casting: str | None = None,
  **kwargs,
) -> xr.Dataset:
  if dtype is not None and casting is not None:
    ds = ds.astype(dtype=dtype, casting=casting)
  elif kwargs is not None:
    data_vars = {}
    for variable, astype_kwargs in kwargs.items():
      if variable in ds.data_vars:
        data_vars[variable] = ds[variable].astype(**astype_kwargs)
      else:
        data_vars[variable] = ds[variable]
    ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def clip_negative(ds: xr.Dataset, variables: Sequence[str]):
  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in variables:
      data_vars[var] = da.clip(min=0.0)
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def flip(ds: xr.Dataset, dim: str) -> xr.Dataset:
  return ds.isel({dim: slice(None, None, -1)})


def gaussian_blur_extrapolate(
  ds: xr.Dataset,
  variables: Iterable[str] | None = None,
  **kwargs,
) -> xr.Dataset:
  if variables is None:
    variables: str = ds.data_vars.keys()  # type: ignore
  gaussian_filter_kwargs = kwargs.get("gaussian_filter_kwargs", {})

  def gauss_fill_nan(data: xr.DataArray) -> xr.DataArray:
    data_u = xr.where(data.isnull(), 0.0, data)
    data_u = xr.apply_ufunc(gaussian_filter, data_u, kwargs=gaussian_filter_kwargs)
    valid_frac = xr.where(data.isnull(), 0.0, 1.0)
    valid_frac = xr.apply_ufunc(gaussian_filter, valid_frac, kwargs=gaussian_filter_kwargs)
    data_u = data_u / valid_frac
    data = xr.where(data.isnull(), data_u, data)
    return data

  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in variables:
      data_vars[var] = da.map_blocks(gauss_fill_nan, template=da)
    else:
      data_vars[var] = da
    ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def get_bottom(ds: xr.Dataset, depth_dim: str) -> xr.Dataset:
  for var, da in ds.data_vars.items():
    ds[var] = _get_bottom_values(da, depth_dim)
  ds = ds.drop_vars(depth_dim)
  return ds


def get_notnull_mask(
  ds: xr.Dataset,
  variable: str,
  mask_name: str,
) -> xr.Dataset:
  ds[mask_name] = ds[variable].notnull()
  return ds


def get_sea_mask(
  ds: xr.Dataset,
  bathymetry="deptho",
  depth_coordinate="depth",
  depth_dim="level",
  threshold: float = 0.5,
  mask_name="sea_land_mask",
) -> xr.Dataset:
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
  ds[mask_name] = (dims, mask)
  return ds


def isel(ds: xr.Dataset, **kwargs) -> xr.Dataset:
  return ds.isel({dim: _get_selection(values) for dim, values in kwargs.items()})


def is_positive_mask(ds: xr.Dataset, variable: str, mask_name: str) -> xr.Dataset:
  ds[mask_name] = ds[variable] > 0.0
  return ds


def masked_fill(
  ds: xr.Dataset,
  variables: Iterable[str],
  fill_value: Number | str | dict[str, Number | str],
  mask_name: str,
) -> xr.Dataset:
  if mask_name not in ds.data_vars:
    raise ValueError(f"Mask {mask_name} does not exist")
  mask = ds[mask_name]
  data_vars = {}
  for var, da in ds.data_vars.items():
    _mask = mask.isel({dim: 0 for dim in mask.dims if dim not in da.dims}, drop=True)
    if var in variables:
      # Check if the fill value is per variable
      _fill_name_or_value = fill_value[var] if isinstance(fill_value, dict) else fill_value  # type: ignore
      #  Check if the fill value is constant or a dataset variable
      _fill_value = (
        ds[_fill_name_or_value] if isinstance(_fill_name_or_value, str) else _fill_name_or_value
      )
      data_vars[var] = xr.where(da.isnull() & _mask, _fill_value, da)
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def regrid(
  ds: xr.Dataset,
  grid: dict[str, Any],
  latitude_dim: str = "latitude",
  longitude_dim: str = "longitude",
  **kwargs,
) -> xr.Dataset:
  """
  Regrids the dataset to a new grid using xarray-regrid (default using nearest neighbor algorithm).

  Args:
    ds (xr.Dataset): The input dataset.
    grid (dict, optional): Grid specification for the target grid.
    **kwargs: Additional arguments for the regridding method.

  Returns:
    xr.Dataset: The regridded dataset.

  Raises:
    ValueError: If grid is not specified.
  """
  new_grid = xarray_regrid.Grid(**grid)
  target_dataset = new_grid.create_regridding_dataset(lat_name=latitude_dim, lon_name=longitude_dim)
  method = kwargs.get("method", "nearest")
  target_dataset = target_dataset.assign_coords(
    {
      latitude_dim: target_dataset[latitude_dim].astype(np.float32),
      longitude_dim: target_dataset[longitude_dim].astype(np.float32),
    }
  )
  regrid_conf = kwargs.get("kwargs", {})
  if method == "conservative":
    regrid_conf |= dict(latitude_coord=latitude_dim)
  ds = getattr(ds.regrid, method)(target_dataset, **regrid_conf)
  return ds


def rename_coordinates(
  ds: xr.Dataset,
  name_dict: dict[str, str],
  set_new_coordinate: dict[str, str | Iterable[Number]] | None = None,
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
      ds = ds.drop_vars(new_name)
      ds = ds.assign_coords({new_name: (new_name, new_coordinate)})
      ds = ds.assign_coords({old_name: (new_name, old_coordinate.data)})
  return ds


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
  ds = getattr(ds_resample, reduce)()
  return ds


def rescale(ds: xr.Dataset, values: dict[str, Number]) -> xr.Dataset:
  data_vars = {}
  for var, da in ds.data_vars.items():
    if var in values:
      data_vars[var] = values[var] * da  # type: ignore
    else:
      data_vars[var] = da
  ds = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
  return ds


def sel(ds: xr.Dataset, **kwargs) -> xr.Dataset:
  return ds.sel({dim: _get_selection(values) for dim, values in kwargs.items()})


def select_variables(ds: xr.Dataset, variables: Iterable[str]) -> xr.Dataset:
  return ds[variables]  # type: ignore


def transpose(ds: xr.Dataset, dims: Sequence[str], **kwargs) -> xr.Dataset:
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

  return bottom


def _get_selection[T: int | float](values: dict[str, T] | list[T] | T) -> slice | list[T] | T:
  if isinstance(values, dict):
    return slice(values.get("start"), values.get("stop"), values.get("step"))
  else:
    return values
