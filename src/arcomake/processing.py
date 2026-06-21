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

Number = int | float | np.float32 | np.float64


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
    for variable, astype_kwargs in kwargs.items():
      if variable in ds.data_vars:
        ds[variable] = ds[variable].astype(**astype_kwargs)
  return ds


def clip_negative(ds: xr.Dataset, variables: Sequence[str]):
  for var, da in ds.data_vars.items():
    if var in variables:
      ds[var] = da.clip(min=0.0)
  return ds


def flip(ds: xr.Dataset, dim: str) -> xr.Dataset:
  return ds.isel({dim: slice(None, None, -1)})


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
    mask_name="sea_land_mask",
) -> xr.Dataset:
  bathymetry_values = ds[bathymetry].values
  depth = ds[depth_coordinate].values

  def get_mask(depth_map: np.ndarray, column_depths: np.ndarray, dtype="bool") -> np.ndarray:
    cell_center_depths = np.insert(column_depths[:-1], 0, 0.0) + 0.5 * np.diff(
      column_depths, prepend=0.0
    )
    mask = np.stack([depth_map >= d for d in cell_center_depths], axis=0)
    return mask.astype(dtype)

  mask = get_mask(bathymetry_values, depth)
  dims = (depth_dim,) + ds[bathymetry].dims
  ds[mask_name] = (dims, mask)
  return ds


def isel_slice(ds: xr.Dataset, **kwargs) -> xr.Dataset:
  return ds.isel({dim: slice(slice_kwargs.get("start"), slice_kwargs.get("stop"), slice_kwargs.get("step"))
                  for dim, slice_kwargs in kwargs.items()})


def masked_fill(
    ds: xr.Dataset,
    variables: Iterable[str],
    fill_value: Number,
    mask_name: str,
) -> xr.Dataset:
  if mask_name not in ds.data_vars:
    raise ValueError(f"Mask {mask_name} does not exist")
  mask = ds[mask_name]
  for var, da in ds.data_vars.items():
    mask = mask.isel({dim: 0 for dim in mask.dims if dim not in da.dims}, drop=True)
    if var in variables:
      ds[var] = xr.where(da.isnull() & mask, fill_value, da)
  return ds


# FIXME: gauss_fill implementation is problematic for several reasons; for each data array in ds:
#    1. It assumes that mask has only spatial dimensions and that data array and mask use the same names for spatial
#    dimensions (however, this is true also for several other processing steps).
#    2. It assumes that if mask has some dimension but data array does not, then 0-th component of the mask along that
#    dimension is the one to use (e.g., for the depth/level).
#    3. It uses map_block, hence need rechunking of both mask and data array.
#  These assumptions and implementation choices should be revisited or clearly documented.
def masked_gauss_fill(
    ds: xr.Dataset,
    mask_name: str,
    variables: Iterable[str],
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
    **kwargs,
) -> xr.Dataset:
  if mask_name not in ds.data_vars:
    raise ValueError(f"Mask {mask_name} does not exist")
  mask = ds[mask_name].copy()
  gaussian_filter_kwargs = kwargs.get("gaussian_filter_kwargs", {})

  def gauss_filter_nan(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    mask = mask.isel({dim: 0 for dim in mask.dims if dim not in data.dims}, drop=True)
    data_u = xr.where(data.isnull() | np.logical_not(mask), 0.0, data)
    data_u.values = gaussian_filter(data_u.values, **gaussian_filter_kwargs)
    data_v = xr.where(data.isnull() | np.logical_not(mask), 0.0, 1.0)
    data_v.values = gaussian_filter(data_v.values, **gaussian_filter_kwargs)
    data_u = data_u / data_v
    data = xr.where(data.isnull() & mask, data_u, data)
    return data

  mask = mask.chunk(chunks={latitude_dim: -1, longitude_dim: -1})
  for var, da in ds.data_vars.items():
    if var in variables:
      # Note: ensuring that da is not chunked along mask dimensions is crucial, otherwise map_block will fail!
      da = da.chunk(chunks={latitude_dim: -1, longitude_dim: -1})
      ds[var] = da.map_blocks(gauss_filter_nan, args=(mask,), template=da)
  return ds


def regrid(
  ds: xr.Dataset,
  grid: dict[str, Any],
  latitude_dim: str= "latitude",
  longitude_dim: str= "longitude",
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
  target_dataset = new_grid.create_regridding_dataset(
    lat_name=latitude_dim, lon_name=longitude_dim
  )
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
        if new_coordinate == 'auto':
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
  for var, da in ds.data_vars.items():
    if var in values:
      ds[var] = values[var] * da # ty: ignore
  return ds


def select_variables(ds: xr.Dataset, variables: Iterable[str]) -> xr.Dataset:
  return ds[variables] # ty: ignore


def transpose(ds: xr.Dataset, dims: Sequence[str], **kwargs) -> xr.Dataset:
  return ds.transpose(*dims, **kwargs)
