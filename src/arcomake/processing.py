# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import logging
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid
from scipy.ndimage import gaussian_filter
from xarray.core.types import InterpOptions

logger = logging.getLogger(__name__)


class Process:
  """
  Applies a sequence of pre|post-processing steps to a xarray.Dataset.

  The steps are provided as a list of dictionaries containing their configurations.
  All methods defined on a xarray.Dataset can be used as processing steps, in addition to the
  methods defined in this class (which have precedence).

  Attributes:
    configs (OrderedDict[str, Any]): Configuration for each processing step.
    mask (xr.DataArray): Values on which to apply the processing step.
  """

  def __init__(self, steps: Sequence[dict[str, Any]] | None = None, mask: xr.DataArray | None = None):
    """
    Initializes the Postprocess object.

    Args:
      steps (Dict[str, Any]): Configuration for each processing step.
      mask (xr.DataArray, optional): Mask to apply to the dataset.
    """

    self.configs = OrderedDict()
    if steps is not None:
      for step in steps:
        # Copy to avoid mutating the arguments
        step = step.copy()
        name = step.pop("name")
        self.configs[name] = step
    self.mask = mask

  def __call__(self, ds: xr.Dataset) -> xr.Dataset:
    """
    Applies the configured processing steps to the provided dataset.
    Each step takes a Dataset as input and returns a Dataset.
    Additional arguments have to be provided as keyword arguments.
    Missing required arguments must issue a warning and step degrades into a no-op.

    Args:
      ds (xr.Dataset): The input dataset to process.

    Returns:
      xr.Dataset: The processed dataset.
    """

    if ("regrid" in self.configs) and ("interpolate" in self.configs):
      logger.debug(
        "Both 'regrid' and 'interpolate' are specified in the configuration. "
        "'regrid' will be applied first, followed by 'interpolate'."
      )
    logger.info(
      "Processing dataset with the following steps: " + ", ".join(self.configs.keys()) + ". "
    )
    for step in self.configs:
      conf = self.configs.get(step, {})
      logger.info(f"Applying {step} with configuration {conf}")
      if step in dir(self):
        ds = getattr(self, step)(ds, **conf)
      elif step in dir(ds):
        ds = getattr(ds, step)(**conf)
      else:
        warnings.warn(f"Unrecognized processing step {step} with configuration {conf}")
    return ds

  def transpose(self, ds: xr.Dataset, dims: Sequence[str] | None = None, **kwargs):
    if dims is None:
      warnings.warn("dims must be specified")
    else:
      ds = ds.transpose(*dims, **kwargs)
    return ds

  def regrid(
    self,
    ds: xr.Dataset,
    grid=None,
    latitude_dim="latitude",
    longitude_dim="longitude",
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

    if grid is None:
      warnings.warn("Grid must be specified")
    else:
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

  def resample(self, ds: xr.Dataset, reduce=None, **kwargs):
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

    if reduce is None:
      warnings.warn("Reduce method must be specified")
    else:
      ds_resample = ds.resample(**kwargs)
      ds = getattr(ds_resample, reduce)()
    return ds

  def rescale(self, ds: xr.Dataset, values=None) -> xr.Dataset:
    if values is None:
      warnings.warn("Values must be specified")
    else:
      for var, da in ds.data_vars.items():
        if var in values:
          ds[var] = values[var] * da
    return ds

  def rename_coordinates(self, ds: xr.Dataset, name_dict=None, set_new_coordinate=None):
    if name_dict is None:
      warnings.warn("Name dictionary must be specified")
    else:
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

  def clip_negative(self, ds: xr.Dataset, variables: Sequence[str]):
    for var, da in ds.data_vars.items():
      if var in variables:
        ds[var] = da.clip(min=0.0)
    return ds

  def select_vars(self, ds: xr.Dataset, variables: Sequence[str] | None = None) -> xr.Dataset:
    if variables is None:
      warnings.warn("Variables must be specified")
    else:
      ds = ds.drop_vars(names=[var for var in ds.data_vars if var not in variables])
    return ds

  def apply_mask(self, ds: xr.Dataset, fill_value: float = np.nan, **kwargs) -> xr.Dataset:
    if self.mask is None:
      raise ValueError("Mask must be provided to apply_mask")
    for var, da in ds.data_vars.items():
      mask = self.mask.isel({dim: 0 for dim in self.mask.dims if dim not in da.dims}, drop=True)
      ds[var] = da.where(mask, fill_value, **kwargs)
    return ds

  def fill(self, ds: xr.Dataset, fill_value: float = 0.0, variables=None) -> xr.Dataset:
    if self.mask is None or variables is None:
      raise ValueError("Mask and variables must be provided to fill")
    else:
      for var, da in ds.data_vars.items():
        mask = self.mask.isel({dim: 0 for dim in self.mask.dims if dim not in da.dims}, drop=True)
        if var in variables:
          ds[var] = xr.where(da.isnull() & mask, fill_value, da)
    return ds

  def gauss_fill(self, ds: xr.Dataset, variables=None, **kwargs) -> xr.Dataset:
    if self.mask is None or variables is None:
      raise ValueError("Mask and variables must be provided to gauss_fill")
    else:
      for var, da in ds.data_vars.items():
        if var in variables:
          mask = self.mask.isel({dim: 0 for dim in self.mask.dims if dim not in da.dims}, drop=True)
          # TODO: check that does as intended (why was I seeing less and less nans with increasing radius?)
          ds[var] = da.map_blocks(gauss_filter_nan, args=(mask,), kwargs=kwargs, template=da)
    return ds

  # FIXME: poor choice of the name, misleading. Rename it to `not_null_mask`, and revise toml configuration files accordingly
  def get_land_mask(self, ds: xr.Dataset, variable=None, mask_name=None) -> xr.Dataset:
    if variable is None:
      warnings.warn("Variable must be specified")
    elif mask_name is None:
      warnings.warn("Mask name must be specified")
    else:
      ds[mask_name] = xr.where(ds[variable].notnull(), True, False)
    return ds

  def isel_slice(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
    return ds.isel({dim: slice(slice_kwargs.get("start"), slice_kwargs.get("stop"), slice_kwargs.get("step"))
                    for dim, slice_kwargs in kwargs.items()})

  def time_shift(self, ds: xr.Dataset, quantity=None) -> xr.Dataset:
    if quantity is None:
      warnings.warn("Shift amount must be specified")
    else:
      ds = ds.assign_coords(time=ds.time - pd.Timedelta(quantity))
    return ds

  def flip(self, ds: xr.Dataset, dim=None) -> xr.Dataset:
    if dim is None:
      warnings.warn("Dim must be specified")
    else:
      ds = ds.isel({dim: slice(None, None, -1)})
    return ds

  def interpolate_na(self, ds, **kwargs) -> xr.Dataset:
    dim = kwargs.get("dim", "time")
    output_chunks = kwargs.pop("output_chunks", {})
    ds = ds.chunk({dim: -1})
    ds = ds.interpolate_na(**kwargs)
    ds = ds.chunk(**output_chunks)
    return ds

  def interpolate(
    self,
    ds: xr.Dataset,
    minimum_latitude: float = -90.0,
    maximum_latitude: float = 90.0,
    minimum_longitude: float = -180.0,
    maximum_longitude: float = 179.0,
    resolution: float = 1.0,
    method: InterpOptions = "linear",
    assume_sorted: bool = True,
    kwargs: dict[str, Any] | None = None,
  ) -> xr.Dataset:
    """
    Interpolates the dataset to a regular latitude/longitude grid.

    Args:
      ds (xr.Dataset): The input dataset.
      minimum_latitude (float): Minimum latitude of the target grid.
      maximum_latitude (float): Maximum latitude of the target grid.
      minimum_longitude (float): Minimum longitude of the target grid.
      maximum_longitude (float): Maximum longitude of the target grid.
      resolution (float): Grid resolution in degrees.
      method (InterpOptions): Interpolation method.
      assume_sorted (bool): Whether to assume the input coordinates are sorted.
      kwargs (Dict[str, Any], optional): Additional arguments for xarray's interp.

    Returns:
      xr.Dataset: The interpolated dataset.
    """

    eps = np.finfo(ds.latitude.dtype).eps
    latitude = np.arange(
      minimum_latitude, maximum_latitude + eps, step=resolution, dtype=np.float32
    )
    longitude = np.arange(minimum_longitude, maximum_longitude, step=resolution, dtype=np.float32)
    ds = ds.interp(
      latitude=latitude,
      longitude=longitude,
      method=method,
      assume_sorted=assume_sorted,
      kwargs=kwargs,
    ).astype(np.float32)
    return ds

  def get_sea_mask(
    self,
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

  # TODO: document astype behaviour
  def astype(self, ds: xr.Dataset, dtype=None, casting=None, **kwargs) -> xr.Dataset:
    if dtype is not None and casting is not None:
      ds = ds.astype(dtype=dtype, casting=casting)
    elif kwargs is not None:
      for variable, astype_kwargs in kwargs.items():
        if variable in ds.data_vars:
          ds[variable] = ds[variable].astype(**astype_kwargs)
    return ds


def gauss_filter_nan(data, mask, **kwargs):
  data_u = xr.where(data.isnull() | np.logical_not(mask), 0.0, data)
  data_u.values = gaussian_filter(data_u.values, **kwargs)
  data_v = xr.where(data.isnull() | np.logical_not(mask), 0.0, 1.0)
  data_v.values = gaussian_filter(data_v.values, **kwargs)
  data_u = data_u / data_v
  data = xr.where(data.isnull() & mask, data_u, data)
  return data
