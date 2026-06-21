from datetime import datetime

import xarray as xr


@xr.register_dataset_accessor("arcomake")
@xr.register_dataarray_accessor("arcomake")
class DateIntervalSelector:
  """XArray DataArray and Dataset sel method is right inclusive, this accessor provides right exclusive selection."""

  def __init__(self, xarray_obj: xr.DataArray | xr.Dataset, time_dim: str = "time"):
    self._obj = xarray_obj
    self._time_dim = time_dim

  def time_sel(self, start_datetime: datetime, end_datetime: datetime):
    if self._time_dim in self._obj.dims:
      time_coordinate = self._obj[self._time_dim]
      time_coordinate = time_coordinate.sel(time=slice(start_datetime, end_datetime))
      time_coordinate = time_coordinate.to_index()
      if not time_coordinate.is_monotonic_increasing:
        raise ValueError(f"Time coordinate is not sorted: {self._obj[self._time_dim]}")
      time_coordinate_last_value: datetime = time_coordinate[-1].to_pydatetime()
      if time_coordinate_last_value == end_datetime:
        end_datetime = time_coordinate[-2].to_pydatetime()
      ds = self._obj.sel(time=slice(start_datetime, end_datetime))
    else:
      ds = self._obj
    return ds
