from datetime import datetime

import xarray as xr

from arcomake.datetime_utils import DateInterval


@xr.register_dataset_accessor("arcomake")
@xr.register_dataarray_accessor("arcomake")
class DateIntervalSelector:
  """XArray DataArray and Dataset sel method is right inclusive, this accessor provides right exclusive selection."""

  def __init__(self, xarray_obj: xr.DataArray | xr.Dataset, time_dim: str = "time"):
    self._obj = xarray_obj
    self._time_dim = time_dim

  def sel(self, date_interval: DateInterval):
    if date_interval is not None and self._time_dim in self._obj.dims:
      time_coordinate = self._obj[self._time_dim]
      time_coordinate = time_coordinate.sel(time=date_interval.to_slice())
      time_coordinate = time_coordinate.to_index()
      if not time_coordinate.is_monotonic_increasing:
        raise ValueError(f"Time coordinate is not sorted: {self._obj[self._time_dim]}")
      time_coordinate_last_value: datetime = time_coordinate[-1].to_pydatetime()
      if time_coordinate_last_value == date_interval.end:
        date_interval = DateInterval(date_interval.start, time_coordinate[-2].to_pydatetime())
      ds = self._obj.sel(time=date_interval.to_slice())
    else:
      ds = self._obj
    return ds
