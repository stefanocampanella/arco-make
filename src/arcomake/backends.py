import datetime
import logging
import os
import pathlib
import tempfile
import warnings
from collections.abc import Callable, Iterable
from typing import Any, override
from urllib.parse import urlparse
from zipfile import ZipFile

import cdsapi
import copernicusmarine as cm
import requests
import xarray as xr
from xarray.backends import AbstractDataStore, BackendEntrypoint
from xarray.core.types import ReadBuffer

from arcomake.datetime_utils import DateInterval

logger = logging.getLogger(__name__)


class CopernicusMarine(BackendEntrypoint):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Avoid annoying copernicusmarine log handling
    cm_logger = logging.getLogger("copernicus_marine_root_logger")
    for handler in cm_logger.handlers:
      cm_logger.removeHandler(handler)
    cm_logger.setLevel(level=logging.getLevelName(logger.getEffectiveLevel()))

  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    variables: str | Iterable[str] | None = None,
    start_datetime: str | datetime.datetime | None = None,
    end_datetime: str | datetime.datetime | None = None,
    dataset_version: str | None = None,
    dataset_part: str | None = None,
    service: str | None = None,
  ) -> xr.Dataset:

    url = urlparse(filename_or_obj)
    if url.path or url.query or url.fragment:
      warnings.warn(
        f"Possibly invalid Copernicus Marine URL: {filename_or_obj}. Ignoring path, query and fragment"
      )
    dataset_id = url.netloc
    if dataset_id == "":
      raise ValueError("Missing dataset ID.")

    if variables is None:
      variables = []
    elif isinstance(variables, str):
      variables = [variables]
    else:
      variables = list(variables)

    # As copenicus marine chunks the dataset here we need to load it upfront and return it using numpy arrays as backend
    with cm.open_dataset(
      dataset_id=dataset_id,
      dataset_version=dataset_version,
      variables=variables,
      dataset_part=dataset_part,
      service=service,
      start_datetime=start_datetime,
      end_datetime=end_datetime,
      chunk_size_limit=False,
    ) as dataset:
      return dataset

  @override
  def guess_can_open(
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | ReadBuffer[Any]
    | bytes
    | memoryview
    | AbstractDataStore,
  ) -> bool:

    if not isinstance(filename_or_obj, str):
      return False
    else:
      url = urlparse(filename_or_obj)
      return url.scheme == "cm"


class NetCDFOverHTTP(BackendEntrypoint):
  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables=None,
  ) -> xr.Dataset:
    logger.info(f"Downloading NetCDF from: {filename_or_obj}")

    # Create a temporary file to download the NetCDF
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temp_file:
      temp_path = temp_file.name

    try:
      # Use requests to download the file
      response = requests.get(filename_or_obj, stream=True)
      response.raise_for_status()  # Raise an exception for HTTP errors

      with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

      # Open the downloaded file as xarray Dataset
      with xr.open_dataset(temp_path, engine="netcdf4", drop_variables=drop_variables) as dataset:
        return dataset.compute()

    finally:
      # Clean up the temporary file
      if pathlib.Path(temp_path).exists():
        pathlib.Path(temp_path).unlink()

  @override
  def guess_can_open(
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | ReadBuffer[Any]
    | bytes
    | memoryview
    | AbstractDataStore,
  ) -> bool:

    if not isinstance(filename_or_obj, str):
      return False
    else:
      url = urlparse(filename_or_obj)
      return url.scheme == "http" or url.scheme == "https"


class EarlyWarningDataStore(BackendEntrypoint):
  """
  Support for downloading from the Early Warning Data Store (EWDS) using cdsapi.
  Minimal implementation: it monkey-patches the cdsapi logger to avoid unformatted output,
  it downloads the whole set of dates at once, loads and returns the dataset.
  For a more robust and feature-rich implementation, see: https://github.com/bopen/xarray-ecmwf/tree/main
  When available, ARCO alternatives should be preferred.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Avoid annoying copernicusmarine log handling
    self.progress = False
    self.client_logger = logger

  @override
  def open_dataset(
    self,
    filename_or_obj,
    *,
    drop_variables: str | Iterable[str] | None = None,
    start_datetime: datetime.datetime | None = None,
    end_datetime: datetime.datetime | None = None,
    system_version: str | None = None,
    hydrological_model: str | None = None,
    product_type: str | None = None,
    variable: str | Iterable[str] | None = None,
    time_dim: str = "time",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
  ) -> xr.Dataset:
    url = urlparse(filename_or_obj)
    if url.path or url.query or url.fragment:
      warnings.warn(
        f"Possibly invalid EWDS URL: {filename_or_obj}. Ignoring path, query and fragment"
      )
    dataset_name = url.netloc
    if any(arg is None for arg in [system_version, hydrological_model, product_type, variable]):
      raise ValueError("Missing required argument.")
    if start_datetime is None or end_datetime is None:
      raise ValueError("Missing required argument.")
    logger.info(f"Downloading {dataset_name} from EWDS")
    datasets = []
    consecutive_dates = self._consecutive_dates_with_same_month_or_year(
      start_datetime, end_datetime
    )
    for dates in consecutive_dates:
      prefix = "h" if dataset_name.endswith("historical") else ""
      request = {
        "system_version": [system_version],
        "hydrological_model": [hydrological_model],
        "product_type": [product_type],
        "variable": [variable] if isinstance(variable, str) else variable,
        prefix + "year": f"{dates[0].year}",
        prefix + "month": [f"{dates[0].month:02}"],
        prefix + "day": [f"{date.day:02}" for date in dates],
        "data_format": "grib2",
        "download_format": "zip",
      }
      dataset = self._process_request(dataset_name, request)
      # TODO:
      #  Unfortunately, it was impossible to get documentation on `valid_time` for historical GLOFAS data, hence
      #  the provider documentation should clarify the difference between the two, e.g., if renaming the time
      #  coordinate is needed (after appropriately resizing the download time interval)
      #    ds = ds.rename(valid_time='time')
      if extra_coords := [
        name for name in dataset.coords if name not in [latitude_dim, longitude_dim, time_dim]
      ]:
        dataset = dataset.drop_vars(extra_coords)
      # _process_request drops the time dimension if of length one
      if len(dates) == 1:
        dataset = dataset.expand_dims(dim=time_dim, axis=0)
        dataset = dataset.assign_coords(time=dates)
      datasets.append(dataset)
    with xr.concat(datasets, dim=time_dim) as dataset:
      # The dataset must be loaded in memory, since the temporary directory will be deleted with all the NetCDFs within it.
      # However, ds should be rather small. Hence, there should be no need to lazily load the dataset.
      return dataset.compute()

  @staticmethod
  def _consecutive_dates_with_same_month_or_year(
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
  ) -> list[list[datetime.datetime]]:
    date_interval = DateInterval(start=start_datetime, end=end_datetime)
    days = date_interval.to_list(freq=datetime.timedelta(days=1))

    def partition[T](f: Callable[[T, T], bool], sequence: Iterable[T]):
      part = []
      subseq = []
      last = None
      for current in sequence:
        if (last is None) or (not subseq) or f(last, current):
          subseq.append(current)
        else:
          part.append(subseq)
          subseq = [current]
        last = current
      part.append(subseq)
      return part

    def same_month_or_year(x: datetime.datetime, y: datetime.datetime) -> bool:
      return x.month == y.month and x.year == y.year

    return partition(same_month_or_year, days)

  def _process_request(self, dataset_name, request, **kwargs) -> xr.Dataset:
    """Submit a request to the Climate Data Store, download some temporary NetCDFs, and returns a dataset.
    Temporary files are deleted on exit.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".zip") as file:
      client = self.get_cdsapi_client(
        progress=self.progress, client_logger=self.client_logger, **kwargs
      )
      client.retrieve(dataset_name, request, file.name)

      with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir)
        with ZipFile(file.name) as zipfile:
          zipfile.extractall(path=path)
        # FIXME: when reading grib files eccodes emits the following warning:
        #    ECCODES WARNING :  g2date:unpack_long: Date is not valid! year=0 month=0 day=0
        #  see: https://git.ecmwf.int/users/erds/repos/eccodes/browse/src/accessor/grib_accessor_class_g2date.cc?at=f10d3f3c1be3737a231d4a2c19f45535de4d4424#71-75
        #  Also, cdsapi download one NetCDF per variable, ugly. :(
        with xr.open_mfdataset(list(path.glob("*.grib")), engine="cfgrib") as dataset:
          return dataset.compute()

  # Credits to the amazing Stefano Piani from OGS
  @staticmethod
  def get_cdsapi_client(url: str | None = None, client_logger=None, **kwargs):
    """Returns a cdsapi.Client instance.

    It also configures the returned client to use a specific logger (if
    submitted)

    Args:
      url (str): the url of the endpoint of the cdsapi. If it is None, it will be
        read from the ~/.cdsapi file (if exists)
      key (str): the key of the cdsapi user account. If it is None, it will be
        read from the ~/.cdsapi file (if exists)
      client_logger (logging.Logger): Logger that the returned client
        will use to print its messages

    Returns:
      cdsapi.Client instance
    """
    if client_logger is not None:

      def debug_callback(*args, **kwargs):
        return client_logger.debug(*args, **kwargs)

      def info_callback(*args, **kwargs):
        return client_logger.info(*args, **kwargs)

      def warning_callback(*args, **kwargs):
        return client_logger.warning(*args, **kwargs)

      def error_callback(*args, **kwargs):
        return client_logger.error(*args, **kwargs)
    else:
      debug_callback = None
      info_callback = None
      warning_callback = None
      error_callback = None

    client_kwargs = {
      "debug_callback": debug_callback,
      "info_callback": info_callback,
      "warning_callback": warning_callback,
      "error_callback": error_callback,
      **kwargs,
    }

    if url is None:
      client_kwargs["url"] = url

    cdsapi_client = cdsapi.Client(**client_kwargs)

    # This is a horrible hack that probably will become not necessary in the
    # next version of cdsapi. It removes the "logging decorator", which is a
    # context manager that changes the configuration of the logger
    if cdsapi_client.__class__.__name__.startswith("Legacy"):  # noqa: SIM102
      if hasattr(cdsapi_client, "logging_decorator"):
        cdsapi_client.logging_decorator = lambda x: x  # ty: ignore

    return cdsapi_client

  @override
  def guess_can_open(
    self,
    filename_or_obj: str
    | os.PathLike[Any]
    | ReadBuffer[Any]
    | bytes
    | memoryview
    | AbstractDataStore,
  ) -> bool:

    if not isinstance(filename_or_obj, str):
      return False
    else:
      url = urlparse(filename_or_obj)
      return url.scheme == "ewds"
