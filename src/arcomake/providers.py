# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import copy
import datetime
import logging
import pathlib
import tempfile
import warnings
from abc import ABC, abstractmethod
from typing import Any, override
from zipfile import ZipFile

import cdsapi
import copernicusmarine as cm
import numpy as np
import requests
import xarray as xr

from arcomake.datetime_utils import DateInterval

logger = logging.getLogger(__name__)


class Provider(ABC):
  def __init__(self, progress=True, log_level="info", client_logger=None):
    """
    Initializes the Provider object.

    Args:
      progress (bool): Whether to show progress bars.
      log_level (str): Logging level.
      client_logger (logging.Logger): Logger for the client.
    """
    self.progress = progress
    self.log_level = log_level
    self.client_logger = client_logger

  @abstractmethod
  def open_dataset(
    self, backend_kwargs: dict[str, Any], date_interval: DateInterval | None, tmpdir: pathlib.Path | None = None,
  ) -> xr.Dataset:
    """Provides a common interface, whether one is downloading from Copernicus Marine, Climate Data Store, etc.
    Depending on the particular implementation, it might download temporary files to `dir`

    Args:
      backend_kwargs: Additional keyword arguments passed to library code (e.g. copernicusmarine)
      date_interval (DateInterval, optional): Date interval to download. Defaults to None (whole dataset).
      tmpdir (pathlib.Path, optional): Directory to download to. Defaults to None (download to tempfile default path).
    """
    pass


class CopernicusMarine(Provider):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Avoid annoying copernicusmarine log handling
    cm_logger = logging.getLogger("copernicus_marine_root_logger")
    for handler in cm_logger.handlers:
      cm_logger.removeHandler(handler)
    cm_logger.setLevel(level=getattr(logging, self.log_level.upper()))

  @override
  def open_dataset(self, backend_kwargs, date_interval=None, tmpdir=None):
    logger.info(f"Opening {backend_kwargs['dataset_id']} from Copernicus Marine")
    if date_interval is not None:
      # Note! Copernicus Marine Data Store uses Python convention (right open)
      backend_kwargs.update({
        "start_datetime": date_interval.start,
        "end_datetime": date_interval.end,
      })
    ds = cm.open_dataset(**backend_kwargs)
    return ds


class ClimateDataStore(Provider):
  """Support for cdsapi. ARCO-ERA5 (WeatherBench datasets) should be preferred."""

  @override
  def open_dataset(self, backend_kwargs, date_interval=None, tmpdir=None):
    logger.info(f"Downloading {backend_kwargs['dataset']} from CDS/EWDS")
    if date_interval is None:
      dataset_name, request = self._get_request(**backend_kwargs)
      ds = self._process_request(dataset_name, request, tmpdir, self.progress, self.client_logger)
    else:
      consecutive_dates = self._consecutive_dates_with_same_month_or_year(date_interval)
      datasets = []
      for dates in consecutive_dates:
        dataset_name, request = self._get_request(dates=dates, **backend_kwargs)
        ds = self._process_request(dataset_name, request, tmpdir, self.progress, self.client_logger)
        # _process_request drops the time dimension if of length one
        if len(dates) == 1:
          ds = ds.expand_dims(dim="time", axis=0)
        datasets.append(ds)
      ds = xr.merge(datasets)
      if date_interval is not None:  # noqa: SIM102
        if not np.array_equal(date_interval.to_numpy(freq=datetime.timedelta(days=1)), ds["time"]):
          warnings.warn(
            f"The requested date interval {date_interval!r} "
            f"is not matching the time coordinate {ds['time']!r}"
          )
    return ds

  @staticmethod
  def _consecutive_dates_with_same_month_or_year(date_interval: DateInterval):
    def partition(f, sequence):
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

    def same_month_or_year(x, y):
      return x.month == y.month and x.year == y.year

    return partition(same_month_or_year, date_interval.to_list(freq=datetime.timedelta(days=1)))

  def _get_request(self, dates=None, **kwargs):
    "Returns a request to the Climate Data Store using the provided dates."
    request = copy.deepcopy(kwargs)
    dataset_name = request.pop("dataset")
    if dates is not None:
      request = {
        **request,
        "hyear": f"{dates[0].year}",
        "hmonth": [f"{dates[0].month:02}"],
        "hday": [f"{date.day:02}" for date in dates],
        "data_format": "grib2",
        "download_format": "zip",
      }
    return dataset_name, request

  def _process_request(self, dataset_name, request, dir, progress, client_logger, **kwargs):
    """Submit a request to the Climate Data Store, download some temporary NetCDFs, and returns a dataset.
    Temporary files are deleted on exit.
    """
    file = tempfile.NamedTemporaryFile("w+", dir=dir, suffix=".zip", delete=False)  # noqa: SIM115
    file.close()

    client = self.get_cdsapi_client(progress=progress, client_logger=client_logger, **kwargs)
    logger.debug(f"Submitting request {request} with destination {file.name}")
    client.retrieve(dataset_name, request, file.name)

    with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
      path = pathlib.Path(tmpdir)
      with ZipFile(file.name) as zipfile:
        zipfile.extractall(path=path)
      # noinspection PyTypeChecker
      ds = xr.open_mfdataset(
        list(path.glob("*")), engine="cfgrib", decode_timedelta=True
      )  # cdsapi download one NetCDF per variable :(
      # ds = ds.rename(valid_time='time')
      if extra_coords := [
        name for name in ds.coords if name not in ["latitude", "longitude", "time"]
      ]:
        ds = ds.drop_vars(extra_coords)
      # The dataset must be loaded in memory, since the temporary directory will be deleted with all the NetCDFs within it.
      # However, ds should be rather small. Hence, there should be no need to lazily load the dataset.
      ds = ds.compute()
    return ds

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


class RemoteZarr(Provider):
  @override
  def open_dataset(self, backend_kwargs, date_interval=None, tmpdir=None):
    logger.info(f"Opening remote zarr at {backend_kwargs['url']}")
    # copy backend_kwargs to avoid modifying the original
    backend_kwargs = copy.deepcopy(backend_kwargs)
    url: str = backend_kwargs.pop("url")
    variables: list[str] | None = backend_kwargs.pop("variables", None)
    ds = xr.open_zarr(url, **backend_kwargs)
    if variables is not None:
      variables_not_found: list[str] = [name for name in variables if name not in ds.data_vars]
      if variables_not_found:
        logger.warning(f"{', '.join(variables_not_found)} variables not found")
      ds = ds.drop_vars(names=[name for name in ds.data_vars if name not in variables])
    # Filter by date interval if specified
    ds = ds.arcomake.sel(date_interval)
    return ds


class RemoteNetCDF(Provider):
  """Provider that downloads a NetCDF file from a URL.

  This provider downloads a NetCDF file from a specified URL and opens it as an xarray Dataset.
  It supports filtering by date interval if the dataset has a time dimension.
  """

  @override
  def open_dataset(self, backend_kwargs, date_interval=None, tmpdir=None):
    """Downloads a NetCDF file from a URL and opens it as an xarray Dataset.

    Args:
      date_interval (DateInterval, optional): Date interval to filter the dataset. Defaults to None.
      tmpdir (pathlib.Path, optional): Directory to download temporary files to. Defaults to None.
      **kwargs: Additional keyword arguments, must include 'url'.
        url (str): URL of the NetCDF file to download.
        variables (list, optional): List of variables to keep in the dataset.

    Returns:
      xr.Dataset: The downloaded dataset.

    Raises:
      ValueError: If 'url' is not provided in kwargs.
    """
    url: str = backend_kwargs.get("url")
    if url is None:
      raise ValueError("URL must be provided for URLProvider")
    logger.info(f"Downloading NetCDF from: {url}")

    # Create a temporary file to download the NetCDF
    with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".nc", delete=False) as temp_file:
      temp_path = temp_file.name

    try:
      # Use requests to download the file
      response = requests.get(url, stream=True)
      response.raise_for_status()  # Raise an exception for HTTP errors

      with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

      # Open the downloaded file as an xarray Dataset
      ds = xr.open_dataset(temp_path)

      # Filter by variables if specified
      variables: list[str] | None = backend_kwargs.get("variables")
      if variables is not None:
        if variables_not_found := [name for name in variables if name not in ds.data_vars]:
          logger.warning(f"{', '.join(variables_not_found)} variables not found")
        ds = ds.drop_vars(names=[name for name in ds.data_vars if name not in variables])

      # Filter by date interval if specified
      ds = ds.arcomake.sel(date_interval)
      return ds

    finally:
      # Clean up the temporary file
      if pathlib.Path(temp_path).exists():
        pathlib.Path(temp_path).unlink()


ProvidersRegistry = {
  "cds": ClimateDataStore,
  "cm": CopernicusMarine,
  "gcs": RemoteZarr,
  "aws": RemoteZarr,
  "url": RemoteNetCDF,
  "http": RemoteNetCDF,
}
