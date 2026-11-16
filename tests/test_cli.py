# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""End-to-end runs of the commands on the sample recipes in `tests/data` (offline)."""

import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from click.testing import CliRunner, Result

from arcomake import cli
from tests.conftest import render_recipe

THREADS = ["--dask-setup", "threads"]


def run(*args: str) -> Result:
  result = CliRunner().invoke(cli, list(args), catch_exceptions=False)
  return result


@pytest.fixture
def recipes(tmp_path: pathlib.Path, sources: pathlib.Path) -> dict[str, pathlib.Path]:
  directory = tmp_path / "recipes"
  directory.mkdir()
  names = ["download", "unpack", "validate", "process", "climatology"]
  return {
    name: render_recipe(f"sample_{name}.toml", directory, SOURCES=str(sources)) for name in names
  }


@pytest.fixture
def downloaded(tmp_path: pathlib.Path, recipes) -> pathlib.Path:
  output = tmp_path / "dataset.zarr"
  result = run("download", *THREADS, str(recipes["download"]), str(output))
  assert result.exit_code == 0, result.output
  return output


def test_download_sample_recipe(downloaded: pathlib.Path):
  """The sample recipe gives a merged, regridded, filled dataset with the requested layout."""
  ds = xr.open_zarr(downloaded)
  assert ds.sizes == {"time": 5, "depth": 3, "lat": 4, "lon": 4}
  assert set(ds.data_vars) == {"t2m", "thetao", "thetao_bottom", "zos", "deptho", "sea_mask"}
  assert ds.attrs["title"] == "Synthetic sample dataset"
  assert (
    str(ds.time.values[0])[:10] == "2020-02-26" and str(ds.time.values[-1])[:10] == "2020-03-01"
  )
  # Inside the sea mask there are no gaps left; outside it, the 3D field is missing.
  assert ds.thetao.where(ds.sea_mask).notnull().sum() == ds.sea_mask.sum() * ds.sizes["time"]
  group = zarr.open_group(str(downloaded), mode="r")
  assert group["thetao"].chunks == (1, 1, 4, 4)
  assert group["thetao"].dtype == np.float32
  assert group["time"].dtype == np.int32


def test_download_refuses_to_overwrite(downloaded: pathlib.Path, recipes):
  """An existing output is never replaced without `--overwrite` (intent 6)."""
  result = CliRunner().invoke(
    cli, ["download", *THREADS, str(recipes["download"]), str(downloaded)]
  )
  assert result.exit_code != 0
  assert "already exists" in result.output
  result = run("download", *THREADS, "--overwrite", str(recipes["download"]), str(downloaded))
  assert result.exit_code == 0, result.output


def test_pieces_and_unpack_equal_one_download(
  tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes
):
  """Downloading in pieces and joining them with `unpack` gives the same dataset as one run (intent 7)."""
  archive = tmp_path / "archive"
  archive.mkdir()
  for start, end in [("2020-02-26", "2020-02-28"), ("2020-02-28", "2020-03-02")]:
    piece = archive / f"{start}_{end}.zarr"
    result = run(
      "download", *THREADS, f"--start={start}", f"--end={end}", str(recipes["download"]), str(piece)
    )
    assert result.exit_code == 0, result.output
  joined = tmp_path / "joined.zarr"
  result = run("unpack", *THREADS, str(recipes["unpack"]), str(archive), str(joined))
  assert result.exit_code == 0, result.output
  xr.testing.assert_identical(xr.open_zarr(joined).load(), xr.open_zarr(downloaded).load())


def test_validate_sample_dataset(downloaded: pathlib.Path, recipes):
  """The sample dataset passes its checks, with `--fail`."""
  result = run("validate", "--fail", str(recipes["validate"]), str(downloaded))
  assert result.exit_code == 0, result.output


def test_process_sample_dataset(tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes):
  """`process` applies the recipe steps (here a time mean of the dynamic variables)."""
  output = tmp_path / "mean.zarr"
  result = run("process", *THREADS, str(recipes["process"]), str(downloaded), str(output))
  assert result.exit_code == 0, result.output
  mean = xr.open_zarr(output)
  expected = xr.open_zarr(downloaded).t2m.mean("time")
  xr.testing.assert_allclose(mean.t2m.load(), expected.load())
  assert "sea_mask" not in mean


def test_climatology_sample_dataset(tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes):
  """`compute-climatology` writes the per-bin mean and the anomaly standard deviation."""
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  result = run(
    "compute-climatology",
    *THREADS,
    str(recipes["climatology"]),
    str(downloaded),
    str(clim),
    str(std),
  )
  assert result.exit_code == 0, result.output
  t2m = xr.open_zarr(downloaded).t2m.isel(time=slice(0, 4)).values
  np.testing.assert_allclose(xr.open_zarr(clim).t2m.values[0], (t2m[0] + t2m[2]) / 2, rtol=1e-6)
  np.testing.assert_allclose(
    xr.open_zarr(std).t2m.values[1], np.abs(t2m[1] - t2m[3]) / np.sqrt(2), rtol=1e-5
  )


def test_climatology_reference_period_and_day_of_year(tmp_path: pathlib.Path):
  """`reference_period` limits the statistics to `[start, end)`; day-of-year bins need no
  calendar conversion (Feb 29 is in Feb 28's bin)."""
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  values = np.random.default_rng(0).standard_normal(time.size).astype("float32")
  xr.Dataset({"x": ("time", values)}, coords={"time": time}).to_zarr(tmp_path / "in.zarr")
  recipe = tmp_path / "climatology.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 1\nweighting = "constant"\n'
    "reference_period = {start = 2020-01-01, end = 2021-01-01}\n\n"
    '[read]\nengine = "zarr"\n\n[save]\n'
  )
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  args = [str(recipe), str(tmp_path / "in.zarr"), str(clim), str(std)]
  result = run("compute-climatology", *THREADS, *args)
  assert result.exit_code == 0, result.output
  out = xr.open_zarr(clim).x.values
  in_2020 = values[time.year == 2020]
  np.testing.assert_array_equal(out[:58], in_2020[:58])  # Jan 1 to Feb 27 of 2020 only
  np.testing.assert_allclose(out[58], in_2020[58:60].mean(), rtol=1e-6)  # Feb 28 and 29
  np.testing.assert_array_equal(out[59:], in_2020[60:])
  assert np.isnan(xr.open_zarr(std).x.values[0])  # one value: no variance


def test_climatology_one_series_at_a_time(tmp_path: pathlib.Path):
  """`split_dims` (S1) writes the same outputs as one Dask graph, series by series."""
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  rng = np.random.default_rng(0)
  ds = xr.Dataset(
    {
      "t3": (("time", "depth", "lat"), rng.standard_normal((time.size, 3, 4)).astype("float32")),
      "t2": (("time", "lat"), rng.standard_normal((time.size, 4)).astype("float32")),
    },
    coords={"time": time, "depth": [0.5, 1.5, 2.5], "lat": [0.0, 1.0, 2.0, 3.0]},
  )
  ds.to_zarr(tmp_path / "in.zarr")
  common = (
    'period = 365\nbins = "day_of_year"\nwindow = 5\nweighting = "gaussian"\n'
    'weighting_scale = 2.0\n{split}\n[read]\nengine = "zarr"\n\n'
    '[[postprocess_climatology]]\nname = "assign_attrs"\ntitle = "clim"\n\n'
    "[save]\nchunks = {{time = 1, depth = 1, lat = -1}}\n"
  )
  outputs = {}
  for label, split in [("graph", ""), ("series", 'split_dims = ["depth"]')]:
    recipe = tmp_path / f"{label}.toml"
    recipe.write_text(common.format(split=split))
    clim, std = tmp_path / f"{label}_clim.zarr", tmp_path / f"{label}_std.zarr"
    result = run(
      "compute-climatology", *THREADS, str(recipe), str(tmp_path / "in.zarr"), str(clim), str(std)
    )
    assert result.exit_code == 0, result.output
    outputs[label] = (xr.open_zarr(clim).load(), xr.open_zarr(std).load())
  for graph, series in zip(outputs["graph"], outputs["series"], strict=True):
    xr.testing.assert_allclose(series, graph, rtol=1e-5)
    assert series.attrs == graph.attrs
  assert outputs["series"][0].attrs["title"] == "clim"
  assert xr.open_zarr(tmp_path / "series_clim.zarr").t3.encoding["chunks"] == (1, 1, 4)
