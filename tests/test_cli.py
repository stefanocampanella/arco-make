# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""End-to-end runs of the commands on the sample recipes in `tests/data` (offline)."""

import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli
from tests.conftest import open_output, output_group, render_recipe

THREADS = ["--dask-setup", "threads"]


def run(*args: str) -> Result:
  result = CliRunner().invoke(cli, list(args), catch_exceptions=False)
  return result


@pytest.fixture
def recipes(tmp_path: pathlib.Path, sources: pathlib.Path) -> dict[str, pathlib.Path]:
  directory = tmp_path / "recipes"
  directory.mkdir()
  names = ["download", "validate", "process", "climatology"]
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
  ds = open_output(downloaded)
  assert ds.sizes == {"time": 5, "depth": 3, "lat": 4, "lon": 4}
  assert set(ds.data_vars) == {"t2m", "thetao", "thetao_bottom", "zos", "deptho", "sea_mask"}
  assert ds.attrs["title"] == "Synthetic sample dataset"
  assert (
    str(ds.time.values[0])[:10] == "2020-02-26" and str(ds.time.values[-1])[:10] == "2020-03-01"
  )
  # Inside the sea mask there are no gaps left; outside it, the 3D field is missing.
  assert ds.thetao.where(ds.sea_mask).notnull().sum() == ds.sea_mask.sum() * ds.sizes["time"]
  group = output_group(downloaded)
  assert group["thetao"].chunks == (1, 1, 4, 4)
  assert group["thetao"].dtype == np.float32
  assert group["time"].dtype == np.int32


def test_download_refuses_to_replace_without_the_flag(downloaded: pathlib.Path, recipes):
  """An existing output is never replaced without `--replace` (intent 6)."""
  result = CliRunner().invoke(
    cli, ["download", *THREADS, str(recipes["download"]), str(downloaded)]
  )
  assert result.exit_code != 0
  assert "already exists" in result.output
  result = run("download", *THREADS, "--replace", str(recipes["download"]), str(downloaded))
  assert result.exit_code == 0, result.output


def test_blocks_equal_one_block(tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes):
  """Downloading in blocks of `step`, appended to the output, gives the same dataset as one block
  (intent 7); static variables are written once."""
  recipe = recipes["download"]
  whole = recipe.with_name("one_block.toml")
  text = recipe.read_text()
  assert 'step = "2D"' in text
  whole.write_text(text.replace('step = "2D"', 'step = "5D"'))
  output = tmp_path / "one_block.zarr"
  result = run("download", *THREADS, str(whole), str(output))
  assert result.exit_code == 0, result.output
  xr.testing.assert_identical(open_output(output).load(), open_output(downloaded).load())
  assert "time" not in open_output(downloaded).deptho.dims


def test_validate_sample_dataset(downloaded: pathlib.Path, recipes):
  """The sample dataset passes its checks, with `--fail`."""
  result = run("validate", "--fail", str(recipes["validate"]), str(downloaded))
  assert result.exit_code == 0, result.output


def test_validate_takes_the_dates_of_the_download(
  tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes
):
  """`--start` and `--end` set the time check's bounds, so that a script passes the same dates to
  `download` and `validate` (the arco-ocean recipe had an end typed years too early, C11); a
  recipe without the check gets it from them."""
  args = ("validate", "--fail", str(recipes["validate"]), str(downloaded))
  late = run(*args[:2], "--end", "2020-03-03", *args[2:])
  assert late.exit_code != 0 and "Validation failed" in late.output
  without = tmp_path / "no_time_check.toml"
  text = recipes["validate"].read_text()
  without.write_text(
    text.split("[checks.valid_time_coordinate]")[0] + text.split('freq = "1D"\n')[1]
  )
  assert "valid_time_coordinate" not in without.read_text()
  dates = ("--start", "2020-02-26", "--end", "2020-03-02")
  assert run("validate", "--fail", *dates, str(without), str(downloaded)).exit_code == 0
  only_start = run("validate", "--fail", "--start", "2020-02-26", str(without), str(downloaded))
  assert only_start.exit_code != 0 and "end" in only_start.output


def test_validate_without_dates_fails_before_reading(
  tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes
):
  """A time check without dates (the recipe's or the command's) fails at once, also without
  `--fail`: as a warning, the time axis went unchecked and validate exited 0."""
  text = recipes["validate"].read_text()
  undated = tmp_path / "undated.toml"
  undated.write_text(
    text.replace("start = 2020-02-26T00:00:00\n", "").replace("end = 2020-03-02T00:00:00\n", "")
  )
  assert "start =" not in undated.read_text()
  result = run("validate", str(undated), str(downloaded))
  assert result.exit_code != 0 and "give --start and --end" in result.output


def test_process_sample_dataset(tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes):
  """`process` applies the recipe steps (here a time mean of the dynamic variables)."""
  output = tmp_path / "mean.zarr"
  result = run("process", *THREADS, str(recipes["process"]), str(downloaded), str(output))
  assert result.exit_code == 0, result.output
  mean = open_output(output)
  expected = open_output(downloaded).t2m.mean("time")
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
  t2m = open_output(downloaded).t2m.isel(time=slice(0, 4)).values
  np.testing.assert_allclose(open_output(clim).t2m.values[0], (t2m[0] + t2m[2]) / 2, rtol=1e-6)
  np.testing.assert_allclose(
    open_output(std).t2m.values[1], np.abs(t2m[1] - t2m[3]) / np.sqrt(2), rtol=1e-5
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
  out = open_output(clim).x.values
  in_2020 = values[time.year == 2020]
  np.testing.assert_array_equal(out[:58], in_2020[:58])  # Jan 1 to Feb 27 of 2020 only
  np.testing.assert_allclose(out[58], in_2020[58:60].mean(), rtol=1e-6)  # Feb 28 and 29
  np.testing.assert_array_equal(out[59:], in_2020[60:])
  assert np.isnan(open_output(std).x.values[0])  # one value: no variance


@pytest.mark.parametrize(
  ("save", "chunks", "shards"),
  [
    ("chunks = {{time = 1, depth = 1, lat = -1}}\n", (1, 1, 4), None),
    ("chunks = {{time = 1, depth = 1, lat = 2}}\nshards = {{time = 73}}\n", (1, 1, 2), (73, 1, 4)),
  ],
  ids=["chunks", "shards"],
)
def test_climatology_one_series_at_a_time(tmp_path: pathlib.Path, save: str, chunks, shards):
  """`--streaming` writes the same outputs as one Dask graph, series by series (`split_dims`);
  shards along the split dimensions are 1, like chunks."""
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
    "[save]\n" + save
  )
  outputs = {}
  for label, split, path in [("graph", "", "--graph"), ("series", 'split_dims = ["depth"]', "")]:
    recipe = tmp_path / f"{label}.toml"
    recipe.write_text(common.format(split=split))
    clim, std = tmp_path / f"{label}_clim.zarr", tmp_path / f"{label}_std.zarr"
    flags = [path or "--streaming", *THREADS]
    result = run(
      "compute-climatology", *flags, str(recipe), str(tmp_path / "in.zarr"), str(clim), str(std)
    )
    assert result.exit_code == 0, result.output
    outputs[label] = (open_output(clim).load(), open_output(std).load())
  for graph, series in zip(outputs["graph"], outputs["series"], strict=True):
    xr.testing.assert_allclose(series, graph, rtol=1e-5)
    assert series.attrs == graph.attrs
  assert outputs["series"][0].attrs["title"] == "clim"
  stored = open_output(tmp_path / "series_clim.zarr").t3.encoding
  assert stored["chunks"] == chunks and stored["shards"] == shards


def climatology_input(tmp_path: pathlib.Path, depth_coordinate: bool = True) -> pathlib.Path:
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  values = np.random.default_rng(2).standard_normal((time.size, 3, 4)).astype("float32")
  coords = {"time": time, "lat": [0.0, 1.0, 2.0, 3.0]}
  if depth_coordinate:
    coords["depth"] = [0.5, 1.5, 2.5]
  ds = xr.Dataset({"t3": (("time", "depth", "lat"), values)}, coords=coords)
  path = tmp_path / ("in.zarr" if depth_coordinate else "in_nodepth.zarr")
  ds.to_zarr(path)
  return path


def run_climatology(
  tmp_path: pathlib.Path, source: pathlib.Path, label: str, extra: str, streaming: bool = True
):
  recipe = tmp_path / f"{label}.toml"
  recipe.write_text(
    f'period = 365\nbins = "day_of_year"\nwindow = 3\nweighting = "constant"\n{extra}\n'
    '[read]\nengine = "zarr"\n'
  )
  clim, std = tmp_path / f"{label}_clim.zarr", tmp_path / f"{label}_std.zarr"
  path = "--streaming" if streaming else "--graph"
  return run(
    "compute-climatology", path, *THREADS, str(recipe), str(source), str(clim), str(std)
  ), clim


def test_the_command_line_chooses_the_path(monkeypatch, caplog):
  """`--streaming` computes one series at a time, else one graph; `split_dims` only says how to
  split, and is warned about without `--streaming`."""
  import logging

  from arcomake import climatology

  calls = []
  monkeypatch.setattr(climatology, "_compute_graph", lambda *a, **k: calls.append("graph"))
  monkeypatch.setattr(climatology, "_compute_streaming", lambda *a, **k: calls.append("series"))
  configs = climatology.ClimatologyConfig.model_validate(
    {"period": 365, "window": 1, "weighting": "constant", "split_dims": ["depth"], "read": {}}
    | {"save": {}}
  )
  with caplog.at_level(logging.WARNING):
    climatology._compute_and_write(xr.Dataset(), configs, ("c", "s"), label="x", streaming=True)
    assert "ignored" not in caplog.text
    climatology._compute_and_write(xr.Dataset(), configs, ("c", "s"), label="x", streaming=False)
  assert calls == ["series", "graph"] and "split_dims ['depth'] ignored" in caplog.text


def test_streaming_without_split_dims_computes_each_variable(tmp_path: pathlib.Path):
  """Without `split_dims`, `--streaming` computes one series per variable, as the graph."""
  source = climatology_input(tmp_path)
  graph, graph_clim = run_climatology(tmp_path, source, "graph", "\n[save]\n", streaming=False)
  series, series_clim = run_climatology(tmp_path, source, "series", "\n[save]\n")
  assert graph.exit_code == 0 and series.exit_code == 0, series.output
  xr.testing.assert_allclose(
    open_output(series_clim).load(), open_output(graph_clim).load(), atol=1e-6
  )


def test_one_series_at_a_time_without_coordinates_or_save_chunks(tmp_path: pathlib.Path):
  """A split dimension without coordinate is written index by index (region "auto" mapped every
  series to index 0); without [save] chunks, the split dimension is stored one index per chunk."""
  source = climatology_input(tmp_path, depth_coordinate=False)
  graph, graph_clim = run_climatology(tmp_path, source, "graph", "\n[save]\n", streaming=False)
  series, series_clim = run_climatology(
    tmp_path, source, "series", 'split_dims = ["depth"]\n[save]\n'
  )
  assert graph.exit_code == 0 and series.exit_code == 0, series.output
  # Float32 rounding differs between the paths (one unit in the last place near zero).
  xr.testing.assert_allclose(
    open_output(series_clim).load(), open_output(graph_clim).load(), atol=1e-6
  )
  assert open_output(series_clim).t3.encoding["chunks"][1] == 1


@pytest.mark.parametrize(
  ("extra", "message"),
  [
    ('split_dims = ["depth"]\n[save]\nchunks = {depth = 3}\n', "must be 1"),
    ('split_dims = ["depth"]\n[save]\nchunks = {depth = 1}\nshards = {depth = 3}\n', "must be 1"),
    ('split_dims = ["depth"]\n[save]\ngroup = "g"\n', "group is not supported"),
    ('split_dims = ["detph"]\n[save]\n', "not dimensions of the input"),
    ('split_dims = ["time"]\n[save]\n', "cannot include the time dimension"),
  ],
)
def test_one_series_at_a_time_checks_before_computing(tmp_path, extra: str, message: str):
  """Settings that cannot work fail before the first series is computed."""
  result, _ = run_climatology(tmp_path, climatology_input(tmp_path), "bad", extra)
  assert result.exit_code != 0 and message in result.output


def test_one_series_at_a_time_with_storage_options_on_a_local_output(tmp_path: pathlib.Path):
  """`[save] storage_options` (e.g. a recipe shared with remote runs) do not break a local output:
  zarr 3 refuses them for local stores, so they are dropped there."""
  source = climatology_input(tmp_path)
  extra = 'split_dims = ["depth"]\n[save]\nstorage_options = {anon = true}\n'
  result, clim = run_climatology(tmp_path, source, "series", extra)
  assert result.exit_code == 0, result.output
  assert open_output(clim).t3.notnull().all()


def test_zip_outputs_are_refused_before_any_work(recipes, tmp_path: pathlib.Path):
  """Zip outputs were removed with Zarr v3: a `.zip` output path fails at once, before any
  download, instead of becoming a directory with that name."""
  output = tmp_path / "piece.zip"
  result = CliRunner().invoke(cli, ["download", *THREADS, str(recipes["download"]), str(output)])
  assert result.exit_code != 0 and "zip outputs are not supported" in result.output
  assert not output.exists()


def test_climatology_series_need_no_temporary_store(tmp_path: pathlib.Path, monkeypatch):
  """Only the graph path has a temporary store: computing one series at a time makes no
  temporary directory."""
  from arcomake import climatology

  def refused(*args, **kwargs):
    raise AssertionError("a temporary directory registry")

  monkeypatch.setattr(climatology, "TempDirectoryRegistry", refused)
  time = pd.date_range("2019-01-01", "2020-01-01", freq="1D", inclusive="left")
  values = np.zeros((time.size, 2, 3), dtype="float32")
  source = tmp_path / "in.zarr"
  xr.Dataset({"t": (("time", "lat", "lon"), values)}, coords={"time": time}).to_zarr(source)
  recipe = tmp_path / "climatology.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 3\nweighting = "constant"\n'
    'split_dims = ["lat"]\n[read]\nengine = "zarr"\n[save]\nchunks = {time = 1}\n'
  )
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  args = [str(recipe), str(source), str(clim), str(std)]
  result = run("compute-climatology", "--streaming", *THREADS, *args)
  assert result.exit_code == 0, result.output


def test_climatology_writes_shards_its_chunks_straddle(tmp_path: pathlib.Path, monkeypatch):
  """Read one level per chunk, stored in shards of all levels (the layout of the historical
  datasets, used by the statistics too): the writes are aligned to the stored layout, as in
  download; xarray refused Dask chunks overlapping a shard.

  The outputs are computed into a temporary store, then copied: the window runs once per chunk
  (it ran once per output), and the store is removed at the end."""
  import tempfile

  from arcomake import climatology
  from arcomake.temporary_utils import TempDirectoryRegistry

  windows = []
  window = climatology._window_bins_last

  def counted(*args, **kwargs):
    windows.append(1)
    return window(*args, **kwargs)

  monkeypatch.setattr(climatology, "_window_bins_last", counted)
  scratch = tmp_path / "scratch"
  scratch.mkdir()
  monkeypatch.setenv("TMPDIR", str(scratch))
  monkeypatch.setattr(tempfile, "tempdir", str(scratch))
  stores = []
  create = TempDirectoryRegistry.create_temp_dir

  def recorded(self, *args, **kwargs):
    stores.append(create(self, *args, **kwargs))
    return stores[-1]

  monkeypatch.setattr(TempDirectoryRegistry, "create_temp_dir", recorded)
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  values = np.random.default_rng(0).standard_normal((time.size, 3, 4, 4)).astype("float32")
  coords = {"time": time, "depth": [0.5, 1.5, 2.5], "lat": np.arange(4.0), "lon": np.arange(4.0)}
  source = tmp_path / "in.zarr"
  xr.Dataset({"t3": (("time", "depth", "lat", "lon"), values)}, coords=coords).to_zarr(source)
  recipe = tmp_path / "climatology.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 5\nweighting = "gaussian"\nweighting_scale = 2\n'
    '[read]\nengine = "zarr"\nchunks = {time = "auto", depth = 1, lat = -1, lon = -1}\n'
    "[save]\nchunks = {time = 1, depth = 1, lat = -1, lon = -1}\n"
    'shards = {time = 8, depth = -1, lat = -1, lon = -1}\nshard_order = "lexicographic"\n'
  )
  clim, std = tmp_path / "clim.zarr", tmp_path / "std.zarr"
  result = run("compute-climatology", *THREADS, str(recipe), str(source), str(clim), str(std))
  assert result.exit_code == 0, result.output
  stored = output_group(clim)["t3"]
  assert stored.shards == (8, 3, 4, 4) and stored.chunks == (1, 1, 4, 4)
  assert open_output(clim).t3.shape == (365, 3, 4, 4)
  assert len(windows) == 3  # one per level: once, not once per output
  ds = xr.Dataset({"t3": (("time", "depth", "lat", "lon"), values)}, coords=coords)
  mean, var = climatology._compute_climatology(
    ds, 365, 5, "gaussian", 2.0, "time", False, "day_of_year"
  )
  xr.testing.assert_allclose(open_output(clim).t3.load(), mean.t3, rtol=1e-6)
  xr.testing.assert_allclose(open_output(std).t3.load(), np.sqrt(var.t3.clip(min=0)), rtol=1e-6)
  (store,) = stores
  assert store.is_relative_to(scratch) and not store.exists()  # removed at the end
