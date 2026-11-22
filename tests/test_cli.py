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


def statistics(path: pathlib.Path) -> tuple[xr.Dataset, xr.Dataset]:
  """The groups of a `compute-climatology` output: the climatology and the anomaly std."""
  return open_output(path, "climatology"), open_output(path, "anomaly_std")


def test_climatology_sample_dataset(tmp_path: pathlib.Path, downloaded: pathlib.Path, recipes):
  """`compute-climatology` writes the per-bin mean and the anomaly standard deviation."""
  output = tmp_path / "statistics.zarr"
  result = run(
    "compute-climatology", *THREADS, str(recipes["climatology"]), str(downloaded), str(output)
  )
  assert result.exit_code == 0, result.output
  clim, std = statistics(output)
  t2m = open_output(downloaded).t2m.isel(time=slice(0, 4)).values
  np.testing.assert_allclose(clim.t2m.values[0], (t2m[0] + t2m[2]) / 2, rtol=1e-6)
  np.testing.assert_allclose(std.t2m.values[1], np.abs(t2m[1] - t2m[3]) / np.sqrt(2), rtol=1e-5)


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
  output = tmp_path / "statistics.zarr"
  result = run("compute-climatology", *THREADS, str(recipe), str(tmp_path / "in.zarr"), str(output))
  assert result.exit_code == 0, result.output
  clim, std = statistics(output)
  out = clim.x.values
  in_2020 = values[time.year == 2020]
  np.testing.assert_array_equal(out[:58], in_2020[:58])  # Jan 1 to Feb 27 of 2020 only
  np.testing.assert_allclose(out[58], in_2020[58:60].mean(), rtol=1e-6)  # Feb 28 and 29
  np.testing.assert_array_equal(out[59:], in_2020[60:])
  assert np.isnan(std.x.values[0])  # one value: no variance


@pytest.mark.parametrize(
  ("save", "chunks", "shards"),
  [
    ("chunks = {{time = 1, depth = 1, lat = -1}}\n", (1, 1, 4), None),
    ("chunks = {{time = 1, depth = 1, lat = 2}}\nshards = {{time = 73}}\n", (1, 1, 2), (73, 3, 4)),
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
    output = tmp_path / f"{label}.zarr"
    flags = [path or "--streaming", *THREADS]
    result = run("compute-climatology", *flags, str(recipe), str(tmp_path / "in.zarr"), str(output))
    assert result.exit_code == 0, result.output
    outputs[label] = tuple(group.load() for group in statistics(output))
  for graph, series in zip(outputs["graph"], outputs["series"], strict=True):
    xr.testing.assert_allclose(series, graph, rtol=1e-5)
    assert series.attrs == graph.attrs
  assert outputs["series"][0].attrs["title"] == "clim"
  stored = open_output(tmp_path / "series.zarr", "climatology").t3.encoding
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
  output = tmp_path / f"{label}.zarr"
  path = "--streaming" if streaming else "--graph"
  result = run("compute-climatology", path, *THREADS, str(recipe), str(source), str(output))
  return result, output


@pytest.mark.parametrize("split_dims", [["depth"], []])
def test_the_command_line_chooses_the_path(monkeypatch, caplog, split_dims):
  """`--streaming` computes one series at a time, else one graph; `split_dims` only says how to
  split, and is warned about without `--streaming` (`[]` too: it selected streaming before)."""
  import logging

  from arcomake import climatology

  calls = []
  monkeypatch.setattr(climatology.graph, "compute", lambda *a, **k: calls.append("graph"))
  monkeypatch.setattr(climatology.streamed, "compute", lambda *a, **k: calls.append("series"))
  configs = climatology.ClimatologyConfig.model_validate(
    {"period": 365, "window": 1, "weighting": "constant", "split_dims": split_dims, "read": {}}
    | {"save": {}}
  )
  climatology.command._compute_and_write(xr.Dataset(), configs, "out", label="x", streaming=True)
  climatology.command._compute_and_write(xr.Dataset(), configs, "out", label="x", streaming=False)
  assert calls == ["series", "graph"]
  with caplog.at_level(logging.WARNING):
    climatology.command._warn_ignored_split_dims(configs, streaming=True)
    assert "ignored" not in caplog.text
    climatology.command._warn_ignored_split_dims(configs, streaming=False)
  assert f"split_dims {split_dims} ignored" in caplog.text


@pytest.mark.parametrize("streaming", [False, True], ids=["graph", "streaming"])
def test_statistics_do_not_replace_a_version_promoted_meanwhile(
  tmp_path: pathlib.Path, monkeypatch, streaming: bool
):
  """The run is opened before computing: a version of `main` promoted while the statistics are
  computed is not replaced (streaming opened its run only at the copy, hours later)."""
  from arcomake import climatology
  from arcomake.dataset_utils import write_output

  source = climatology_input(tmp_path)
  output = tmp_path / "statistics.zarr"
  write_output(xr.Dataset({"old": ("x", np.arange(3.0))}), output, message="first")
  stage = climatology.staging.stage

  def meanwhile(*args, **kwargs):
    write_output(xr.Dataset({"other": ("x", np.arange(2.0))}), output, message="meanwhile")
    return stage(*args, **kwargs)

  monkeypatch.setattr(climatology.staging, "stage", meanwhile)
  recipe = tmp_path / "promoted.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 3\nweighting = "constant"\n'
    + ('split_dims = ["depth"]\n' if streaming else "")
    + '[read]\nengine = "zarr"\n[save]\n'
  )
  path = "--streaming" if streaming else "--graph"
  args = [str(recipe), str(source), str(output)]
  result = run("compute-climatology", path, "--replace", *THREADS, *args)
  assert result.exit_code != 0 and "changed" in result.output
  assert list(open_output(output).data_vars) == ["other"]


def test_streaming_needs_no_worker_to_see_its_store(tmp_path: pathlib.Path, monkeypatch):
  """Streaming computes and copies in this process: its temporary store is not checked against
  a distributed client's workers (a node-local TMPDIR refused it under `mpi`)."""
  from arcomake import climatology

  def blind(path, what):
    raise RuntimeError(f"Dask workers tcp://b do not see {what}")

  monkeypatch.setattr(climatology.staging, "check_workers_reach", blind)
  source = climatology_input(tmp_path)
  series, _ = run_climatology(tmp_path, source, "series", 'split_dims = ["depth"]\n[save]\n')
  assert series.exit_code == 0, series.output
  graph, _ = run_climatology(tmp_path, source, "graph", "\n[save]\n", streaming=False)
  assert graph.exit_code != 0 and "do not see the temporary store" in graph.output


def test_streaming_stores_one_index_per_chunk_whatever_the_postprocess(tmp_path: pathlib.Path):
  """A postprocess step chunking a split dimension does not make the temporary store's chunks
  span several series (each series writes its region alone)."""
  source = climatology_input(tmp_path)
  extra = (
    'split_dims = ["depth"]\n[[postprocess_climatology]]\nname = "chunk"\n'
    "chunks = {depth = -1}\n[save]\nchunks = {time = 1, depth = 1}\nshards = {time = 5}\n"
  )
  result, output = run_climatology(tmp_path, source, "series", extra)
  assert result.exit_code == 0, result.output
  assert output_group(output)["climatology/t3"].shards == (5, 3, 4)


@pytest.mark.parametrize("streaming", [False, True], ids=["graph", "streaming"])
def test_replacing_statistics_keeps_nothing_else(tmp_path: pathlib.Path, streaming: bool):
  """`--replace` writes a repository holding only the two groups: what it held before (here
  variables at its root, as the former layout had) goes, on both paths."""
  from arcomake.dataset_utils import write_output

  source = climatology_input(tmp_path)
  output = tmp_path / "series.zarr" if streaming else tmp_path / "graph.zarr"
  old = xr.Dataset({"old": ("x", np.arange(3.0))}, attrs={"title": "former layout"})
  write_output(old, output, message="former")
  recipe = tmp_path / "replace.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 3\nweighting = "constant"\n'
    '[read]\nengine = "zarr"\n[save]\n'
  )
  path = "--streaming" if streaming else "--graph"
  args = [str(recipe), str(source), str(output)]
  result = run("compute-climatology", path, "--replace", *THREADS, *args)
  assert result.exit_code == 0, result.output
  root = output_group(output)
  assert sorted(name for name, _ in root.members()) == ["anomaly_std", "climatology"]
  assert dict(root.attrs) == {}


@pytest.mark.parametrize("streaming", [False, True], ids=["graph", "streaming"])
def test_statistics_keep_every_coordinate(tmp_path: pathlib.Path, streaming: bool):
  """Coordinates other than indexes, along a split dimension or not, are written as read
  (streaming wrote them as zeros: neither its template's nor its series' carried them)."""
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  values = np.random.default_rng(4).standard_normal((time.size, 3, 4)).astype("float32")
  bounds = np.stack([np.arange(4.0) - 0.5, np.arange(4.0) + 0.5], axis=1)
  ds = xr.Dataset(
    {"t3": (("time", "depth", "lat"), values)},
    coords={
      "time": time,
      "depth": [0.5, 1.5, 2.5],
      "lat": np.arange(4.0),
      "thickness": ("depth", [1.0, 1.0, 2.0]),
      "lat_bnds": (("lat", "bnds"), bounds),
    },
  )
  ds.to_zarr(tmp_path / "in.zarr")
  split = 'split_dims = ["depth"]' if streaming else ""
  result, output = run_climatology(
    tmp_path, tmp_path / "in.zarr", "coords", f"{split}\n[save]\n", streaming=streaming
  )
  assert result.exit_code == 0, result.output
  for group in statistics(output):
    np.testing.assert_array_equal(group.thickness, [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(group.lat_bnds, bounds)


def test_streamed_series_see_the_coordinates_the_graph_sees(tmp_path: pathlib.Path):
  """Postprocess steps on a series see every coordinate, as on the graph path: `coverage_attrs`
  with `vertical_dim = "depth"` failed on a 2D variable's series (the ARCO-OCEAN recipe). Both
  paths write the same statistics (to rounding: they sum in different orders), and the same
  attributes."""
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  rng = np.random.default_rng(5)
  coords = {"time": time, "depth": [0.5, 1.5, 2.5], "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]}
  ds = xr.Dataset(
    {
      "t3": (("time", "depth", "lat", "lon"), rng.standard_normal((time.size, 3, 2, 3))),
      "t2": (("time", "lat", "lon"), rng.standard_normal((time.size, 2, 3))),
    },
    coords=coords,
  ).astype("float32")
  ds.to_zarr(tmp_path / "in.zarr")
  steps = "".join(
    f'[[postprocess_{group}]]\nname = "coverage_attrs"\nvertical_dim = "depth"\ntime_dim = false\n'
    for group in ("climatology", "anomaly_std")
  )
  outputs = []
  for streaming in (True, False):
    label = "series" if streaming else "graph"
    extra = f'split_dims = ["depth"]\n{steps}[save]\n'
    result, output = run_climatology(tmp_path, tmp_path / "in.zarr", label, extra, streaming)
    assert result.exit_code == 0, result.output
    outputs.append(tuple(group.load() for group in statistics(output)))
  for series, graph in zip(*outputs, strict=True):
    xr.testing.assert_allclose(series, graph, rtol=1e-6, atol=1e-6)  # float32
    assert series.attrs == graph.attrs and series.attrs["geospatial_vertical_max"] == 2.5
    assert all(series[name].attrs == graph[name].attrs for name in series.variables)


@pytest.mark.parametrize("split", ['split_dims = ["depth"]', ""], ids=["depth", "variables"])
def test_series_at_once_write_what_one_at_a_time_writes(
  tmp_path: pathlib.Path, monkeypatch, split: str
):
  """Series computed at once write the same output, bit for bit, as one at a time; coordinates
  along a split dimension (written with the metadata) included."""
  from arcomake import climatology

  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  rng = np.random.default_rng(3)
  ds = xr.Dataset(
    {
      "t3": (("time", "depth", "lat"), rng.standard_normal((time.size, 3, 4)).astype("float32")),
      "s3": (("time", "depth", "lat"), rng.standard_normal((time.size, 3, 4)).astype("float32")),
      "t2": (("time", "lat"), rng.standard_normal((time.size, 4)).astype("float32")),
    },
    coords={
      "time": time,
      "depth": [0.5, 1.5, 2.5],
      "thickness": ("depth", [1.0, 1.0, 2.0]),
      "lat": [0.0, 1.0, 2.0, 3.0],
    },
  )
  ds.to_zarr(tmp_path / "in.zarr")
  outputs = []
  for at_once in (1, 4):
    monkeypatch.setattr(climatology.streamed, "_series_at_once", lambda *args, n=at_once: n)
    result, output = run_climatology(
      tmp_path, tmp_path / "in.zarr", f"at_once_{at_once}", f"{split}\n[save]\n"
    )
    assert result.exit_code == 0, result.output
    outputs.append(tuple(group.load() for group in statistics(output)))
  for one, several in zip(*outputs, strict=True):
    xr.testing.assert_identical(several, one)
  np.testing.assert_array_equal(outputs[1][0].thickness, [1.0, 1.0, 2.0])


def test_series_are_reported_as_they_are_written(tmp_path: pathlib.Path, monkeypatch):
  """Progress is logged when each series is written, not once all are (a long run reports how
  far it is): the last series waits until the first ones are reported."""
  import threading

  from arcomake import climatology

  reported = threading.Event()
  seen = []
  compute, info = climatology.streamed._compute_series, climatology.streamed.logger.info

  def last_waits(piece, **kwargs):
    if piece.depth.item() == 2.5:
      seen.append(reported.wait(timeout=20))
    return compute(piece, **kwargs)

  def logged(message, *args, **kwargs):
    if str(message).startswith("Series "):
      reported.set()
    return info(message, *args, **kwargs)

  monkeypatch.setattr(climatology.streamed, "_compute_series", last_waits)
  monkeypatch.setattr(climatology.streamed.logger, "info", logged)
  monkeypatch.setattr(climatology.streamed, "_series_at_once", lambda *args: 1)
  result, _ = run_climatology(
    tmp_path, climatology_input(tmp_path), "progress", 'split_dims = ["depth"]\n[save]\n'
  )
  assert result.exit_code == 0, result.output
  assert seen == [True]


def test_a_failing_series_fails_the_run(tmp_path: pathlib.Path, monkeypatch):
  """A series that fails fails the run (the others not started are cancelled): `main` is not
  moved."""
  from arcomake import climatology

  compute = climatology.streamed._compute_series

  def failing(piece, **kwargs):
    if piece.depth.item() == 1.5:
      raise RuntimeError("series failed")
    return compute(piece, **kwargs)

  monkeypatch.setattr(climatology.streamed, "_compute_series", failing)
  monkeypatch.setattr(climatology.streamed, "_series_at_once", lambda *args: 2)
  result, output = run_climatology(
    tmp_path, climatology_input(tmp_path), "failing", 'split_dims = ["depth"]\n[save]\n'
  )
  assert result.exit_code != 0 and "RuntimeError" in result.output
  with pytest.raises(zarr.errors.GroupNotFoundError):  # `main` still at the repository's creation
    output_group(output)


def test_an_error_outside_the_series_cancels_those_not_started(tmp_path: pathlib.Path, monkeypatch):
  """Any error leaving the loop (an interrupt, here the progress log) cancels the series not
  started: the pool waited for every queued series (hours) before the error surfaced."""
  from arcomake import climatology

  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  rng = np.random.default_rng(5)
  shape = (time.size, 3, 4)
  ds = xr.Dataset(
    {
      name: (("time", "depth", "lat"), rng.standard_normal(shape).astype("float32"))
      for name in "ab"
    },
    coords={"time": time, "depth": [0.5, 1.5, 2.5], "lat": np.arange(4.0)},
  )
  ds.to_zarr(tmp_path / "in.zarr")
  computed = []
  compute, info = climatology.streamed._compute_series, climatology.streamed.logger.info

  def counted(piece, **kwargs):
    computed.append(piece)
    return compute(piece, **kwargs)

  def interrupted(message, *args, **kwargs):
    if str(message).startswith("Series "):
      raise KeyboardInterrupt
    return info(message, *args, **kwargs)

  monkeypatch.setattr(climatology.streamed, "_compute_series", counted)
  monkeypatch.setattr(climatology.streamed.logger, "info", interrupted)
  monkeypatch.setattr(climatology.streamed, "_series_at_once", lambda *args: 1)
  result, _ = run_climatology(
    tmp_path, tmp_path / "in.zarr", "stop", 'split_dims = ["depth"]\n[save]\n'
  )
  assert result.exit_code != 0  # click: "Aborted!"
  assert len(computed) <= 2  # of 6: the first, and at most one its worker had started


def test_streaming_without_split_dims_computes_each_variable(tmp_path: pathlib.Path):
  """Without `split_dims`, `--streaming` computes one series per variable, as the graph."""
  source = climatology_input(tmp_path)
  graph, graph_clim = run_climatology(tmp_path, source, "graph", "\n[save]\n", streaming=False)
  series, series_clim = run_climatology(tmp_path, source, "series", "\n[save]\n")
  assert graph.exit_code == 0 and series.exit_code == 0, series.output
  xr.testing.assert_allclose(
    statistics(series_clim)[0].load(), statistics(graph_clim)[0].load(), atol=1e-6
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
    statistics(series_clim)[0].load(), statistics(graph_clim)[0].load(), atol=1e-6
  )
  assert statistics(series_clim)[0].t3.encoding["chunks"][1] == 1


@pytest.mark.parametrize(
  ("extra", "message"),
  [
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
  assert statistics(clim)[0].t3.notnull().all()


def test_zip_outputs_are_refused_before_any_work(recipes, tmp_path: pathlib.Path):
  """Zip outputs were removed with Zarr v3: a `.zip` output path fails at once, before any
  download, instead of becoming a directory with that name."""
  output = tmp_path / "piece.zip"
  result = CliRunner().invoke(cli, ["download", *THREADS, str(recipes["download"]), str(output)])
  assert result.exit_code != 0 and "zip outputs are not supported" in result.output
  assert not output.exists()


@pytest.mark.parametrize("streaming", [False, True], ids=["graph", "streaming"])
def test_climatology_writes_shards_its_chunks_straddle(
  tmp_path: pathlib.Path, monkeypatch, streaming: bool
):
  """Read one level per chunk, stored in shards of all levels (the layout of the historical
  datasets, used by the statistics too): the writes are aligned to the stored layout, as in
  download; xarray refused Dask chunks overlapping a shard. Streaming stored one level per shard
  before it wrote into a temporary store too.

  Both paths compute into a temporary store, then copy: the window runs once per level (it ran
  once per output), and the store is removed at the end."""
  import tempfile

  from arcomake import climatology
  from arcomake.temporary_utils import TempDirectoryRegistry

  windows = []
  window = climatology.kernels.window_bins_last

  def counted(*args, **kwargs):
    windows.append(1)
    return window(*args, **kwargs)

  monkeypatch.setattr(climatology.kernels, "window_bins_last", counted)
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
    + ('split_dims = ["depth"]\n' if streaming else "")
    + '[read]\nengine = "zarr"\nchunks = {time = "auto", depth = 1, lat = -1, lon = -1}\n'
    "[save]\nchunks = {time = 1, depth = 1, lat = -1, lon = -1}\n"
    'shards = {time = 8, depth = -1, lat = -1, lon = -1}\nshard_order = "lexicographic"\n'
  )
  output = tmp_path / "statistics.zarr"
  path = "--streaming" if streaming else "--graph"
  result = run("compute-climatology", path, *THREADS, str(recipe), str(source), str(output))
  assert result.exit_code == 0, result.output
  stored = output_group(output)["climatology/t3"]
  assert stored.shards == (8, 3, 4, 4) and stored.chunks == (1, 1, 4, 4)
  clim, std = statistics(output)
  assert clim.t3.shape == (365, 3, 4, 4)
  assert len(windows) == 3  # one per level: once, not once per output
  ds = xr.Dataset({"t3": (("time", "depth", "lat", "lon"), values)}, coords=coords)
  mean, var = climatology.graph._compute_climatology(
    ds, 365, 5, "gaussian", 2.0, "time", "day_of_year"
  )
  # Streaming accumulates in float64, the graph in float32: equal to float32 precision.
  tolerance = {"rtol": 1e-5, "atol": 1e-6} if streaming else {"rtol": 1e-6}
  xr.testing.assert_allclose(clim.t3.load(), mean.t3, **tolerance)
  xr.testing.assert_allclose(std.t3.load(), np.sqrt(var.t3.clip(min=0)), **tolerance)
  (store,) = stores
  assert store.is_relative_to(scratch) and not store.exists()  # removed at the end


@pytest.mark.parametrize("setup", ["localcluster", "processes"])
def test_download_on_worker_processes_writes_what_threads_write(
  tmp_path: pathlib.Path,
  recipes,
  downloaded: pathlib.Path,
  isolated_tempdir: pathlib.Path,
  setup: str,
):
  """Worker processes (the backends and steps pickled to them) write the same output: a local
  cluster's workers see the run's temporary directory, and filter the warnings this process
  filters; Dask's process pool too (it was warned about, untested)."""
  import dask

  output = tmp_path / "cluster.zarr"
  options = {"n_workers": 2, "threads_per_worker": 1, "local_directory": str(isolated_tempdir)}
  with dask.config.set({"localcluster": options, "distributed.dashboard.enabled": False}):
    result = run("download", "--dask-setup", setup, str(recipes["download"]), str(output))
  assert result.exit_code == 0, result.output
  xr.testing.assert_identical(open_output(output).load(), open_output(downloaded).load())


@pytest.mark.parametrize("streaming", [True, False])
def test_bins_are_checked_before_the_output_is_opened(tmp_path: pathlib.Path, streaming: bool):
  """Streaming found sub-daily steps (or partial periods) in every series, after opening the
  output's run and writing the temporary store."""
  time = pd.date_range("2020-01-01", periods=48, freq="1h")
  xr.Dataset({"x": ("time", np.zeros(48, "float32"))}, coords={"time": time}).to_zarr(
    tmp_path / "in.zarr"
  )
  result, output = run_climatology(tmp_path, tmp_path / "in.zarr", "hourly", "[save]\n", streaming)
  assert result.exit_code != 0 and "one time step per day" in result.output
  assert not output.exists()
