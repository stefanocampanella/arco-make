# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`download` chunks the parts of each block as planned (6.1): without chunk settings, a recipe
writes the same output as with the sizes set by hand, and the plan follows the memory budget."""

import datetime
import importlib
import logging
import pathlib
import re

import pytest
import xarray as xr
import zarr
from click.testing import CliRunner, Result
from pydantic import ValidationError

from arcomake import cli, icechunk_utils
from arcomake.chunk_plan import Budget
from arcomake.dataset_utils import (
  DatasetConfig,
  download_and_process,
  maybe_checkpointing_download_and_process,
)
from arcomake.processing_utils import ProcessingStepConfig
from tests.conftest import open_output, render_recipe


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  return render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))


@pytest.fixture
def logs(caplog, monkeypatch: pytest.MonkeyPatch):
  """The log of the commands run in a test (the CLI's logging setup would remove caplog's)."""
  module = importlib.import_module("arcomake.download")  # `arcomake.download` is the command
  monkeypatch.setattr(module, "set_default_logger", lambda *args, **kwargs: None)
  with caplog.at_level(logging.INFO, logger="arcomake"):
    yield caplog


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "synchronous", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args, catch_exceptions=False)


def array_metadata(path: pathlib.Path) -> dict[str, dict]:
  group = zarr.open_group(icechunk_utils.readonly_store(path, None), mode="r")
  return {name: array.metadata.to_dict() for name, array in group.arrays()}


def planned_chunks(text: str) -> dict[str, str]:
  """The chunks logged for each dataset in its first block: name -> chunks."""
  plans: dict[str, str] = {}
  for label, chunks in re.findall(r"(\w+): chunks (\{[^}]*\})", text):
    plans.setdefault(label, chunks)
  return plans


def test_the_plan_does_not_change_the_output(recipe, tmp_path: pathlib.Path, logs):
  """The sample recipe (no chunk sizes) planned for a large and a small budget: same values,
  attributes and array metadata; each dataset of each block is planned, and logged."""
  result = run(recipe, tmp_path / "large.zarr")
  assert result.exit_code == 0, result.output
  plans = planned_chunks(logs.text)
  assert set(plans) == {"atmosphere", "ocean", "ocean_static"}
  assert plans["atmosphere"] == "{time: 48, latitude: 8, longitude: 8}"  # a whole block
  logs.clear()
  result = run(recipe, tmp_path / "small.zarr", "--memory-limit", "100kB")
  assert result.exit_code == 0, result.output
  assert planned_chunks(logs.text)["atmosphere"].startswith("{time: 24,")
  xr.testing.assert_identical(
    open_output(tmp_path / "small.zarr").load(), open_output(tmp_path / "large.zarr").load()
  )
  assert array_metadata(tmp_path / "small.zarr") == array_metadata(tmp_path / "large.zarr")


def test_a_budget_too_small_for_one_task_fails(recipe, tmp_path: pathlib.Path, logs):
  result = run(recipe, tmp_path / "out.zarr", "--memory-limit", "20kB")
  assert result.exit_code != 0
  assert "PlanError" in result.output
  assert "give more memory (--memory-limit)" in logs.text


@pytest.mark.parametrize("blocks", ["1", "2"])
def test_chunks_are_aligned_to_the_stored_chunks(recipe, tmp_path: pathlib.Path, blocks: str):
  """Stored chunks of two days over the ocean's daily checkpoints (Dask chunks of one day):
  rechunked before writing, as a `chunk` step did by hand; the values are those of daily
  stored chunks."""
  assert run(recipe, tmp_path / "daily.zarr").exit_code == 0
  text = recipe.read_text()
  layout = "chunks = {time = 1, depth = 1, lat = -1, lon = -1}\n"
  assert text.count(layout) == 1
  two_days = recipe.with_name("two_days.toml")
  two_days.write_text(text.replace(layout, layout.replace("time = 1", "time = 2")))
  result = run(two_days, tmp_path / "out.zarr", "--blocks", blocks)
  assert result.exit_code == 0, result.output
  out = open_output(tmp_path / "out.zarr")
  assert out["thetao"].encoding["chunks"][0] == 2
  xr.testing.assert_identical(out.load(), open_output(tmp_path / "daily.zarr").load())


def recipe_with(dataset: dict | None = None, postprocess: list | None = None) -> dict:
  return {
    "step": "1D",
    "start": "2020-01-01T00:00:00",
    "end": "2020-01-02T00:00:00",
    "datasets": {"a": {"parts": [{"filename_or_obj": "a.zarr"}], **(dataset or {})}},
    "postprocess": postprocess or [],
    "save": {},
  }


REGRID = {
  "name": "regrid",
  "grid": {"south": 0, "north": 1, "west": 0, "east": 1, "resolution_lat": 1, "resolution_lon": 1},
}


@pytest.mark.parametrize(
  ("recipe", "message"),
  [
    (
      recipe_with({"parts": [{"filename_or_obj": "a.zarr", "chunks": {"time": 24}}]}),
      "part 1: chunks are planned for each block .*: remove `chunks`",
    ),
    (
      recipe_with({"parts": [{"filename_or_obj": "cm://x", "chunks": {"time": 2}}]}),
      "part 1: chunks are planned",
    ),
    (
      recipe_with({"postprocess": [{"name": "chunk", "chunks": {"time": 32}}]}),
      r"step 1 \(chunk\): chunks are planned .*: remove the step",
    ),
    (
      recipe_with(postprocess=[{"name": "sel"}, {"name": "chunk", "time": 32}]),
      r"step 2 \(chunk\)",
    ),
    (
      recipe_with({"postprocess": [{**REGRID, "kwargs": {"output_chunks": {"time": 16}}}]}),
      r"step 1 \(regrid\): chunks are planned .*: remove `output_chunks`",
    ),
  ],
)
def test_download_recipes_set_no_chunk_sizes(recipe: dict, message: str):
  """Part `chunks`, `chunk` steps and regrid `output_chunks` are refused at load, with what to
  do; `[save] chunks`, the stored layout, stay."""
  download_config = importlib.import_module("arcomake.download").DownloadConfig
  with pytest.raises(ValidationError, match=message):
    download_config.model_validate(recipe)
  download_config.model_validate(recipe_with(postprocess=[REGRID]) | {"save": {"chunks": 1}})


def test_plan_prints_the_chunks_and_writes_nothing(recipe, tmp_path: pathlib.Path):
  """`--plan`: the first block's variables with their chunks; no output, no data read."""
  output = tmp_path / "out.zarr"
  result = run(recipe, output, "--plan", "--memory-limit", "100kB")
  assert result.exit_code == 0, result.output
  assert "Block [2020-02-26T00:00, 2020-02-28T00:00), budget: target" in result.output
  assert re.search(
    r"  thetao \(time, depth, lat, lon\) float32: chunks \(1, 3, 4, 4\), ", result.output
  )
  assert re.search(
    r"  t2m \(time, lat, lon\) float64: chunks \(1, 4, 4\), 128 B each, 2 chunks", result.output
  )
  assert not output.exists()


ATMOSPHERE_DAYS = (datetime.datetime(2020, 2, 26), datetime.datetime(2020, 2, 28))


def atmosphere(sources: pathlib.Path, **dataset) -> DatasetConfig:
  part = {"filename_or_obj": str(sources / "atmosphere.zarr"), "engine": "zarr"}
  return DatasetConfig.model_validate({"parts": [part], **dataset})


def test_checkpoints_take_planned_chunks(sources: pathlib.Path):
  """Daily checkpoints of hourly data planned in chunks of 11 hours (6 kB a task): the second day
  is appended at hour 24, inside a stored chunk of the first; it is aligned, not refused."""
  budget = Budget(memory_limit=30_000, threads=1)
  config = atmosphere(sources, checkpointing={"step": "1D"})
  with maybe_checkpointing_download_and_process(
    config, *ATMOSPHERE_DAYS, budget=budget, progress=False
  ) as dataset:
    expected = xr.open_zarr(sources / "atmosphere.zarr").sel(time=slice("2020-02-26", "2020-02-27"))
    xr.testing.assert_equal(dataset.load(), expected.load())


def test_steps_after_merging_are_planned_for(sources: pathlib.Path):
  """A resample in the recipe's steps after merging: the parts are read in whole days (20 kB a
  task: 39 hours would fit, 24 are a day)."""
  budget = Budget(memory_limit=100_000, threads=1)
  resample = ProcessingStepConfig.model_validate(
    {"name": "resample", "time": "1D", "reduce": "mean"}
  )
  alone = download_and_process(atmosphere(sources), *ATMOSPHERE_DAYS, budget=budget)
  assert alone["t2m"].chunks[0] == (39, 9)
  dataset = download_and_process(
    atmosphere(sources), *ATMOSPHERE_DAYS, budget=budget, later_steps=[resample]
  )
  assert dataset["t2m"].chunks[0] == (24, 24)


def test_parts_of_a_dataset_share_their_chunks(tmp_path: pathlib.Path, sources: pathlib.Path):
  """A float32 part beside a float64 one on the same axes: planned together, so merged variables
  have the same chunks (steps combining them would otherwise rechunk)."""
  source = xr.open_zarr(sources / "atmosphere.zarr")[["t2m"]].astype("float32").rename(t2m="t2")
  source.drop_encoding().to_zarr(tmp_path / "single.zarr")
  config = DatasetConfig.model_validate(
    {
      "parts": [
        {"filename_or_obj": str(sources / "atmosphere.zarr"), "engine": "zarr"},
        {"filename_or_obj": str(tmp_path / "single.zarr"), "engine": "zarr"},
      ]
    }
  )
  budget = Budget(memory_limit=30_000, threads=1)
  dataset = download_and_process(config, *ATMOSPHERE_DAYS, budget=budget)
  assert dataset["t2"].chunks == dataset["t2m"].chunks == dataset["tp"].chunks
