# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`download` chunks the parts of each block as planned (6.1): without chunk settings, a recipe
writes the same output as with the sizes set by hand, and the plan follows the memory budget."""

import importlib
import logging
import pathlib
import re

import pytest
import xarray as xr
import zarr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
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


PART_CHUNKS = (
  "chunks = {time = 24, latitude = -1, longitude = -1}\n",
  "chunks = {time = 1, depth = 1, latitude = -1, longitude = -1}\n",
)


def without_chunks(recipe: pathlib.Path) -> pathlib.Path:
  text = recipe.read_text()
  for line in PART_CHUNKS:
    assert text.count(line) == 1
    text = text.replace(line, "")
  planned = recipe.with_name("planned.toml")
  planned.write_text(text)
  return planned


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "synchronous", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args, catch_exceptions=False)


def array_metadata(path: pathlib.Path) -> dict[str, dict]:
  group = zarr.open_group(icechunk_utils.readonly_store(path, None), mode="r")
  return {name: array.metadata.to_dict() for name, array in group.arrays()}


def planned_chunks(text: str) -> dict[str, str]:
  """The chunks logged for each part in its first block: label -> chunks."""
  plans: dict[str, str] = {}
  for label, chunks in re.findall(r"(\w+, part \d+): chunks (\{[^}]*\})", text):
    plans.setdefault(label, chunks)
  return plans


def test_planned_chunks_write_the_same_output(recipe, tmp_path: pathlib.Path, logs):
  """The sample recipe without its parts' `chunks`, against the recipe: same values, attributes
  and array metadata."""
  assert run(recipe, tmp_path / "set.zarr").exit_code == 0
  result = run(without_chunks(recipe), tmp_path / "planned.zarr")
  assert result.exit_code == 0, result.output
  xr.testing.assert_identical(
    open_output(tmp_path / "planned.zarr").load(), open_output(tmp_path / "set.zarr").load()
  )
  assert array_metadata(tmp_path / "planned.zarr") == array_metadata(tmp_path / "set.zarr")
  # Each part of each block (and checkpointed sub-block) is planned, and logged.
  plans = planned_chunks(logs.text)
  assert set(plans) == {"atmosphere, part 1", "ocean, part 1", "ocean_static, part 1"}
  assert plans["atmosphere, part 1"] == "{time: 48, latitude: 8, longitude: 8}"  # a whole block


def test_the_plan_follows_the_budget(recipe, tmp_path: pathlib.Path, logs):
  """With 20 kB per task, one day of hourly data (12 kB) fits and two do not: whole days, as the
  resampling needs."""
  result = run(without_chunks(recipe), tmp_path / "out.zarr", "--memory-limit", "100kB")
  assert result.exit_code == 0, result.output
  assert planned_chunks(logs.text)["atmosphere, part 1"].startswith("{time: 24,")


def test_a_budget_too_small_for_one_task_fails(recipe, tmp_path: pathlib.Path, logs):
  result = run(without_chunks(recipe), tmp_path / "out.zarr", "--memory-limit", "20kB")
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
