# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`[checks]` in a download recipe: each block is checked before its commit, like pre-commit
hooks; a failing block is not committed, and the run can resume once the cause is fixed."""

import datetime
import importlib
import pathlib
import re
import tomllib

import numpy as np
import pydantic
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
from arcomake.checks import ChecksConfig, ValidationError
from arcomake.datetime_utils import DateInterval
from tests.conftest import make_ocean_dataset, open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command

CHECKS = """
[checks.valid_time_coordinate]
freq = "1D"

[checks.valid_global_ecmwf_coordinates]
latitude_dim = "lat"
longitude_dim = "lon"

[checks.ensure_no_nans]
sea_mask = ["thetao"]
"""


def recipe_with(tmp_path: pathlib.Path, sources: pathlib.Path, checks: str) -> pathlib.Path:
  recipe = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  recipe.write_text(recipe.read_text().replace("[save]", checks + "\n[save]"))
  return recipe


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def test_blocks_that_pass_are_written_as_without_checks(tmp_path, sources):
  checked = run(recipe_with(tmp_path, sources, CHECKS), tmp_path / "checked")
  assert checked.exit_code == 0, checked.output
  plain_recipe = tmp_path / "plain_recipe"
  plain_recipe.mkdir()
  plain = run(
    render_recipe("sample_download.toml", plain_recipe, SOURCES=str(sources)), tmp_path / "plain"
  )
  assert plain.exit_code == 0, plain.output
  xr.testing.assert_identical(open_output(tmp_path / "checked"), open_output(tmp_path / "plain"))


def test_a_failing_block_is_not_committed_and_the_run_resumes(tmp_path, sources):
  """The ocean has no values on 28 Feb, the first day of the second block: the check fails while
  that block is written, so only the first block is on the run's branch and `main` is not moved.
  Without the check (it is not part of the recipe's fingerprint), --resume writes the rest."""
  ocean = make_ocean_dataset()
  ocean["thetao"] = ocean.thetao.where(ocean.time != np.datetime64("2020-02-28"))
  ocean.to_zarr(sources / "ocean.zarr", mode="w", zarr_format=2)
  output = tmp_path / "out"
  failed = run(recipe_with(tmp_path, sources, CHECKS), output)
  assert failed.exit_code != 0
  assert "the block was not committed: ensure_no_nans:" in failed.output
  # The first chunk to fail is reported: thetao (inside the mask) or thetao_bottom (everywhere).
  assert re.search(
    r"NaN values found for variable thetao\S* .*in a block, first in its chunk at 2020-02-28",
    failed.output,
  )
  waiting = icechunk_utils.waiting_run(output, None, "download")
  assert waiting is not None and len(waiting.blocks) == 1
  resumed = run(recipe_with(tmp_path, sources, ""), output, "--resume")
  assert resumed.exit_code == 0, resumed.output
  assert open_output(output).sizes["time"] == 5


def test_the_time_check_needs_no_dates_and_checks_each_block():
  """The table takes the recipe's dates when loaded; each block's axis is checked over the block."""
  config = download_module.DownloadConfig.model_validate(
    {
      "start": datetime.datetime(2020, 2, 26),
      "end": datetime.datetime(2020, 3, 2),
      "step": "2D",
      "datasets": {},
      "save": {"chunks": {"time": 1}},
      "checks": {"valid_time_coordinate": {"freq": "1D"}},
    }
  )
  assert config.checks is not None and config.checks.valid_time_coordinate is not None
  block = DateInterval(datetime.datetime(2020, 2, 28), datetime.datetime(2020, 3, 1))
  days = xr.Dataset(coords={"time": np.array(["2020-02-28", "2020-02-29"], "datetime64[ns]")})
  download_module._checked_block(days, config.checks, block, "time")
  with pytest.raises(ValidationError, match="valid_time_coordinate: .*Missing"):
    download_module._checked_block(days.isel(time=[0]), config.checks, block, "time")


def test_checks_are_not_part_of_the_fingerprint(tmp_path, sources):
  with_checks = download_module.DownloadConfig.model_validate(
    tomllib.loads(recipe_with(tmp_path, sources, CHECKS).read_text())
  )
  without = with_checks.model_copy(update={"checks": None})
  assert download_module.recipe_fingerprint(with_checks) == download_module.recipe_fingerprint(
    without
  )
  assert isinstance(with_checks.checks, ChecksConfig)


def test_the_time_check_takes_no_dates():
  """They would go unused: each block is checked over the block."""
  recipe = {
    "start": datetime.datetime(2020, 2, 26),
    "end": datetime.datetime(2020, 3, 2),
    "step": "2D",
    "datasets": {},
    "save": {"chunks": {"time": 1}},
    "checks": {"valid_time_coordinate": {"start": datetime.datetime(2020, 2, 26)}},
  }
  with pytest.raises(pydantic.ValidationError, match="remove start and end"):
    download_module.DownloadConfig.model_validate(recipe)


def test_block_checks_take_the_recipes_time_dimension():
  """Tables without `time_dim` use the recipe's: here a static variable under a mask that varies
  in time is refused, and the time axis is checked along `valid_time`."""
  config = ChecksConfig.model_validate(
    {
      "valid_time_coordinate": {
        "start": datetime.datetime(2020, 2, 28),
        "end": datetime.datetime(2020, 3, 1),
      },
      "ensure_no_nans": {"ice": ["depth"]},
    }
  )
  stamps = np.array(["2020-02-28", "2020-02-29"], "datetime64[ns]")
  ds = xr.Dataset(
    {"ice": (("valid_time", "x"), np.ones((2, 3))), "depth": ("x", np.ones(3))},
    coords={"valid_time": stamps},
  )
  block = DateInterval(datetime.datetime(2020, 2, 28), datetime.datetime(2020, 3, 1))
  with pytest.raises(ValidationError, match="'depth' has no 'valid_time' but its mask 'ice' has"):
    download_module._checked_block(ds, config, block, "valid_time")
