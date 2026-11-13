# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Recipe loading and command options: a recipe either runs as written or fails (intents 3, 4)."""

import pathlib

import click
import pydantic
import pytest

from arcomake import cli_utils, climatology
from arcomake.cli_utils import check_if_overwriting, read_configs
from arcomake.download import DownloadConfig
from arcomake.process import ProcessConfig
from tests.conftest import make_ocean_dataset, render_recipe


def test_read_configs_injects_cli_dates(tmp_path: pathlib.Path, sources: pathlib.Path):
  """`--start`/`--end` replace the recipe's dates before validation."""
  path = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  configs = read_configs(str(path), schema=DownloadConfig, inject={"end": "2020-02-27T00:00:00"})
  assert configs.end.day == 27


def test_read_configs_reports_invalid_recipes(tmp_path: pathlib.Path):
  """A recipe failing validation is reported with its path, as a CLI error."""
  path = tmp_path / "bad.toml"
  path.write_text('start = 2020-01-02T00:00:00\nend = 2020-01-01T00:00:00\nstep = "1D"\n')
  with pytest.raises(click.ClickException, match="bad.toml"):
    read_configs(str(path), schema=DownloadConfig)


def test_download_config_rejects_unknown_top_level_keys():
  """Unknown keys at the top level of a recipe are errors."""
  with pytest.raises(pydantic.ValidationError):
    DownloadConfig.model_validate(
      {
        "start": "2020-01-01",
        "end": "2020-01-02",
        "step": "1D",
        "datasets": {},
        "save": {},
        "chekcs": {},
      }
    )


@pytest.mark.xfail(
  reason="B15: ProcessConfig accepts `time_dim` and `skipna` and ignores them", strict=True
)
def test_process_config_has_no_ignored_options():
  """Options that have no effect are not accepted."""
  with pytest.raises(pydantic.ValidationError):
    ProcessConfig.model_validate({"skipna": True})


@pytest.mark.xfail(reason="B4: the recipe's `use_flox` never reaches the computation", strict=True)
def test_climatology_honours_use_flox(tmp_path: pathlib.Path, sources: pathlib.Path, monkeypatch):
  """`use_flox = true` in the recipe selects the flox implementation."""
  seen = {}

  def fake_compute(dataset, **kwargs):
    seen.update(kwargs)
    raise RuntimeError("stop after recording the arguments")

  monkeypatch.setattr(climatology, "_compute_climatology", fake_compute)
  recipe = render_recipe("sample_climatology.toml", tmp_path, SOURCES=str(sources))
  recipe.write_text(
    recipe.read_text().replace('weighting = "constant"', 'weighting = "constant"\nuse_flox = true')
  )
  dataset = tmp_path / "in.zarr"
  make_ocean_dataset().isel(time=slice(0, 4)).to_zarr(dataset)
  with pytest.raises(click.ClickException):
    climatology.compute_climatology.callback(  # type: ignore[misc]
      str(recipe),
      str(dataset),
      str(tmp_path / "c.zarr"),
      str(tmp_path / "s.zarr"),
      dask_setup="threads",
    )
  assert seen.get("use_flox") is True


def test_check_if_overwriting(tmp_path: pathlib.Path):
  """An existing destination needs `--overwrite`; a missing one is fine."""
  check_if_overwriting(str(tmp_path / "absent.zarr"))
  with pytest.raises(click.ClickException, match="already exists"):
    check_if_overwriting(str(tmp_path))
  check_if_overwriting(str(tmp_path), overwrite=True)


@pytest.mark.xfail(
  reason="B14: a failed existence check is treated as 'does not exist'", strict=True
)
def test_check_if_overwriting_fails_when_destination_is_unreachable(
  monkeypatch: pytest.MonkeyPatch,
):
  """If the destination cannot be checked (e.g. no access), the run stops before downloading."""

  class Unreachable:
    def exists(self, path):
      raise PermissionError("access denied")

  monkeypatch.setattr(cli_utils.fsspec, "url_to_fs", lambda path, **kwargs: (Unreachable(), path))
  with pytest.raises((click.ClickException, PermissionError)):
    check_if_overwriting("s3://bucket/output.zarr")
