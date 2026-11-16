# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Recipe loading and command options: a recipe either runs as written or fails (intents 3, 4)."""

import pathlib

import click
import pydantic
import pytest

from arcomake import cli_utils, climatology
from arcomake.checks import ChecksConfig
from arcomake.cli_utils import check_if_overwriting, read_configs
from arcomake.dataset_utils import DatasetConfig, ReadConfig, SaveConfig, VariableEncodingConfig
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


def test_process_config_has_no_ignored_options():
  """Options that have no effect are not accepted."""
  with pytest.raises(pydantic.ValidationError):
    ProcessConfig.model_validate({"skipna": True})


@pytest.mark.parametrize(
  ("model", "table"),
  [
    (SaveConfig, {"chunk": {"time": 1}}),  # typo for `chunks` (arco-ocean stats recipes)
    (SaveConfig, {"chunks": {"time": "auto"}}),  # N4: not a stored layout
    (SaveConfig, {"chunks": {"time": 0}}),
    (ReadConfig, {"engin": "zarr"}),
    (VariableEncodingConfig, {"unit": "days since 2000-01-01"}),
    (ChecksConfig, {"valid_time_coordinates": {}}),
    (ChecksConfig, {"ensure_no_nans": {"sea_mask": "thetao"}}),  # a list of variables is required
  ],
)
def test_nested_tables_reject_unknown_or_invalid_keys(model, table):
  """Typos one level down in a recipe are errors too (intent 3)."""
  with pytest.raises(pydantic.ValidationError):
    model.model_validate(table)


@pytest.mark.parametrize(
  "part",
  [
    {"filename_or_obj": "gs://bucket/store.zarr", "engine": "zarr", "consolidate": True},
    {"filename_or_obj": "cm://dataset", "engine": "copernicusmarine", "variable": ["uo"]},
    {"filename_or_obj": "ewds://cems", "engine": "earlywarningdatastore", "variables": ["dis"]},
  ],
)
def test_parts_with_typos_do_not_fall_back_to_generic_parts(part: dict):
  """A typo in a part of a dedicated engine is an error, not a generic `open_dataset` call."""
  with pytest.raises(pydantic.ValidationError):
    DatasetConfig.model_validate({"parts": [part]})


def test_generic_parts_take_open_dataset_arguments():
  """Other engines accept the arguments of `xarray.open_dataset`."""
  part = {"filename_or_obj": "data.nc", "engine": "h5netcdf", "decode_times": False}
  assert DatasetConfig.model_validate({"parts": [part]}).parts[0].decode_times is False


def test_drop_variables_survive_repeated_dumps():
  """`drop_variables` reaches every interval: the config is dumped once per part and interval.

  Typed as an `Iterable`, pydantic stored a one-shot iterator, and only the first dump had the
  names.
  """
  part = {"filename_or_obj": "data.nc", "engine": "h5netcdf", "drop_variables": ["a", "b"]}
  config = DatasetConfig.model_validate({"parts": [part]}).parts[0]
  assert [config.model_dump()["drop_variables"] for _ in range(2)] == [["a", "b"]] * 2
  read = ReadConfig(drop_variables=["a"])
  assert [read.model_dump()["drop_variables"] for _ in range(2)] == [["a"]] * 2


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
