# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Recipes without dates (given by the command) and `download --extend`: the days after an
output's last appended as a new version, with the recipe it was written with."""

import importlib
import pathlib
import tomllib

import icechunk
import pytest
import xarray as xr
from click.testing import CliRunner, Result

from arcomake import cli, icechunk_utils
from arcomake.dataset_utils import write_output
from tests.conftest import make_static_dataset, open_output, render_recipe

download_module = importlib.import_module("arcomake.download")  # the module, not the command
START, MIDDLE, END = "2020-02-26T00:00:00", "2020-02-28T00:00:00", "2020-03-02T00:00:00"


@pytest.fixture
def recipe(tmp_path: pathlib.Path, sources: pathlib.Path) -> pathlib.Path:
  """The sample recipe without its dates."""
  path = render_recipe("sample_download.toml", tmp_path, SOURCES=str(sources))
  text = path.read_text()
  dates = "start = 2020-02-26T00:00:00\nend = 2020-03-02T00:00:00\n"
  assert text.count(dates) == 1
  path.write_text(text.replace(dates, ""))
  return path


def run(recipe: pathlib.Path, output: pathlib.Path, *options: str) -> Result:
  args = ["download", "--dask-setup", "threads", *options, str(recipe), str(output)]
  return CliRunner().invoke(cli, args)


def versions(output: pathlib.Path) -> int:
  repo = icechunk.Repository.open(icechunk_utils.storage(output))
  return sum(1 for _ in repo.ancestry(branch="main"))


def test_a_recipe_without_dates_takes_them_from_the_command(recipe, tmp_path):
  missing = run(recipe, tmp_path / "out")
  assert missing.exit_code != 0 and "give --start and --end" in missing.output
  result = run(recipe, tmp_path / "out", f"--start={START}", f"--end={END}")
  assert result.exit_code == 0, result.output
  assert open_output(tmp_path / "out").sizes["time"] == 5


def test_an_extension_equals_one_download(recipe, tmp_path):
  whole = run(recipe, tmp_path / "whole", f"--start={START}", f"--end={END}")
  assert whole.exit_code == 0, whole.output
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  before = versions(output)
  extended = run(recipe, output, "--extend", f"--end={END}")
  assert extended.exit_code == 0, extended.output
  xr.testing.assert_identical(open_output(output), open_output(tmp_path / "whole"))
  assert versions(output) > before  # the earlier version stays in the history
  again = run(recipe, output, "--extend", f"--end={END}")
  assert again.exit_code == 0 and "Nothing to extend" in again.output


def test_an_extension_keeps_the_recipe(recipe, tmp_path):
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  recipe.write_text(recipe.read_text().replace('title = "Synthetic sample dataset"', 'title = "x"'))
  other = run(recipe, output, "--extend", f"--end={END}")
  assert other.exit_code != 0 and "another recipe" in other.output


@pytest.mark.parametrize("option", [f"--start={MIDDLE}", "--replace"])
def test_an_extension_sets_its_own_start(recipe, tmp_path, option):
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  refused = run(recipe, output, "--extend", option, f"--end={END}")
  assert refused.exit_code != 0 and "not with --replace nor --start" in refused.output


def test_an_extension_compares_the_static_variables(recipe, tmp_path, sources):
  """Appends do not rewrite them: changed static variables fail the extension."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  static = make_static_dataset()
  static["deptho"] = static.deptho + 1.0
  static.to_zarr(sources / "static.zarr", mode="w", zarr_format=2)
  failed = run(recipe, output, "--extend", f"--end={END}")
  assert failed.exit_code != 0 and "differs from the stored one" in failed.output


def test_a_failed_extension_resumes(recipe, tmp_path, monkeypatch):
  whole = run(recipe, tmp_path / "whole", f"--start={START}", f"--end={END}")
  assert whole.exit_code == 0, whole.output
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  real = download_module.maybe_checkpointing_download_and_process

  def failing(configs, start, end, **kwargs):
    if str(start).startswith("2020-03-01"):
      raise RuntimeError("simulated failure")
    return real(configs, start, end, **kwargs)

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", failing)
  assert run(recipe, output, "--extend", f"--end={END}").exit_code != 0
  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", real)
  resumed = run(recipe, output, "--extend", "--resume", f"--end={END}")
  assert resumed.exit_code == 0, resumed.output
  xr.testing.assert_identical(open_output(output), open_output(tmp_path / "whole"))


def fail_every_block(monkeypatch):
  def failing(configs, start, end, **kwargs):
    raise RuntimeError("simulated failure")

  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", failing)


def test_a_failed_extension_resumes_only_as_one(recipe, tmp_path, monkeypatch):
  """Resumed as a download, an extension that committed no block would write its first block
  in place of the output (mode "w"); with another recipe it would append unchecked."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  real = download_module.maybe_checkpointing_download_and_process
  fail_every_block(monkeypatch)
  assert run(recipe, output, "--extend", f"--end={END}").exit_code != 0
  monkeypatch.setattr(download_module, "maybe_checkpointing_download_and_process", real)
  plain = run(recipe, output, "--resume", f"--start={MIDDLE}", f"--end={END}")
  assert plain.exit_code != 0 and "is a failed extension: --extend --resume" in plain.output
  again = run(recipe, output, "--extend", f"--end={END}")
  assert again.exit_code != 0 and "--extend --resume continues it" in again.output
  recipe.write_text(recipe.read_text().replace('title = "Synthetic sample dataset"', 'title = "x"'))
  other = run(recipe, output, "--extend", "--resume", f"--end={END}")
  assert other.exit_code != 0 and "another recipe" in other.output
  assert open_output(output).sizes["time"] == 2  # `main` as it was


def test_an_extension_starts_where_the_last_block_ended(recipe, tmp_path):
  """Not from the stamps (monthly or right-labelled axes, cftime): from the last block."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  configs = download_module.DownloadConfig.model_validate(
    tomllib.loads(recipe.read_text()) | {"end": END}
  )
  tip = download_module._main_tip(str(output), configs)
  assert str(download_module._extension_start(str(output), tip)) == "2020-02-28 00:00:00"


def test_only_a_download_output_can_be_extended(recipe, tmp_path):
  output = tmp_path / "other"
  write_output(xr.Dataset({"x": ("time", [1.0, 2.0])}), output, message="not a download")
  refused = run(recipe, output, "--extend", f"--end={END}")
  assert refused.exit_code != 0 and "records no download block" in refused.output


def test_access_and_retention_do_not_change_the_fingerprint(recipe):
  """A renamed profile, another retention or another way to read the sources must not block
  extensions."""
  configs = download_module.DownloadConfig.model_validate(tomllib.loads(recipe.read_text()))
  other_save = configs.save.model_copy(
    update={"keep_versions": 5, "storage_options": {"anon": True}}
  )
  other = configs.model_copy(update={"save": other_save})
  assert download_module.recipe_fingerprint(other) == download_module.recipe_fingerprint(configs)
  # Nor the sources' access (anonymous or a profile, an endpoint): the same data, read otherwise.
  data = tomllib.loads(recipe.read_text())
  for dataset in data["datasets"].values():
    for part in dataset["parts"]:
      part["storage_options"] = {"anon": True}
  sources = download_module.DownloadConfig.model_validate(data)
  assert download_module.recipe_fingerprint(sources) == download_module.recipe_fingerprint(configs)


def test_an_extension_refuses_a_main_that_moved_meanwhile(recipe, tmp_path, monkeypatch):
  """Its start was read from `main`: if another run was promoted before this one started, it
  would append days already stored."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  real = download_module._main_tip

  def stale(output_path, configs):
    tip = real(output_path, configs)
    return icechunk_utils.Tip(snapshot="0" * 20, metadata=tip.metadata)

  monkeypatch.setattr(download_module, "_main_tip", stale)
  moved = run(recipe, output, "--extend", f"--end={END}")
  assert moved.exit_code != 0 and "moved while the extension started" in moved.output
  assert open_output(output).sizes["time"] == 2


def test_extension_messages(recipe, tmp_path, monkeypatch):
  """A failed extension says how to continue it; without an end, only --end is asked for."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  no_end = run(recipe, output, "--extend")
  assert no_end.exit_code != 0 and "give --end" in no_end.output and "--start" not in no_end.output
  fail_every_block(monkeypatch)
  failed = run(recipe, output, "--extend", f"--end={END}")
  assert failed.exit_code != 0 and "--extend --resume continues from there" in failed.output


def test_a_recipe_end_in_utc_extends(recipe, tmp_path):
  """TOML dates with an offset are aware: an end in UTC is that stamp (an aware end failed the
  comparison with the extension's start, a TypeError); another offset is refused at load."""
  output = tmp_path / "out"
  assert run(recipe, output, f"--start={START}", f"--end={MIDDLE}").exit_code == 0
  text = recipe.read_text()
  recipe.write_text("end = 2020-03-02T00:00:00Z\n" + text)
  extended = run(recipe, output, "--extend")
  assert extended.exit_code == 0, extended.output
  assert open_output(output).sizes["time"] == 5
  recipe.write_text("end = 2020-03-02T00:00:00+02:00\n" + text)
  refused = run(recipe, tmp_path / "other", f"--start={START}")
  assert refused.exit_code != 0 and "without a time zone, or in UTC" in refused.output


def test_extending_a_repository_without_a_download_points_to_resume(recipe, tmp_path):
  """A first download that failed leaves `main` at the repository's initial snapshot: --replace
  alone, as the message said, would throw its committed blocks away."""
  output = tmp_path / "out"
  icechunk.Repository.create(icechunk_utils.storage(output))
  refused = run(recipe, output, "--extend", f"--end={END}")
  assert refused.exit_code != 0 and "records no download block" in refused.output
  assert "--resume" in refused.output
