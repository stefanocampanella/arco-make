# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`ecmwfopendata`: forecast runs of ECMWF's open data, read field by field through the runs'
indexes, from a local copy of the tree written here with ecCodes (as on data.ecmwf.int and AWS's
`s3://ecmwf-forecasts`: a GRIB2 file per step, a JSON-lines `.index` beside it)."""

import datetime
import json
import pathlib

import numpy as np
import pytest
import xarray as xr

from arcomake.backends import ECMWFOpenData, RunUnavailable
from arcomake.temporary_utils import TempDirectoryRegistry

eccodes = pytest.importorskip("eccodes")

RUN = datetime.datetime(2026, 11, 20)
HOUR = datetime.timedelta(hours=1)
# IFS: 3-hourly steps to 144 h, then 6-hourly (here shortened: 3-hourly to 12 h, then 6-hourly).
STEPS = [0, 3, 6, 9, 12, 18, 24, 30, 36]


def _message(param: str, step: int, values: np.ndarray, level: int | None = None) -> bytes:
  sample = "regular_ll_sfc_grib2" if level is None else "regular_ll_pl_grib2"
  handle = eccodes.codes_grib_new_from_samples(sample)
  try:
    grid = {
      "Ni": 4,
      "Nj": 3,
      "latitudeOfFirstGridPointInDegrees": 1.0,
      "longitudeOfFirstGridPointInDegrees": 0.0,
      "latitudeOfLastGridPointInDegrees": -1.0,
      "longitudeOfLastGridPointInDegrees": 3.0,
      "iDirectionIncrementInDegrees": 1.0,
      "jDirectionIncrementInDegrees": 1.0,
      "dataDate": int(f"{RUN:%Y%m%d}"),
      "dataTime": 0,
    }
    for key, value in grid.items():
      eccodes.codes_set(handle, key, value)
    eccodes.codes_set(handle, "shortName", param)
    if level is not None:
      eccodes.codes_set(handle, "level", level)
    if eccodes.codes_get(handle, "stepType") == "accum":
      eccodes.codes_set(handle, "stepRange", f"0-{step}")
    else:
      eccodes.codes_set(handle, "step", step)
    eccodes.codes_set_values(handle, values.ravel())
    return eccodes.codes_get_message(handle)
  finally:
    eccodes.codes_release(handle)


def accumulated(step: int) -> float:
  """An accumulation since the run's start that grows unevenly: its increments tell which steps a
  sum took."""
  return float(step**2)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
  """A run of `RUN` with 2t (the step, in kelvin) and tp (`accumulated`), plus msl at some steps
  only, and a field of another level type, in a copy of the open-data tree."""
  try:
    _message("2t", 0, np.zeros((3, 4)))
  except Exception as error:  # an ecCodes without samples, or none
    pytest.skip(f"ecCodes cannot write GRIB here: {error}")
  root = tmp_path / "ecmwf-forecasts"
  run_dir = root / f"{RUN:%Y%m%d}" / "00z" / "ifs" / "0p25" / "oper"
  run_dir.mkdir(parents=True)
  for step in STEPS:
    fields = [
      ("2t", "sfc", np.full((3, 4), float(step)), None),
      ("tp", "sfc", np.full((3, 4), accumulated(step)), None),
      # Another level type, never read as sfc: two levels, so two messages per step.
      ("t", "pl", np.full((3, 4), 250.0), 500),
      ("t", "pl", np.full((3, 4), 280.0), 850),
    ]
    if step % 6 == 0:
      fields.append(("msl", "sfc", np.full((3, 4), 1e5), None))
    name = run_dir / f"{RUN:%Y%m%d%H}0000-{step}h-oper-fc"
    blob, lines = b"", []
    for param, levtype, values, level in fields:
      message = _message(param, step, values, level)
      lines.append(
        json.dumps(
          {"param": param, "levtype": levtype, "step": str(step), "_offset": len(blob)}
          | {"_length": len(message)}
        )
      )
      blob += message
    name.with_suffix(".grib2").write_bytes(blob)
    name.with_suffix(".index").write_text("\n".join(lines) + "\n")
  # A GRIB file without its index is not a step of the run (still being published).
  (run_dir / f"{RUN:%Y%m%d%H}0000-42h-oper-fc.grib2").write_bytes(b"")
  return root


def _open(root: pathlib.Path, start, end, **kwargs) -> xr.Dataset:
  with TempDirectoryRegistry(base_dir=root.parent / "registry") as registry:
    out = ECMWFOpenData().open_dataset(
      str(root), run_datetime=RUN, start_datetime=start, end_datetime=end, **kwargs
    )
    assert not any(registry.job_dir.rglob("*.grib2"))  # fetched fields are removed at once
  return out


def test_a_run_along_valid_time(tree: pathlib.Path):
  """The steps whose valid times are in [start, end] (the end included), fields of `levtype`
  only, along `time`; cfgrib's names (`t2m` for "2t")."""
  out = _open(tree, RUN, RUN + 24 * HOUR, param=["2t"])
  hours = (out.time.values - np.datetime64(RUN)) / np.timedelta64(1, "h")
  assert hours.tolist() == [0, 3, 6, 9, 12, 18, 24]
  assert list(out.data_vars) == ["t2m"]
  assert out.t2m.dims == ("time", "latitude", "longitude")
  # cfgrib's history names the temporary file: datasets of one run would not merge, and the
  # output would publish a local path.
  assert "history" not in out.attrs and out.attrs["GRIB_centre"] == "ecmf"
  assert out.t2m.mean(["latitude", "longitude"]).values.tolist() == hours.tolist()


def test_accumulations_become_increments_stamped_at_their_start(tree: pathlib.Path):
  """An accumulation since the run's start becomes the increment to the next step, stamped at
  the start of its interval: a daily sum is then exactly the accumulation at the day's end minus
  that at its start, whatever the steps; the last stamp has no increment."""
  out = _open(tree, RUN, RUN + 36 * HOUR, param="tp")
  increments = out.tp.isel(latitude=0, longitude=0).values
  steps = [0, 3, 6, 9, 12, 18, 24, 30, 36]
  expected = [accumulated(b) - accumulated(a) for a, b in zip(steps, steps[1:], strict=False)]
  np.testing.assert_allclose(increments[:-1], expected)
  assert np.isnan(increments[-1])
  day = out.tp.sel(time=slice(RUN, RUN + 23 * HOUR)).sum("time").values
  np.testing.assert_allclose(day, accumulated(24) - accumulated(0))


def test_accumulations_kept_without_deaccumulate(tree: pathlib.Path):
  out = _open(tree, RUN, RUN + 12 * HOUR, param="tp", deaccumulate=False)
  assert out.tp.isel(latitude=0, longitude=0).values.tolist() == [
    accumulated(s) for s in (0, 3, 6, 9, 12)
  ]


def test_a_window_inside_the_run(tree: pathlib.Path):
  """Valid times from a later start read the steps from there only."""
  out = _open(tree, RUN + 12 * HOUR, RUN + 30 * HOUR, param=["2t", "tp"])
  hours = (out.time.values - np.datetime64(RUN)) / np.timedelta64(1, "h")
  assert hours.tolist() == [12, 18, 24, 30]
  assert set(out.data_vars) == {"t2m", "tp"}


def test_a_run_not_there_is_unavailable(tree: pathlib.Path):
  """A run the copy does not have (not published yet, older than the archive) raises
  `RunUnavailable`: the forecast download leaves the dataset out of that run."""
  with (
    pytest.raises(RunUnavailable, match="2026-11-21"),
    TempDirectoryRegistry(base_dir=tree.parent / "registry"),
  ):
    ECMWFOpenData().open_dataset(
      str(tree),
      run_datetime=RUN + 24 * HOUR,
      start_datetime=RUN + 24 * HOUR,
      end_datetime=RUN + 48 * HOUR,
      param="2t",
    )


def test_valid_times_beyond_the_run_fail(tree: pathlib.Path):
  """A window past the run's last step is a recipe's mistake (a horizon the model does not
  reach): it fails, naming the steps."""
  with pytest.raises(ValueError, match="steps to 36 h"):
    _open(tree, RUN, RUN + 48 * HOUR, param="2t")


def test_a_field_missing_at_some_steps_fails(tree: pathlib.Path):
  """A field the run lacks at some steps (msl here only every 6 hours) fails, naming it."""
  with pytest.raises(ValueError, match="msl"):
    _open(tree, RUN, RUN + 12 * HOUR, param=["2t", "msl"])


def test_fields_on_several_levels(tree: pathlib.Path):
  """A field with several messages per step (pressure levels; ensemble members alike) is read
  whole: a level dimension, as cfgrib makes it."""
  out = _open(tree, RUN, RUN + 12 * HOUR, param="t", levtype="pl")
  assert out.t.sizes["isobaricInhPa"] == 2
  assert sorted(out.t.isel(time=0, latitude=0, longitude=0).values.tolist()) == [250.0, 280.0]


def test_a_step_missing_inside_the_run_fails(tree: pathlib.Path):
  """A step missing among those read (a file being published, or lost) fails: a daily mean would
  average fewer samples, silently. (One missing where the spacing grows looks like the change of
  spacing: not seen.)"""
  run_dir = tree / f"{RUN:%Y%m%d}" / "00z" / "ifs" / "0p25" / "oper"
  (run_dir / f"{RUN:%Y%m%d%H}0000-6h-oper-fc.index").unlink()
  with pytest.raises(ValueError, match="lacks a step between 3 and 9 h"):
    _open(tree, RUN, RUN + 24 * HOUR, param="2t")
  (run_dir / f"{RUN:%Y%m%d%H}0000-30h-oper-fc.index").unlink()
  with pytest.raises(ValueError, match="lacks a step between 24 and 36 h"):
    _open(tree, RUN + 12 * HOUR, RUN + 36 * HOUR, param="2t")


def test_a_block_ends_on_a_step(tree: pathlib.Path):
  """A block whose end is not a step would leave its last increment NaN inside the block."""
  with pytest.raises(ValueError, match="no step at"):
    _open(tree, RUN, RUN + 15 * HOUR, param="tp")


@pytest.mark.parametrize(
  "options", [{"stream": "opr"}, {"resolution": "0p4"}, {"type": "ef"}], ids=str
)
def test_a_recipe_mistake_is_not_a_missing_run(tree: pathlib.Path, options):
  """A run that is there, but not with the recipe's model, resolution, stream or type, is a
  recipe's mistake: it fails, instead of becoming NaN as a run not published."""
  with pytest.raises(ValueError, match="check"):
    _open(tree, RUN, RUN + 12 * HOUR, param="2t", **options)


def test_steps_need_their_index():
  """Steps are the run's GRIB files that have their index beside them."""
  prefix = f"{RUN:%Y%m%d%H}0000"
  listed = [
    f"x/{prefix}-0h-oper-fc.grib2",
    f"x/{prefix}-0h-oper-fc.index",
    f"x/{prefix}-3h-oper-fc.grib2",
    f"x/{prefix}-144h-oper-fc.grib2",
    f"x/{prefix}-144h-oper-fc.index",
    f"x/{prefix}-6h-enfo-ef.grib2",
    f"x/{prefix}-6h-enfo-ef.index",
  ]
  assert ECMWFOpenData._steps(listed, prefix, "oper", "fc") == [0, 144]
