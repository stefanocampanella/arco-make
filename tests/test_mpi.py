# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`--dask-setup mpi` end to end, under `mpirun`: skipped without it or without the `mpi` extra
(CI installs both)."""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner

from arcomake import cli
from tests.conftest import open_output

pytestmark = pytest.mark.skipif(
  shutil.which("mpirun") is None or importlib.util.find_spec("dask_mpi") is None,
  reason="needs mpirun and the `mpi` extra",
)

# The command's arguments become click's (`sys.argv[1:]`).
ARCO_MAKE = [sys.executable, "-c", "from arcomake import cli; cli()"]


def test_compute_climatology_on_mpi_writes_what_threads_write(
  tmp_path: pathlib.Path, isolated_tempdir: pathlib.Path
):
  """4 ranks: the scheduler, the command and two workers, which write the temporary store."""
  time = pd.date_range("2019-01-01", "2021-01-01", freq="1D", inclusive="left")
  values = np.random.default_rng(0).standard_normal((time.size, 3, 4)).astype("float32")
  coords = {"time": time, "depth": [0.5, 1.5, 2.5], "lat": [0.0, 1.0, 2.0, 3.0]}
  xr.Dataset({"t": (("time", "depth", "lat"), values)}, coords=coords).to_zarr(tmp_path / "in.zarr")
  recipe = tmp_path / "climatology.toml"
  recipe.write_text(
    'period = 365\nbins = "day_of_year"\nwindow = 5\nweighting = "gaussian"\n\n'
    '[read]\nengine = "zarr"\n\n[save]\nchunks = {time = 73}\n'
  )
  args = [str(recipe), str(tmp_path / "in.zarr")]
  threads = CliRunner().invoke(cli, ["compute-climatology", *args, str(tmp_path / "threads.zarr")])
  assert threads.exit_code == 0, threads.output
  # Open MPI refuses more ranks than cores unless told; MPICH ignores the variable.
  env = os.environ | {"TMPDIR": str(isolated_tempdir), "OMPI_MCA_rmaps_base_oversubscribe": "1"}
  mpi = subprocess.run(
    ["mpirun", "-n", "4", *ARCO_MAKE, "compute-climatology", "--dask-setup", "mpi", *args]
    + [str(tmp_path / "mpi.zarr")],
    env=env,
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert mpi.returncode == 0, mpi.stdout[-3000:] + mpi.stderr[-3000:]
  assert "Using dask-mpi" in mpi.stderr
  for group in ("climatology", "anomaly_std"):
    xr.testing.assert_identical(
      open_output(tmp_path / "mpi.zarr", group).load(),
      open_output(tmp_path / "threads.zarr", group).load(),
    )
