# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`arco-make` (package `arcomake`, Python ≥ 3.13) builds Analysis-Ready Cloud-Optimized (ARCO) Zarr timeseries from heterogeneous sources (Copernicus Marine, Climate Data Store / Early Warning Data Store, remote NetCDF over HTTP, Zarr on GCS/S3). Every command is driven by a TOML "recipe" plus CLI arguments. It was built to produce ARCO-OCEAN; the reference recipe lives in the external `inogs/arco-ocean` repo under `configs/download/`.

## Commands

```bash
uv sync --frozen --extra dev                    # dev tools: pytest, ruff, ty, reuse, pre-commit
uv run --frozen pytest -q                       # offline test suite (synthetic data, no credentials)
uv run --frozen pytest tests/test_cli.py -k blocks   # a single test
uv run --frozen ruff check && uv run --frozen ruff format --check
uv run --frozen ty check                        # all ty rules are errors
uv run --frozen pre-commit run --all-files      # ruff, ty, reuse lint, gitleaks
uv run arco-make --help                         # CLI entry point (arcomake:cli)
```

Tests live in `tests/`: synthetic fixtures in `conftest.py` (grid with a coastline and slope, Feb 29 crossed), sample recipes in `tests/data/` (rendered with `@SOURCES@` replaced), and CLI end-to-end runs via click's `CliRunner`. Known bugs are pinned as `xfail(strict=True)`; the commit fixing a bug removes its marker. `process` and `compute-climatology` default to `--dask-setup mpi`: pass `threads` in tests.

CI (`.github/workflows/ci.yml`): lint, reuse, gitleaks secret scan, tests with a coverage gate of 80 %, on Python 3.13 as in the image. CI never downloads data or uses credentials, by design: the repository and its Actions logs are public, and pull-request code would run with the secrets. Checks against the live services are in `tests/live/` (marker `network`, skipped unless pytest gets `--live`): one tiny recipe per backend, checked for structure and plausible values. `.github/workflows/live-services.yml` runs them weekly, on demand and after merges touching the backends or `uv.lock`, never on pull requests, with the secrets of the `live-services` environment, each backend getting only its own. Locally: `uv run --frozen --extra dev --extra live pytest --live tests/live` with the credentials in the environment or the services' credential files (the `live` extra adds `eccodeslib`, an ecCodes build recent enough for GloFAS GRIB). The Apptainer image (`apptainer-build-deploy.yml`) is built on pull requests and pushed as `:latest` only from master.

Code style: **2-space indentation**, line length 100 (ruff config in `pyproject.toml`). Files start with an SPDX MIT header.

## CLI subcommands

Registered in `src/arcomake/__init__.py`: `download`, `process`, `compute-climatology` (from `climatology.py`), `validate`, and `steps` (prints the step reference). Each data command has a Pydantic config model (`DownloadConfig`, `ProcessConfig`, `ClimatologyConfig`, `ValidateConfig`) loaded via `cli_utils.read_configs(path, schema=..., inject=...)`; CLI options like `--start/--end` are injected into the parsed TOML before validation, and the output check runs after (it uses `[save] storage_options`). Checkpointing is configured as `[datasets.<name>.checkpointing]` with `step` and optional `compressors`.

**Documentation is tested:** `docs/steps.md` must equal `arco-make steps` (regenerate with `uv run arco-make steps > docs/steps.md` after changing a step or its docstring), every TOML example in a step docstring must validate, and the README recipe must load as a `DownloadConfig`.

## Architecture

**Download pipeline** (`download.py` → `dataset_utils.py`): the recipe's `[start, end)` range is split into `step`-sized intervals (`datetime_utils.IterableDateInterval`). For each interval, every dataset in `datasets` is produced by `maybe_checkpointing_download_and_process`: each `part` is opened with `open_input(**part_conf)` (`xr.open_dataset`; with zarr 3 a failed chunk read of a Zarr source raises instead of becoming NaN, N8, as `tests/test_read_errors.py` checks for sources in both formats; the `[read]` tables use the same), time-sliced right-exclusively (`ds.arcomake.time_sel`), merged with `join="exact"`, then the dataset's `postprocess` steps run. If `checkpointing.step` is smaller than the interval, sub-intervals are written to a temp Zarr and re-opened lazily, to bound memory on HPC nodes. All datasets are then merged (`join="exact"`, `combine_attrs="identical"`), the global `postprocess` runs, and `safe_to_zarr` writes the first interval with `mode="w"` and appends later ones along `time_dim`. Output can be a directory or an fsspec URL (`unpack` and zip outputs were removed with the move to Zarr v3).

**Part configs** (`DatasetPartConfig` in `dataset_utils.py`) are a plain Pydantic union of per-engine models. The custom engines are xarray backend entrypoints in `backends.py`, registered through `[project.entry-points."xarray.backends"]` in `pyproject.toml`: `copernicusmarine`, `netcdfoverhttp`, `earlywarningdatastore`. For `copernicusmarine` and `earlywarningdatastore`, `start_datetime`/`end_datetime` are injected per interval. `copernicusmarine` is upstream main pinned at a commit (`f66a8b8`, for `COPERNICUSMARINE_MAX_POOL_CONNECTIONS`, not yet released): move to the first release that has it. The backend passes `chunk_size_limit=False` (undocumented: the toolbox then returns lazily indexed arrays, not Dask arrays of its own chunking, so a part's `chunks` decide the reads; without `chunks` a variable is read in one piece); `tests/live` checks it by calling the backend directly, so run the live checks after any toolbox update. The backend imports the toolbox at first use: it reads the pool size at import, and xarray imports every backend module whenever it lists engines.

**Processing steps** (`steps.py`, `processing_utils.py`, `xarray_accessors.py`): importing `arcomake` registers the `ds.arcomake` accessor. A recipe step `name` is either a function of `processing_utils.py` registered with `@step` (called as `fn(ds, **args)`) or one of the `xr.Dataset` methods in `steps.DATASET_METHODS`; registered functions win (`sel`, `isel`, `transpose`, `astype`). `ProcessingStepConfig` (extra keys = arguments) checks the name, unknown/missing arguments and argument types **when the recipe is loaded**, so typos fail before any download. To add a step: write a pure function `def name(ds: xr.Dataset, ...) -> xr.Dataset` with typed parameters, decorate it with `@step`, and add it to `STEPS` in `tests/test_steps_purity.py` (a test fails otherwise). Steps never modify their input (intent 1, tested for every step).

**Climatology** (`climatology.py`): with `split_dims`, `compute-climatology` computes one series (variable × index of the split dims) at a time: per-bin Welford statistics streamed in blocks of about 1 GB (numba kernel), the window in NumPy over point bands, each series written into its region of outputs whose metadata are written first. Without it, everything is one Dask graph (`_compute_climatology`); the two agree (tested).

**Checks** (`checks.py`): `ds.arcomake.validate(ChecksConfig)` runs the functions in `checks.CHECKS`, one per field of `ChecksConfig`; they raise `checks.ValidationError` on failure. All config models reject unknown keys, except `ProcessingStepConfig` (arguments) and `EnsureNoNansCheck` (mask name → list of variables).

**Dask setup** (`dask_utils.get_client`, a context manager): `--dask-setup` is one of `synchronous|threads|processes|mpi|localcluster`. `download` defaults to `threads` because the custom remote backends don't serialize to distributed workers. `process` and `compute-climatology` default to `mpi` (dask-mpi under SLURM). `mpi` and `localcluster` read extra kwargs from Dask env config (`DASK_MPI__*`, `DASK_LOCALCLUSTER__*`) via `get_dask_env_options`. Optional `frisky` (from the `experimental` extra) can wrap the client.

**User settings** (`settings.py`): `ArcoMakeSettings` (pydantic-settings) reads the environment, then the dotenv file `~/.arcomakerc` (or `$ARCOMAKE_RC`; empty disables it); the `cli` group exports the values the environment lacks, because the libraries read only the environment (`TMPDIR`, `AWS_PROFILE`/`AWS_DEFAULT_PROFILE`, Copernicus Marine and CDS/EWDS credentials, `COPERNICUSMARINE_MAX_POOL_CONNECTIONS`; `DASK_*` keys go to Dask). Runs with `s3://` paths or Copernicus Marine parts fail at once if botocore's profile (`AWS_DEFAULT_PROFILE` first, then `AWS_PROFILE`) is not configured (`cli_utils.check_aws_profile`, also in the path checks of arguments). Errors name variables, never values (`settings.SettingsError`). Secrets are `SecretStr`; unknown keys are ignored with a warning naming the key. To support a new setting, add a field with the variable as `validation_alias`. Such files hold secrets: never read a real one; tests point `ARCOMAKE_RC` at temporary files (an autouse fixture disables it, except for live checks).

**Temp storage** (`temporary_utils.TempDirectoryRegistry`): a context-var-scoped registry of temp dirs, cleaned up on exit, atexit and signals, with stale-dir detection by PID. Checkpoints and remote downloads use it heavily, so `TMPDIR` should point to disk, not RAM.

**Zarr v3**: outputs and checkpoints are written in Zarr format 3 with zarr-python 3 (`dataset_utils._write_zarr`); there is no support for writing format 2. Sources in either format are read. Recipes give compressors as Zarr v3 codec metadata, `compressors = [{name = "blosc", configuration = {...}}]` (`CodecConfig`, built when the recipe is loaded). Codecs inherited from format 2 sources (numcodecs objects in the encodings) are dropped before writing (`_without_zarr2_codecs`): zarr 3 cannot write them in format 3. Consolidated metadata are written by default (maintainer's choice), without zarr's warning that they are not in the v3 specification; xarray stores a float `_FillValue` base64-encoded in format 3, apart from the array's own fill value.

## Working rules (from the maintainer)

- **Local notes:** machine-specific paths, host names and file names live in `CLAUDE.local.md` (gitignored, not committed). Keep them out of committed files, this one and `docs/logbook.md` included.
- **Secrets:** a credentials file on the project machine (named in `CLAUDE.local.md`) holds credentials. Never read, print or copy it. Source it only in the shell of the command that needs it (`set -a; . FILE; set +a`), without `set -x`, `env` or `printenv`. Commit files by name (never `git add -A`) and read `git diff --cached --stat` first. Logs may contain tokens: keep them out of the repo and read them through redacting filters.
- **Commits:** one purpose each; tests pass after each; a fix comes with its test; a commit that changes output for an unchanged recipe starts with `output:`. Nothing specific to ARCO-OCEAN or to a cluster goes into arco-make.
- **Real-data baseline:** a directory outside the repo (location and scripts in `CLAUDE.local.md`) holds the frozen code per commit (from `git archive`, each with its own venv), outputs per window, logs with `/usr/bin/time -v`, and a pinned copy of the 0.25° levels-50 recipe. "Same output" is checked bit for bit, metadata and encodings included. Runs use the image's eccodes 2.48 (via `ECCODES_DIR` and `ECCODES_DEFINITION_PATH`): older host eccodes cannot read GloFAS GRIB. Run a watchdog next to long runs: a hang (N7) sends no notification.
- **Experiments:** for distributed runs (`localcluster`/`mpi`), `--frisky` (the `experimental` extra) gives readable diagnostics. It is experimental and closed-source, so treat what it shows as a lead to confirm.

## Download performance (to revisit before a production download)

The maintainer will revisit performance after all planned changes, aiming at the shortest full download; details and figures in `docs/logbook.md` (16 Nov). In short, on the project VM:
- A full 0.25° levels-50 download took 166 h of running time and 7.8 days of calendar time (manual restarts, stalls), 46.6 s per day of data. The output resolution does not change the time: 10-level runs at 0.25°, 0.5° and 1.0° took the same.
- GLORYS 3D is about 75 % of the time. One process keeps 2–6 of 32 cores busy, and two processes each keep their speed: the limit is how the work of one process overlaps (its Dask graph), not CPU or network. A larger Copernicus Marine pool (32) cuts reading by about 20 % but processing by only about 5 %.
- Expected: about 6 days per resolution with group 7 (no manual restarts), about 3–3.5 with two blocks at once (memory allows two).
- ERA5 stays on Google's GCS store (maintainer's decision). Two consecutive 600 s read timeouts on one baseline window (Feb–Mar 2020) remain unexplained; group 7's retries are the remedy. The GloFAS static file is fetched again in every block, so a server outage (502 on 16 Nov) fails the block it hits.

## Findings for Moby-Dick

A report for Moby-Dick (the consumer of ARCO-OCEAN) is due after the PRs are merged. Record every finding that affects consumers of the data here, and with details in `docs/logbook.md` (entries tagged **[Moby-Dick]**). So far:
- The published 0.25° dataset was downloaded with arco-make `6f0e2f1` (image of 30 Sep 2026), not `847b6f2`. Six pieces were rerun after network failures.
- N1 (uncompressed output without `[save] compressor`) and N2 (`astype` pitfalls) do not affect the published data: the arco-ocean recipes set lz4 and both `astype` arguments.
- N6 (EWDS requests of different months opened GRIB concurrently, which crashed ecCodes) can abort a download piece; it cannot write wrong data. Hypothesis, not checkable: some of the six rerun pieces of the 0.25° download failed this way.
- N7: a download can hang forever on a read from Google Cloud Storage (ERA5): gcsfs sends requests with no timeout by default (seen twice on the baseline). A killed piece leaves a partial store; it is safe only if rerun with `--overwrite`. Do not set `requests_timeout` in recipes run with code older than `f9fcd45` (N8).
- N8: a source chunk whose read fails with a timeout or connection error (any `OSError`) is read as NaN by zarr 2 `FSStore`, silently; a daily mean then averages the remaining hours and shows no NaN. Seen on real ERA5 data (two hours lost, bit-exact) and reproduced offline. Published ERA5 days may be means over fewer than 24 hours wherever a connection error outlasted gcsfs's retries; only recomputation from the source can tell. Fixed for Zarr inputs in `f9fcd45` (`dataset_utils.strict_zarr_store`, used by `open_input` and `open_archive`).
- N9: the Copernicus Marine fork's Zarr store turns any S3 error response into a missing chunk (NaN) after 9 retries (about 8.5 min). Code reading only; fix belongs in the fork. Exposes every Copernicus Marine variable to the same silent loss under a long outage.
- N10: `[save] chunks = -1` over Dask chunks smaller than the dimension wrote overlapping chunks concurrently and lost most values, silently (since before `847b6f2`; fixed: such a write now fails). ARCO-OCEAN is not affected: its recipes make `lat` and `lon` whole Dask chunks before saving. Other users of arco-make might be.
- Peak memory of the 0.25° levels-50 download: 36–53 GB for 2–5 days on 32 threads.
- B6, measured on synthetic data: a missing year biases the anomaly variance of its bin low (6 % with 1 of 5 years missing). An all-NaN bin spreads NaN to its window neighbours. Fixed in group 6, with Feb 29 in Feb 28's bin (`bins = "day_of_year"`) and a `reference_period` setting (arco-ocean: 1993–2018): the climatology and anomaly_std have to be recomputed.
