# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`arco-make` (package `arcomake`, Python ≥ 3.13) builds Analysis-Ready Cloud-Optimized (ARCO) Zarr timeseries from heterogeneous sources (Copernicus Marine, Climate Data Store / Early Warning Data Store, remote NetCDF over HTTP, Zarr on GCS/S3). Every command is driven by a TOML "recipe" plus CLI arguments. It was built to produce ARCO-OCEAN; the reference recipe lives in the external `inogs/arco-ocean` repo under `configs/download/`.

## Commands

```bash
uv sync --frozen --extra dev                    # dev tools: pytest, ruff, ty, reuse, pre-commit
uv run --frozen pytest -q                       # offline test suite (synthetic data, no credentials)
uv run --frozen pytest tests/test_cli.py -k unpack   # a single test
uv run --frozen ruff check && uv run --frozen ruff format --check
uv run --frozen ty check                        # all ty rules are errors
uv run --frozen pre-commit run --all-files      # ruff, ty, reuse lint, gitleaks
uv run arco-make --help                         # CLI entry point (arcomake:cli)
```

Tests live in `tests/`: synthetic fixtures in `conftest.py` (grid with a coastline and slope, Feb 29 crossed), sample recipes in `tests/data/` (rendered with `@SOURCES@` replaced), and CLI end-to-end runs via click's `CliRunner`. Known bugs are pinned as `xfail(strict=True)`; the commit fixing a bug removes its marker. `unpack`, `process` and `compute-climatology` default to `--dask-setup mpi`: pass `threads` in tests.

CI (`.github/workflows/ci.yml`): lint, reuse, gitleaks secret scan, tests with a coverage gate of 80 %. The Apptainer image build is separate.

Code style: **2-space indentation**, line length 100 (ruff config in `pyproject.toml`). Files start with an SPDX MIT header.

## CLI subcommands

Registered in `src/arcomake/__init__.py`: `download`, `unpack`, `process`, `compute-climatology` (from `climatology.py`), `validate`, and `steps` (prints the step reference). Each data command has a Pydantic config model (`DownloadConfig`, `UnpackConfig`, `ProcessConfig`, `ClimatologyConfig`, `ValidateConfig`) loaded via `cli_utils.read_configs(path, schema=..., inject=...)`; CLI options like `--start/--end` are injected into the parsed TOML before validation, and the output check runs after (it uses `[save] storage_options`). Checkpointing is configured as `[datasets.<name>.checkpointing]` with `step` and optional `compressor`.

**Documentation is tested:** `docs/steps.md` must equal `arco-make steps` (regenerate with `uv run arco-make steps > docs/steps.md` after changing a step or its docstring), every TOML example in a step docstring must validate, and the README recipe must load as a `DownloadConfig`.

## Architecture

**Download pipeline** (`download.py` → `dataset_utils.py`): the recipe's `[start, end)` range is split into `step`-sized intervals (`datetime_utils.IterableDateInterval`). For each interval, every dataset in `datasets` is produced by `maybe_checkpointing_download_and_process`: each `part` is opened with `open_input(**part_conf)` (an `xr.open_dataset` that opens remote Zarr through `strict_zarr_store`, so a failed chunk read raises instead of becoming NaN; `open_archive` and the `[read]` tables use the same), time-sliced right-exclusively (`ds.arcomake.time_sel`), merged with `join="exact"`, then the dataset's `postprocess` steps run. If `checkpointing.step` is smaller than the interval, sub-intervals are written to a temp Zarr and re-opened lazily, to bound memory on HPC nodes. All datasets are then merged (`join="exact"`, `combine_attrs="identical"`), the global `postprocess` runs, and `safe_to_zarr` writes the first interval with `mode="w"` and appends later ones along `time_dim`. Output can be a directory, a `.zip` (ZipStore), or an fsspec URL.

**Part configs** (`DatasetPartConfig` in `dataset_utils.py`) are a plain Pydantic union of per-engine models. The custom engines are xarray backend entrypoints in `backends.py`, registered through `[project.entry-points."xarray.backends"]` in `pyproject.toml`: `copernicusmarine`, `netcdfoverhttp`, `earlywarningdatastore`. For `copernicusmarine` and `earlywarningdatastore`, `start_datetime`/`end_datetime` are injected per interval. `copernicusmarine` is a fork pinned to a git branch (`stefanocampanella/copernicus-marine-toolbox@arco-make`).

**Processing steps** (`steps.py`, `processing_utils.py`, `xarray_accessors.py`): importing `arcomake` registers the `ds.arcomake` accessor. A recipe step `name` is either a function of `processing_utils.py` registered with `@step` (called as `fn(ds, **args)`) or one of the `xr.Dataset` methods in `steps.DATASET_METHODS`; registered functions win (`sel`, `isel`, `transpose`, `astype`). `ProcessingStepConfig` (extra keys = arguments) checks the name, unknown/missing arguments and argument types **when the recipe is loaded**, so typos fail before any download. To add a step: write a pure function `def name(ds: xr.Dataset, ...) -> xr.Dataset` with typed parameters, decorate it with `@step`, and add it to `STEPS` in `tests/test_steps_purity.py` (a test fails otherwise). Steps never modify their input (intent 1, tested for every step).

**Checks** (`checks.py`): `ds.arcomake.validate(ChecksConfig)` runs the functions in `checks.CHECKS`, one per field of `ChecksConfig`; they raise `checks.ValidationError` on failure. All config models reject unknown keys, except `ProcessingStepConfig` (arguments) and `EnsureNoNansCheck` (mask name → list of variables).

**Dask setup** (`dask_utils.get_client`, a context manager): `--dask-setup` is one of `synchronous|threads|processes|mpi|localcluster`. `download` defaults to `threads` because the custom remote backends don't serialize to distributed workers. `unpack` and `compute-climatology` default to `mpi` (dask-mpi under SLURM). `mpi` and `localcluster` read extra kwargs from Dask env config (`DASK_MPI__*`, `DASK_LOCALCLUSTER__*`) via `get_dask_env_options`. Optional `frisky` (from the `experimental` extra) can wrap the client.

**Temp storage** (`temporary_utils.TempDirectoryRegistry`): a context-var-scoped registry of temp dirs, cleaned up on exit, atexit and signals, with stale-dir detection by PID. Checkpoints and remote downloads use it heavily, so `TMPDIR` should point to disk, not RAM.

**Zarr v2 only**: dependencies pin `zarr >=2,<3` and `numcodecs <0.16`, and the code uses `zarr.storage.DirectoryStore`/`ZipStore` and `numcodecs.Blosc`. Porting to v3 is a TODO.

## Working rules (from the maintainer)

- **Secrets:** a credentials file on the project machine holds credentials. Never read, print or copy it. Source it only in the shell of the command that needs it (`set -a; . FILE; set +a`), without `set -x`, `env` or `printenv`. Commit files by name (never `git add -A`) and read `git diff --cached --stat` first. Logs may contain tokens: keep them out of the repo and read them through redacting filters.
- **Commits:** one purpose each; tests pass after each; a fix comes with its test; a commit that changes output for an unchanged recipe starts with `output:`. Nothing specific to ARCO-OCEAN or to a cluster goes into arco-make.
- **Real-data baseline:** a directory outside the repo. It holds the frozen code per commit (`code-<sha>`, from `git archive`, with its own venv), outputs per window, logs with `/usr/bin/time -v`, and the pinned recipe copy. Runs use the image's eccodes 2.48 (via `ECCODES_DIR` and `ECCODES_DEFINITION_PATH`): the host's eccodes 2.45 cannot read GloFAS GRIB.
- **Experiments:** for distributed runs (`localcluster`/`mpi`), `--frisky` (the `experimental` extra) gives readable diagnostics. It is experimental and closed-source, so treat what it shows as a lead to confirm.

## Findings for Moby-Dick

A report for Moby-Dick (the consumer of ARCO-OCEAN) is due after the PRs are merged. Record every finding that affects consumers of the data here, and with details in `docs/logbook.md` (entries tagged **[Moby-Dick]**). So far:
- The published 0.25° dataset was downloaded with arco-make `6f0e2f1` (image of 30 Sep 2026), not `847b6f2`. Six pieces were rerun after network failures.
- N1 (uncompressed output without `[save] compressor`) and N2 (`astype` pitfalls) do not affect the published data: the arco-ocean recipes set lz4 and both `astype` arguments.
- N6 (EWDS requests of different months opened GRIB concurrently, which crashed ecCodes) can abort a download piece; it cannot write wrong data. Hypothesis, not checkable: some of the six rerun pieces of the 0.25° download failed this way.
- N7: a download can hang forever on a read from Google Cloud Storage (ERA5): gcsfs sends requests with no timeout by default (seen twice on the baseline). A killed piece leaves a partial store; it is safe only if rerun with `--overwrite`. Do not set `requests_timeout` in recipes run with code older than `efd3d60` (N8).
- N8: a source chunk whose read fails with a timeout or connection error (any `OSError`) is read as NaN by zarr 2 `FSStore`, silently; a daily mean then averages the remaining hours and shows no NaN. Seen on real ERA5 data (two hours lost, bit-exact) and reproduced offline. Published ERA5 days may be means over fewer than 24 hours wherever a connection error outlasted gcsfs's retries; only recomputation from the source can tell. Fixed for Zarr inputs in `efd3d60` (`dataset_utils.strict_zarr_store`, used by `open_input` and `open_archive`).
- N9: the Copernicus Marine fork's Zarr store turns any S3 error response into a missing chunk (NaN) after 9 retries (about 8.5 min). Code reading only; fix belongs in the fork. Exposes every Copernicus Marine variable to the same silent loss under a long outage.
- Peak memory of the 0.25° levels-50 download: 36–53 GB for 2–5 days on 32 threads.
- B6, measured on synthetic data: a missing year biases the anomaly variance of its bin low (6 % with 1 of 5 years missing). An all-NaN bin spreads NaN to its window neighbours.
