# Logbook

Dated records of what was run, measured or decided while working on arco-make. Newest last.
Entries marked **[Moby-Dick]** are relevant to consumers of ARCO-OCEAN and feed the report due
after the correctness PRs are merged.

## 2026-11-13: correctness PR (groups 0–5 and 10 of the findings guide)

**Decisions (maintainer):**
- Scope: groups 0–5 and 10 now; climatology (6), blocks (7), chunk planner (8) and recipe
  capabilities (9) in follow-up PRs.
- `gaussian_blur_extrapolate` is removed, not fixed. `diff_std` is removed (tendency std moves to
  Moby-Dick).
- Feb 29, for group 6: keep 365 bins with a documented Feb 28 mapping.

**Provenance of the published 0.25° dataset [Moby-Dick]:**
- The 0.25° levels-50 download (30 Sep – 8 Oct 2026, on the project VM) ran in the cached
  Apptainer image of 30 Sep. Its `src/arcomake` is identical to commit **`6f0e2f1`**
  (2026-09-30), not to `847b6f2`. `6f0e2f1..847b6f2` changes `download.py`, `dataset_utils.py`,
  `climatology.py` and the Dask setup.
- It ran as 45 pieces of 256 days, about 3.3 h each and 165 h in total. Pieces 1, 16, 19, 20, 34 and
  41 failed on the first attempt and were rerun with `--overwrite` (7–8 Oct); 41 needed a third try.
  The errors in the logs are transient network ones: EWDS retries, DNS failures, connect timeouts.
- In the arco-ocean checkout, the logs of that 0.25° run sit in a folder named for the 0.5° recipe
  (machine details: `CLAUDE.local.md`, not committed).

**Environment:**
- The host's apt eccodes 2.45 cannot decode the GloFAS GRIB served by EWDS (`Unable to find
  template localSection from grib2/local.98.65535.def`, then `KeyError: 'gridType'`). The image
  builds eccodes 2.48 from source, which works. Local runs use the image's library and
  definitions via `ECCODES_DIR`, `ECCODES_DEFINITION_PATH` and `ECCODES_SAMPLES_PATH`.

**Real-data baseline at `847b6f2`** (a pinned copy of the 0.25° levels-50 recipe, on the project VM
with 32 cores and 110 GB, the image's eccodes 2.48, threads scheduler; locations in
`CLAUDE.local.md`, not committed):

| Window | Result | Wall time | Peak RSS | Output |
|---|---|---|---|---|
| 1993-01-01 → 01-03 | ok | 22 min | 36 GB | 832 MB |
| 2012-12-30 → 2013-01-02 | attempt 1: `ConnectTimeout` on `confluence.ecmwf.int` (GloFAS `uparea`, NetCDF over HTTP: no retry) after 25 min; attempt 2: **segmentation fault** after 22 s; attempts 3 and 4: **hung** (no progress for 2 h 11 min and for 15 min, stopped by hand); attempt 5, with a GCS timeout: ok, but **two ERA5 hours silently lost** (N8) | 5 min | 31 GB | 1.3 GB |
| 2020-02-26 → 03-02 | ok | 3 min | 53 GB | 2.1 GB |

- Wall time is dominated by the remote services' queues (EWDS above all) and varies by an order of
  magnitude between runs. Peak memory is the meaningful number: 36–53 GB for 2–5 days. The
  recipe header's "32 CPUs and about 110 GB" is consistent with this; `download.slurm`'s 8 CPUs and
  30 GB is not.
- **N6, new bug:** the segfault (`fatal flex scanner internal error--end of buffer missed`) came from
  ecCodes. The EWDS backend sends one request per month on a thread pool, and each thread opens its
  GRIB files with cfgrib. cfgrib locks reads but not opens, and ecCodes' definition parser is not
  thread-safe. So any interval spanning two or more months can crash; the 2020 window, also
  spanning two months, happened not to.
- **[Moby-Dick]** The production pieces of 256 days always span several months. *Hypothesis:* some of
  the pieces that failed and were rerun in the 0.25° download (1, 16, 19, 20, 34, 41) crashed this
  way. Their logs were overwritten by the reruns, so this cannot be checked. A crash aborts the
  piece and writes nothing wrong, so the published data are not affected.
- The ConnectTimeout is the failure class seen in production: a transient error aborts the whole
  block (group 7).
- **Hang, a third failure class (N7):** attempts 3 and 4 both stopped about two minutes in, while
  writing, with 1.3 GB on disk: every array was complete except ERA5 `10v`, which had no chunk.
  The process then sat at 0 % CPU with no I/O (stopped by hand after 2 h 11 min and 15 min).
  - A py-spy dump of attempt 4: the main thread waits in `Dataset.to_zarr` for Dask; two Dask
    workers wait in `fsspec.asyn.sync` (`Event.wait`, no timeout) on `FSMap.getitems` of the ERA5
    store on Google Cloud Storage; the fsspec event loop is idle in `epoll`. Nothing deadlocked: a
    read never got an answer and nothing gave up on it. (The 30 CLOSE-WAIT connections to Copernicus
    Marine are idle leftovers.)
  - Cause: gcsfs (2026.10.0) passes `requests_timeout=None` to every aiohttp request by default,
    and aiohttp (3.14.5) turns an explicit `None` into `ClientTimeout(total=None)`, which removes
    its own 5-minute default. Any GCS read can wait forever, at any arco-make commit.
  - Setting `requests_timeout` in the recipe is **not** a fix with code older than `f9fcd45`: see
    N8. Without the N8 fix, a timeout turns the hang into silent data loss.
  - The PR's code finished this window once (2 min 20 s); it has the same dependencies, so it is
    exposed the same way.
- **N8, silent loss of source chunks (all commits) [Moby-Dick]:** attempt 5 ran the old code with
  `requests_timeout = 120` on the ERA5 parts (a copy of the pinned recipe, nothing else changed).
  It finished in 5 min 20 s, 31 GB, exit 0, with no error or warning in the log. Its output differs
  from the PR's in two fields only: `2t` on 2012-12-30 (max 0.70 K, mean 0.043 K) and `msl` on
  2013-01-01 (max 74 Pa, mean 7.2 Pa).
  - Recomputed from the public ERA5 store: the PR's two fields equal the 24-hour means bit for bit;
    the old run's are, bit for bit, the means without 09:00 (`2t`) and without 05:00 (`msl`).
  - Mechanism: zarr 2.18's `FSStore.getitems` treats `KeyError`, `PermissionError` and `IOError` as
    "chunk missing" and returns the fill value (NaN). `TimeoutError`, aiohttp's `ClientOSError` and
    `ServerTimeoutError` are all `OSError`. So a chunk read that times out, or fails with a
    connection error once gcsfs's retries are spent, becomes NaN with no message. The daily
    `resample` mean skips NaN, so the output is a mean over 23 hours and has no NaN to detect.
  - Reproduced offline with the PR's environment: a `TimeoutError` on one hourly chunk of a
    memory-filesystem Zarr store gives NaN at that hour and a wrong mean, silently.
  - Exposure: every source opened with `engine = "zarr"` over fsspec (ERA5 in ARCO-OCEAN). The
    published download ran with no timeout, so the trigger there is a connection error that
    outlasts gcsfs's 6 retries. Days without such an error are exact.
  - Fixed in this PR (`f9fcd45`, maintainer's decision): remote Zarr inputs (download parts,
    `unpack` pieces, `[read]` tables) are opened as `FSStore(..., exceptions=(KeyError,))`, so a
    failed read raises; a missing object is still a fill value. Tests in
    `tests/test_read_errors.py`. Timeouts and retries per block stay in group 7: with this fix, a
    timeout aborts the piece instead of losing data. Window 1 with `f9fcd45` and the pinned recipe:
    identical to `847b6f2` (37 arrays, bit for bit), 3 min 7 s, 37 GB.
  - Side effect: on S3 buckets that answer 403 for a missing object (no list permission), a
    sparse store now fails instead of reading fill values. No ARCO-OCEAN source is affected (ERA5
    on GCS answers 404).
- **N9, the Copernicus Marine toolbox has the same flaw [Moby-Dick]:** the fork used by arco-make
  (`copernicusmarine` 2.4.1, branch `arco-make`, `3009add`) reads Zarr through its own
  `CustomS3StoreZarrV2`. Its `__getitem__` turns every S3 error response (`ClientError`: 403,
  500, 503 SlowDown...) into `KeyError`, which zarr reads as a missing chunk. Its retry wrapper
  tries 9 times over about 8.5 minutes first, so only a long outage or expired credentials during
  a read would lose data. Connection errors (no response) are raised. Seen in the code, not in a
  run. Fix belongs in the fork: raise unless the error is 404/`NoSuchKey`. Exposed: every
  Copernicus Marine variable (GLORYS12, WAVERYS, ...).
- **Window 2, result:** apart from those two fields, explained by the old run's lost chunks, the
  37 arrays are identical. The PR's output matches the source where the old one does not.
- **[Moby-Dick]** A production piece could hang the same way rather than fail. A killed piece leaves
  a partial store that must be rewritten with `--overwrite`, as the failed pieces were; the cost is
  lost time, not wrong data, provided every killed piece was rerun.
- **Comparison, `b6b2cd7` vs `847b6f2` on window 1:** the array metadata were identical (dtype,
  chunks, compressor, fill value, attributes), but ERA5 values (`2t`, `msl`, `ssrd`...) and a few
  wave points differed by a few float32 ULP, e.g. 1.2e-4 K on `2t`.
  - Cause: the B5 fix selected the time range with an index array. That rechunked the hourly
    source (24 → 16 steps per chunk), so the daily resample summed in another order.
  - Fixed in `210bdbd` (slice selection), with a test on the chunks.
  - Note: any rechunking before a reduction changes float32 results at this level. "Same output"
    has to be checked bit for bit, not with a tolerance.
- **PR head `210bdbd`, same recipe and host:** window 1: 3 min 27 s, 35.5 GB, identical; window 2:
  2 min 20 s, 44 GB, identical apart from the old run's lost chunks (N8); window 3: 5 min 12 s,
  50 GB, identical (37 arrays each, compared bit for bit with metadata and encodings). Peak memory
  is unchanged by the PR.

**Strict recipes (group 3):**
- All arco-ocean recipes at the local checkout (`c67b8ab` plus uncommitted and untracked files) were
  loaded with the strict models; nothing was downloaded.
- These load: the download recipes for levels-50 at 0.25°, 0.5° and 1.0°, every unpack and
  climatology recipe, and both validate recipes.
- These are rejected, for arco-ocean A13/A17:
  - levels-10 (0.25°, 0.5°, 1.0°) and arco-med: `step` is missing (already rejected before), and
    `[save] chunk` is a typo for `chunks`;
  - anfc: `step` is missing;
  - both stats recipes: `[save] chunk`, plus `time_dim` and `skipna`, which never had an effect
    (B15, C2).
- The step names and arguments of every recipe pass the new load-time checks.

**Code findings beyond the guide (confirmed by tests in `tests/`):**
- N1: `safe_to_zarr` writes uncompressed arrays when `[save]` has no compressor. **[Moby-Dick]**
  Not affected: the arco-ocean download, unpack and climatology recipes all set lz4.
- N2: `astype` with `dtype` and no `casting` is a no-op, and its per-variable form drops the
  unlisted variables. **[Moby-Dick]** Not affected: every recipe call sets both.
- N3: the ECMWF-coordinates check has different lat/lon defaults in its model and its function.
- B12 went further than a deprecation: `valid_time_coordinate` built the expected cftime axis
  with `datetime` bounds, which makes xarray ignore both `calendar` and `inclusive`. The check
  therefore failed on every cftime dataset. Published data were not affected: the dataset uses
  numpy datetimes and the climatology integer bins.
- B6, measured on synthetic data **[Moby-Dick]:**
  - with one of five years missing in one bin, the anomaly variance of that bin is 6 % too low
    (window 1);
  - a bin missing in every year makes its neighbours NaN through the window.

## 2026-11-16: CI on GitHub

- The first PR runs failed before reaching the code. uv picked the runner's newest Python (3.14),
  for which numcodecs 0.15 has no wheel, and its source build failed. CI now pins 3.13, as the
  image does. `requires-python = ">=3.13"` still admits 3.14; that resolves with the zarr v3 port.
- The image workflow pushed to `oras://ghcr.io/...:${tag}` with `tag` never set (C15). The master
  runs of October passed that step (their logs have expired); the current Apptainer rejects the
  empty tag. It now pushes as `latest`, the tag the arco-ocean scripts pull, and only from master
  pushes and manual runs: a pull request builds the image without publishing it.
- **Decision (maintainer): CI stays offline.** The repository has secrets for Copernicus Marine and
  EWDS, but PR checks do not use them:
  - the repository and its Actions logs are public, and GitHub masks only the secrets themselves,
    not tokens derived from them (Copernicus Marine S3 credentials, for instance);
  - a pull request from a branch of the repository runs its own code with the secrets;
  - the services fail and queue for reasons unrelated to the code (window 2 of the baseline failed
    four times), and their data are revised, so only structural checks would be possible;
  - a real recipe needs 36–53 GB, a standard runner has 16 GB.
- What only the live services catch (changes in the services, the toolbox fork or cdsapi; GRIB
  decoding; N6–N8 were found on real data) goes into a separate, non-blocking workflow: scheduled
  and manual only, secrets in a protected environment, one tiny request per backend.
- The NetCDF-over-HTTP backend's download had no test; it is now tested against a local server.
- Live checks (`tests/live/`, `live-services.yml`): one day of 2020 over the Adriatic per backend
  (ERA5 on GCS, GLORYS12 from Copernicus Marine, GloFAS from EWDS, the GloFAS upstream-area NetCDF
  over HTTPS). On the project VM with the project credentials: all four pass in 82 s, EWDS included
  (no queue that day). They use `eccodeslib` 2.49, ECMWF's binary wheel, which also avoids the
  host's old ecCodes. Secrets go to an environment restricted to master, without required
  reviewers, so that the weekly run needs no approval; only reviewed code on master can read them.
  - After the code review of the live checks: `eccodeslib` is locked (the `live` extra) instead of
    installed unpinned next to the secrets; actions are pinned to commits; the tests check the
    fraction of finite values (ERA5 1.0; GLORYS 0.41, GloFAS 0.62 measured on 16 Nov), so a lost
    chunk fails; a failure reports the chain of exception types (the services' messages stay out
    of the public log); the live tests are skipped without `--live` instead of deselected; they
    also run after merges touching the backends, which keeps the schedule from being disabled
    after 60 days without activity. Still open: the NetCDF-over-HTTP backend has no request
    timeout (group 7); the job timeout bounds a hang.

## 2026-11-16: code review of the PR (effort high)

A review of the whole PR found ten issues; each was reproduced before being fixed, one commit and
test each.

- **N10, silent corruption with `-1` save chunks (also at `847b6f2`) [Moby-Dick]:** `[save]
  chunks` passed -1 ("whole dimension") into the Zarr encoding. xarray's check that Dask chunks
  do not straddle Zarr chunks computes `n % -1 == 0` and passes, so Dask chunks smaller than the
  dimension wrote into the same Zarr chunk concurrently. On a 2 × 400 × 400 array in 10 × 10 Dask
  chunks, about 95 % of the values read back wrong, with no error, at both commits. -1 is now
  resolved to the dimension's size, and such a write fails.
  - Not affected: ARCO-OCEAN. Its download, unpack and climatology recipes all rechunk `lat` and
    `lon` to -1 (a `chunk` step, or `[read] chunks`) before saving, so Dask and Zarr chunks align;
    the baseline outputs were identical across runs and commits, which a race would not give.
- The other findings, fixed:
  - `drop_variables` in `[read]` and generic parts was a one-shot iterator after validation: only
    the first interval dropped the variables.
  - `[save.encoding]` forbade `_FillValue` and passed `fill_value`, which `to_zarr` rejects: no
    recipe could set a fill value. `_FillValue`, `scale_factor` and `add_offset` are accepted.
  - `[read]` had no `storage_options`, so remote inputs needing credentials could not be read.
  - The N8 fix applied only with `engine = "zarr"` spelled out; with the engine guessed from a
    `.zarr` name (the arco-ocean stats recipes), timeouts still became NaN.
  - A bare string passed the load-time check of list arguments, and steps then matched substrings
    (`clip_negative`, `apply_mask` and `masked_fill` would act on the wrong variables).
  - `time_sel` failed on cftime calendars with the `datetime` bounds `download` passes.
  - A part without `engine` matched the Copernicus Marine model; Zarr parts refused the other
    `open_dataset` arguments.
- Not fixed, recorded: three commits change output for some recipes without the `output:` prefix
  of the commit rules, `astype` with `dtype` alone or per variable (N2), `apply_mask` no longer
  masking the mask by default, and `get_sea_mask`'s default `depth_dim`. None applies to the
  arco-ocean recipes: they set both `astype` arguments, pass `depth_dim` to `get_sea_mask` and do
  not use `apply_mask`. The PR description lists them under "Behaviour changes". Rewording would
  rename every later commit.

## 2026-11-16: climatology statistics (group 6)

**Decisions (maintainer):** Feb 29 counts in Feb 28's bin (365 bins; the lookup rule Moby-Dick
already uses); the arco-ocean climatology uses the reference period 1993–2018; the single-node
implementation (S1) is built and timed on the project VM, and the maintainer times it on the HPC cluster.

- **B6, fixed:** each bin counts the values it has. Per bin, the count, mean and sum of squared
  deviations are computed skipping NaN; the window combines bins with reliability weights
  (weight of the offset × count). With complete data this is the previous formula. A bin missing
  in every period no longer spreads NaN through the window; a value missing in some years no
  longer biases the variance low. Where the window holds fewer than two values, the variance is
  NaN (it was 0).
- **Calendar:** `bins = "day_of_year"` puts daily steps in the bin of their date in a 365-day year
  and Feb 29 in Feb 28's bin, so recipes no longer need `convert_calendar` (which dropped Feb 29).
  Incomplete first or last years are allowed.
- **Reference period:** `reference_period = {start, end}`, right-open; the default stays the whole
  input.
- Output: per-bin statistics and the window are computed in the input's float type (float32 for
  ARCO-OCEAN) and in another order than before, so the published climatology would differ in the
  last bits even for complete data, besides the changes above (commit marked `output:`).
- After the code review: position bins again require a whole number of periods (a missing
  `convert_calendar` would otherwise shift every bin after a leap day, silently); the one-step-
  per-day check of day-of-year bins also covers flox; `reference_period` (ordered, no time zone)
  and `day_of_year` with `period = 365` are checked when the recipe is loaded; flox again uses the
  communication-free "blockwise" method when time chunks hold whole periods.

## 2026-11-16: user settings file

**Request (maintainer):** support a user file for secrets and settings, like the one kept on the
project VM (which must never be read here), including the default S3 profile. Parsed with
pydantic-settings (maintainer's choice) rather than by hand.

- `ArcoMakeSettings` (`settings.py`) reads the environment, then the dotenv file `~/.arcomakerc`
  (or `$ARCOMAKE_RC`; empty disables it); every command exports the values the environment lacks,
  since the libraries read only the environment. The environment wins. python-dotenv accepts
  `export KEY=VALUE`, so job scripts can `source` the same file.
- Settings: `TMPDIR`, `AWS_PROFILE` (default profile of `s3://` paths), `AWS_CONFIG_FILE`,
  `AWS_SHARED_CREDENTIALS_FILE`, `GOOGLE_APPLICATION_CREDENTIALS`, Copernicus Marine and CDS/EWDS
  credentials; `DASK_*` keys go to Dask. Secrets are `SecretStr`; values are checked at start
  (`CDSAPI_URL` must be HTTP) and never logged. Unknown keys are ignored with a warning naming the
  key (maintainer's choice: the file may hold exports for job scripts). A file other users can
  access is reported.
- `TMPDIR` and `DASK_*` from the file take effect although `tempfile` and Dask read them earlier.
- Tests never read a real file: an autouse fixture sets `ARCOMAKE_RC=""`, except for the live
  checks, whose credentials may come from it.
- **S3 profile and Copernicus Marine.** The toolbox (fork 2.4.1) reaches CloudFerro's S3-compatible
  storage through botocore: `boto3.Session()` (which reads the profile variables) and a client
  with `signature_version=UNSIGNED` and an explicit `endpoint_url`. So a profile's credentials are
  never used to sign, and nothing of the AWS identity goes to CloudFerro; explicit endpoint and
  retries override the profile's. But an unknown profile fails the session: seen live, the
  Copernicus part failed with `ProfileNotFound` after about 4 min of retries. arco-make now checks
  the profile botocore will use when a data command starts.
  - botocore takes `AWS_DEFAULT_PROFILE` before `AWS_PROFILE`; the session environment here sets
    the former, which hid the test profile in the first two live runs. Both are settings now, and
    the test fixture unsets both.
  - Live check with a profile holding fake credentials and `addressing_style = virtual`: the
    Copernicus Marine download passes (complete, plausible values) in 3 s. The profile's
    credentials and S3 options do not reach the toolbox's unsigned CloudFerro client.
- After the code review of this PR: values from the environment are checked even without a file;
  errors become CLI messages naming the variable (no value, no traceback); the AWS profile is
  checked only for runs with `s3://` paths or Copernicus Marine parts (local runs and `--help`
  are not affected); empty values count as unset (a template `CDSAPI_KEY=` hid `~/.cdsapirc`);
  `TMPDIR` must be an existing writable directory (tempfile silently fell back to `/tmp`); a
  missing file named by `ARCOMAKE_RC` is reported; AWS credentials, region, endpoint and the
  `ECCODES_*` paths are settings too; `DASK_*` keys are merged into Dask's configuration instead
  of `dask.config.refresh()`, which dropped values set in the process; the test fixture no longer
  leaks the exported values into later tests. Commands no longer reduce their own `ClickException`
  messages (e.g. the overwrite refusal) to "An error occurred (ClickException)".

## 2026-11-16: statistics on the project VM (S1), zarr v3 and Icechunk

**Decisions (maintainer):** move arco-make to zarr v3, then to Icechunk (in this order, after S1
and before group 7). Shard sizes will come from Moby-Dick's access patterns and the best object
size on S3: experiments that read less than a whole shard must still perform well.

**Measured on the project VM, published 0.25° dataset on S3 (read only):**
- 223 series (23 surface variables + 4 × 50 levels), 11,322 daily steps; 9,496 in 1993–2018.
  Time-varying data: 10.5 TB raw, 3.86 TB stored (LZ4), 3.24 TB within the reference period.
- S3 to the VM: 36 chunks/s (85 MB/s) with s3fs's default pool of 10 connections, whatever the
  number of threads; 136 MB/s (about 1.1 Gbit/s, likely the VM's limit) with 64 connections;
  29 MB/s with 256.
- One full series (thetao, depth 0, 1993–2018) with the group 6 code: 234 s, peak 55 GB, limited
  by reading. Projection: about 15 h for the 223 series one at a time, at least about 7 h if
  reading is overlapped; the outputs are about 0.3 TB.
- **S1:** `split_dims` computes one series at a time: per-bin count, mean and sum of squared
  deviations in one read pass, then the window in NumPy over latitude bands, written into the
  series' region of outputs whose metadata were written first. Same statistics as the Dask path
  (tested). Postprocess steps must work on one series.
- **S1 on real data** (one series per pass, input and outputs on S3 under `experiments/`):
  - First version (Dask fold per series): 473 s per series and a 105 GB peak, with the recipe's
    `time = "auto"` read chunks: worse than one graph.
  - Streaming per-bin statistics (Welford, float64) and the window over point bands, in threads:
    `2t` 207 s reading and binning, 22 s window, 85 s writing (one chunk at a time), 26 GB.
  - memray on a `thetao` level: the window took 263 s and 55 GB because its bands were cut along
    `depth` (size 1 in a series); bands are now cut over flattened points (24 s). Binning was
    CPU-bound (same 200 s for `2t` and `thetao`, whose chunks are 40 % smaller): a numba kernel
    (one fused pass per step, parallel over points) replaced eight NumPy passes.
  - Reads were latency-bound with 64-step blocks (49 steps/s for `thetao`); about 1 GB per read
    (a whole number of the input's time chunks) gives 71 steps/s. Writes are chunked like the
    outputs (parallel) and run in the background while the next series is read.
  - Result for `thetao` depth 0: 122 s reading and binning, 24 s window, 62 s writing (overlapped
    in a full run), 32 GB peak: about 150–210 s per series, 9–13 h for the 223 series.
  - One run hung (all threads waiting, no I/O, 14 min) right after writing the outputs' metadata,
    with `numba.set_num_threads` called among Dask and pool threads; that call is gone, and the
    rerun finished. Not reproduced, cause not confirmed; long runs keep faulthandler on (SIGABRT
    dumps the stacks) and a watchdog.
  - These figures are for today's layout (zarr v2, daily chunks, no shards). The statistics will
    be recomputed on the new zarr v3, sharded dataset: the read block follows the input's chunks
    (or shards), and grouping several levels per pass waits for the shard layout.
- After the code review of S1: the Dask path now runs the same window function as the per-series
  path (`xr.apply_ufunc`, block by block), so the formulas exist once; writes use explicit regions
  (`region="auto"` put every series of a dimension without coordinate at index 0, silently); the
  split dimensions are stored one index per chunk (other sizes, `[save] group`, unknown
  `split_dims` and the time dimension are refused before any work); Dask-backed coordinates are
  written with the template; the per-series path checks one step per day like the Dask path; read
  blocks follow the stored chunks when the input has no Dask chunks (they can still straddle
  stored chunks when `reference_period` starts inside one); the per-series path uses local threads
  whatever `--dask-setup` says; numba is a declared dependency, compiled on first use.

## 2026-11-16: download times, GloFAS outage, Copernicus Marine toolbox, ERA5 sources

**Question (maintainer):** how long does a full download take per resolution, and how will the
planned changes move it?

**From the logs of past runs on the project VM** (code of the 30 Sep image, blocks of 256 days run
one after another, written to S3):
- 0.25°, 50 levels, 1993–2023 (11,322 days): 166 h of running time, 7.8 days of calendar time
  (30 Sep – 8 Oct). Median 46.6 s per day of data. The 21 h difference is failed attempts and
  waits before manual restarts (six blocks); stalled requests inside completed blocks cost another
  ~19 h (one block lost 8.5 h).
- 10 levels, 10,408 days, all three resolutions (Aug–Sep): 88.5 h (0.25°), 85.2 h (0.5°), 85.7 h
  (1.0°): about 30 s per day each. The output resolution does not matter: the time goes into
  reading the sources, the same for all three. 0.5° and 1.0° at 50 levels were never run here;
  estimate: as 0.25°.

**Per source** (4-day windows, PR stack code, values discarded, s per day of data): GLORYS 3D at
50 levels 26–36 (18–27 reading alone), Waverys 3.8, ERA5 3.6, ERA5 accumulated 2.2, sea ice 1.6,
GloFAS about 2.5 (from the logs), static datasets about 1: about 40–50 in total, as measured in
production. GLORYS is about 75 %. A run keeps 2–6 of 32 cores busy; two GLORYS benchmarks run at
the same time each kept their speed (1.4 GB/s decoded together): neither the CPU nor the network
is the limit, but how the work of one process overlaps. Run-to-run noise is large (26–36 s/day
for the same window). Single requests stall for 76–91 s now and then.

**Expected effect of the planned changes** (±20 %):
- Group 7 (blocks with timeouts, retries, restart): about 146 h per resolution, unattended (no
  stalls or manual restarts).
- Two blocks at once (independent blocks of group 7): about 3–3.5 days per resolution; memory
  allows two (36–53 GB each), not three.
- Downloading the whole series directly removes the unpack pass on the HPC cluster.
- zarr v3 with shards: little effect on time (writing averages about 7 MB/s); the block length must
  be a multiple of the shard's time length, or appends rewrite partial shards.
- Group 8 (chunk planner): possible gain, not quantified.
- Not planned, for the maintainer: computing 0.5° and 1.0° from the 0.25° product would take hours
  instead of days, but the values would differ from regridding the native grids.
- **Maintainer's decision:** the performance of the full download is revisited after all planned
  changes, before a production download (the Dask graph of one block is the first lead).

**GloFAS static file outage.** `confluence.ecmwf.int` answered 502 on 16 Nov from about 12:05 UTC;
it worked at 12:04 and again at 19:28 (not checked in between). Every block reads that file again,
so an outage fails the block it hits. (A benchmark error first read as the outage was a bug in the
benchmark script: the NetCDF-over-HTTP backend needs the temp directory registry of `download`.)

**Baseline windows 2 and 3 with PR #1 code** (600 s GCS timeout):
- 2012-12-30 – 2013-01-02: the differences from `847b6f2` are the two ERA5 hours the old output
  lost (N8: hour 09 of `2t` on 30 Dec, hour 05 of `msl` on 1 Jan); the new output equals the full
  daily means recomputed from the source, bit for bit. 2 min 48 s, 36 GB.
- 2020-02-26 – 2020-03-02: failed twice, each time after about 11 min, on a GCS read that received
  nothing for 600 s (`TimeoutError`; the run stops, no partial data hidden). The same window ran in
  3–5 min on 13 Nov with older code. Windows 1 and 2 pass. Not investigated (a blocked network
  thread would look the same as a slow server); group 7's retries are the planned remedy.

**Copernicus Marine toolbox.** The fork's only change (pool size from
`COPERNICUSMARINE_MAX_POOL_CONNECTIONS`, default 10) is merged upstream (#532, 21 Oct) but in no
release yet (2.5.1 of 3 Nov lacks it). arco-make now pins upstream main at `f66a8b8` (reports
2.5.0); move to the first release that has it.
- (Wrong, corrected on 17 Nov.) The backend passes `chunk_size_limit=False`, which upstream
  documents as -1 or a positive limit. Upstream main, 2.5.1 and the fork (2.4.1) all treat a falsy
  limit as "no rechunking" and give the store's native chunks (one day, 512 × 2048); the toolbox
  default doubles the time chunk. A live check now fails if that changes. Offline suite and live
  checks pass on all three.
- (Partly wrong, corrected on 17 Nov.) Upstream reads the pool size once, when the toolbox is
  imported (the fork, per client). arco-make does not import the toolbox itself: xarray loads it
  at the first use of the engine, after the CLI applied the settings file, where the pool size is
  now a setting.
- GLORYS on 4 days with 10, 32, 64 connections: 21.4, 16.6, 17.4 s/day reading; 21.9, 20.9, 20.6
  s/day reading and processing. Reading gains about 20 %; the processing graph is the limit.
  With 10 connections, urllib3 logs "Connection pool is full, discarding connection".

**ERA5 sources** (looked for an alternative to Google's GCS store because of the timeouts):
- Earthmover's Icechunk ERA5 on AWS (`earthmover-icechunk-era5`, us-east-1, anonymous, zarr v3,
  CC-BY 4.0, 1940 – June 2026, updated quarterly; one hour per global map, as Google's): has 7 of
  the 8 hourly variables used (as `u10`, `v10`, `d2m`, `t2m`, `msl`, `ssrd`, `tp`) and the two
  static ones (`z_sfc`, `lsm`), not `surface_thermal_radiation_downwards`; needs
  `numcodecs[pcodec]`. Values match Google's within GRIB packing (t2m 0.0008 K, msl 0.08 Pa), not
  bit for bit. One day of 7 variables from the VM: 10 s (Earthmover) against 15–20 s (Google),
  but 79 s on a repeat of the same Earthmover read.
- ECMWF's own ARCO Zarr (CDS, time-chunked and geo-chunked): a subset of variables, token access,
  beta ("access may be closed at any time").
- **Maintainer's decision:** stay with Google's store; the timeouts are to be handled by the
  refactoring (group 7).


## 2026-11-17: review of PR #7 (Copernicus Marine toolbox)

Corrects the toolbox bullets of 16 Nov.
- **`chunk_size_limit=False` does not give native chunks: it gives no Dask arrays.** A falsy
  limit makes the toolbox call `open_zarr(chunks=None)`, so it returns lazily indexed arrays, and
  xarray chunks them as the recipe's `chunks` say (the arco-ocean recipes set them, e.g. time 2,
  depth 1, whole maps). Its default returns Dask arrays of its own chunking (time 2, all 50
  depths), which the recipe's `chunks` then only rechunk. The live check of 16 Nov opened the data
  with `chunks={}`, which makes xarray rechunk to the store's chunks whatever the toolbox returns:
  it could not fail, and the comparison of 16 Nov had the same flaw. The check now calls the
  backend directly, with the recipes' service and version, and requires unchunked arrays; it
  fails when the backend's call loses `chunk_size_limit=False` (checked). Upstream main, 2.5.1
  and the fork (2.4.1) all return unchunked arrays (checked again, the same way).
- A recipe without `chunks` on a Copernicus Marine part reads each variable in one piece, without
  Dask parallelism.
- The toolbox was imported whenever xarray listed its backends (any `open_dataset`), which in
  library use or tests can be before the settings file is applied, fixing the pool size at its
  default. The backend now imports it at first use.
- The pool size is checked as the toolbox reads it (`int()` of the text): `32.0` was accepted by
  the check, then ignored by the toolbox, which falls back to 10 without a message.
- The toolbox writes its version into the dataset attributes (`copernicusmarine_version`), now
  `2.5.0` (upstream main reports the version of the last release branch). Outputs of recipes that
  keep those attributes change; ARCO-OCEAN's do not (its recipes drop them).

## 2026-11-17: Zarr v3 migration

**Maintainer's decisions (17 Nov):** outputs in Zarr format 3 only, no compatibility with format 2
outputs; `unpack` and zip outputs removed; shards set by the recipe, unsharded without `shards`;
compressors as Zarr v3 codec metadata (a list of `{name, configuration}`), as flexible as zarr
and free to adopt in the arco-ocean recipes; consolidated metadata kept.

**Versions:** zarr 3.4.2, numcodecs 0.17.1 (no longer a direct dependency, so the `<0.16` pin and
its Python 3.13 constraint are gone; CI and the image stay on 3.13).

**Findings while porting:**
- Variables read from format 2 stores (ERA5, GLORYS) carry numcodecs codecs (`Blosc`) in their
  encodings, coordinates included; zarr 3 refuses them in format 3 ("Expected a BytesBytesCodec").
  They are dropped before writing (the recipe's compressors or zarr's default, zstd, apply); codecs
  of format 3 sources are kept. The synthetic test sources are now format 2 with Blosc.
- zarr 3's fsspec store reads only `FileNotFoundError` (and directory errors) as missing chunks;
  timeouts and connection errors raise. The N8 safeguard (`strict_zarr_store`) is no longer
  needed; the read-error tests run on sources of both formats.
- The toolbox, under zarr 3, uses its own zarr 3 store: only 403 and 404 answers become missing
  chunks, other errors raise after its 9 retries (N9 narrower; code reading).
- Consolidated metadata are not in the format 3 specification: zarr warns at every write (silenced
  at arco-make's writes). Icechunk will not use them.
- xarray keeps the CF `_FillValue` as an attribute (a float is stored base64-encoded) apart from
  the array's own fill value (used for chunks never written).
- Shards: xarray refuses Dask chunks straddling a shard; appending into a partial last shard works.
  Blocks need not be multiples of the time shard, but each partial shard is rewritten on append.
- Appending in format 3 to a format 2 store fails with an unclear xarray message ("append_dim ...
  does not match any existing dataset dimensions {}"); no compatibility, by decision.

**Copernicus Marine toolbox (PR #7) breaks the ARCO-OCEAN recipes.** Found by baseline window 2:
upstream main refuses coordinates in `variables` ("The variable 'depth' is neither a variable or a
standard name in the dataset"). Upstream #534 (`e34a30b`, 8 Oct, "standard names can also be used
for sparse subset") checks requested names against the catalogue's variables, which exclude
coordinates; before, against the dataset's variables. It precedes #532 on main, so every commit
with the pool setting has it; 2.5.1 (cut from 2.5.0) and the fork accept `depth`. The GLORYS
static parts of four arco-ocean recipes (0.25°, 0.5°, 1.0° levels-50, anfc levels-10) ask for
`["deptho", "depth"]`: `depth` is needed by `get_sea_mask`. Workaround in the recipes: ask for
`["deptho", "mask"]` (the 3D mask brings the same depth coordinate, checked on both static
datasets) and drop `mask` as the first step, before anything reads it. The live checks did not
catch it: their recipe asks for `thetao` only.

**Baseline window 2** (2012-12-30 – 2013-01-02, code `5eb0587`, the 0.25° levels-50 recipe with
`compressors` and the static workaround): all 37 variables equal the PR #1 output (`656503f`, format
2) bit for bit, values, dtypes and attributes; same size on disk (1.3 GB). 3 min 58 s and 37.5 GB
peak, against 2 min 48 s and 35.6 GB for `656503f` (one run each; not investigated, for the
performance pass). New script `compare_values.py` (baseline scripts) compares across formats.

**arco-ocean (fork stack):** the static workaround (for arco-make #7); `compressor` to `compressors`
in every recipe (same Blosc lz4 settings, byte shuffle); the unpack recipes, `unpack.slurm` and
the `--zip` options removed. The download scripts still write one store per piece, which nothing
joins any more: they are to be reworked with group 7 (resumable blocks).

## 2026-11-17: review of PR #8, Icechunk decisions, forecast time axes

**Review of PR #8 (Zarr v3), high effort: 9 findings, all fixed** (one commit, each fix with a
test that fails on the previous commit):
- `[save.encoding.VAR]` went to xarray as `encoding=`, which replaces a variable's whole
  encoding: a variable listed for its dtype lost the `[save]` chunks, shards and compressors
  (also true before Zarr v3). Now merged key by key; appends reuse the stored encodings, so the
  pruning of encodings in append mode (which read the existing store) is gone.
- Shards read from a format 3 source survived new `[save] chunks`; an uncompressed source gave an
  uncompressed output (N1 again); zarr 3 refuses `storage_options` for local stores (plain paths,
  `file://`), which broke `open_input` and the one-series-at-a-time climatology on local
  outputs; `.zip` outputs silently became directories; shard/chunk divisibility was checked at
  load only when both were tables.
- After the fixes: offline suite 261 passed, live checks 5 passed. Baseline window 2 (code
  `9e9fa8c`) is still bit for bit equal to the PR #1 output in values and attributes, `time`
  stored as before; 3 min 54 s, peak 44.5 GB (37.5 GB on `5eb0587`, 35.6 GB on `656503f`: single
  runs, for the performance pass). A first attempt failed after 10 min on a GCS read timeout
  (600 s) while opening the ERA5 store's consolidated metadata, before any download: the network
  failures group 7 is to retry.

**Maintainer's decisions (17 Nov):**
- The upstream toolbox refusing coordinates in `variables`: keep the recipe workaround, with a
  comment stating the issue (mercator-ocean/copernicus-marine-toolbox#534) for later.
- Icechunk: every command writes Icechunk repositories; `process` and `compute-climatology`
  should ideally read both Zarr v3 and Icechunk, weighed against code complexity; one commit per
  block, its message naming the block's time range. Measure the effect of Icechunk on memory and
  time.
- Forecasts: the Icechunk code will have to handle forecast time series (initialization time
  and lead time) as well as historical ones.

**Forecast time axes** (investigation, nothing implemented):
- Mercator GLO12 (GLOBAL_ANALYSISFORECAST_PHY_001_024): one `time` axis, the valid time (checked:
  June 2022 to today + 9 days for the daily dataset), no run dimension or forecast metadata. Run
  daily at 00 UTC, 10-day forecast, published by 12:00 UTC. The series is a best estimate
  overwritten daily (product user manual): each validity date gets about 15 releases (forecast,
  simulation, near-real-time analysis on Thursdays, best analysis), the last 11-15 days change
  every day, and dates older than 15-21 days are final. Original files carry the run date
  (`..._R<run date>.nc`) but are overwritten too: past forecasts are not archived. Daily means
  run midnight to midnight, stamped 00:00.
- ECMWF IFS: runs at 00/06/12/18 UTC (00 and 12 UTC to 240 h: 3-hourly to 144 h, then
  6-hourly). GRIB through cfgrib gives `time` (initialization) x `step` and a 2D `valid_time`.
  Open data keep the last ~12 runs; full history needs a service agreement. dynamical.org
  archives IFS ENS (00 UTC since April 2024) as Icechunk, as `init_time` x `lead_time` x
  `ensemble_member`, `valid_time` = init + lead.
- **Maintainer's decisions (17 Nov):** forecast runs, not best-estimate series; GLO12 runs built
  from the release date (an archive collected from now on); products: GLO12 physics
  (GLOBAL_ANALYSISFORECAST_PHY_001_024), GLO12 waves (GLOBAL_ANALYSISFORECAST_WAV_001_027), ECMWF
  IFS open data at 0.25° (0.1° possibly later) and the GloFAS river discharge forecast
  (`cems-glofas-forecast`, EWDS); dynamical.org's names (`init_time`, `lead_time`,
  `ensemble_member`, `valid_time`).
- GLO12 waves (Météo-France MFWAM, 1/12°): one `time` axis, 3-hourly instantaneous (checked: Nov
  2022 to today + 10 days, 19 variables). Two bulletins a day, 00 and 12 UTC: each releases a
  10-day forecast and 24 h (00 UTC) or 12 h (12 UTC) of hindcast, overwriting the series; original
  files are named `mfwamglocep_<field date>_R<bulletin date>_<00H|12H>.nc` (product user manual).
- GloFAS forecast (EWDS `cems-glofas-forecast`, GloFAS v4, 0.05°, global without Antarctica):
  river discharge in the last 24 hours, runoff, snow depth water equivalent, soil wetness; one
  run a day since 5 Nov 2019; `leadtime_hour` 24 to 720 by 24; `product_type` control forecast or
  perturbed ensemble members; GRIB2 or NetCDF (request form). The value at lead 24 h is the
  discharge of the first day: aligned with daily means stamped at the start of the day, it is lead
  day 0 (the C10 question, here for forecasts).
- Still open: which IFS stream (HRES `oper` or ENS `enfo`, 51 members), runs other than 00 UTC
  (waves and IFS have 12 UTC), and whether 3-hourly fields (waves, IFS) stay 3-hourly or become
  daily means.

## 2026-11-17: forecast design decisions; what "day D" means for every source

**Maintainer's decisions (17 Nov):** IFS HRES (one member), 00 UTC runs only, daily resolution,
`lat`/`lon` for the horizontal coordinates (dynamical.org's names for the time dimensions:
`init_time`, `lead_time`, `valid_time`).

**Day D in each source** (checked on the data and GRIB keys, 17 Nov; the convention to keep:
day D is [D 00:00, D+1 00:00) UTC, stamped D 00:00):
- GLORYS reanalysis and GLO12 forecast daily means: stamped 00:00, midnight to midnight (GLO12
  product user manual). In sync.
- WAVERYS and GLO12 waves: instantaneous every 3 h, 00:00 to 21:00; `resample` with
  `label = closed = "left"` averages 00:00-21:00 of D. In sync.
- ERA5 instantaneous variables: hourly 00:00-23:00 of D, averaged the same way. In sync.
- **ERA5 accumulations (`tp`, `ssrd`, `strd`): one hour early.** ECMWF: an hourly accumulation
  covers the hour ending at its stamp, so the value stamped 00:00 is the last hour of the previous
  day. The recipe sums the stamps 00:00-23:00 of D: [D-1 23:00, D 23:00). **[Moby-Dick]** this
  holds for the published data (C10 confirmed). A recipe-only fix (`closed = "right"`) would
  break at block edges (the first stamp of a block forms a one-hour bin, the last day misses its
  24:00 hour): it needs a part-level time offset applied before the block's time selection, or
  reading one hour past the block (arco-make, group 9).
- GloFAS historical (v5, `time_mean`): the request for day D is the mean over [D 00:00, D+1
  00:00) (GRIB `dataDate` D, `stepRange` 0-24, `validityDate` D+1); arco-make stamps it D. In
  sync: the feared one-day offset of `dis24` (C10, B9) does not exist for this variable.
- GloFAS forecast: lead 24 h is the mean over the run day (`stepRange` 0-24), lead 48 h the next
  day: lead day k = `leadtime_hour` 24(k+1), so a shift of -24 h gives lead days 0-29.
- IFS HRES 00 UTC (to build): daily mean of lead day k from the instantaneous steps 24k to 24k+21
  (3-hourly to 144 h, then 6-hourly: 8 then 4 samples, ERA5 has 24); accumulations as
  differences, acc(24(k+1)) - acc(24k), exactly lead day k. Lead days 0-9.
- GLO12 physics and waves forecast runs from the 00 UTC release of day R: lead day k is the
  daily value of R + k; lead days 0-9 (negative leads, the hindcast part of a release, dropped).

## 2026-11-17: ERA5 accumulations corrected (maintainer's decision)

- arco-make: parts take a `time_offset`, added to their time coordinate before each block's
  time selection (backends with a time range get it shifted the other way). The arco-ocean
  recipes set `time_offset = "-1h"` on their ERA5 accumulation parts (five download recipes).
- Baseline window 2 (code `6fa30bb`, recipe with the offset): tp, ssrd and strd equal ERA5 summed
  over the stamps 01:00 of D to 00:00 of D+1 (largest relative difference about 1e-7, float32
  rounding); the previous output equals the sums over 00:00-23:00 of D. The two windows differ by
  up to 8-12 % of the field's maximum for tp and ssrd, 1.3 % for strd. The other 34 variables are
  bit for bit the same. 4 min 29 s, peak 37.4 GB (single run; the shifted hours no longer align
  with the source's 24-hour Dask chunks, so each day spans two of them).
- New script `check_era5_accumulations.py` (baseline scripts): which 24 hours an output's daily
  accumulations cover.
- **[Moby-Dick]** The published ERA5 accumulations are the one-hour-early sums; the corrected ones
  come with the next download.

## 2026-11-17: Icechunk outputs

**Maintainer's decisions (17 Nov):** every command writes an Icechunk repository; `process` and
`compute-climatology` read Zarr v3 and Icechunk inputs (weighed against code complexity: an
explicit `engine = "icechunk"`); one commit per block, named by its interval; measure the effect
on memory and time.

**Design:** icechunk 2.3.1. `download` commits each block (`download [start, end)`), `process`
once, `compute-climatology` once per output with the same message. `--overwrite` is a new commit
(older versions stay) and refuses anything that is not a repository. Checkpoints stay plain Zarr
v3; `[save] consolidated` is gone. On a distributed client the two climatology outputs are
computed in turn (each repository's writes are merged by computing them; one shared computation
would need icechunk's and xarray's private writers); with local schedulers once.

**S3 access:** icechunk's own AWS client cannot use the project VM's profile ("credentials-login"
feature missing) and needs a region the profile does not set: credentials come from botocore
(as for every other S3 access), the region from S3 (`eu-south-1` for the project bucket).

**Measurements** (window 2, 0.25° levels-50, 1.3 GB; the same in both formats, values and
attributes bit for bit):
- End to end on the VM, one successful run each (two other runs, one per format, failed on GCS
  read timeouts of ERA5, 600 s): plain Zarr v3 4 min 09 s, 43.4 GB peak; Icechunk 5 min 40 s,
  36.9 GB. The download dominates; the difference is within the run-to-run spread seen so far
  (3:54-4:29 for the same code).
- The format alone (`bench_formats.py`, writing and reading window 2's output, one process each):
  - Local disk: write 5.1 s (Zarr) / 5.8 s (Icechunk); whole read 2.8 / 3.4 s; one point of
    every variable 2.2 / 3.0 s; peak memory equal (0.67 GB writing, 4.5 GB reading).
  - Project bucket: write 58 s / 266 s at first, 57.6 s once credentials were resolved once
    (botocore session and credential process for every client or fork before); open 6.5 / 0.7
    s, whole read 20 / 6-7 s, one point 14 / 6 s. Icechunk reads stalled now and then for about
    its stream timeout (60 s; up to 107 s for a whole read): with 10 s at most 22 s. arco-make
    uses 20 s (`network_stream_timeout_seconds`). **[Moby-Dick]** consumers reading from S3 with
    icechunk's defaults may see such stalls: a shorter stream timeout avoids them.
- Repositories of the measurements: `experiments/icechunk-probe-20261117/` and
  `experiments/bench-formats-20261117/` in the project bucket (about 4 GB, may be deleted).

**Review (high effort), 10 findings:** 9 fixed (bucket region over the profile's, asked at the
endpoint and for anonymous access; unusable destinations fail before any download; GCS tokens;
equal commit messages for the two climatology outputs; synchronous scheduler kept; one helper for
writes into sessions); one not: the proposed shared computation for distributed clients relied
on `write_lazy` returning lazily, which it does not (it computes).

## 2026-11-17: runs on their own branch, --replace, retention

**Maintainer's decisions (17 Nov):** a failed run must not hide the current version (gap 2 of
`--overwrite`); the history of replaced versions is limited in the recipe (gap 3), by age or by
count; `--overwrite` becomes `--replace` (nothing is deleted when it runs).

- Every run writes on `run/<command>-<start time>` from the tip of `main`; `main` moves to the
  run's last commit only when the run succeeds and only if `main` has not changed meanwhile
  (conditional reset, `icechunk.ConflictError` otherwise). A failed run leaves `main` as it was,
  its branch holding what it wrote; a repository whose `main` has no version yet is written again
  without `--replace`. `compute-climatology` promotes its outputs once both are written.
- `[save] keep_history` (counted from when a version was replaced) or `keep_versions`. Garbage
  collection keeps objects younger than 7 days (`GRACE`): another run writing to the same
  repository has uncommitted chunks no snapshot references, and the review showed that an
  immediate collection would delete them, with its branch.
- Measured: three versions of 8 MB take 22.5 MB, 7.5 MB with `keep_versions = 1` (grace 0);
  the same cycle on the project bucket (`experiments/runs-probe-20261117/`) kept only the current
  version and deleted the others' manifests (constant test data is inlined in manifests).
- Review (high effort), 10 findings, all fixed: keep_history counted from writing (a version
  replaced today could be deleted at once); unitless keep_history ("30" is 30 ns); retention could
  delete another run's branch and uncommitted chunks; a failed first run left a repository that
  needed `--replace`; any error at promotion was reported as a conflict; a branch deletion error
  after promotion failed the run; keep_versions read the whole history (a commit per block);
  garbage collection ran with nothing expired; process runs were labelled "write"; one commit had
  three purposes (now three commits).
- Still to decide (maintainer): the retention of the ARCO-OCEAN recipes (they keep everything).

## 2026-11-17: forecast retention, storage budget

**Maintainer's decisions (17 Nov):**
- Forecasts keep a rolling window of runs (option B): each new run drops the oldest one from the
  current version, so `init_time` keeps a fixed length. Window: 6 years (about 2,190 daily
  00 UTC runs). Snapshot retention alone does not bound a forecast dataset: it only drops old
  versions, and the current version would keep every run.
- Every dataset is 50 levels; the 10-level recipes (e.g. anfc) are remnants, to be updated or
  deleted.
- Retention of the arco-ocean recipes: downloads `keep_versions = 2`, statistics
  `keep_versions = 1` (fork stack).

**How a rolling window can work** (prototype, not in arco-make yet): Icechunk's
`Session.shift_array` moves the chunks of an array along a dimension by whole chunks, discarding
those shifted out, without copying data (`init_time` chunked one run per chunk); the new run is
written into the vacated slot, and the dropped chunks go with snapshot expiration and garbage
collection. Tested on `init_time` x `lead_time` x `lat` with a 2D `valid_time`, keeping 3 runs
over 8: coordinates and data right, size constant. xarray's region writes leave index
coordinates out: the new `init_time` value has to be written explicitly. Others: dynamical.org
keeps full, growing archives (GEFS 115 TB compressed in March 2025; costs about $100,000 a year,
growing about 30 % a year); Earthmover's Arraylake schedules expiration and garbage collection.

**Storage budget** (extrapolated from window 2 and the published metadata; the bucket's own
listing would count old versions, archived pieces and experiments):
- One day at 0.25°, 50 levels: 432 MB (388 MB 3D ocean, 44 MB all 2D fields), 47 % of its
  uncompressed size. A year of statistics output (365 bins) at 0.25°: about 0.16 TB per output.
- Historical datasets 1993-2025 (0.25°, 0.5°, 1.0°) and their statistics: 7.3 TB per version;
  about 14 TB in steady state (2 versions of the datasets, 1 of the statistics), about 21 TB if
  everything is replaced at once.
- A forecast run (10 lead days of GLO12 physics and waves and IFS HRES at 0.25°, 50 levels, plus
  30 days of GloFAS discharge): about 4.4 GB. Six years: about 9.6 TB, reached after six years
  (about 1.6 TB a year).
- Total: about 24 TB in steady state, about 31 TB at worst, of the 50 TB of the sponsorship.
  Until the production download replaces it, the data in use now (unpacked dataset, its archived
  pieces, an older version: about 15 TB) also stays.

## 2026-11-17: group 7, resumable and fault-tolerant downloads

Maintainer's decisions: retry defaults in arco-make, overridable per source; an explicit
`--resume` (a plain rerun refuses and names the waiting branch); blocks at once through a CLI
option, default 1. Five commits:
- **Request timeouts.** Parts take `timeout` (seconds, default 120): gcsfs `requests_timeout`
  (None by default: the N7 hangs), botocore's connect and read timeouts on S3 when given,
  `requests` for NetCDF over HTTP (connect at most 30 s), cdsapi for EWDS. Not for Copernicus
  Marine, which has its own.
- **Block retries.** A block failing with a transient error (timeouts, connection errors, HTTP
  408/425/429/5xx, also raised from or grouped in another error) is redone, waiting 1 min,
  doubled up to 30 min, 5 attempts by default (`[retries]`). Others fail at once; the log says
  which. Per block, not per source: a block reads all its sources in one computation. The
  errors logged by the 0.25° download's six failed pieces (EWDS retries, DNS failures, connect
  timeouts, Copernicus Marine resets) are of the transient kind; an ecCodes crash (N6) kills the
  process and cannot be retried, but N6 is fixed.
- **Fetched once per run.** The GloFAS static file was downloaded for every block; an outage of
  its server (502, 16 Nov) failed the block it met. It is now fetched once per run.
- **`--resume`.** Every block's commit records its interval and a fingerprint of the recipe
  (start and end included; retries, request timeouts and checkpointing not, so a run that failed
  on timeouts resumes with longer ones). `--resume` continues the newest failed download run on
  its branch and writes only the missing blocks; it refuses another recipe, or a run that
  started before `main` last moved. A plain rerun refuses beside a failed run that can still be
  promoted; `--replace` starts over. The six manual reruns of the 0.25° download become one
  command each, without rereading the blocks already written.
- **`--blocks N`.** N blocks are read and processed at once, each staged on local disk by a
  thread of its own and committed in order with the encodings and Dask chunks its direct write
  would have had: the output is the same as with one block at a time (tested, array metadata
  included). Memory and disk grow with N: the 0.25° levels-50 baseline windows (2-5 days)
  peaked at 36-53 GB in one process. Not yet measured on real data: two processes each kept
  their speed (logbook, 16 Nov), so about 2x is expected with N = 2 (measured: about 1.2×,
  18 Nov).

Review (high effort) of the five commits: ten findings, all fixed in six follow-up commits, each
with a test that fails without it. The serious ones: gcsfs options were added to `engine =
"icechunk"` inputs on GCS, which then failed to open; a commit that landed but whose answer was
lost would be retried and append its block twice (each attempt now checks the branch tip's
block first); `--blocks N` held N + 1 blocks. Found while fixing: checkpoints were removed only
at the end of a run, harmless for 256-day pieces but filling the disk over a whole resumable
download; they now go with their block.

## 2026-11-17: ARCO-OCEAN path scheme; Icechunk repositories can be moved

Maintainer's decisions for the new version of ARCO-OCEAN:
- One repository per product type and resolution, hive-style, levels left out (every dataset has
  50): `historical/res=0p25` (every product that is not a forecast, GloFAS's hindcast included),
  `forecast/res=0p25` (one recipe, all products homogenized; GloFAS keeps 10 lead days like the
  others), `climatology/res=0p25`, `anomaly-std/res=0p25`, later `observations/res=0p25`; the
  same for `0p5` and `1p0`. Downloads write there directly: no pieces, no `archive/` (the
  download scripts of the fork are reworked accordingly, recipes renamed after the scheme, e.g.
  `historical_res-0p25.toml`).
- `anomaly-std` takes a hyphen (the published path is `anomaly_std`), so only the published
  climatology occupies a new path. The published statistics may be deleted or overwritten once
  the historical dataset is downloaded (Moby-Dick keeps a local copy).

**An Icechunk repository can be moved by copying its objects** (icechunk 2.3.1, checked locally
and on S3 with server-side copies): no object names the repository's location, so a copy opens
at its new path with its branches, tags and history, reads identical data at `main` and at a
tag, takes new commits, and expires and garbage-collects there; the original is unaffected.
Conditions: copy while nothing writes to it (a reference copied before the snapshot it names
would be dangling), and only native chunks move (virtual references, which arco-make does not
write, keep their absolute URLs).


## 2026-11-17: --blocks 2 on real data; one lock for netCDF4 and HDF5

Baseline window 2 (30 Dec 2012 – 2 Jan 2013, 0.25°, 50 levels, one-day blocks, 600 s GCS
timeout), without GloFAS (EWDS queued requests for hours that day), on the project VM, one run
at a time:

| Run | Code | Wall time | Peak RSS | 600 s read timeouts (each retried once, successfully) |
|---|---|---|---|---|
| `--blocks 1` | `083d8a4` | 38:22 | 22.0 GiB | 2 (days 2, 3) |
| `--blocks 1`, again | `083d8a4` | 33:37 | 20.9 GiB | 2 (day 2, twice) |
| `--blocks 2` | `083d8a4` | failed after 36 s | | |
| `--blocks 2` | first fix (static file read whole) | 19:27 | 28.6 GiB | 1 (day 1) |
| `--blocks 2` | second fix | 20:37 | 23.7 GiB | 2 (days 2, 3) |
| `--blocks 2` | final fix (`hdf5_lock`) | 8:34 | 23.8 GiB | none |

All five complete outputs are identical bit for bit (36 variables: values, attributes, array
metadata). Reads that sent nothing until the 600 s timeout (empty `TimeoutError`, so GCS:
ERA5) dominate the wall times: each costs about 11 minutes (timeout, wait, redone block). The
one run without them did three days in 8.5 minutes with two blocks at once; the one-block runs
less their stalls took about 15 minutes, consistent with the expected factor of about 2 (wrong:
about 1.2× measured, 18 Nov). The
stalls are not explained: they hit any day, also day 1 at the start of a process, so idle
connections cannot explain all of them. To investigate before the production download (a
shorter GCS timeout would bound their cost).

**`--blocks 2` failed at once**: two blocks opened the GloFAS static file (`netcdfoverhttp`) at
the same moment and netCDF4 raised "NetCDF: Can't open HDF5 attribute". netCDF-C and HDF5 are
not thread-safe, and xarray locks reads of array values only, not what it reads while opening a
file; eight threads opening files at once crash the interpreter (tests, for the backend and for
generic NetCDF parts and `[read]` tables, also when xarray guesses the engine). Every netCDF4 or
h5netcdf file is now opened through `hdf5_lock.open_dataset`, which holds one reentrant
process-wide lock during the open and gives it to xarray for the later reads; a test fails if
other code opens datasets with xarray directly. Two reviews shaped it: a first fix read the
static file whole under xarray's own lock and covered one backend only; a version that also took
xarray's two locks inside ours could deadlock (xarray combines locks in arbitrary order). GRIB
opens were already serialized by one module lock (N6); 240 GRIB opens overlapping reads of
other files ran without error on ecCodes 2.48 and 2.49, so they are left as they are.

## 2026-11-18: GCS stalls traced; timeouts gcsfs retries; --blocks 2 over eight days

**The stalls.** Window 2 was run again with every GCS request traced (a wrapper around gcsfs's
request method, outside the repository: requests over 5 s or failed, requests in flight and the
connection pool every 15 s). Now and then one GET gets no answer for minutes (212 s once, for a
2 MB chunk; 183 s, 201 s; four at once past 600 s) while every other chunk of the block is done,
then answers 200 if still waiting. The pool was never full (at most 81 of 100 connections, nobody
waiting), so the stalls are GCS's or the network's, not ours. gcsfs passed the part's timeout to
aiohttp as a number, a total for the request: past it aiohttp raises a plain `TimeoutError`,
which gcsfs does not retry, so the whole block failed and was redone after the retry wait.

**The change.** A GCS part's `timeout` (default 120 s) now reaches gcsfs as an aiohttp
`ClientTimeout` bounding the connection (at most 30 s) and each wait for data, with no total: a
stalled answer raises a socket timeout, which gcsfs requests again (6 attempts in all), so a
stall costs about the timeout, not the block, and large chunks over slow links are not cut.
`requests_timeout` in a GCS part's storage options is refused at load (a total, not retried).
Tested against a local GCS endpoint whose first answer stalls. Not covered: `[read]` tables
(`process`, `compute-climatology`, `validate`) get no GCS timeout from arco-make; recipes that set
`requests_timeout` on GCS parts must drop it (the baseline's `timeout600` copies are refused).

**Eight days.** 30 Dec 2012 – 7 Jan 2013, 0.25°, 50 levels, GloFAS included, one-day blocks, on
the project VM, one run at a time, traced:

| Run | Code | GCS timeout | Wall time | Peak RSS | GCS stalls | Blocks redone |
|---|---|---|---|---|---|---|
| `--blocks 1` | `99ebb11` | 600 s total | 1:27:52 | 23.0 GiB | 6 requests at 600 s | 3 |
| `--blocks 1` | `526b465` | 120 s of silence | 30:49 | 22.4 GiB | 5, retried by gcsfs | 0 |
| `--blocks 2` | `526b465` | 120 s of silence | 25:40 | 32.2 GiB | 2, retried by gcsfs | 1 (Copernicus Marine S3 read timeout) |

The three outputs are identical bit for bit (37 variables). In the new runs gcsfs also retried
connections refused for a moment ("Network is unreachable", three in the one-block run), and no
GCS stall reached the block level. The old run's three redone blocks cost at least 33 minutes
(600 s, the retry wait, the block again); the rest of its extra hour is not attributed (EWDS
queues and Copernicus Marine timeouts vary between runs).

**Correction: `--blocks 2` is about 1.2× as fast, not 2×.** Over eight days, two blocks at once
took 25:40 against 30:49 (1.2×; one redone block included, so at most about 1.3× without it),
for 44 % more peak memory. GloFAS window 2 (three days, `99ebb11`) gave the same: 11:03 against
9:11. The 17 Nov entry's "consistent with the expected factor of about 2" compared one clean run
with an estimate of the one-block runs less their stalls; on three days two blocks cannot be
more than 1.5× faster anyway (two rounds instead of three). Two processes each kept their speed
(16 Nov), but two blocks in one process do not: they share its Dask thread pool, the GIL, the
HDF5 lock and the Copernicus Marine connection pool. Which of these limits it is for the
performance pass; until then, expect about 5 days per resolution with `--blocks 2`, not 3–3.5.

## 2026-11-18: group 8, chunks planned instead of set by recipes

Download recipes set 21 chunk sizes by hand (part `chunks`, regrid `output_chunks`, `chunk`
steps), tuned by trial and error for one machine. They are now refused at load, and each dataset
of each block (or checkpointed sub-block) is chunked by `chunk_plan.plan`:
- **Shapes.** The plan uses the shapes and dtypes of the variables the dataset's first selection
  steps keep.
- **What the steps need (`Needs`).** Whole latitude and longitude for `regrid`. Whole bins for
  `resample`, when its bins start at the block's start. For `get_bottom`, the whole depth: it
  counts when sizing that step's tasks.
- **Source chunks.** The plan prefers multiples of the sources' stored chunks.
- **A memory budget.** `--memory-limit`, `ARCOMAKE_MEMORY_LIMIT`, or what the process may use,
  shared among threads and blocks. Each task holds about 4 chunks, and chunks aim at Dask's
  `array.chunk-size` at most.

Space stays whole unless one step is above the target; then outer dimensions are split, and time
grows in whole bins. Writes (output and checkpoints) align Dask chunks to the stored ones
(xarray's `align_chunks`). `download --plan` prints the plan without computing. Download recipes
must give `[save] chunks`: otherwise the stored chunks would follow the plan, which depends on
the machine. The recipes' `output_chunks` had no effect: xarray-regrid reads only the entries of
the regridded dimensions. A high-effort review found ten issues, fixed in six commits, each with
a test (three checked to fail on the code before).

On the shapes of the 0.25 deg sources, the plan gives the sizes set by hand:
- ERA5: one day of hourly data;
- GLORYS: one day and one level (decoded to float64, two levels exceed 128 MiB);
- WAVERYS: one day of 3-hourly data.

GLORYS `get_bottom` tasks hold a whole column (3.3 GiB, above the 705 MiB per task the budget
allows on the VM), which is reported as a warning.

Window 30 Dec 2012 – 7 Jan 2013, GloFAS included, one run at a time:

| Run | Code | Blocks | Chunks | Wall time | Peak RSS |
|---|---|---|---|---|---|
| t8-b1 (18 Nov, above) | before | 8 x 1 day | by hand | 30:49 | 22.4 GiB |
| p8-b1 | planner | 8 x 1 day | planned | 35:53 (one block redone) | 23.4 GiB |
| p8-b2 | planner | 8 x 1 day, `--blocks 2` | planned | 19:00 | 35.1 GiB |
| o8d-b1 | before | 1 x 8 days | by hand | 7:21 | 101.8 GiB |
| p8d-b1 | planner | 1 x 8 days | planned | 8:42 | 43.7 GiB |
| c128-8d | final | 1 x 8 days | planned, 128 MiB | 7:59 | 45.2 GiB |
| c256-8d | final | 1 x 8 days | planned, 256 MiB | 4:42 | 46.5 GiB |
| c512-8d | final | 1 x 8 days | planned, 512 MiB | 8:10 | 60.5 GiB |
| ck1d-8d | final | 1 x 8 days, GLORYS checkpointed daily | planned, 128 MiB | 27:35 | 19.0 GiB |

All data variables are identical bit for bit across the runs. One difference: the `time`
coordinate is stored in chunks as long as the first block (8, not 1). That depends on the block
length, not on the planner, and happens with the old code too.

What the figures say (single runs, network noise included):
- **Block length.** Long blocks are fast. One-day blocks pay fixed costs in each block: 20 s to
  5 min opening sources, mostly EWDS queueing, and the static masks recomputed for
  `masked_fill`. They also compute at 100–270 s per day, against about 40 s per day in an 8-day
  block.
- **Memory.** It grows with block length. With hand-set chunks, the 8-day block came within a few
  GB of the VM's 110 GB: GLORYS chunks of 2 days make each `get_bottom` task gather 2 days x 50
  levels. Planned chunks halve that.
- **Chunk size.** 256 MiB was the fastest at the same memory; 512 MiB was slower, with more memory.
- **Checkpointing.** Checkpointing GLORYS per day halves memory but is 3.5x slower: it computes
  the sub-blocks one after another, and GLORYS no longer overlaps the other sources. The
  maintainer's direction is to avoid it in production.

Next:
- a 32-day window with blocks of 8, 16 and 32 days and `--blocks 2`, to choose the production
  block length (128 days now) and decide whether `--blocks` stays;
- profiling one 8-day block;
- one commit per run (Icechunk's `Session.amend`).

## 2026-11-18: performance: block length, connections, EWDS requests, shards

Measured on the project VM (32 cores, 110 GB), one run at a time, 256 MiB chunks unless said,
single runs with network noise; every output compared bit for bit (data variables) with its
reference: all identical. "Computing" is from the first write to the commit of a block (log
timestamps); "opening" is before it (sources, EWDS queue).

**Block length** (window 30 Dec 2012 – 31 Jan 2013, GloFAS included, code `428d812`):

| Blocks | Wall time | Opening | Computing | Peak RSS |
|---|---|---|---|---|
| 4 x 8 days | 32:47 | 16:00 (one EWDS wait of 11 min) | 16:40 (31 s/day) | 49.2 GiB |
| 4 x 8 days, 128 MiB chunks | 28:13 | 8:03 | 20:03 (38 s/day) | 60.5 GiB |
| 2 x 16 days | 22:54 | | | 66.8 GiB |
| 1 x 32 days | 15:43 | 1:59 | 13:34 (25 s/day) | 63.3 GiB |
| 2 x 16 days, `--blocks 2` | 17:38 | | | 87.3 GiB |
| 4 x 8 days, `--blocks 2` | 17:46 | | | 55.5 GiB |
| 1 x 64 days (to 4 Mar 2013) | 42:17 | 15:16 (EWDS) | 26:50 (25 s/day) | 73.6 GiB |

Computing costs about 25 s per day from 32-day blocks on; memory levels off. `--blocks 2` gains
on short blocks, but a longer block is faster, with less memory: `--blocks` was removed. The
64-day output's first 32 days equal the 32-day run's. The maintainer chose 32-day blocks for
production, without checkpointing. Chunk sizes do not depend on block length from 2-day blocks
on: ERA5 grows in whole days (one hour of a variable is 4 MB) to 48 h, 190 MiB, the most below
256 MiB.

**Profile** (one 8-day block, py-spy as the parent process, `--idle`, and a Dask callback timing
tasks by name): the processor averaged 3.4 of 32 cores. Dask's threads spent 81 % of their task
time waiting for zarr reads; GLORYS 3D reads were 58 % of all task time, Icechunk writes 10 %,
regridding 7 %, `get_bottom` 1.6 %, decoding and masking about 3 %. No step is worth a numba
kernel (`get_bottom`'s own code: about 15 of 9,500 thread-seconds). 29 % of the Copernicus
Marine reading time went into opening connections (DNS, TCP, TLS): the toolbox keeps 10 per
client, Dask runs 32 threads, so connections were discarded ("Connection pool is full", 773 in a
block) and opened again.

**Copernicus Marine pool** (8-day block without GloFAS):

| Pool | Computing | Peak RSS |
|---|---|---|
| 10 (the toolbox's default) | 4:21 | 48.3 GiB |
| 32 | 2:45 | 35.2 GiB |
| 64 | 2:45 | 35.8 GiB |

On a 32-day block (GloFAS included) pool 32 computed in 12:11 instead of 13:34 and peaked at
47 GiB instead of 63. arco-make now sets the pool to Dask's thread count (at least 10) unless
given. With it, the 32-day block took 0:59 + 12:21 at 43.9 GiB. Network during that run: 62 GB
received, 43 GB from Copernicus Marine (32 connections, 58 ms round trips: about 2 MB/s each),
18 GB from Google Cloud Storage (13 ms); the interface averaged 77 MB/s and peaked at 328 MB/s,
so the link is not the limit.

**More requests in flight do not help** (8-day block without GloFAS):

| Dask threads | zarr executor (pool alike) | Computing | Peak RSS |
|---|---|---|---|
| 32 | 36 (Python's default) | 2:53–2:59 | 37–39 GiB |
| 64 | 36 | 4:45 | 49.3 GiB |
| 96 (chunks shrink to 235 MiB) | 36 | 6:24 | 58.8 GiB |
| 32 | 96 | 4:28 | 48.3 GiB |
| 32 | 128 | 4:06 | 42.9 GiB |
| 24 | 36 | 3:14 | 35.9 GiB |
| 16 | 36 | 3:08 | 38.9 GiB |

The toolbox's Zarr store runs its blocking requests on zarr's executor (36 threads by default),
so about 36 requests are in flight at most. Raising that, or the threads, slowed the reads:
Copernicus Marine seems to limit a client, and sends nothing for minutes now and then whatever
the settings (151 s once while opening). Fewer threads do not save memory: it is mostly static
fields, regridding weights and data waiting to be written. The defaults stay: 32 threads, pool
alike, executor 36.

**EWDS requests.** The EWDS runs a user's GloFAS requests one at a time, about two minutes apart.
Measured on months no one in the project had requested (submitted together, server timestamps):

| Request | Queued | Processing | Size |
|---|---|---|---|
| Jan + Feb 1985, one request | 14 s | 53 s | 303 MiB |
| Jan 1986 | 195 s | 26 s | 162 MiB |
| Feb 1986 | 357 s | 32 s | 145 MiB |
| all of 1987 | 394 s | 298 s | 2,065 MiB |

The service takes lists of `year`, `month` and `day` and gives every valid date of their product,
up to 500 days per request; date ranges and the old `hyear`/`hmonth`/`hday` names are refused.
The backend now makes one request per year of a block (its months x its days) and drops the days
outside the block: a 32-day block across two months waits once instead of twice, for up to a
month more data.

**[Moby-Dick] Shards** (32-day block, chunks of 1 day x 1 level x the whole grid, shards of
8 days x 50 levels; Morton order, the code then had no other):

| Layout | Computing | Peak RSS | Objects | Size |
|---|---|---|---|---|
| chunks only | 12:21 | 43.9 GiB | 7,193 | 13 GB |
| shards | 12:58 | 40.6 GiB | 116 | 13 GB |

Each 3D shard (1.55 GiB) is written whole by one task, as warned. The maintainer chose this
layout for all three resolutions, with `[save] shard_order = "lexicographic"`, new: a day's 50
levels lie next to each other in a shard, so a whole depth profile is read in few requests (zarr
merges ranges less than 1 MiB apart). The order is not in the metadata: readers need nothing,
and find chunks through each shard's index.

Next (maintainer, 18 Nov): the stack is merged before the production run, which uses the
updated image; group 9 and the Moby-Dick findings PR come before it. One commit per run
(Icechunk's `Session.amend`) is dropped from the plan: a TODO at `icechunk_utils.Run.write`.

A high-effort review of the branch found five issues, fixed in five commits with tests (two
checked to fail before): with `shard_order`, the first block's metadata write checked unaligned
Dask chunks against the shards, so a download whose planned chunks straddle shards (ERA5's
2-day chunks in 8-day shards) failed; the recipe fingerprint counted options at their default,
so any new option blocked `--resume` across an upgrade; per-year EWDS requests could exceed the
500-field limit (it counts days x variables: a year of two variables costs 730) or name months
not published yet (refused whole; a request mixing published and unpublished months is accepted
and the latter left out): the service's cost estimate is now asked first, with per-month
requests as before when it refuses, and a block missing days fails naming them; the pool
default stayed in the environment; a name from `--blocks`. Its claim that distributed workers
keep the shard order was right: zarr pickles the order with the codec, so the warning saying
otherwise was removed. On real data (8-day block without GloFAS, shards of 8 days x 50 levels,
lexicographic): values identical to the unsharded run, the 400 chunks of each 3D shard in
lexicographic order.

## 2026-11-18: group 9, general capabilities for the recipes

Six capabilities the arco-ocean recipes need, each general, with a synthetic test; none changes
the output of an unchanged recipe:
- **`ensure_no_nans` in one computation** (S3): it looped over every day (11,322 in the published
  dataset) and variable, two computations each. Now one Dask computation counts the missing
  values inside each mask per day; the report is the same, static variables once. On the 32-day
  output (hyperfine, 2 runs each): 0.5 min against 1.4-3.9 min; both found no missing value.
- **`validate --start/--end`** (C11): the time check's bounds as given to `download`; the
  arco-ocean validate recipe ended years before the data.
- **`sin_cos`** (C1, 5.3): an angle replaced by its sine and cosine before regridding and daily
  means (350 and 10 degrees averaged to 180 as numbers). The maintainer chose the names `mwd_sin`
  and `mwd_cos` for the wave direction.
- **`static_time`** (C5, findings 3 and 4): a part read at one stamp, without time: ERA5's `z` and
  `lsm` once instead of 24 times a day, and without the glitches of a few days.
- **`[grids]`** (A4): a target grid written once per recipe and named by its regrid steps.
- **`coverage_attrs`** (C13, finding 15): ACDD extent attributes from the coordinates. xarray
  replaces a store's attributes with each appended block's, so `download` sets the time coverage
  of the whole output (stored stamps and the block's) for each block.

Masking after regridding needs nothing new: `apply_mask` does it; the arco-ocean recipes will use
it in the Moby-Dick findings PR (NaN outside the masks).

A high-effort review found nine issues, fixed in five commits (tests checked to fail before where
the fix changes behaviour): the time coverage was per block (no resolution with one-stamp blocks,
removed by a short last block; a checkpointed dataset's start), now the whole output's; grid
resolutions of 0.1 and float32 1/12 degree were missing; `[grids]` changed the resume
fingerprint, and a grid that is not a table raised a traceback; a zoned `static_time` was
converted; two comments named one dataset's data; `validate` repeated `read_configs`. Its claim
that `ensure_no_nans` lost an exact join was checked and dropped: variables of one Dataset share
its indexes, so a variable and its mask cannot be misaligned.

A second high-effort review, of the fixes, found six more, fixed in three commits (tests checked
to fail before): `coverage_attrs` in a dataset's steps fails the merge of the datasets (they need
identical attributes), so it is refused there; fine float32 grids far from zero (0.01 degree to
360) still lacked a resolution (the tolerance now follows the type's rounding); download dropped
every `time_coverage_*` attribute, not only those it computes; `validate` read its recipe twice
(`read_configs` now merges injected tables); a UTC `static_time` was refused; this entry counted
six fix commits. The earlier fix had said the whole-output coverage also fixed a checkpointed
dataset: it did not need to, the recipe's steps run on the reopened block. Left as they are: no
`output:` prefix (no recipe could use `coverage_attrs` before this branch), the fingerprint
change for runs of older code (unreleased), and reading the output's time coordinate for each
block (about 380 blocks in production).

## 2026-11-19: coarsening GLORYS and GloFAS, decided with the maintainer

The reasons and the figures are in `docs/coarsen-ocean-datasets.md`, measured at 0.25, 0.5 and
1 degree by `docs/coarsen_analysis.py`; the decisions, in order of the discussion:

- **Sea is any water** (**[Moby-Dick]**): a coarse cell is sea at a level where GLORYS has some
  sea cell under it. `seafrac` (static), GLORYS's 3D mask regridded conservatively, is published:
  the share of the cell where the model has a sea cell. The recipes' rule (the regridded depth
  below the level's centre) is GLORYS's own rule applied to a flat column: right in total, wrong
  by layer (+1.2 % of the water at the surface, -1.6 % at 5275 m, at 0.25 deg), and 8,869 surface
  cells it called sea were less than half sea. With any water, regridded values (nan_threshold 1)
  and sea cells coincide; masks count every sea cell as full (+5 % of the volume at 0.25 deg, +16 %
  at 1 deg): volumes, heat contents and a model's loss should weigh by `seafrac` x layer thickness x
  area (a note for Moby-Dick). `glorys_mask` is gone; `deptho` is the mean depth with land as 0.
- **Level 49 dropped** (**[Moby-Dick]**): NEMO never computes it (no value on any day); 49 levels.
- **Gaps filled from above** (**[Moby-Dick]**), velocities included, 0 where nothing is above: the
  few sea cells without values (1,505 at 0.25 deg, the same every day) take the nearest value
  above in the column. Leaving a level out and refilling it, this beats the regridded bottom
  values the recipes used at every level and resolution (temperature RMSE at 5275 m: 0.043 against
  0.084 at 0.25 deg, 0.041 against 0.63 at 1 deg). `*_bottom` variables are gone.
- **No latitude cut**: the rows it removed (-77, 90) are sea in GLORYS's static mask without values,
  but leave no coarse sea cell empty; the cut itself caused round-off fractions (1e-8) at three
  0.25 deg columns.
- **Sea ice** (**[Moby-Dick]**): GLORYS's `sithick`, `usi`, `vsi` are means over the ice cover
  (`cell_methods = "area: mean where sea_ice"`), which the recipes averaged with open water and
  land as 0 (thickness neither of the ice nor per area; ice drift slowed by about 17 % where the
  ice covers less than half a cell). Published now, over the sea part: `siconc`, `sivol` (siconc x
  sithick), `usic`, `vsic` (siconc x velocity); GLORYS's thickness and velocity come back as ratios
  to `siconc` (relative error 1.5e-7 in float32), ice area and volume are kept.
- **Rivers** (**[Moby-Dick]**): the recipes' `dis24` was a mean over coarse cells, kept only where
  a cell is all land (nan_threshold 0): 1.7 % of the discharge reaching the coast was kept, the
  Amazon's none; inland it was a fifth of the main river at 0.25 deg, a twentieth at 1 deg.
  `dis24_outlets` replaces it: the discharge at the pits of GloFAS's drainage network (LDD v4.0,
  152,629 pits; their upstream area equals the land area), in metres per day over each cell,
  regridded conservatively with its total checked (1,079,449 m3/s on 1 Mar 2013). `dis24`,
  `glofas_mask` and `uparea` are gone.
- **WAVERYS dropped**: its sea differs from GLORYS's (none north of about 81 N), waves are missing
  where ERA5's ice cover exceeds about 0.3, and the recipes averaged the direction arithmetically
  and filled it with 0 under ice.

arco-make, generic: steps `fraction_mask`, `fill_from_above`, `products`, `divide_by`,
`keep_where`, `per_unit_area`; `regrid`'s `conservation_tolerance`; parts' `rename` and
`coordinate_tolerance` (the LDD's coordinates differ from the discharge's by 4e-12); the check
`fraction_domain` (values exactly where a static fraction is positive); static datasets computed
once per run, and compared with the stored ones by a resumed run (they were recomputed for every
block and discarded by the appends). Two fixes: `regrid` with `stat` crashed on datasets
(xarray-regrid 0.4.2 reads `.dtype` of a Dataset when the target extends beyond the source; also
on upstream main; NaN is now passed as `fill_value`), and `apply_mask` made 2D fields 3D with a 3D
mask. xarray-regrid blanks target cells centred outside the source's centres, whatever the
method (GloFAS loses its -60 and 90 rows: no rivers there); the conservation check catches it.

Checked: the full suite; the notes' doctests, offline and live; the analysis at three
resolutions; a two-day download with the new 1 deg recipe (static datasets computed once, river
total equal to the pits', `validate` with `fraction_domain` passed).

A high-effort review found ten issues; nine were fixed in six commits (tests checked to fail
before where behaviour changed): `products` and `fraction_mask` removed every coordinate's
attributes; the static comparison ran at the second block of every run, not only on `--resume`,
and failed on static variables packed by `[save.encoding]` (`scale_factor`); `fraction_domain`
crashed on a missing fraction, kept one fraction's counts for a variable under two, and accepted
`inclusive` at 0; the conservation check could not be built on outputs split along latitude or
longitude; `coordinate_tolerance` let NaN coordinates through and did nothing on the first part;
`fraction_mask`'s docstring on `tolerance = 0`. A second review, of the fixes, found nine more;
seven were fixed in four commits: a fraction stored as a coordinate was "not found"; the static
comparison read units such as "seconds" as durations (NaT never equal) and compared again after
a first block retried once its commit had landed; the conservation check copied and promoted
values through `where` (now passed through unchanged, coordinates joined exactly);
`coordinate_tolerance` refused equal NaN positions and ignored a tolerance matching no coordinate.
The two points left by the reviews were done afterwards: `ensure_no_nans` and `fraction_domain`
share their count-and-report code (same reports, pinned by a test that passes before and after).
The attribute fix has no `output:` prefix (the commit was pushed): it changes the output only of
recipes using `products` or `fraction_mask`, new on this branch, where no other dataset gives the
coordinates their attributes. The 1 deg fork recipe's two-day output is identical before and
after it (values, encodings, coordinate attributes; the merge took them from other datasets).
A third review, of these commits, found a defect in older code, fixed: `ensure_no_nans` raised
KeyError on a mask missing from the dataset, stopping `validate`. Left to the maintainer, since
they change what existing recipes accept: a listed variable missing from the dataset only warns
in `ensure_no_nans` (an error in `fraction_domain`), and a mask with NaN counts them as inside.
Both were decided on 20 Nov (below).

### 20 Nov: cell measures, one check for masks

Maintainer's decisions: GLORYS's sea and ERA5's grid stay separate domains (the grids do not
overlap, and users may postprocess them differently; no routing of outlets to the sea); ERA5's
`z` and `lsm` stay as they are; the dataset publishes cell bounds, areas and sea volumes, with
areas on the sphere of GLORYS12's static product; `ensure_no_nans` absorbs `fraction_domain`
and fails on a listed variable missing from the dataset.

- GLORYS12's static `e1t` and `e2t` are spherical (`e1t` = R cos(lat) dlon) with R = 6,378,100 m,
  not NEMO's 6,371,229 m nor the mean radius. Any sphere is off the WGS84 ellipsoid by +0.45 %
  (cells at the equator) to -0.89 % (near the poles); the geoid's departures (about 100 m) are
  negligible. The static product's two parts (`bathy`, `coords`) give depths differing by float32
  rounding (at most 0.5 mm): the `e3t` part takes `coordinate_tolerance`.
- New: `regrid` passes variables without the horizontal dimensions through (the conservative
  method failed on them, linear and nearest gave them a latitude dimension: `output:`); steps
  `cell_bounds` (CF bounds as coordinates; depth from thicknesses) and `cell_area`; a resumed
  download compares static coordinates too; `ensure_no_nans` reads masks as `mask > threshold`
  (boolean masks and fractions alike, NaN and negative round-off outside), takes `exact` (true or
  a list of masks), names the mask of each count, and refuses a static variable under a
  time-varying mask.
- At 1 deg (1-2 Mar 2013, the fork recipe): the total of `volcello` equals GLORYS12's own volume
  (sum of `e1t` x `e2t` x `e3t` x `mask`, levels 0-48, 1.330e18 m3) to 1.8e-7, every level to
  3e-7; the sea area at the surface agrees; the river total rebuilt with `areacello` equals the
  pits' (1,079,448.71 m3/s); `areacello` sums to 4 pi R^2 to 7e-10; `validate --fail` passes.
- **[Moby-Dick]** The recipes kept GLORYS12's attributes on `thetao`, `so`, `uo`, `vo`, `zos`:
  `cell_methods = "area: mean"` (the coarse values are means over the sea part) and the
  `valid_min`/`valid_max` of GLORYS12's packed int16 values (thetao: -32766 to 21306), meaningless
  on unpacked floats. The new recipes replace them; the published dataset probably carries them
  (not checked).
- A high-effort review of these commits found nine points; eight were fixed before pushing.
  Left: a mask whose inside is negative (-1 for sea, say) is now all outside (the maintainer's
  `mask > threshold` rule).
- Correction (maintainer, later on 20 Nov): "`z` and `lsm` stay" meant that they stay in the
  dataset, as static data. The fork recipes had never adopted `static_time` and still stored them
  as daily means; they now read them at 2000-01-01T00:00, once per run, without time. The
  source's values at that stamp equal those of 1993-01-01, 2010-07-01 and 2021-06-30; `z`
  differs from 2022-10-01 on at the two polar rows (2,880 cells, up to 364 m2 s-2). At 1 deg the
  static fields differ from the old daily means by float32 rounding only; the statistics leave
  them out (Moby-Dick notes, item 3: their z-scores were rounding noise).

### 20 Nov: findings for Moby-Dick, checks on each block

The Moby-Dick findings plan (maintainer, 17 Nov), with what the coarsening already did:
- `msl` becomes `mslp` in the arco-ocean recipes (the fork; a typo, "in ARCO-OCEAN v2").
- NaN outside the masks: done by the coarsening (`apply_mask` on the sea, `ensure_no_nans` with
  `exact`).
- Checks before each block's commit: a download recipe may give `[checks]`. The time axis is
  checked over the block, the coordinates and attributes before writing, and missing values
  while the block is written: `checks.guard_no_nans` wraps each variable in a pass-through
  `apply_ufunc` that raises on a chunk breaking the rule, so nothing is read twice (reading a
  32-day block back from the output would mean about 13 GB at 0.25 deg). A failing block is not
  committed; the message names the check, the variable, the mask and the first day wrong in the
  chunk that failed (chunks fail in any order); `[checks]` is out
  of the fingerprint, so the run resumes once the cause or the check is fixed. On the 1 deg fork
  recipe (1-9 Mar 2013) the block passed; the run took 5 min (its cost against a run without
  checks was not measured).
- The notes of Moby-Dick's development (`arco-ocean-notes.md`, 19 items): items 1-2, 10-13 were
  handled by group 6 and the dropped tendency std; 3-4 by static `z` and `lsm` (read at
  2000-01-01T00:00; the source's `z` differs from 2022-10-01 on at the two polar rows, 2,880
  cells); 5-6 and 15 by dropping WAVERYS; 7-9, 16-17 by the coarsening; 14 by the shard layout;
  18 by the new check `attributes` (required attributes, distinct `long_name`s); 19 by the
  attribution of every source in the fork's recipes.
- The notebook `notebooks/regrid_glorys12.ipynb` was reviewed in the coarsening (its notes,
  `docs/coarsen-ocean-datasets.md`).
- The pre-commit hook `ruff format` checks Markdown too, as CI does.

Fork recipes (`production-recipes`): `mslp`; `[grids]` (the grid once per recipe, same
fingerprints); the production layout (32-day blocks, shards of 8 days and all levels,
lexicographic); `coverage_attrs` instead of hand-written extents (they said longitudes -180 to
180 over 0 to 359 and a coverage to 2023); attribution of ERA5, GLORYS12 and GloFAS; `[checks]`;
and a fix: `scripts/download.sh` still passed `--blocks`, removed from arco-make, so it failed at
once. Open for the maintainer: the recipes end on 2024-01-01 but GLORYS12's multi-year product
ends on 2021-06-30 (end there, or add its interim product); how outputs record the arco-make
version (A15: the version is a fixed "0.1.0").

### 20 Nov: sources to mid-2026, statistics over 1993-2023, provenance, extensions

Maintainer's decisions: statistics in the datasets' shard layout and over 1993-2023 (a 30-year
normal, as climatological normals are; was 1993-2018 with later years held out); the time mean
is no longer computed; the analysis/forecast recipe is removed, the Mediterranean one kept
loadable but not run; the extent maximal; recipes without dates; provenance in the commits'
metadata, not their messages.

- **How far the sources reach** (20 Nov 2026): GLORYS12's multi-year dataset id
  (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, version 202311) continues past 2021-06-30 to
  2026-10-06, values continuous across mid-2021 (no separate interim id is served); ERA5's ARCO
  store has final data to 2026-06-30 and ERA5T, preliminary and revised within months, to
  2026-11-04; GloFAS v5 consolidated to 2026-08-31 (the service's cost estimate, day by day; its
  intermediate product refuses these requests, v4 and v5). The historical recipes end on
  2026-07-01, the last day every source has as final data, so revisions stay out. An earlier
  commit of the fork had set 2021-07-01 on the wrong belief that the multi-year id ends there.
- **Provenance:** every commit's metadata hold `arcomake_provenance` (arco-make's version and
  commit, Python and OS, CPUs and memory, the versions of the packages that decide the values, a
  hash of every installed distribution; nothing naming the machine or its paths) and the
  recipe's fingerprint, for `process` and `compute-climatology` too. The image build writes the
  commit into `src/arcomake/BUILD_COMMIT` (the image has no history). Icechunk keeps nested
  metadata (checked).
- **Recipes without dates; `download --extend`:** the fingerprint leaves the dates out; a recipe
  may omit them (the command gives them); `--extend` appends the days after `main`'s last stamp
  as a new version, with the same recipe and the static variables compared; an extension equals
  one download over the whole period (tested), and a failed one resumes. Estimated before:
  about 100 lines in `download`; it took about 70 in `download` and `icechunk_utils` (the dates
  made optional included), plus 125 of tests. Rewriting days already
  stored (for revised data) is left out: ending at final data makes it unnecessary.
- `coverage_attrs` takes `time_dim = false` (a recipe cannot give None), for statistics whose
  time axis holds the bins of a year.
- The xarray-regrid issue (`stat` on a Dataset beyond the source) still reproduces on 0.4.2 and
  upstream main; posting it needs the maintainer (the VM's token cannot open issues elsewhere).

### 21 Nov: the climatology's graph path, memory and time

The statistics recipe (shards of 8 bins and all levels) ran out of memory on the graph path even
at 1 deg over 8 days. Traced task by task on the local scheduler (results alive in Dask's cache,
each buffer counted once, and RSS), on synthetic data and the 1 deg test output:

- **Task explosion:** Dask's rechunk to whole bins for the window cut the grid into small tiles,
  747,727 tasks for one 3D variable; folded data are now chunked once (33,384 tasks; 670 -> 531 s
  on the 1 deg output, memory unchanged).
- **All levels at once:** the variance's second pass was tied with `bind` to a checkpoint of every
  chunk's mean, so every level waited for all levels' means: the statistics of all levels were
  computed first and held until their windows (10 levels: 3.54 GiB alive instead of 0.44). Untied,
  each level needs only its own chunks. Reading 4 levels at 1 deg from disk, window included:

  | Fold statistics | 6 years, 1 thread | 6 years, 8 threads | 12 years, 1 thread | 12 years, 8 threads |
  |---|---|---|---|---|
  | two passes, `bind` (before) | 62 s, 5.5 GiB | 14 s, 7.7 GiB | 79 s, 8.6 GiB | 22 s, 10.2 GiB |
  | two passes, untied (now) | 52 s, 2.8 GiB | 14 s, 8.1 GiB | 76 s, 4.4 GiB | 23 s, 8.1 GiB |
  | Dask's one-pass variance | 85 s, 7.4 GiB | 21 s, 19.1 GiB | | |
  | one pass, lean reduction | 68 s, 5.9 GiB | 18 s, 18.8 GiB | | |

  `bind` never made the computation faster; it read the input twice and serialized the levels.
  One-pass reductions keep larger per-chunk intermediates (counts and moments in 64 bits per
  value of one period). Not measured on a distributed cluster.
- **Shared work computed twice:** `write_outputs_together` computed the two outputs'
  `to_zarr(compute=False)` tasks together, but `dask.delayed` optimizes each on its own (and
  makes its keys unique), so a task with one consumer in each output was fused into both: the
  window and everything before it ran once per output. Within one dataset, shared work runs once
  (checked); download is not affected. Now one xarray `ArrayWriter` for both outputs (441 ->
  216 s, 10 levels, 1 thread). A DataTree of two groups in one store, or one dataset with
  suffixed variables, also share the work, with public API only (checked).
- **The window** was NumPy over float64 temporaries (19.2 s per 1 deg level, 1.4 GB per band); a
  numba kernel gives the same bits in 6.6 s, in the output's type. Both paths use it.
- Together, writing 10 levels of 8 days at 1 deg (1 thread): 410 s and 6.35 GiB -> 43 s and
  1.32 GiB. The 1 deg test output (3 variables, 49 levels, 4 threads): 670 s and 26.3 GiB ->
  80 s and 4.1 GiB without shards; with shards of 8 bins and all levels 739 s and 35.1 GiB ->
  74 s and 18.3 GiB: a shard spanning all levels and the grid can be written only once every
  level's window is done, so all of a variable's statistics are held (about 150 GB for both
  outputs of a 3D variable at 0.25 deg).
- **Graph against streaming** (`split_dims`), `compute-climatology`, 4 levels at 1 deg from local
  disk, 32 threads: 6 years 23 s and 6.5 GiB against 34 s and 2.4 GiB; 12 years 27 s and 8.7 GiB
  against 38 s and 3.3 GiB. Streaming reads one block at a time per series; the graph reads in
  parallel, which should matter more from S3 (not measured).
- **Checkpoint** (maintainer's suggestion, accepted): the graph path computes both outputs once
  into a temporary plain Zarr v3 store (a DataTree, one group each; unsharded, Dask's chunks, and
  along time whole stored units up to Dask's chunk size), then copies each into its repository,
  one task per group of stored units. The shared session writer (`write_into_sessions`, on
  xarray's private API) is gone: writes computed together share no work, so the outputs are
  written in turn, with any scheduler. The 1 deg test output (4 threads): without shards 80 s
  and 4.1 GiB -> 97 s and 4.0 GiB; with shards of 8 bins and all levels 74 s and 18.3 GiB ->
  104 s and 4.1 GiB; outputs identical bit for bit (values, attributes, encodings, array
  metadata). With one-bin units the temporary store first had a chunk per bin and level (224 s:
  small files); whole units up to Dask's chunk size fixed it. Temporary disk at 0.25 deg: about
  150 GB per 3D variable for both outputs, before compression (warned about if the disk has less
  free). On a distributed client every worker must see the store (checked: a chunk written where
  the copy does not look would be read as NaN).
- **Maintainer's decisions:** the statistics are one repository, a DataTree with the groups
  `climatology` and `anomaly_std`, written in one commit (both paths); the streaming path is
  chosen with `--streaming` (`split_dims` only says how it splits). The temporary store's layout,
  measured on the 1 deg test output with 32 threads (without shards / with shards of 8 bins and
  all levels in the output): chunks of whole stored units, unsharded, 66 s / 73 s; one shard per
  Dask chunk with chunks of one unit, 80 s / 74 s; Dask's chunks, 68 s / 106 s (each copy reads
  whole chunks of 365 bins for 8). Memory 26.5-28.5 GiB in all six runs (it follows the threads:
  4.1 GiB with 4). Shards in the temporary store bring nothing: unsharded chunks of whole units
  are kept.
