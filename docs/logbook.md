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
- The 0.25° levels-50 download (30 Sep – 8 Oct 2026, on the project VM) ran in the cached image
  (built 30 Sep). Its `src/arcomake` is identical to commit **`6f0e2f1`**
  (2026-09-30), not to `847b6f2`. `6f0e2f1..847b6f2` changes `download.py`, `dataset_utils.py`,
  `climatology.py` and the Dask setup.
- It ran as 45 pieces of 256 days, about 3.3 h each and 165 h in total. Pieces 1, 16, 19, 20, 34 and
  41 failed on the first attempt and were rerun with `--overwrite` (7–8 Oct); 41 needed a third try.
  The errors in the logs are transient network ones: EWDS retries, DNS failures, connect timeouts.
- The arco-ocean log folder named for the 0.5° recipe holds that 0.25° run,
  despite its name.

**Environment:**
- The host's apt eccodes 2.45 cannot decode the GloFAS GRIB served by EWDS (`Unable to find
  template localSection from grib2/local.98.65535.def`, then `KeyError: 'gridType'`). The image
  builds eccodes 2.48 from source, which works. Local runs use the image's library and
  definitions via `ECCODES_DIR`, `ECCODES_DEFINITION_PATH` and `ECCODES_SAMPLES_PATH`.

**Real-data baseline at `847b6f2`** (0.25° levels-50 recipe, copied outside the repo,
on the project VM with the image's eccodes 2.48, threads scheduler):

| Window | Result | Wall time | Peak RSS | Output |
|---|---|---|---|---|
| 1993-01-01 → 01-03 | ok | 22 min | 36 GB | 832 MB |
| 2012-12-30 → 2013-01-02 | attempt 1: `ConnectTimeout` on `confluence.ecmwf.int` (GloFAS `uparea`, NetCDF over HTTP: no retry) after 25 min; attempt 2: **segmentation fault** after 22 s | | | |
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
- **Comparison, `29a9edd` vs `847b6f2` on window 1:** the array metadata were identical (dtype,
  chunks, compressor, fill value, attributes), but ERA5 values (`2t`, `msl`, `ssrd`...) and a few
  wave points differed by a few float32 ULP, e.g. 1.2e-4 K on `2t`.
  - Cause: the B5 fix selected the time range with an index array. That rechunked the hourly
    source (24 → 16 steps per chunk), so the daily resample summed in another order.
  - Fixed in `d7b49dc` (slice selection), with a test on the chunks.
  - Note: any rechunking before a reduction changes float32 results at this level. "Same output"
    has to be checked bit for bit, not with a tolerance.
  - Peak memory was 27 GB instead of 36 GB; to be re-measured with the fix.

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
