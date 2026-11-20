# ARCO-MAKE

Arco-make prepares Analysis-Ready Cloud-Optimized (ARCO) timeseries datasets in Zarr from individual data
sources. It works from *recipes*: TOML files that say where each source is, how to process it (regridding,
resampling, masks, renaming...), and how to store the result. It also computes statistics of the result,
such as a climatology, for standardizing the inputs of data-driven models.

It was built to produce [ARCO-OCEAN](https://github.com/inogs/arco-ocean), whose recipes live in that
repository.

## Installation

Arco-make requires Python 3.13 or later. Install it with [uv](https://github.com/astral-sh/uv) or `pip`:

```bash
pip install .
```

For development, with the test and lint tools:

```bash
uv sync --extra dev
```

A container image with an MPI build is published on the GitHub Container Registry:

```bash
apptainer exec oras://ghcr.io/stefanocampanella/arco-make:latest arco-make download --help
```

## Commands

| Command | What it does |
|---|---|
| `arco-make download RECIPE OUTPUT` | Downloads, processes and merges the datasets of a recipe into one Icechunk repository |
| `arco-make validate RECIPE INPUT` | Checks a dataset: time axis, grid, missing values inside masks; `--start`/`--end` give the expected time axis, as to `download` |
| `arco-make process RECIPE INPUT OUTPUT` | Applies recipe steps to a dataset (e.g. a time mean) |
| `arco-make compute-climatology RECIPE INPUT CLIMATOLOGY ANOMALY_STD` | Computes a periodic climatology and the standard deviation of the anomalies |
| `arco-make steps` | Prints the reference of the processing steps ([docs/steps.md](docs/steps.md)) |

Every command documents its options with `--help`. Paths may be local or fsspec URLs (e.g. `s3://`, `gs://`).
The usual workflow is `download`, then `validate` and `compute-climatology`.

## Settings and credentials

Credentials and machine settings come from the environment, never from recipes: Copernicus
Marine (`COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD`), EWDS and CDS
(`CDSAPI_URL`, `CDSAPI_KEY`), cloud storage (`AWS_PROFILE` or `AWS_DEFAULT_PROFILE`, the default
profile of `s3://` paths;
`AWS_CONFIG_FILE`, `AWS_SHARED_CREDENTIALS_FILE`, `GOOGLE_APPLICATION_CREDENTIALS`), `TMPDIR` (use a
disk, not RAM: checkpoints and downloads go there) and Dask (`DASK_*`).

Every command first reads `~/.arcomakerc` (or the file named by `ARCOMAKE_RC`; an empty value
disables it), a dotenv file, and sets the variables the environment lacks. `export KEY=VALUE` lines
are allowed, so a shell can `source` the same file:

```sh
# ~/.arcomakerc: holds secrets, so `chmod 600 ~/.arcomakerc`
export TMPDIR=/scratch/${USER}/tmp
export AWS_PROFILE=my-bucket-profile
export CDSAPI_URL=https://ewds.climate.copernicus.eu/api
export CDSAPI_KEY='...'
```

Variables already set win, and empty values (`KEY=`) count as unset. Values are checked when a
command starts, from the environment or the file (`CDSAPI_URL` must be an HTTP URL, `TMPDIR` an
existing writable directory); errors name the variable, never its value. Keys that are not
settings are ignored with a warning naming the key; a file other users can read, or a missing file
named by `ARCOMAKE_RC`, is reported. Also accepted: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_SESSION_TOKEN`, `AWS_REGION`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`, the `ECCODES_*`
paths, and `COPERNICUSMARINE_MAX_POOL_CONNECTIONS` (connections per Copernicus Marine client, a
positive integer; by default Dask's thread count, at least the toolbox's own default of 10), and
`ARCOMAKE_MEMORY_LIMIT` (the memory `download` plans its chunks for, e.g. `100GB`: see
`--memory-limit`). A
recipe's `storage_options` (e.g. `profile`) override `AWS_PROFILE`. Runs with `s3://`
paths or Copernicus Marine parts (both use botocore) fail at once if the AWS profile does not
exist.

## Recipes

A download recipe has a top table for the whole dataset, one table per source dataset, steps applied
after merging, and the stored layout. A complete example, run by the tests on synthetic sources, is
[`tests/data/sample_download.toml`](tests/data/sample_download.toml); its shape is:

```toml
time_dim = "time"
start = 2020-02-26T00:00:00    # optional: or --start (the recipe is then the same for any period)
end = 2020-03-02T00:00:00      # optional: or --end; excluded, intervals are [start, end)
step = "2D"                    # blocks written (then appended) one at a time

[datasets.ocean.checkpointing] # optional: process this source in sub-blocks saved to temporary Zarr
step = "1D"

[[datasets.ocean.parts]]       # one or more parts, merged into the dataset
filename_or_obj = "gs://bucket/ocean.zarr"
engine = "zarr"

[[datasets.ocean.postprocess]] # steps applied to this dataset, in order
name = "regrid"
method = "conservative"
grid = "one_degree"            # a grid of [grids], or the table itself
kwargs = {skipna = true, nan_threshold = 1.0}

[grids.one_degree]             # optional: a grid written once, named by regrid steps
south = -1.5
north = 1.5
west = 0.5
east = 3.5
resolution_lat = 1.0
resolution_lon = 1.0

[[postprocess]]                # steps applied after merging all datasets
name = "masked_fill"
mask_name = "sea_mask"
fill_value = {thetao = "thetao_bottom"}
variables = ["thetao"]

[save]                         # the layout on disk
chunks = {time = 1, depth = 1, lat = -1, lon = -1}
compressors = [{name = "blosc", configuration = {cname = "lz4", clevel = 1, shuffle = "shuffle"}}]
```

Sources are read with xarray: any of its engines (e.g. `zarr`, `h5netcdf`), or one of the backends of
arco-make, `copernicusmarine` (Copernicus Marine, `cm://DATASET_ID`), `earlywarningdatastore` (EWDS,
`ewds://DATASET`), `netcdfoverhttp` (a NetCDF file served over HTTP) and `icechunk` (an Icechunk
repository, such as an output of arco-make: the tip of `main`, or the version named by `branch`, `tag`
or `snapshot_id`). Credentials come from the environment of each service's client, never from recipes.

Day D is the interval from 00:00 of D to 00:00 of D+1, stamped D 00:00 (`resample` with `label` and
`closed` "left"). Sources that stamp a value at the end of its period, such as ERA5 hourly
accumulations (the value stamped 00:00 covers the last hour of the previous day), take a part option
`time_offset = "-1h"`: the time coordinate is moved before each block is selected, and the part is
read over the block moved the other way, so that every day holds its own hours.

Fields constant in time that a source stores as a time series (ERA5's geopotential at the surface
and land-sea mask) take a part of their own with `static_time = 2000-01-01T00:00:00`, a stamp of the
source: the part is read at that instant, without the time dimension, the same in every block.
Zarr, NetCDF-over-HTTP and generic parts take it; Copernicus Marine and EWDS parts are read over the
block's range.

Every request to a source times out: a part's `timeout` (seconds a source may stay silent, default 120)
applies to Zarr stores on Google Cloud Storage (gcsfs waits forever otherwise), NetCDF files over HTTP
and EWDS requests, and to Zarr stores on S3 when given (botocore's default is 60 s). Connecting takes
at most 30 s of it; no timeout bounds a whole request, so large chunks over slow links are not cut.
On GCS, a stalled answer is requested again by gcsfs (6 attempts): GCS answers stall now and then for
minutes. gcsfs's own `requests_timeout` is refused in a part's storage options (it bounds a whole
request and gcsfs does not retry it): set `timeout`. Copernicus Marine parts and Icechunk inputs rely
on their own timeouts. A block that fails with a transient error (a timeout, a dropped connection, a server
or throttling answer) is redone, after a wait that doubles each time: `[retries] attempts = 5`, `wait =
"1min"`, `max_wait = "30min"` by default. Other errors (a recipe or data error) stop the run at once; the
log says which kind it was.

A download that fails leaves `main` as it was and its committed blocks on its branch: rerun it with
`--resume` (same recipe and `--start`) to write only the missing blocks. Each block's commit
records its interval and a fingerprint of the recipe, its dates aside; `--resume` refuses another
recipe (request timeouts, checkpointing and `[retries]` may change), blocks that are not the first
ones of the interval, or a run that started before `main` last moved; it may stop at another `--end`.
A plain rerun refuses to start beside a failed run it could continue; `--replace` starts over. Do not
resume a run that is still going.

`--extend` appends the days after an output's last stored day, up to `--end`, as a new version: with
the recipe the output was written with (its dates aside), the static variables compared with the
stored ones, every block appended. A failed extension continues with `--extend --resume`. With the
dates left to the command, one recipe serves the first download and every extension:
`download --start 1993-01-01 --end 2026-07-01 R OUT`, then `download --extend --end 2026-08-01 R OUT`.

A download recipe may give `[checks]`, the tables of a `validate` recipe, run on each block before its
commit, like pre-commit hooks: `valid_time_coordinate` over the block (its table takes no dates),
`valid_global_ecmwf_coordinates` and `attributes` before writing, and `ensure_no_nans` while the block
is written (no second reading). A block that fails is not committed: the run stops, naming the check
and, for `ensure_no_nans`, the variable, the mask and the first day wrong in the chunk that failed
(not necessarily the first in the block); `--resume` continues once the cause or the check is fixed
(`[checks]` is not part of the recipe's fingerprint). `validate` still reports everything wrong in a whole output.

Long blocks are much faster than short ones (each block opens its sources and recomputes what
static fields the steps need, and a long block gives Dask more work to overlap), and memory levels
off with block length: on the project machine, a 32-day block of the 0.25 deg recipe took half the
time of 8-day blocks, at 63 GiB (logbook, 18 Nov).

Download recipes give no chunk sizes (part `chunks`, `chunk` steps and regrid `output_chunks` are
refused): each part of each block is chunked as planned from the shapes and dtypes of its variables,
what the dataset's steps need (whole latitude and longitude for `regrid`, whole bins for `resample`,
whole columns for `get_bottom`) and a memory budget. The budget is `--memory-limit` (or
`ARCOMAKE_MEMORY_LIMIT`), by default the memory the process may use (its cgroup's or Slurm job's
limit included), shared among Dask's threads: a task holds about four chunks,
and chunks aim at Dask's `array.chunk-size` (128 MiB) at most. Space stays whole unless one time step
is above that; then outer dimensions (depth) are split, and time grows in whole bins. The plan of each
part is logged, with what does not fit; a task that cannot fit the budget fails the block. Writes
rechunk what straddles the stored chunks of `[save]`. `download --plan` prints the chunks of the first
block and its stored chunks, and exits, without reading data or writing the output (sources are
opened: EWDS parts download their files to do so; checkpointed datasets are planned over the whole
block, a run plans each sub-block). Stored chunks (or shards) above a task's share are reported: each
is written whole by one task.

Every output is an [Icechunk](https://icechunk.io) repository (Zarr format 3 with versions): a local
directory, an `s3://` or a `gs://` URL. Every run writes on a branch of its own (`run/<command>-<start
time>`), in commits: one per block for `download` (its message names the block's interval), one for
`process` and for each output of `compute-climatology`. Each commit's metadata (not its message)
record what made it: `arcomake_recipe`, a hash of the recipe, and `arcomake_provenance`, arco-make's
version and commit, the Python and operating system, the machine's CPUs and memory, the versions of
the packages that decide the values and a hash of every installed package (no host name nor path);
`repo.ancestry(branch="main")` lists them. Only when the run succeeds does branch `main`,
the version readers see, move to the run's output (if `main` changed meanwhile, the run stays on its
branch); a failed run leaves `main` as it was. An existing output is written only with `--replace`,
which keeps the version it replaces in the history and refuses anything at the path that is not a
repository. `[save] keep_history` (a duration with its unit, e.g. `"30D"`: a version is kept while it
was replaced less than that long ago) or `keep_versions` (a number, the current version included)
limits that history: after each successful run, older versions are expired and the data no version
uses any more is deleted, once it is older than a week (another run may still be writing to the same
repository); without them, every version is kept. Read outputs with
`engine = "icechunk"`, or with Icechunk and xarray directly. `[save] storage_options` for an S3
repository: `profile`, `region` (by default the bucket's, asked to S3, else the profile's),
`endpoint_url`, `anon`, `network_stream_timeout_seconds` (default 20: a request without progress for
that long is retried); credentials come from botocore, as for the rest of arco-make. For GCS: `token`
is `"anon"` or a service account file (by default, the environment's credentials). Options Icechunk
cannot honour are an error, before any download. In `[save]`:
- `chunks` set the stored chunks, per dimension (-1, or a dimension not listed: the whole dimension);
  required in download recipes, whose chunks for reading are planned for each machine.
- `shards`, in the same form, group whole chunks into one stored object each (Zarr v3 sharding):
  fewer, larger objects, while readers still fetch single chunks. Without `shards`, every chunk is an
  object. Dask chunks may not straddle shards when writing (`download` rechunks them; `process` and
  `compute-climatology` refuse them).
- `shard_order` is the order of the chunks inside each shard: `"morton"` (zarr's default),
  `"lexicographic"` (C order over the chunk grid: with chunks of one time step and one level,
  a time step's levels lie next to each other, so a whole depth profile is read in few requests)
  or `"colexicographic"` (Fortran order). It is not part of the array metadata: readers need no
  setting, and find chunks through each shard's index whatever the order.
- `compressors` is a list of Zarr v3 codecs, `{name, configuration}` as in Zarr's array metadata
  (e.g. `blosc`, `zstd`, `gzip`); an empty list stores the data uncompressed, and without it zarr's
  default (zstd) applies. `[save.encoding.VARIABLE]` can override chunks, shards and compressors per
  variable.

Each step is a table whose `name` is a step of arco-make or one of a few `xarray.Dataset` methods; the
other keys are its arguments. See [docs/steps.md](docs/steps.md). The `process`, `validate` and
`compute-climatology` recipes have the same `[read]`, step and `[save]` tables; see the other samples in
[`tests/data`](tests/data).

## Guarantees

- **A recipe runs as written, or fails before reading any data.** Unknown keys, unknown steps, unknown or
  missing step arguments, and arguments of the wrong type are errors when the recipe is loaded.
- **Steps never modify their input** (values, attributes, encodings): each returns a new dataset. Steps are
  lazy, and the data are computed once, when written.
- **Time intervals are right-open**, `[start, end)`, for the recipe, the blocks of `download` and the time
  selection of each source, so consecutive pieces never overlap.
- **An existing output is never replaced without `--replace`**, a run that fails leaves the current
  version as it was, and a destination whose existence cannot be checked stops the command.
- **Downloads proceed in blocks**, so that memory stays bounded: by `step` (and per source by
  `checkpointing.step`). Blocks appended to the output equal a single block.
- **Static variables** (without the time dimension) are written once, with the first block.
- **Temporary data never outlive the process**: they are removed on exit, on errors and on SIGTERM/SIGHUP,
  and leftovers of killed runs are removed by the next run.

## HPC and cluster considerations

The download phase is the most demanding part of the workflow, and its design follows the constraints of
HPC clusters, where nodes with internet access have strict limits on memory, I/O and compute time.
High-resolution sources can yield blocks larger than memory: each source is processed separately, and
`checkpointing.step` splits it further into sub-blocks saved to temporary Zarr.

> [!IMPORTANT]
> Temporary storage is used heavily for intermediate results and for downloads from the Climate Data Store
> or remote NetCDF files. If your system's default temporary directory is in RAM, set `TMPDIR` to a disk.

> [!NOTE]
> Arco-make relies on `xarray` and `dask`; regridding uses `xarray-regrid`. GRIB sources (EWDS) need a
> recent ecCodes (2.48 is used in the container image).

## Development

```bash
uv run pytest                            # offline tests on synthetic data
uv run ruff check && uv run ruff format --check
uv run ty check
uv run pre-commit run --all-files        # also REUSE and secret scanning
```

See [CLAUDE.md](CLAUDE.md) for the architecture and [docs/logbook.md](docs/logbook.md) for dated records.
Contributions are welcome. *Arco-make is a best-effort project; use at your own peril!*
