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
| `arco-make download RECIPE OUTPUT` | Downloads, processes and merges the datasets of a recipe into one Zarr store |
| `arco-make validate RECIPE INPUT` | Checks a dataset: time axis, grid, missing values inside masks |
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
positive integer; the toolbox's default, 10, is fewer than Dask's threads on a large machine). A
recipe's `storage_options` (e.g. `profile`) override `AWS_PROFILE`. Runs with `s3://`
paths or Copernicus Marine parts (both use botocore) fail at once if the AWS profile does not
exist.

## Recipes

A download recipe has a top table for the whole dataset, one table per source dataset, steps applied
after merging, and the stored layout. A complete example, run by the tests on synthetic sources, is
[`tests/data/sample_download.toml`](tests/data/sample_download.toml); its shape is:

```toml
time_dim = "time"
start = 2020-02-26T00:00:00
end = 2020-03-02T00:00:00      # excluded: intervals are [start, end)
step = "2D"                    # blocks written (then appended) one at a time

[datasets.ocean.checkpointing] # optional: process this source in sub-blocks saved to temporary Zarr
step = "1D"

[[datasets.ocean.parts]]       # one or more parts, merged into the dataset
filename_or_obj = "gs://bucket/ocean.zarr"
engine = "zarr"
chunks = {time = 1, depth = 1, latitude = -1, longitude = -1}

[[datasets.ocean.postprocess]] # steps applied to this dataset, in order
name = "regrid"
method = "conservative"
grid = {south = -1.5, north = 1.5, west = 0.5, east = 3.5, resolution_lat = 1.0, resolution_lon = 1.0}
kwargs = {skipna = true, nan_threshold = 1.0}

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
`ewds://DATASET`) and `netcdfoverhttp` (a NetCDF file served over HTTP). Credentials come from the
environment of each service's client, never from recipes.

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
- **An existing output is never replaced without `--overwrite`**, and a destination whose existence cannot
  be checked stops the command.
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
