# Changelog

Versions follow [Semantic Versioning](https://semver.org). The public interface is the recipes, the
commands and their options, the layout of outputs, and the commit metadata that `download --resume` and
`--extend` read (`arcomake_block`, `arcomake_recipe`). Before 1.0.0 a minor version may change it in ways
that break: 1.0.0 will declare it stable, once a production dataset has been made with it. A version that
changes what an unchanged recipe writes says so here.

## 0.2.0 (2026-11-25)

Most of arco-make was rewritten (pull requests #1 to #22; reasons and measurements are in
[docs/logbook.md](docs/logbook.md)). Recipes and outputs of 0.1.0 do not carry over: see below.

The image of this version is `oras://ghcr.io/stefanocampanella/arco-make:v0.2.0`: from this version on,
each version tag publishes its own image, besides `latest` (the tip of `master`).

### Upgrading from 0.1.0

Commands:

- Outputs are [Icechunk](https://icechunk.io) repositories (Zarr format 3, with versions), not Zarr
  format 2 stores; read them with xarray's `engine = "icechunk"` or icechunk itself (README, "Recipes").
  Zip outputs and the `unpack` command are removed.
- `--overwrite` is now `--replace`: it writes a new version of an existing repository and refuses
  anything else at the path.
- `compute-climatology RECIPE INPUT OUTPUT` writes one repository with the groups `climatology` and
  `anomaly_std`, in place of two output paths.
- Every command runs on `--dask-setup threads` by default (`compute-climatology` used `mpi`);
  `--dask-setup mpi` needs the `mpi` extra.

Recipes (unknown keys, unknown steps and wrong arguments are now errors when the recipe is loaded):

- `[save] compressor` is now `compressors`, a list of Zarr v3 codecs, e.g.
  `compressors = [{name = "blosc", configuration = {cname = "lz4", clevel = 1, shuffle = "shuffle"}}]`.
  Outputs are compressed unless the recipe says `compressors = []`. `[save] consolidated` is removed.
- Download recipes give no chunk sizes: parts' `chunks`, `chunk` steps and regrid `output_chunks` are
  refused, and arco-make plans the chunks of each block. They must give the stored layout, `[save] chunks`
  (and optionally `shards`).
- `start` and `end` may be left out of a recipe and given as `--start`/`--end`.
- A Google Cloud Storage part's `storage_options.requests_timeout` is refused: parts take `timeout`.
- Statistics recipes: `use_flox` is removed; `bins`, `reference_period` and `split_dims` are new.
- Removed steps: `diff_std`, `gaussian_blur_extrapolate`. Some steps take other arguments
  ([docs/steps.md](docs/steps.md)): a 0.1.0 recipe that no longer fits fails when loaded, naming the key.

### New

- `download --resume` continues a failed run; `--extend` appends to an output from its last block. Every
  run writes on its own branch and moves `main` only once it succeeds; retention of old versions
  (`[save] keep_history`, `keep_versions`). Blocks with transient errors are redone (`[retries]`).
- Forecast runs along `init_time` x `lead_time`, with a rolling window (`[forecast]`); the engine
  `ecmwfopendata` (ECMWF open data) and forecast requests to the Early Warning Data Store.
- Checks run on each block before it is committed (`[checks]`).
- Provenance in each commit's metadata: arco-make's version and commit, Python, the system, key package
  versions, the recipe's fingerprint.
- Shards (`[save] shards`, `shard_order`), regrid grids named once (`[grids]`), parts' `rename`,
  `coordinate_tolerance`, `time_offset` and `static_time`, and the steps `sin_cos`, `coverage_attrs`,
  `cell_bounds` and `cell_area`, among others (`arco-make steps`, [docs/steps.md](docs/steps.md)).
- Settings and credentials from `~/.arcomakerc`; `--memory-limit`; `download --plan`.
- `compute-climatology --streaming`: each series apart, as many at a time as fit in memory.

### Changed output for an unchanged recipe

- Outputs without `[save] compressor` were uncompressed; they are now compressed (N1).
- The climatology counts the values each bin has: missing years no longer bias it (with the new
  `bins = "day_of_year"`, Feb 29 falls in Feb 28's bin).
- Statistics keep every coordinate of the input without time (cell bounds included).
- `regrid` passes variables without the horizontal dimensions through.

### Fixed (bugs of 0.1.0)

- A failed read of a Zarr source chunk (a timeout, a connection error) was read as NaN, silently (N8).
- `[save] chunks = -1` over smaller Dask chunks wrote overlapping chunks concurrently and lost values,
  silently (N10).
- Reads from Google Cloud Storage could wait forever (N7); concurrent GRIB opens could crash ecCodes (N6).
- These and the other fixes are listed with their tests in [docs/logbook.md](docs/logbook.md) (N and B
  items).
