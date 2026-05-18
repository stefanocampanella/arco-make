# ARCO-MAKE

Arco-make is a tool for automating the preparation of an analysis-ready cloud-optimized timeseries dataset from individual data sources. It does so by means of recipes, i.e., TOML configuration files, which specify where each dataset part is located, along with pre- and post-processing steps to apply to that part. Finally, it allows computing some statistics; useful, for example, for the development of data-driven (deep learning) models.

## Installation

Arco-make requires Python 3.13 or later. It is recommended to install it using [uv](https://github.com/astral-sh/uv) or `pip`:

```bash
pip install .
```

For development, you can install the package with extra dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

The typical workflow is:

1. Download a collection of Zarr datasets using `arco-make download`, for example, using SLURM job arrays or GNU parallel; eventually save them as zip archives.
2. If needed, unpack the zip archives into a single dataset using `arco-make unpack`.
3. Compute statistics using `arco-make stats`

The documentation of these commands is available via the command line interface.

The download phase is the most critical part, and most of its design choices were motivated by the limitations of running `arco-make` on HPC clusters. Indeed, on such shared machines the internet access is typically available on nodes with restrictions on memory, IO, and compute time. To counteract these limitations, `arco-make download` takes the following actions:

1. It allows downloading different segments of long timeseries in parallel using SLURM job arrays. The lenght of such segments is specified in TOML configurations using the `array_step` key, in the top-level table, and it can be set to stay within maximum job time.
2. As source datasets may have fine time or spatial resolutions, resulting in segments which might not fit into memory, `arco-make` downloads and preprocesses each dataset separately, saving regridded/resampled intermediate results on disk. To further reduce memory requirements, it does so by subdividing timeseries segments into smaller temporary segments, whose length can be specified using the `tmp_step` key in the top-level TOML table.
3. It allows saving the resulting Zarr to zip archives to avoid the creation of a plethora of small files, which could impact parallel filesystems like LUSTRE.

Arco-make allows downloading from many different data providers (as the Copernicus Marine or the Climate Data Store) and online resources (i.e., Google Cloud Storage or NetCDF files served via HTTP). Notice that temporary storage is used both for saving intermediate results and when downloading from the Climate Data Store or remote NetCDFs. If your temporary directory uses RAM, it might be a good idea to set up the `TMPDIR` environment variable to point elsewhere.

>[!NOTE]
> `arco-make` leverage `xarray` and `dask`, regridding operations uses `xarray-regrid`.

## Download recipes

Arco-make (unlike GNU Make, contrary to what the name might suggest) does not use a declarative syntax for its recipes. Instead, it follows the sequence of operations specified in the recipe in an imperative style, as its purpose is to automate repetitive operations to process multiple data sources at once.

Arco-make has been used to produce [ARCO-OCEAN](https://github.com/inogs/arco-ocean), using the recipe file located at `configs/arco-ocean_tres-1d_res-0p25_levels-10.toml`. This recipe file is an illustrative example of the schema that TOML configurations should follow, and we suggest using it as a starting point and editing it to adapt to other use cases. A valid recipe should look like the follwing.

```toml
start = 1970-01-01T00:00:00
end = 2026-06-01T00:00:00
array_step = '10 W'
tmp_step = '2 W'

[array]
step = '180 D'

[temporary]
step = '18 D'

[[datasets]]
name = 'fictious-dataset'
provider = 'gcs'
type = 'timeseries'

[[datasets.parts]]
# info to download dataset part 1

[[datasets.parts]]
# info to download dataset part 2

[[datasets.parts]]
# info to download dataset part 3

[[datasets.preprocess]]
# preprocess step 1

[[datasets.preprocess]]
# preprocess step 2

[[datasets.postprocess]]
# postprocess step 1

[[datasets.postprocess]]
# postprocess step 2

[datasets.mask]
provider = 'cm'
type = 'static'
variable = 'mask_variable_name'

[[datasets.mask.parts]]
# info to download mask for dataset (i.e. parts 1, 2, and 3)

[[datasets.mask.preprocess]]
# preprocess step 1 applied to mask

[[datasets.mask.preprocess]]
# preprocess step 2 applied to mask

[[datasets.mask.postprocess]]
# postprocess step 1 applied to mask

[[datasets.mask.postprocess]]
# postprocess step 2 applied to mask

[save]
# save options
```

Currently, the implementation is scarcely documented, and there is no unit testing. There are plans to add both, and contributions in that regard are welcome (to add features and fixes as well).

Arco-make is a best-effort project, use at your own peril!