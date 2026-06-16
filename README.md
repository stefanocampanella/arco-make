# ARCO-MAKE

Arco-make is a tool for automating the preparation of Analysis-Ready Cloud-Optimized (ARCO) timeseries datasets from individual data sources. It operates via "recipes": TOML configuration files that specify where each dataset component is located, alongside the post-processing steps to apply to it. Finally, it allows you to compute data statistics, which are useful for developing data-driven (deep learning) models.

## Installation

Arco-make requires Python 3.13 or later. We recommend installing it using [uv](https://github.com/astral-sh/uv) or `pip`:

```bash
pip install .

```

For development, you can install the package with its extra dependencies:

```bash
pip install -e ".[dev]"

```

Alternatively, you can pull the latest image from the GitHub Container Registry. For example:

```bash
apptainer exec oras://ghcr.io/arco-make:latest arco-make download --help

```

## Usage

The typical workflow consists of three steps:

1. Download a collection of Zarr datasets using `arco-make download` (optionally using SLURM job arrays or GNU Parallel) and optionally save them as zip archives.
2. If needed, unpack the zip archives into a single dataset using `arco-make unpack`.
3. Compute statistics using `arco-make stats`.

Detailed documentation for these commands is available directly via the command-line interface.

### HPC and Cluster Considerations

The download phase is the most critical part of the workflow. Most of its design choices were motivated by the constraints of running workflows on HPC clusters, where nodes with internet access face strict limits on memory, I/O, and compute time. To navigate these limitations, `arco-make download` takes the following actions:

1. **Parallelized Segments:** It allows you to download different segments of long timeseries in parallel using SLURM job arrays. The length of these segments is specified in the top-level table of the TOML configuration using the `array_step` key, allowing you to tune jobs to stay within maximum walltime limits.
2. **Memory Management:** High-resolution spatial or temporal source datasets can yield segments that exceed available memory. To prevent this, `arco-make` downloads and postprocesses from each source separately, saving regridded/resampled intermediate results to disk. To reduce the memory footprint further, it splits timeseries segments into smaller, temporary segments defined by the `tmp_step` key in the top-level TOML table.
3. **Storage Optimization:** It can save the resulting Zarr datasets directly into zip archives. This prevents the creation of a plethora of small files, which would otherwise degrade performance on parallel filesystems like Lustre or on tape storage systems.

Arco-make supports downloading from various data providers (such as Copernicus Marine or the Climate Data Store) and online resources (including Google Cloud Storage or remote NetCDF files served over HTTP).

> [!IMPORTANT]
> Temporary storage is used heavily for saving intermediate results, as well as during downloads from the Climate Data Store or remote NetCDFs. If your system's default temporary directory resides in RAM, we recommend setting the `TMPDIR` environment variable to point to physical disk storage instead.

> [!NOTE]
> `arco-make` leverages `xarray` and `dask`. Regridding operations are powered by `xarray-regrid`.

## Download Recipes

Contrary to what the name might suggest, Arco-make does not use a declarative syntax like GNU Make. Instead, it processes operations sequentially in an imperative style, as its primary purpose is to automate repetitive processing pipelines across multiple data sources simultaneously.

Arco-make was originally built to produce [ARCO-OCEAN](https://github.com/inogs/arco-ocean), utilizing the recipe file located at `configs/arco-ocean_tres-1d_res-0p25_levels-10.toml`. This file serves as an illustrative example of the expected TOML schema. We suggest using it as a template and adapting it to your specific use case.

A valid recipe follows this pattern:

```toml
# Top table contains info related to the whole dataset
start = 1970-01-01T00:00:00
end = 2026-06-01T00:00:00
array_step = '10 W'

[[datasets]]
name = 'fictitious-dataset'
provider = 'gcs'
type = 'timeseries'
tmp_step = '2 W'

[[datasets.parts]]
# Info to download dataset part 1

[[datasets.parts]]
# Info to download dataset part 2

[[datasets.parts]]
# Info to download dataset part 3

[[datasets.postprocess]]
# Preprocessing step 1 (applies to parts 1, 2, and 3; also, dataset postprocessing steps can use mask, when defined)

[[datasets.postprocess]]
# Preprocessing step 2

[datasets.mask]
provider = 'cm'
type = 'static'
variable = 'mask_variable_name'

[[datasets.mask.parts]]
# Info to download mask for 'fictitious-dataset'

[[datasets.mask.postprocess]]
# Preprocessing step 1 applied to mask

[[datasets.mask.postprocess]]
# Preprocessing step 2 applied to mask

[save]
# Save options

```

## Contributing & Roadmap

Currently, the codebase has minimal documentation and lacks comprehensive unit testing. We plan to address both of these gaps in the future, and contributions in these areas, as well as new features or bug fixes, are welcome!

*Arco-make is a best-effort project; use at your own peril!*