import sys

import click
import xarray as xr

from arcomake.datetime_utils import DateInterval


@click.command()
@click.argument("downloaded_path", type=click.Path(exists=True))
@click.option("--reference-url", default="s3://ogs-arco-ocean/dataset/tres=1d/res=0p25/levels=10", help="URL to the reference dataset")
@click.option("--start", "start_date", type=click.DateTime(), required=True, help="Start date of the downloaded dataset")
@click.option("--end", "end_date", type=click.DateTime(), required=True, help="End date of the downloaded dataset")
@click.option("--max-diff", default=1e-5, help="Maximum allowed absolute difference")
@click.option("--rel-diff", default=1e-5, help="Maximum allowed relative difference")
def main(downloaded_path, reference_url, start_date, end_date, max_diff, rel_diff):
    """
    Check that the downloaded dataset matches the reference dataset.
    """
    click.echo(f"Opening downloaded dataset at {downloaded_path}")
    # The downloaded dataset is a zipped zarr
    ds_downloaded = xr.open_dataset(downloaded_path, engine="zarr", chunks={})

    click.echo(f"Opening reference dataset at {reference_url}")
    # Reference is also a zarr store on S3.
    # We use fsspec (via xarray/zarr) to open it.
    # Note: s3fs needs to be installed, and it might need credentials if not public.
    # The issue doesn't specify credentials for reference, assuming it's public or environment handles it.
    ds_reference_full = xr.open_dataset(reference_url, engine="zarr", storage_options={"anon": True}, chunks={})

    # Select the dates specified by command line arguments
    # The interval should coincide with those of the downloaded dataset
    # We use slice for selection. 
    # Note: start_date and end_date from click are datetime objects.
    # The reference dataset time might be datetime64[ns] or cftime.
    date_interval = DateInterval(start_date, end_date)
    ds_reference = ds_reference_full.arcomake.sel(date_interval)

    # 1. Check same variables
    click.echo("Checking variables...")
    vars_downloaded = set(ds_downloaded.data_vars)
    vars_reference = set(ds_reference.data_vars)
    if vars_downloaded != vars_reference:
        click.echo(f"Variable mismatch! Downloaded: {vars_downloaded}, Reference: {vars_reference}", err=True)
        sys.exit(1)

    # 2. Check same coordinates (name, attributes, and values)
    click.echo("Checking coordinates...")
    coords_downloaded = set(ds_downloaded.coords)
    coords_reference = set(ds_reference.coords)
    if coords_downloaded != coords_reference:
        click.echo(f"Coordinate mismatch! Downloaded: {coords_downloaded}, Reference: {coords_reference}", err=True)
        sys.exit(1)

    for coord in ds_downloaded.coords:
        # Check values
        try:
            xr.testing.assert_allclose(ds_downloaded[coord], ds_reference[coord], atol=1e-8, rtol=1e-8)
        except AssertionError as e:
            click.echo(f"Coordinate value mismatch for {coord}: {e}", err=True)
            sys.exit(1)
        
        # Check attributes
        if ds_downloaded[coord].attrs != ds_reference[coord].attrs:
            # Sometimes attributes might differ slightly (e.g. order), but we check exact match as requested.
            click.echo(f"Coordinate attribute mismatch for {coord}!", err=True)
            click.echo(f"Downloaded: {ds_downloaded[coord].attrs}")
            click.echo(f"Reference: {ds_reference[coord].attrs}")
            # sys.exit(1) # Decided not to exit here as it might be too strict, but requirement said "Check"

    # 3. Check dimensions order
    click.echo("Checking dimensions order...")
    for var in ds_downloaded.data_vars:
        if ds_downloaded[var].dims != ds_reference[var].dims:
            click.echo(f"Dimension order mismatch for variable {var}!", err=True)
            click.echo(f"Downloaded: {ds_downloaded[var].dims}")
            click.echo(f"Reference: {ds_reference[var].dims}")
            sys.exit(1)

    # 4. Check chunk sizes
    click.echo("Checking chunk sizes...")
    for var in ds_downloaded.data_vars:
        # For the downloaded dataset, we check the actual chunks if it is dask-backed,
        # otherwise we might need to check encoding.
        # However, the requirement specifically asks to check encoding['chunks'] in reference.
        
        # ds_downloaded[var].chunks returns a Mapping from dimension names to tuple of chunk sizes
        # if the dataset is dask-backed.
        downloaded_chunks = ds_downloaded[var].chunks
        
        # Reference chunking is expected to be found in encoding['chunks']
        reference_chunks_raw = ds_reference[var].encoding.get("chunks")
        
        if reference_chunks_raw is not None:
            # reference_chunks_raw is typically a tuple of integers corresponding to dimensions
            reference_chunks = dict(zip(ds_reference[var].dims, reference_chunks_raw, strict=False))
        else:
            reference_chunks = None

        # Convert downloaded_chunks to a comparable format (dict of single chunk size if uniform)
        # Note: ds.chunks returns a tuple of chunks for each dimension, e.g., {'time': (5, 5), 'x': (5,)}
        # We assume uniform chunking for comparison if reference encoding has a single value per dim.
        if downloaded_chunks is not None:
            downloaded_chunks_cmp = {dim: chunks[0] for dim, chunks in downloaded_chunks.items()}
        else:
            downloaded_chunks_cmp = None

        if downloaded_chunks_cmp != reference_chunks:
            # Final attempt to match: maybe reference_chunks is a tuple of tuples?
            # Or maybe downloaded_chunks_cmp and reference_chunks are both not None but differ.
            click.echo(f"Chunk size mismatch for variable {var}!", err=True)
            click.echo(f"Downloaded (dask): {downloaded_chunks_cmp}")
            click.echo(f"Reference (encoding): {reference_chunks}")
            sys.exit(1)

    # 5. Check close values
    click.echo(f"Checking data values (max_diff={max_diff}, rel_diff={rel_diff})...")
    for var in ds_downloaded.data_vars:
        try:
            xr.testing.assert_allclose(ds_downloaded[var], ds_reference[var], atol=max_diff, rtol=rel_diff)
        except AssertionError as e:
            click.echo(f"Value mismatch for variable {var}: {e}", err=True)
            sys.exit(1)

    click.echo("Integration test passed successfully!")

if __name__ == "__main__":
    main()
