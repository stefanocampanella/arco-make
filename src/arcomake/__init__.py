# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""arco-make: build Analysis-Ready Cloud-Optimized (ARCO) Zarr datasets from TOML recipes."""

import click

import arcomake.xarray_accessors  # noqa: F401  (registers the `Dataset.arcomake` accessor)
from arcomake import settings
from arcomake.climatology import compute_climatology
from arcomake.download import download
from arcomake.process import process
from arcomake.steps import render_reference
from arcomake.unpack import unpack
from arcomake.validate import validate


@click.group()
@click.pass_context
def cli(ctx: click.Context):
  """Download, join, process, validate and summarize ARCO datasets, as described by TOML recipes.

  Every command takes a recipe (TOML) and input/output paths, which may be local or fsspec
  URLs. Outputs are never overwritten without --overwrite.

  Settings missing from the environment (credentials, AWS_PROFILE, TMPDIR, DASK_*) are read
  from ~/.arcomakerc, or the file named by ARCOMAKE_RC: a dotenv file (`export KEY=VALUE`).
  """
  settings.load()
  # Data commands open botocore sessions (S3 paths, Copernicus Marine): fail now, not after retries.
  if ctx.invoked_subcommand != "steps" and (profile := settings.missing_aws_profile()) is not None:
    raise click.ClickException(
      f"The AWS profile {profile!r} (AWS_DEFAULT_PROFILE or AWS_PROFILE) is not configured "
      "(AWS_CONFIG_FILE, ~/.aws/config). Fix or unset it."
    )


@click.command(name="steps")
def steps_reference():
  """Print the reference of the processing steps a recipe can use (Markdown)."""
  click.echo(render_reference(), nl=False)


cli.add_command(download)
cli.add_command(unpack)
cli.add_command(process)
cli.add_command(compute_climatology)
cli.add_command(validate)
cli.add_command(steps_reference)

if __name__ == "__main__":
  cli()
