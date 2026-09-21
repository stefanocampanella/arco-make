# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import click

import arcomake.xarray_accessors  # noqa: F401
from arcomake.climatology import compute_climatology
from arcomake.download import download
from arcomake.process import process
from arcomake.unpack import unpack
from arcomake.validate import validate


@click.group()
def cli():
  pass


cli.add_command(download)
cli.add_command(unpack)
cli.add_command(process)
cli.add_command(compute_climatology)
cli.add_command(validate)

if __name__ == "__main__":
  cli()
