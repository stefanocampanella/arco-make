# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import click

import arcomake.xarray_accessors  # noqa: F401
from arcomake.climatology import compute_climatology
from arcomake.dataset import download, unpack
from arcomake.stats import compute_stats


@click.group()
def cli():
  pass


cli.add_command(download)
cli.add_command(unpack)
cli.add_command(compute_stats)
cli.add_command(compute_climatology)

if __name__ == "__main__":
  cli()
