# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
import click

from arcomake.dataset import array_range, download, unpack_zips
from arcomake.stats import compute


@click.group()
def cli():
  pass


cli.add_command(array_range)
cli.add_command(download)
cli.add_command(unpack_zips)
cli.add_command(compute)

if __name__ == "__main__":
  cli()
