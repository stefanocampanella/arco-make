# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`compute-climatology`: a periodic climatology and the standard deviation of the
anomalies.

Modules: `command` (the command), `config` (its recipe), `graph` and `streamed` (the two
ways of computing), `binning` (bins and window weights), `kernels` (numba), `staging` (the
temporary store and the copy into the output).
"""

from arcomake.climatology.command import compute_climatology
from arcomake.climatology.config import (
  GROUPS,
  METHOD_KEY,
  BinningType,
  ClimatologyConfig,
  ReferencePeriod,
  WeightingType,
)

__all__ = [
  "GROUPS",
  "METHOD_KEY",
  "BinningType",
  "ClimatologyConfig",
  "ReferencePeriod",
  "WeightingType",
  "compute_climatology",
]
