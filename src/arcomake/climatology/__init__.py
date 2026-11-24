# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""`compute-climatology`: a periodic climatology and the standard deviation of the anomalies.

What is computed. Every time step falls in one of `period` bins (`binning.time_bins`: by
position in each period, or by day of a 365-day year). For bin b, the values of the bins b + k,
for the offsets k of a centred window (wrapping around the period), are pooled, each weighted by
the weight of its offset (`binning.window`). The climatology of bin b is their weighted mean; the
anomaly standard deviation, the square root of their unbiased weighted variance (reliability
weights). Missing values are skipped. Both come from three statistics per bin, which the window
combines (`kernels.window_bins_last`): the number of values, their mean, and the sum of their
squared deviations from it (count, mean, m2).

Two ways of computing them (`command._compute_and_write`), which agree (tested):

- `graph` (`--graph`, the default): every variable in one Dask graph, the per-bin statistics by
  folding the time axis into periods x bins. Its memory grows with the years at 0.25 deg.
- `streamed` (`--streaming`): every series (a variable, or one index of the recipe's
  `split_dims`) apart, in memory, by Welford updates over blocks of steps read in turn
  (`kernels.welford`), as many series at once as fit in memory. Its memory stays flat.

Both compute into a temporary store and copy it into the output with the layout of `[save]`
(`staging`): one Icechunk repository, one commit, the groups `climatology` and `anomaly_std`.

Modules: `command` (the command), `config` (its recipe), `graph` and `streamed` (the two ways),
`binning` (bins, window, output type), `kernels` (numba), `staging` (the temporary store and
the copy).
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
