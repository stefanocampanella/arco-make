# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The examples of `docs/coarsen-ocean-datasets.md` that read the real sources (```pycon live):
GLORYS's mask and depth, its empty deepest level, the footnote's premise, on regional boxes.
Copernicus Marine credentials are needed (environment, `~/.arcomakerc` or the toolbox's file).
"""

import pytest

from arcomake import settings
from tests.test_coarsening_notes import run_examples

pytestmark = pytest.mark.network


def test_live_examples_of_the_notes():
  settings.load()
  run_examples(live=True)
