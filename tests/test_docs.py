# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The documentation matches the code: generated reference, and valid examples."""

import pathlib
import re
import tomllib

import pytest

from arcomake.processing_utils import ProcessingStepConfig
from arcomake.steps import registered_steps, render_reference

ROOT = pathlib.Path(__file__).parents[1]


def test_step_reference_is_up_to_date():
  """docs/steps.md is the output of `arco-make steps` (regenerate it after changing a step)."""
  assert (ROOT / "docs" / "steps.md").read_text() == render_reference()


def toml_examples() -> list[tuple[str, str]]:
  reference = render_reference()
  examples = []
  for name in sorted(registered_steps()):
    section = reference.split(f"## `{name}`")[1].split("\n## ")[0]
    examples += [(name, block) for block in re.findall(r"```toml\n(.*?)```", section, re.S)]
  return examples


@pytest.mark.parametrize(("name", "example"), toml_examples())
def test_step_examples_are_valid_recipe_tables(name: str, example: str):
  """Every TOML example in a step's docstring parses and passes the recipe checks."""
  table = tomllib.loads(example)
  assert table["name"] == name
  ProcessingStepConfig.model_validate(table)
