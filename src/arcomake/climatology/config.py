# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The recipe of `compute-climatology` (`ClimatologyConfig`) and the output's groups."""

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arcomake.dataset_utils import (
  ReadConfig,
  SaveConfig,
)
from arcomake.processing_utils import ProcessingStepConfig

# Weighting scheme applied to the steps of the (centered) window used to
# compute the climatology. The weights depend only on the distance from the
# center of the window and are therefore constant from one year to the next.
WeightingType = Literal["constant", "gaussian", "geometric"]


# How time steps fall into bins: by position in each period, or by day of a 365-day year.
BinningType = Literal["position", "day_of_year"]


# The groups of the output repository: the climatology and the anomaly standard deviation.
GROUPS = ("climatology", "anomaly_std")


# Commit metadata: how the statistics were computed ("graph" or "streaming").
METHOD_KEY = "arcomake_method"


class ReferencePeriod(BaseModel):
  """The right-open interval `[start, end)` of the input the statistics are computed over."""

  model_config = ConfigDict(extra="forbid")

  start: datetime.datetime
  end: datetime.datetime

  @model_validator(mode="after")
  def _ordered_and_naive(self) -> "ReferencePeriod":
    if self.start.tzinfo is not None or self.end.tzinfo is not None:
      raise ValueError("reference_period dates have no time zone (time axes are naive)")
    if self.start >= self.end:
      raise ValueError(f"reference_period start ({self.start}) must be before end ({self.end})")
    return self


class ClimatologyConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  time_dim: str = "time"
  period: int = Field(ge=1, description="Number of climatology bins")
  bins: BinningType = Field(
    default="position",
    description=(
      'How steps fall into bins: "position" (step k of each period of `period` steps is in bin k)'
      ' or "day_of_year" (daily data, 365 bins, Feb 29 in Feb 28\'s bin)'
    ),
  )
  reference_period: ReferencePeriod | None = Field(
    default=None, description="Compute the statistics over [start, end) only (default: all)"
  )
  window: int = Field(ge=1, description="Number of bins to average over")
  weighting: WeightingType
  weighting_scale: float | None = Field(
    default=None, ge=0.0, description="Scale of the weighting function in bins"
  )
  split_dims: list[str] | None = Field(
    default=None,
    description=(
      "With --streaming (series computed apart): each data variable, and each index along these"
      ' dimensions, is a series (e.g. ["depth"]; default: one series per variable). Warned'
      " about without --streaming"
    ),
  )
  read: ReadConfig
  preprocess: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_climatology: list[ProcessingStepConfig] = Field(default_factory=list)
  postprocess_anomaly_std: list[ProcessingStepConfig] = Field(default_factory=list)
  save: SaveConfig

  @model_validator(mode="after")
  def _no_save_group(self) -> "ClimatologyConfig":
    if self.save.group is not None:
      raise ValueError(f"[save] group is not supported: the output's groups are {GROUPS}")
    return self

  @model_validator(mode="after")
  def _split_dims_not_time(self) -> "ClimatologyConfig":
    if self.split_dims is not None and self.time_dim in self.split_dims:
      raise ValueError(f"split_dims cannot include the time dimension {self.time_dim!r}")
    return self

  @model_validator(mode="after")
  def _day_of_year_bins(self) -> "ClimatologyConfig":
    if self.bins == "day_of_year" and self.period != 365:
      raise ValueError(f'bins = "day_of_year" needs period = 365, got {self.period}')
    return self
