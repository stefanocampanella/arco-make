# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""The registry of processing steps a recipe may name.

A step is either a function of `arcomake.processing_utils` registered with `@step`, called as
`fn(dataset, **arguments)`, or one of an explicit list of `xarray.Dataset` methods, called as
`dataset.method(**arguments)`. Registered functions take precedence over methods of the same name
(`sel`, `isel`, `transpose`, `astype`).

Recipes are checked when they are loaded: an unknown step, an unknown argument, a missing
argument or an argument of the wrong type is an error before any data is read.
"""

import difflib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import xarray as xr
from pydantic import ConfigDict, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

StepFunction = Callable[..., xr.Dataset]

_REGISTRY: dict[str, StepFunction] = {}

# `xarray.Dataset` methods usable as steps. Extend deliberately: each name is part of the recipe
# language.
DATASET_METHODS: frozenset[str] = frozenset(
  {
    "assign_attrs",
    "chunk",
    "convert_calendar",
    "drop_attrs",
    "drop_vars",
    "fillna",
    "mean",
    "rename",
    "squeeze",
  }
)

_TYPE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


def step[F: StepFunction](function: F) -> F:
  """Register a function of `processing_utils` as a recipe step, under its own name."""
  name = getattr(function, "__name__", None)
  if not isinstance(name, str):
    raise TypeError(f"Steps must be named functions, got {function!r}")
  if name in _REGISTRY:
    raise ValueError(f"Step {name!r} is already registered")
  _REGISTRY[name] = function
  return function


def registered_steps() -> dict[str, StepFunction]:
  """The registered arco-make steps, by name."""
  return dict(_REGISTRY)


def step_names() -> list[str]:
  """All names a recipe may use: arco-make steps and allowed `Dataset` methods."""
  return sorted(set(_REGISTRY) | DATASET_METHODS)


def resolve(name: str) -> Callable[..., xr.Dataset]:
  """Return a callable `f(dataset, **arguments)` for the step `name`.

  Raises:
    ValueError: if `name` is not a step.
  """
  if name in _REGISTRY:
    return _REGISTRY[name]
  if name in DATASET_METHODS:
    method = getattr(xr.Dataset, name)
    return lambda dataset, **arguments: method(dataset, **arguments)
  raise ValueError(f"Unknown processing step {name!r}. Available steps: {', '.join(step_names())}")


def check_arguments(name: str, arguments: Mapping[str, Any]) -> None:
  """Check a recipe step: the name exists, and the arguments bind to its signature with valid types.

  Arguments collected by `**kwargs` are not type-checked here; the steps that take them
  check them when called.

  Raises:
    ValueError: describing the first problem found.
  """
  function = _REGISTRY.get(name) or (getattr(xr.Dataset, name) if name in DATASET_METHODS else None)
  if function is None:
    raise ValueError(
      f"Unknown processing step {name!r}. Available steps: {', '.join(step_names())}"
    )
  signature = inspect.signature(function)
  parameters = list(signature.parameters.values())[1:]  # the first one is the dataset
  if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
    known = [p.name for p in parameters]
    for argument in arguments:
      if argument not in known:
        close = difflib.get_close_matches(argument, known, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else f" (arguments: {', '.join(known)})"
        raise ValueError(f"Unknown argument {argument!r} of step {name!r}{hint}")
  try:
    bound = signature.bind(None, **arguments)
  except TypeError as exc:
    raise ValueError(f"Invalid arguments for step {name!r}: {exc}") from None
  if name not in _REGISTRY:
    return  # Dataset methods: binding is the check (their annotations are not pydantic-friendly).
  for parameter_name, value in bound.arguments.items():
    parameter = signature.parameters[parameter_name]
    if parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter_name == next(
      iter(signature.parameters)
    ):
      continue
    if parameter.annotation is inspect.Parameter.empty:
      continue
    try:
      TypeAdapter(parameter.annotation, config=_TYPE_CONFIG).validate_python(value)
    except PydanticValidationError as exc:
      message = "; ".join(error["msg"] for error in exc.errors()[:3])
      raise ValueError(f"Invalid argument {parameter_name!r} of step {name!r}: {message}") from None
