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
import types
from collections.abc import Callable, Hashable, Mapping
from typing import Any, Literal, Union, get_args, get_origin

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
  return _lookup(name)[0]


def _lookup(name: str) -> tuple[StepFunction, bool]:
  """The function of step `name`, taking the dataset first, and whether it is a registered step.

  The one place that decides what a name means, for loading (`check_arguments`) and running
  (`resolve`) alike.
  """
  if name in _REGISTRY:
    return _REGISTRY[name], True
  if name in DATASET_METHODS:
    # The method as a plain function of the class: `method(dataset, **arguments)`.
    return getattr(xr.Dataset, name), False
  raise ValueError(f"Unknown processing step {name!r}. Available steps: {', '.join(step_names())}")


def check_arguments(name: str, arguments: Mapping[str, Any]) -> None:
  """Check a recipe step: the name exists, and the arguments bind to its signature with valid types.

  Arguments collected by `**kwargs` are not type-checked here; the steps that take them
  check them when called.

  Raises:
    ValueError: describing the first problem found.
  """
  function, registered = _lookup(name)
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
  if not registered:
    return  # Dataset methods: binding is the check (their annotations are not pydantic-friendly).
  dataset_parameter = next(iter(signature.parameters))
  for parameter_name, value in bound.arguments.items():
    parameter = signature.parameters[parameter_name]
    if parameter_name == dataset_parameter or parameter.annotation is inspect.Parameter.empty:
      continue
    adapter = TypeAdapter(parameter.annotation, config=_TYPE_CONFIG)
    # `**kwargs: T` annotates each value collected by kwargs.
    items = (
      value.items()
      if parameter.kind is inspect.Parameter.VAR_KEYWORD
      else [(parameter_name, value)]
    )
    for argument_name, argument_value in items:
      # A string is an iterable of hashables: without this, `variables = "thetao"` would pass and
      # then match every variable whose name is a substring of it.
      if isinstance(argument_value, str) and not _accepts_a_string(parameter.annotation):
        raise ValueError(
          f"Invalid argument {argument_name!r} of step {name!r}: expected a list, got a string"
        )
      try:
        adapter.validate_python(argument_value)
      except PydanticValidationError as exc:
        message = "; ".join(error["msg"] for error in exc.errors()[:3])
        raise ValueError(
          f"Invalid argument {argument_name!r} of step {name!r}: {message}"
        ) from None


def _accepts_a_string(annotation: Any) -> bool:
  """Whether a parameter annotated `annotation` means to accept a single string."""
  if annotation in (str, Any, Hashable):
    return True
  origin = get_origin(annotation)
  if origin in (Union, types.UnionType):
    return any(_accepts_a_string(argument) for argument in get_args(annotation))
  if origin is Literal:
    return any(isinstance(argument, str) for argument in get_args(annotation))
  return False


def render_reference() -> str:
  """The step reference (`docs/steps.md`), generated from the registry and the docstrings."""
  import arcomake.processing_utils  # noqa: F401  (registers the steps)

  lines = [
    "# Processing steps",
    "",
    "<!-- Generated by `arco-make steps > docs/steps.md`; do not edit by hand. -->",
    "",
    "A recipe step is a table whose `name` is one of the steps below; every other key is an",
    "argument. Steps and arguments are checked when the recipe is loaded. Every step returns a new",
    "dataset and never modifies its input.",
    "",
  ]
  for name, function in sorted(_REGISTRY.items()):
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())[1:]
    lines += [f"## `{name}`", ""]
    doc = inspect.getdoc(function)
    if doc is not None:
      lines += _docstring_to_markdown(doc) + [""]
    if parameters:
      lines += ["| Argument | Type | Default |", "|---|---|---|"]
      for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
          argument = "*any other key*"
        else:
          argument = f"`{parameter.name}`"
        annotation = (
          ""
          if parameter.annotation is inspect.Parameter.empty
          else _type_name(parameter.annotation)
        )
        default = (
          "required" if parameter.default is inspect.Parameter.empty else f"`{parameter.default!r}`"
        )
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
          default = ""
        lines.append(f"| {argument} | {annotation.replace('|', '&#124;')} | {default} |")
      lines.append("")
  lines += [
    "## `xarray.Dataset` methods",
    "",
    "These methods of `xarray.Dataset` are steps too, with their own arguments: "
    + ", ".join(f"`{method}`" for method in sorted(DATASET_METHODS))
    + ". The steps above take precedence over methods of the same name.",
    "",
  ]
  return "\n".join(lines)


def _docstring_to_markdown(doc: str) -> list[str]:
  """Indented blocks after a line ending in `::` become TOML code; `Args:` items become a list."""
  out: list[str] = []
  lines = doc.splitlines()
  i = 0
  while i < len(lines):
    line = lines[i]
    if line.rstrip().endswith("::") or line.strip() == "Args:":
      is_code = line.rstrip().endswith("::")
      out += [line.rstrip()[:-1] if is_code else "Arguments:", ""]
      # The block is the next run of non-blank lines (docstrings keep examples contiguous).
      block: list[str] = []
      i += 1
      while i < len(lines) and not lines[i].strip():
        i += 1
      while i < len(lines) and lines[i].strip():
        block.append(lines[i])
        i += 1
      indent = min((len(b) - len(b.lstrip()) for b in block if b.strip()), default=0)
      block = [b[indent:] for b in block]
      if is_code:
        out += ["```toml", *block, "```"]
      else:
        for item in block:
          if item.startswith(" ") and out:
            out[-1] += " " + item.strip()  # continuation of the previous argument
          elif item.strip():
            key, _, text = item.partition(":")
            out.append(f"- `{key.strip()}`: {text.strip()}")
        out.append("")
      continue
    out.append(line)
    i += 1
  while out and not out[-1].strip():
    out.pop()
  return out


def _type_name(annotation: Any) -> str:
  text = str(annotation) if not isinstance(annotation, type) else annotation.__name__
  for prefix in ("collections.abc.", "typing.", "xarray.core.dataset."):
    text = text.replace(prefix, "")
  return f"`{text}`"
