# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Determine bounds of symbolic integers."""

from __future__ import annotations
from typing import Any
from absl import logging
import sympy
import torch
from torch.utils._sympy.numbers import int_oo
from torch_tpu._internal.compile.dynamic import sym_utils


def _is_valid_bound(s: sympy.Expr) -> bool:
  return s.is_integer and s.is_constant() and (s not in (int_oo, -int_oo))  # pyrefly: ignore[missing-attribute, bad-return]


def _lookup_bounds_in_shape_env(
    expr: sympy.Expr, shape_env
) -> tuple[int, int] | None:
  """Queries the shape environment using all available lookups and math solvers."""
  # When expr is a symbol.
  if isinstance(expr, sympy.Symbol):
    vr = shape_env.var_to_range.get(expr, None)
    if vr and _is_valid_bound(vr.lower) and _is_valid_bound(vr.upper):
      logging.debug(
          "shape_env.var_to_range bounds: %s -> [%s, %s]",
          expr,
          vr.lower,
          vr.upper,
      )
      return int(vr.lower), int(vr.upper)
    return None

  # When expr is an arithmetic expression, try bound_sympy.
  if hasattr(shape_env, "bound_sympy"):
    try:
      vr = shape_env.bound_sympy(expr)
      if _is_valid_bound(vr.lower) and _is_valid_bound(vr.upper):
        min_val = int(vr.lower)
        max_val = int(vr.upper)
        logging.debug(
            "shape_env.bound_sympy bounds: %s -> [%d, %d]",
            expr,
            min_val,
            max_val,
        )
        return min_val, max_val
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(
          "Failed to get bounds from shape_env.bound_sympy for %s: %s", expr, e
      )

  return None


def get_symint_bounds(sym_int: torch.SymInt) -> tuple[int, int]:
  """Gets lower and upper bounds for a given SymInt, evaluating expressions if needed.

  Args:
    sym_int: The SymInt to get the bounds for.

  Returns:
    A tuple of (lower_bound, upper_bound) for the SymInt.

  Raises:
    RuntimeError: If bounds are not found via shape_env and no hint is provided
      for the symbol.
  """
  if vr := _lookup_bounds_in_shape_env(
      sym_int.node.expr, sym_int.node.shape_env  # pyrefly: ignore[bad-argument-type]
  ):
    return vr

  # Fallback: resolve using concrete runtime profiling tracing hints
  hint = sym_int.node.hint
  if hint is None:
    raise RuntimeError(
        f"Cannot determine bounds for dynamic symbol or expression: {sym_int}."
    )

  lower_bound = hint
  upper_bound = lower_bound * 2

  logging.debug(
      "Fallback hint bounds: %s, lb: %s, ub: %s",
      sym_int,
      lower_bound,
      upper_bound,
  )
  return lower_bound, upper_bound  # pyrefly: ignore[bad-return]


def get_upper_bound(val: Any) -> int:
  """Extracts concrete integer upper bound for a value, int, or SymInt node."""
  if sym_utils.is_symint(val):
    symint = val.meta["val"] if sym_utils.is_symint_node(val) else val
    _, upper = get_symint_bounds(symint)
    assert upper is not None, f"Failed to get upper bound for SymInt {symint}"
    return upper
  elif isinstance(val, int):
    return val
  raise ValueError(f"Unexpected type for upper bound extraction: {type(val)}")
