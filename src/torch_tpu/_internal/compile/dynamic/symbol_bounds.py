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
from collections.abc import Mapping, Sequence
from typing import Any
from absl import logging
import sympy
import torch
from torch.utils._sympy import value_ranges
from torch.utils._sympy.functions import Mod as TorchMod, PythonMod
from torch.utils._sympy.numbers import int_oo
from torch_tpu._internal.compile.dynamic import sym_utils


def _is_valid_bound(s: sympy.Expr) -> bool:
  return s.is_integer and s.is_constant() and (s not in (int_oo, -int_oo))  # pyrefly: ignore[missing-attribute, bad-return]


def _lookup_bounds_in_shape_env(
    expr: sympy.Expr, shape_env
) -> tuple[int, int | sympy.Expr] | None:
  """Queries the shape environment using all available lookups and math solvers."""
  # When expr is a symbol.
  if isinstance(expr, sympy.Symbol):
    vr = shape_env.var_to_range.get(expr, None)
    if vr and _is_valid_bound(vr.lower):
      if _is_valid_bound(vr.upper):
        logging.debug(
            "shape_env.var_to_range bounds: %s -> [%s, %s]",
            expr,
            vr.lower,
            vr.upper,
        )
        return int(vr.lower), int(vr.upper)
      if vr.upper in (int_oo, sympy.oo):
        logging.debug(
            "shape_env.var_to_range bounds: %s -> [%s, %s]",
            expr,
            vr.lower,
            int_oo,
        )
        return int(vr.lower), int_oo
    return None

  # When expr is an arithmetic expression, try bound_sympy.
  if hasattr(shape_env, "bound_sympy"):
    try:
      vr = shape_env.bound_sympy(expr)
      if _is_valid_bound(vr.lower):
        min_val = int(vr.lower)
        if _is_valid_bound(vr.upper):
          max_val = int(vr.upper)
          logging.debug(
              "shape_env.bound_sympy bounds: %s -> [%d, %d]",
              expr,
              min_val,
              max_val,
          )
          return min_val, max_val
        if vr.upper in (int_oo, sympy.oo):
          logging.debug(
              "shape_env.bound_sympy bounds: %s -> [%d, %s]",
              expr,
              min_val,
              int_oo,
          )
          return min_val, int_oo
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(
          "Failed to get bounds from shape_env.bound_sympy for %s: %s", expr, e
      )

  return None


def get_fallback_upper_bound(lower_bound: int) -> int:
  """Computes the fallback upper bound when candidate bounds violate shape guards."""
  return 2 * lower_bound


def find_violated_guard(
    shape_env: Any,
    upper_bounds: Mapping[str, int],
) -> sympy.Expr | None:
  """Checks whether the assignment of symbol upper bounds satisfies active shape guards.

  Args:
    shape_env: The ShapeEnv containing active shape guards.
    upper_bounds: Mapping from symbol name to its upper bound value.

  Returns:
    The first violated guard expression, or None if all guards are satisfied.
  """
  if shape_env is None:
    return None

  guards = getattr(shape_env, "guards", None)
  if not guards:
    return None

  replacements = getattr(shape_env, "replacements", None)

  for guard in guards:
    guard_expr = getattr(guard, "expr", None)
    if guard_expr is None or not hasattr(guard_expr, "free_symbols"):
      continue

    if replacements:
      try:
        guard_expr = guard_expr.xreplace(replacements)
      except Exception:  # pylint: disable=broad-except
        pass

    # Only evaluate guards where all free symbols are bounded by upper_bounds.
    guard_sym_names = {str(s) for s in guard_expr.free_symbols}
    if not guard_sym_names or not guard_sym_names.issubset(upper_bounds.keys()):
      continue

    # Multi-symbol non-modular constraints (such as sum conservation
    # Eq(s0 + s1 + ... , C)) describe joint runtime relationships between
    # dynamic dimensions. They do not hold at the joint upper-bound corner where
    # all symbols simultaneously equal their individual maximum capacities.
    # We only validate modular arithmetic guards (e.g. Mod(expr, c) == 0 for
    # reshape divisibility) or single-symbol guards.
    if len(guard_expr.free_symbols) > 1 and not guard_expr.has(
        sympy.Mod, TorchMod, PythonMod
    ):
      continue

    substitutions = {
        sym: upper_bounds[str(sym)] for sym in guard_expr.free_symbols
    }

    try:
      eval_guard = guard_expr.subs(substitutions)
      if not eval_guard.free_symbols:
        if eval_guard == sympy.true:
          continue
        if eval_guard == sympy.false:
          logging.debug(
              "Joint upper bounds %s violate guard %s",
              upper_bounds,
              guard_expr,
          )
          return guard_expr
    except Exception:  # pylint: disable=broad-except
      pass

  return None


def get_symint_bounds(
    sym_int: torch.SymInt,
) -> tuple[int, int | sympy.Expr]:
  """Gets lower and upper bounds for a given SymInt, evaluating expressions if needed.

  If explicit finite bounds are not found in `shape_env`, returns
  `(int(hint), int_oo)` to indicate an unbounded upper bound.

  Args:
    sym_int: The SymInt to get the bounds for.

  Returns:
    A tuple of (lower_bound, upper_bound) for the SymInt, where `upper_bound`
    is `int_oo` when no upper bound is set in `shape_env`.

  Raises:
    RuntimeError: If bounds are not found via shape_env and no hint is provided
      for the symbol.
  """
  if vr := _lookup_bounds_in_shape_env(
      sym_int.node.expr, sym_int.node.shape_env  # pyrefly: ignore[bad-argument-type]
  ):
    return vr

  # Unbounded in shape_env: resolve lower bound from concrete tracing hint and
  # return int_oo as upper bound.
  hint = sym_int.node.hint
  if hint is None:
    raise RuntimeError(
        f"Cannot determine bounds for dynamic symbol or expression: {sym_int}."
    )

  lower_bound = int(hint)
  logging.debug(
      "Unbounded hint bounds: %s, lb: %s, ub: %s",
      sym_int,
      lower_bound,
      int_oo,
  )
  return lower_bound, int_oo


class SymbolBounds:
  """Holds lower and upper bounds for atomic symbols and evaluates symbolic expressions."""

  _symbol_ranges: dict[sympy.Symbol, value_ranges.ValueRanges]

  def __init__(
      self, symbol_ranges: Mapping[sympy.Symbol, value_ranges.ValueRanges]
  ):
    self._symbol_ranges = dict(symbol_ranges)

  @property
  def symbol_ranges(
      self,
  ) -> Mapping[sympy.Symbol, value_ranges.ValueRanges]:
    return self._symbol_ranges

  def __repr__(self) -> str:
    def _sym_sort_key(sym: sympy.Symbol) -> tuple[str, int]:
      name = str(sym)
      for i, ch in enumerate(name):
        if ch.isdigit() and name[i:].isdigit():
          return (name[:i], int(name[i:]))
      return (name, -1)

    grouped: dict[tuple[int | str, int | str], list[str]] = {}
    for sym in sorted(self._symbol_ranges, key=_sym_sort_key):
      vr = self._symbol_ranges[sym]
      bounds = (
          int(vr.lower) if _is_valid_bound(vr.lower) else str(vr.lower),
          int(vr.upper) if _is_valid_bound(vr.upper) else str(vr.upper),
      )
      grouped.setdefault(bounds, []).append(str(sym))

    parts = []
    for (lower, upper), syms in grouped.items():
      key_str = syms[0] if len(syms) == 1 else f"({', '.join(syms)})"
      parts.append(f"{key_str}: [{lower}, {upper}]")
    return f"SymbolBounds({{{', '.join(parts)}}})"

  @classmethod
  def merge_bounds(cls, bounds_seq: Sequence[SymbolBounds]) -> SymbolBounds:
    """Returns a SymbolBounds spanning the union of ranges across `bounds_seq`."""
    if not bounds_seq:
      raise ValueError("Cannot merge an empty sequence of SymbolBounds.")
    merged_ranges: dict[sympy.Symbol, value_ranges.ValueRanges] = {}
    for bounds in bounds_seq:
      for sym, vr in bounds.symbol_ranges.items():
        if sym not in merged_ranges:
          merged_ranges[sym] = vr
        else:
          merged_ranges[sym] = value_ranges.ValueRanges(
              min(merged_ranges[sym].lower, vr.lower),
              max(merged_ranges[sym].upper, vr.upper),
          )
    return cls(merged_ranges)

  def get_symint_bounds(self, sym_int: torch.SymInt) -> tuple[int, int]:
    """Gets lower and upper bounds for a given SymInt or symbolic expression."""
    expr = getattr(getattr(sym_int, "node", None), "expr", None)
    if isinstance(expr, sympy.Symbol) and expr in self._symbol_ranges:
      vr = self._symbol_ranges[expr]
      return int(vr.lower), int(vr.upper)

    if expr is not None and expr.free_symbols.issubset(
        self._symbol_ranges.keys()
    ):
      try:
        vr = value_ranges.bound_sympy(expr, self._symbol_ranges)
        if _is_valid_bound(vr.lower) and _is_valid_bound(vr.upper):
          return int(vr.lower), int(vr.upper)
      except Exception as e:  # pylint: disable=broad-except
        logging.debug(
            "Could not evaluate finite int bounds for %s via bound_sympy: %s",
            expr,
            e,
        )
    raise ValueError(f"No bounds found for SymInt {sym_int}")

  def get_upper_bound(self, val: Any) -> int:
    """Gets the upper bound for an int, SymInt, or SymInt FX node."""
    if sym_utils.is_symint(val):
      symint = val.meta["val"] if sym_utils.is_symint_node(val) else val
      _, upper = self.get_symint_bounds(symint)
      return upper
    if isinstance(val, int):
      return val
    raise ValueError(f"Unexpected type for upper bound extraction: {type(val)}")
