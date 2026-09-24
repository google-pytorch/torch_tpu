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

"""Symbolic bounds manager for dynamic shape compilation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from absl import logging
import sympy
import torch
from torch._inductor.utils import InputType
from torch.fx.experimental import symbolic_shapes
from torch.utils._sympy import value_ranges
from torch_tpu._internal.compile.dynamic import symbol_bounds


class SymBoundsManager:
  """Manages lower and upper bounds for dynamic SymInts."""

  _atomic_symints: list[torch.SymInt]
  _user_defined_bound_for_symbols: set[str]
  _symbol_bounds: symbol_bounds.SymbolBounds

  def __init__(self, example_inputs: Sequence[InputType]):
    self._atomic_symints = []
    self._user_defined_bound_for_symbols = set()
    self._populate_bounds_metadata(example_inputs)
    self._validate_bounds()

  def get_symbol_bounds(self) -> symbol_bounds.SymbolBounds:
    """Returns the SymbolBounds object holding symbol ranges and expression evaluator."""
    return self._symbol_bounds

  def _register_atomic_symbol(
      self,
      s: torch.SymInt,
      symbol_ranges: dict[sympy.Symbol, value_ranges.ValueRanges],
  ) -> None:
    """Registers atomic symbol with its bounds."""
    sym_expr = getattr(getattr(s, "node", None), "expr", None)
    if isinstance(sym_expr, sympy.Symbol):
      # sym_expr is a symbol, e.g. s0, s1, etc.
      symbol = sym_expr
      if symbol not in symbol_ranges:
        lower, upper = symbol_bounds.get_symint_bounds(s)
        if symbol_bounds.is_user_defined_bound(s):
          self._user_defined_bound_for_symbols.add(str(symbol))
        symbol_ranges[symbol] = value_ranges.ValueRanges(lower, upper)
        self._atomic_symints.append(s)

  def _populate_bounds_metadata(
      self, example_inputs: Sequence[InputType]
  ) -> None:
    """Populates bounds metadata for input tensors and SymInt scalars."""
    self._atomic_symints = []
    symbol_ranges: dict[sympy.Symbol, value_ranges.ValueRanges] = {}

    for arg in example_inputs:
      if isinstance(arg, torch.Tensor):
        for s in arg.shape:
          if isinstance(s, torch.SymInt):
            self._register_atomic_symbol(s, symbol_ranges)
      elif isinstance(arg, torch.SymInt):
        self._register_atomic_symbol(arg, symbol_ranges)

    self._symbol_bounds = symbol_bounds.SymbolBounds(symbol_ranges)

  def _update_symint_upper_bounds(
      self, upper_bounds: Mapping[str, int]
  ) -> None:
    """Updates the upper bounds for symbols across internal metadata."""
    self._symbol_bounds = self._symbol_bounds.with_upper_bounds(upper_bounds)

  def add_bound_checks(self) -> None:
    """Adds bound checks for all SymInts in the graph."""
    for symint in self._atomic_symints:
      lower, upper = self._symbol_bounds.get_symint_bounds(symint)
      torch._check(symint >= lower, "Dynamic shape lower bound check failed")  # pylint: disable=protected-access
      torch._check(symint <= upper, "Dynamic shape upper bound check failed")  # pylint: disable=protected-access

  def _validate_bounds(self) -> None:
    """Validates that all SymInt upper bounds satisfy active shape guards."""
    if not self._atomic_symints:
      return

    shape_env = getattr(self._atomic_symints[0].node, "shape_env", None)
    if shape_env is None:
      return

    upper_bounds = {
        str(sym): int(vr.upper)
        for sym, vr in self._symbol_bounds.symbol_ranges.items()
    }
    lower_bounds = {
        str(sym): int(vr.lower)
        for sym, vr in self._symbol_bounds.symbol_ranges.items()
    }

    violated_guard = symbol_bounds.find_violated_guard(shape_env, upper_bounds)
    if violated_guard is None:
      return

    # In case of failure, fall back compiler-selected bounds.
    has_fallen_back = False
    for sym_name, lower in lower_bounds.items():
      fallback_upper = symbol_bounds.get_fallback_upper_bound(lower)
      if (
          sym_name not in self._user_defined_bound_for_symbols
          and upper_bounds[sym_name] != fallback_upper
      ):
        logging.debug(
            "Bound %s violated guard %s; falling back SymInt %s to %s",
            upper_bounds[sym_name],
            violated_guard,
            sym_name,
            fallback_upper,
        )
        upper_bounds[sym_name] = fallback_upper
        has_fallen_back = True

    if not has_fallen_back:
      raise symbolic_shapes.ConstraintViolationError(
          f"Upper bounds {upper_bounds} violate active shape guard: "
          f"{violated_guard}."
      )

    self._update_symint_upper_bounds(upper_bounds)

    if (
        still_violated := symbol_bounds.find_violated_guard(
            shape_env, upper_bounds
        )
    ) is not None:
      raise symbolic_shapes.ConstraintViolationError(
          f"Upper bounds {upper_bounds} violate active shape guard: "
          f"{still_violated}."
      )
