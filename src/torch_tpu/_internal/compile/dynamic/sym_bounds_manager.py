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

from collections.abc import Sequence

from absl import logging
import sympy
import torch
from torch._inductor.utils import InputType
from torch.fx.experimental import symbolic_shapes
from torch.utils._sympy import value_ranges
from torch.utils._sympy.numbers import int_oo
from torch_tpu._internal.compile.dynamic import output_sym_shape
from torch_tpu._internal.compile.dynamic import symbol_bounds


class SymBoundsManager:
  """Manages lower and upper bounds for dynamic SymInts."""

  _atomic_symints: list[torch.SymInt]

  def __init__(self, example_inputs: Sequence[InputType]):
    self._atomic_symints = []
    self._populate_atomic_symbols(example_inputs)

  def get_fallback_symbol_bounds(self) -> symbol_bounds.SymbolBounds:
    """Computes and validates a single SymbolBounds using explicit or fallback bounds."""
    symbol_ranges: dict[sympy.Symbol, value_ranges.ValueRanges] = {}
    upper_bounds: dict[str, int] = {}

    for symint in self._atomic_symints:
      symbol = symint.node.expr
      if isinstance(symbol, sympy.Symbol):
        lower, upper = symbol_bounds.get_symint_bounds(symint)
        hint = getattr(symint.node, "hint", None)
        start_val = max(lower, int(hint)) if hint is not None else lower
        upper_int = (
            symbol_bounds.get_fallback_upper_bound(start_val)
            if upper == int_oo
            else int(upper)
        )
        sym_name = str(symbol)
        upper_bounds[sym_name] = upper_int
        symbol_ranges[symbol] = value_ranges.ValueRanges(lower, upper_int)

    if self._atomic_symints:
      shape_env = getattr(self._atomic_symints[0].node, "shape_env", None)
      if (
          violated_guard := symbol_bounds.find_violated_guard(
              shape_env, upper_bounds
          )
      ) is not None:
        raise symbolic_shapes.ConstraintViolationError(
            f"Upper bounds {upper_bounds} violate active shape guard: "
            f"{violated_guard}."
        )

    return symbol_bounds.SymbolBounds(symbol_ranges)

  def get_bucketed_symbol_bounds(
      self,
      output_sym_shapes: Sequence[output_sym_shape.OutputSymShape],
      num_buckets: int = 1,
      bucket_step: int = 128,
  ) -> list[symbol_bounds.SymbolBounds]:
    """Computes a sequence of bucketed SymbolBounds using 128k - c alignment.

    For each atomic symbol `s` with lower bound `L` and positive constant offset
    `c` extracted from `s + c` output shape expressions (defaulting to `c = 0`),
    computes bucket upper bounds:
      first_bucket_multiple = ceil((start_val + c) / bucket_step) * bucket_step
      first_bucket_upper = first_bucket_multiple - c
      B_i = first_bucket_upper + i * bucket_step   for i in range(num_buckets)

    Each candidate bucket is validated against active shape guards. If any
    bucket violates a guard, returns an empty list.

    Args:
      output_sym_shapes: Sequence of OutputSymShape specs for graph outputs.
      num_buckets: Maximum number of buckets to generate.
      bucket_step: Bucket step size (defaults to 128).

    Returns:
      A list of SymbolBounds, one per valid bucket in ascending order, or an
      empty list if any candidate bucket violates an active shape guard.
    """
    if not self._atomic_symints:
      return [symbol_bounds.SymbolBounds({})]

    # Extract positive constant offsets `c` from `s + c` output expressions so
    # that bucket upper bounds `128k - c` make output shapes multiples of 128.
    symbol_offsets: dict[str, int] = {}
    for out_shape in output_sym_shapes:
      for sym_name, offset in out_shape.extract_symbol_offsets().items():
        if sym_name in symbol_offsets and symbol_offsets[sym_name] != offset:
          logging.warning(
              "Conflicting output shape offsets for symbol %s: %d vs %d;"
              " keeping %d.",
              sym_name,
              symbol_offsets[sym_name],
              offset,
              symbol_offsets[sym_name],
          )
        else:
          symbol_offsets[sym_name] = offset
    shape_env = getattr(self._atomic_symints[0].node, "shape_env", None)

    # Precompute per-symbol parameters:
    # (symbol, sym_name, initial_lower, initial_upper, first_bucket_upper).
    symbol_configs: list[
        tuple[sympy.Symbol, str, int, int | sympy.Expr, int]
    ] = []
    for symint in self._atomic_symints:
      symbol = symint.node.expr
      if isinstance(symbol, sympy.Symbol):
        sym_name = str(symbol)
        initial_lower, initial_upper = symbol_bounds.get_symint_bounds(symint)
        offset = symbol_offsets.get(sym_name, 0)
        # Ensure the first bucket is large enough to cover both `initial_lower`
        # (e.g. from `mark_dynamic(..., min=...)`) and the concrete input size
        # (`hint`) that triggered compilation.
        hint = getattr(symint.node, "hint", None)
        start_val = (
            max(initial_lower, int(hint)) if hint is not None else initial_lower
        )
        # Round `start_val + offset` up to the next multiple of `bucket_step`,
        # then subtract `offset` to get the first bucket's upper bound.
        first_bucket_multiple = (
            (start_val + offset + bucket_step - 1) // bucket_step
        ) * bucket_step
        first_bucket_upper = first_bucket_multiple - offset
        # Avoid a degenerate single-point first bucket `[L, L]` (which causes
        # ShapeEnv to specialize the symbol to a constant when L == U) when
        # `initial_lower + offset` is already an exact multiple of `bucket_step`.
        if (
            first_bucket_upper <= initial_lower
            and initial_upper > initial_lower
        ):
          first_bucket_upper += bucket_step
        symbol_configs.append(
            (symbol, sym_name, initial_lower, initial_upper, first_bucket_upper)
        )

    buckets: list[symbol_bounds.SymbolBounds] = []
    prev_ranges: dict[sympy.Symbol, value_ranges.ValueRanges] | None = None

    for bucket_idx in range(num_buckets):
      is_last_bucket = bucket_idx == num_buckets - 1
      bucket_ranges: dict[sympy.Symbol, value_ranges.ValueRanges] = {}
      candidate_upper_bounds: dict[str, int] = {}
      for (
          symbol,
          sym_name,
          initial_lower,
          initial_upper,
          first_bucket_upper,
      ) in symbol_configs:
        # If this symbol already reached its finite user-defined `initial_upper`
        # in the previous bucket, keep its capped range so we don't advance
        # `bucket_lower` (`prev_upper + 1`) past `initial_upper`.
        if (
            prev_ranges is not None
            and int(prev_ranges[symbol].upper) >= initial_upper
        ):
          bucket_ranges[symbol] = prev_ranges[symbol]
          candidate_upper_bounds[sym_name] = int(prev_ranges[symbol].upper)
          continue
        # Step upper bound by `bucket_step` for each bucket, clamped to
        # `initial_upper` (which is `int_oo` when unbounded). On the final
        # bucket, extend to `initial_upper` if finite so Dynamo's
        # StrictMinMaxConstraint is satisfied.
        if is_last_bucket and initial_upper != int_oo:
          candidate_upper = int(initial_upper)
        else:
          candidate_upper = int(
              min(first_bucket_upper + bucket_idx * bucket_step, initial_upper)
          )
        # Partition buckets into contiguous non-overlapping intervals:
        # Bucket 0 covers [initial_lower, B_0], Bucket i covers [B_{i-1} + 1, B_i].
        bucket_lower = (
            initial_lower
            if prev_ranges is None
            else int(prev_ranges[symbol].upper) + 1
        )
        bucket_ranges[symbol] = value_ranges.ValueRanges(
            bucket_lower, candidate_upper
        )
        candidate_upper_bounds[sym_name] = candidate_upper

      # Stop early if all symbols have reached their `initial_upper` and no
      # symbol range changed from the previous bucket.
      if prev_ranges is not None and bucket_ranges == prev_ranges:
        break

      # Validate that the candidate bucket's upper bounds satisfy all active
      # shape guards (e.g. reshape divisibility constraints).
      violated_guard = symbol_bounds.find_violated_guard(
          shape_env, candidate_upper_bounds
      )
      if violated_guard is not None:
        logging.debug(
            "Bucket %s violated guard %s",
            candidate_upper_bounds,
            violated_guard,
        )
        return []

      buckets.append(symbol_bounds.SymbolBounds(bucket_ranges))
      prev_ranges = bucket_ranges

    return buckets

  def _populate_atomic_symbols(
      self, example_inputs: Sequence[InputType]
  ) -> None:
    """Populates atomic SymInts from inputs."""
    self._atomic_symints = []
    seen_symbols: set[sympy.Symbol] = set()

    for arg in example_inputs:
      if isinstance(arg, torch.SymInt):
        sym_expr = getattr(getattr(arg, "node", None), "expr", None)
        if isinstance(sym_expr, sympy.Symbol) and sym_expr not in seen_symbols:
          seen_symbols.add(sym_expr)
          self._atomic_symints.append(arg)

    for arg in example_inputs:
      if isinstance(arg, torch.SymInt):
        symints = [arg]
      elif isinstance(arg, torch.Tensor):
        symints = [
            s
            for s in (*arg.shape, *arg.stride(), arg.storage_offset())
            if isinstance(s, torch.SymInt)
        ]
      else:
        continue

      for s in symints:
        sym_expr = getattr(getattr(s, "node", None), "expr", None)
        if sym_expr is not None:
          missing = sym_expr.free_symbols - seen_symbols
          if missing:
            raise ValueError(
                f"Input SymInt {s} depends on symbols {missing} that are not"
                " present as standalone atomic SymInt inputs in"
                f" example_inputs: {seen_symbols}"
            )

  def add_bound_checks(
      self,
      bounds: symbol_bounds.SymbolBounds | Sequence[symbol_bounds.SymbolBounds],
  ) -> None:
    """Adds bound checks for all SymInts in the graph across the given bounds."""
    if not isinstance(bounds, symbol_bounds.SymbolBounds):
      if not bounds:
        raise ValueError(
            "Cannot add bound checks for an empty bounds sequence."
        )
      bounds = symbol_bounds.SymbolBounds.merge_bounds(bounds)

    for symint in self._atomic_symints:
      lower, upper = bounds.get_symint_bounds(symint)
      torch._check(symint >= lower, "Dynamic shape lower bound check failed")  # pylint: disable=protected-access
      torch._check(symint <= upper, "Dynamic shape upper bound check failed")  # pylint: disable=protected-access
