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

"""Unit tests for symbol_bounds."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import sympy
import torch
from torch.fx.experimental import sym_node
from torch.utils._sympy import value_ranges
from torch.utils._sympy.functions import Mod as TorchMod, PythonMod
from torch.utils._sympy.numbers import int_oo
from torch_tpu._internal.compile.dynamic import output_sym_shape
from torch_tpu._internal.compile.dynamic import sym_bounds_manager
from torch_tpu._internal.compile.dynamic import symbol_bounds
from tests import seed_test_utils


class FakeGuard:

  def __init__(self, expr: sympy.Expr):
    self.expr = expr


class FakeShapeEnv:

  def __init__(
      self,
      var_to_range: dict[sympy.Symbol, value_ranges.ValueRanges] | None = None,
      guards: list[FakeGuard] | None = None,
  ):
    self.var_to_range = var_to_range if var_to_range is not None else {}
    self.guards = guards if guards is not None else []
    self.replacements = {}
    self._translation_validation_enabled = False
    self._replacements_version_counter = 1

  def replace(self, expr: sympy.Expr) -> sympy.Expr:
    return expr.xreplace(self.replacements) if self.replacements else expr


def _make_symint(
    name: str,
    hint: int,
    lower: int | None = None,
    upper: int | sympy.Expr = int_oo,
    shape_env: FakeShapeEnv | None = None,
) -> tuple[torch.SymInt, sympy.Symbol]:
  """Creates a torch.SymInt backed by FakeShapeEnv without running torch.compile."""
  sym = sympy.Symbol(name, integer=True)
  if shape_env is None:
    shape_env = FakeShapeEnv()
  if lower is not None:
    shape_env.var_to_range[sym] = value_ranges.ValueRanges(lower, upper)
  node = sym_node.SymNode(sym, shape_env, int, hint=hint)
  return torch.SymInt(node), sym


class SymbolBoundsTest(seed_test_utils.RepeatableTest):

  def test_get_symint_bounds_explicit_dynamic_mark(self):
    s0, _ = _make_symint("s0", hint=4, lower=2, upper=8)
    self.assertEqual(symbol_bounds.get_symint_bounds(s0), (2, 8))

  def test_get_symint_bounds_unconstrained_hint(self):
    s0, _ = _make_symint("s0", hint=100)
    self.assertEqual(symbol_bounds.get_symint_bounds(s0), (100, int_oo))

  def test_get_symint_bounds_lower_from_shape_env_unbounded_upper(self):
    s0, _ = _make_symint("s0", hint=128, lower=4, upper=int_oo)
    self.assertEqual(symbol_bounds.get_symint_bounds(s0), (4, int_oo))
    s1, _ = _make_symint("s1", hint=128, lower=4, upper=sympy.oo)
    self.assertEqual(symbol_bounds.get_symint_bounds(s1), (4, int_oo))

  def test_find_violated_guard_multi_symbol_sum_equality_skipped(self):
    s0, s1, s2 = sympy.symbols("s0 s1 s2")
    # Eq(s0 + s1 + s2, 64) is a sum conservation guard (e.g. MoE routing / split)
    guard = FakeGuard(sympy.Eq(s0 + s1 + s2, 64))
    shape_env = FakeShapeEnv(guards=[guard])
    upper_bounds = {"s0": 32, "s1": 32, "s2": 32}

    # Sum of upper bounds (96) != 64, but multi-symbol non-modular equalities
    # are skipped when checking individual capacity bounds.
    violated = symbol_bounds.find_violated_guard(shape_env, upper_bounds)
    self.assertIsNone(violated)

  @parameterized.named_parameters(
      ("sympy_mod", sympy.Mod),
      ("torch_mod", TorchMod),
      ("python_mod", PythonMod),
  )
  def test_find_violated_guard_multi_symbol_mod_equality_checked(self, mod_fn):
    s0, s1 = sympy.symbols("s0 s1")
    # Eq(mod_fn(s0 * s1, 3), 0) is a reshape divisibility guard
    guard = FakeGuard(sympy.Eq(mod_fn(s0 * s1, 3), 0))
    shape_env = FakeShapeEnv(guards=[guard])

    # 128 * 128 % 3 = 1 != 0 -> violated
    violated = symbol_bounds.find_violated_guard(
        shape_env, {"s0": 128, "s1": 128}
    )
    self.assertIsNotNone(violated)

    # 12 * 12 % 3 = 0 -> satisfied
    satisfied = symbol_bounds.find_violated_guard(
        shape_env, {"s0": 12, "s1": 12}
    )
    self.assertIsNone(satisfied)

  def test_symbol_bounds_merge_bounds(self):
    s0, s1, s2 = sympy.symbols("s0 s1 s2", integer=True)
    sb0 = symbol_bounds.SymbolBounds({
        s0: value_ranges.ValueRanges(9, 32),
        s1: value_ranges.ValueRanges(4, 16),
    })
    sb1 = symbol_bounds.SymbolBounds({
        s0: value_ranges.ValueRanges(2, 8),
        s2: value_ranges.ValueRanges(17, 64),
    })

    merged = symbol_bounds.SymbolBounds.merge_bounds([sb0, sb1])

    self.assertEqual(int(merged.symbol_ranges[s0].lower), 2)
    self.assertEqual(int(merged.symbol_ranges[s0].upper), 32)
    self.assertEqual(int(merged.symbol_ranges[s1].lower), 4)
    self.assertEqual(int(merged.symbol_ranges[s1].upper), 16)
    self.assertEqual(int(merged.symbol_ranges[s2].lower), 17)
    self.assertEqual(int(merged.symbol_ranges[s2].upper), 64)

  def test_symbol_bounds_repr(self):
    s0, s2, s10 = sympy.symbols("s0 s2 s10", integer=True)
    sb = symbol_bounds.SymbolBounds({
        s10: value_ranges.ValueRanges(2, 128),
        s0: value_ranges.ValueRanges(2, 128),
        s2: value_ranges.ValueRanges(4, int_oo),
    })
    self.assertEqual(
        repr(sb), f"SymbolBounds({{(s0, s10): [2, 128], s2: [4, {int_oo}]}})"
    )

  def test_symbol_bounds_get_symint_bounds_and_get_upper_bound(self):
    shape_env = FakeShapeEnv()
    s0_symint, s0 = _make_symint(
        "s0", hint=4, lower=2, upper=8, shape_env=shape_env
    )
    sb = symbol_bounds.SymbolBounds({s0: value_ranges.ValueRanges(2, 8)})

    # 1. Atomic symbol lookup
    self.assertEqual(sb.get_symint_bounds(s0_symint), (2, 8))
    self.assertEqual(sb.get_upper_bound(s0_symint), 8)

    # 2. Compound expression evaluated via bound_sympy: 3 * s0 + 1 -> [7, 25]
    compound_node = sym_node.SymNode(3 * s0 + 1, shape_env, int, hint=13)
    compound_symint = torch.SymInt(compound_node)
    self.assertEqual(sb.get_symint_bounds(compound_symint), (7, 25))
    self.assertEqual(sb.get_upper_bound(compound_symint), 25)

    # 3. get_upper_bound on concrete int
    self.assertEqual(sb.get_upper_bound(42), 42)

  @parameterized.named_parameters(
      ("offset_0", 0, [(16, 128), (129, 256), (257, 384)]),
      ("offset_1", 1, [(16, 127), (128, 255), (256, 383)]),
      ("offset_4", 4, [(16, 124), (125, 252), (253, 380)]),
  )
  def test_get_bucketed_symbol_bounds_with_offsets(
      self, concat_offset, expected_bucket_bounds
  ):
    s0_symint, s0 = _make_symint("s0", hint=16, lower=16, upper=int_oo)
    out_shape = output_sym_shape.OutputSymShape([{"expr": s0 + concat_offset}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    buckets = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=3)

    self.assertEqual(
        [b.get_symint_bounds(s0_symint) for b in buckets],
        expected_bucket_bounds,
    )
    merged = symbol_bounds.SymbolBounds.merge_bounds(buckets)
    self.assertEqual(
        merged.get_symint_bounds(s0_symint),
        (expected_bucket_bounds[0][0], expected_bucket_bounds[-1][1]),
    )

  def test_get_bucketed_symbol_bounds_hint_larger_than_min(self):
    # User set min=1, max=2048, but the concrete input size (hint) is 1024.
    # Bucket 0 covers [1, 1024], Bucket 1 covers [1025, 1152], and the final
    # bucket is extended to initial_upper [1153, 2048].
    s0_symint, s0 = _make_symint("s0", hint=1024, lower=1, upper=2048)
    out_shape = output_sym_shape.OutputSymShape([{"expr": s0}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    buckets = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=3)

    self.assertEqual(
        [b.get_symint_bounds(s0_symint) for b in buckets],
        [(1, 1024), (1025, 1152), (1153, 2048)],
    )

    single_bucket = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=1)
    self.assertEqual(
        [b.get_symint_bounds(s0_symint) for b in single_bucket],
        [(1, 2048)],
    )

  def test_get_bucketed_symbol_bounds_lower_at_multiple_of_128(self):
    # When initial_lower == 128 and offset == 0, Bucket 0 should advance to 256
    # ([128, 256]) rather than collapsing to a degenerate [128, 128] point.
    s0_symint, s0 = _make_symint("s0", hint=128, lower=128, upper=int_oo)
    out_shape = output_sym_shape.OutputSymShape([{"expr": s0}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    buckets = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=1)

    self.assertEqual(
        [b.get_symint_bounds(s0_symint) for b in buckets],
        [(128, 256)],
    )

  def test_get_bucketed_symbol_bounds_clamped_by_explicit_max(self):
    # User set min=4, max=32 (< 128). Bucket 0 is clamped to [4, 32] and
    # loop stops early without generating duplicate or invalid [33, 32] buckets.
    s0_symint, s0 = _make_symint("s0", hint=16, lower=4, upper=32)
    out_shape = output_sym_shape.OutputSymShape([{"expr": s0}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    buckets = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=3)

    self.assertEqual(
        [b.get_symint_bounds(s0_symint) for b in buckets],
        [(4, 32)],
    )

  def test_get_bucketed_symbol_bounds_multi_symbol_partial_cap(self):
    # s0 is capped at max=32 while s1 is unbounded (int_oo).
    # s0 stays at [4, 32] while s1 continues stepping across buckets.
    shape_env = FakeShapeEnv()
    s0_symint, s0 = _make_symint(
        "s0", hint=16, lower=4, upper=32, shape_env=shape_env
    )
    s1_symint, s1 = _make_symint(
        "s1", hint=16, lower=16, upper=int_oo, shape_env=shape_env
    )
    out_shape = output_sym_shape.OutputSymShape([{"expr": s0}, {"expr": s1}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint, s1_symint])
    buckets = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=3)

    self.assertEqual(
        [
            (b.get_symint_bounds(s0_symint), b.get_symint_bounds(s1_symint))
            for b in buckets
        ],
        [
            ((4, 32), (16, 128)),
            ((4, 32), (129, 256)),
            ((4, 32), (257, 384)),
        ],
    )

  def test_get_bucketed_symbol_bounds_returns_empty_on_guard_violation(self):
    s0 = sympy.Symbol("s0", integer=True)
    guard = FakeGuard(sympy.Eq(sympy.Mod(s0, 3), 0))
    shape_env = FakeShapeEnv(guards=[guard])
    s0_symint, _ = _make_symint(
        "s0", hint=12, lower=12, upper=int_oo, shape_env=shape_env
    )
    out_shape = output_sym_shape.OutputSymShape([{"expr": s0}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    buckets = bm.get_bucketed_symbol_bounds([out_shape], num_buckets=3)

    # 128 % 3 != 0 violates the guard, so get_bucketed_symbol_bounds returns []
    # and the caller falls back to get_fallback_symbol_bounds() -> (12, 24).
    self.assertEqual(buckets, [])
    fallback_bounds = bm.get_fallback_symbol_bounds()
    self.assertEqual(fallback_bounds.get_symint_bounds(s0_symint), (12, 24))

  def test_conflicting_symbol_offsets_logs_warning(self):
    s0_symint, s0 = _make_symint("s0", hint=16, lower=16, upper=int_oo)
    out_shape_1 = output_sym_shape.OutputSymShape([{"expr": s0 + 1}])
    out_shape_2 = output_sym_shape.OutputSymShape([{"expr": s0 + 4}])

    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    with self.assertLogs(level="WARNING") as cm:
      buckets = bm.get_bucketed_symbol_bounds(
          [out_shape_1, out_shape_2], num_buckets=1
      )

    self.assertTrue(
        any(
            "Conflicting output shape offsets for symbol s0" in m
            for m in cm.output
        )
    )
    self.assertEqual(buckets[0].get_symint_bounds(s0_symint), (16, 127))

  def test_empty_bounds_raises_value_error(self):
    s0_symint, _ = _make_symint("s0", hint=16, lower=16, upper=int_oo)
    bm = sym_bounds_manager.SymBoundsManager([s0_symint])
    with self.assertRaises(  # ASSERT_RAISES_OK=Cannot happen from user end
        ValueError
    ):
      bm.add_bound_checks([])
    with self.assertRaises(  # ASSERT_RAISES_OK=Cannot happen from user end
        ValueError
    ):
      symbol_bounds.SymbolBounds.merge_bounds([])


if __name__ == "__main__":
  absltest.main()
