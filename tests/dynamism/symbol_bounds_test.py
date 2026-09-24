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
from torch._dynamo.backends.common import aot_autograd
from torch.utils._sympy import value_ranges
from torch.utils._sympy.functions import Mod as TorchMod, PythonMod
from torch_tpu._internal.compile.dynamic import symbol_bounds
from tests import seed_test_utils


class SymbolBoundsTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("zero", 0, 2),
      ("negative_one", -1, 2),
      ("one", 1, 2),
      ("two", 2, 2),
      ("three", 3, 4),
      ("four", 4, 4),
      ("five", 5, 8),
      ("six", 6, 8),
      ("eight", 8, 8),
      ("ten", 10, 16),
      ("sixteen", 16, 16),
      ("twenty", 20, 32),
      ("thirty_two", 32, 32),
      ("fifty", 50, 64),
      ("sixty_four", 64, 64),
      ("sixty_five", 65, 128),
      ("below_first_boundary", 127, 128),
      ("exact_first_boundary", 128, 128),
      ("above_first_boundary", 129, 256),
      ("below_second_boundary", 255, 256),
      ("exact_second_boundary", 256, 256),
      ("above_second_boundary", 257, 384),
      ("exact_third_boundary", 384, 384),
      ("exact_fourth_boundary", 512, 512),
      ("large_value", 2000, 2048),
      ("large_value_on_boundary", 2048, 2048),
      ("large_value_above_boundary", 2050, 2176),
  )
  def test_round_up_bound(self, val: int, expected: int):
    actual = symbol_bounds._round_up_bound(val)
    self.assertEqual(actual, expected)

  def test_get_symint_bounds_explicit_dynamic_mark(self):
    captured_bounds = None
    captured_is_user_defined = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_bounds, captured_is_user_defined
      for arg in example_inputs:
        if isinstance(arg, torch.Tensor) and isinstance(
            arg.shape[0], torch.SymInt
        ):
          captured_bounds = symbol_bounds.get_symint_bounds(arg.shape[0])
          captured_is_user_defined = symbol_bounds.is_user_defined_bound(
              arg.shape[0]
          )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)
    f(t)

    self.assertIsNotNone(captured_bounds)
    self.assertEqual(captured_bounds, (2, 8))
    self.assertTrue(captured_is_user_defined)

  def test_get_symint_bounds_unconstrained_hint(self):
    captured_bounds = None
    captured_is_user_defined = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_bounds, captured_is_user_defined
      for arg in example_inputs:
        if isinstance(arg, torch.Tensor) and isinstance(
            arg.shape[0], torch.SymInt
        ):
          captured_bounds = symbol_bounds.get_symint_bounds(arg.shape[0])
          captured_is_user_defined = symbol_bounds.is_user_defined_bound(
              arg.shape[0]
          )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(100)
    torch._dynamo.mark_dynamic(t, 0)
    f(t)

    self.assertIsNotNone(captured_bounds)
    # Hint 100 without divisibility constraints rounds up to multiple of 128 (256)
    self.assertEqual(captured_bounds, (100, 256))
    self.assertFalse(captured_is_user_defined)

  def test_find_violated_guard_multi_symbol_sum_equality_skipped(self):
    class FakeGuard:

      def __init__(self, expr):
        self.expr = expr

    class FakeShapeEnv:

      def __init__(self, guards):
        self.guards = guards
        self.replacements = {}

    s0, s1, s2 = sympy.symbols("s0 s1 s2")
    # Eq(s0 + s1 + s2, 64) is a sum conservation guard (e.g. MoE routing / split)
    guard = FakeGuard(sympy.Eq(s0 + s1 + s2, 64))
    shape_env = FakeShapeEnv([guard])
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
    class FakeGuard:

      def __init__(self, expr):
        self.expr = expr

    class FakeShapeEnv:

      def __init__(self, guards):
        self.guards = guards
        self.replacements = {}

    s0, s1 = sympy.symbols("s0 s1")
    # Eq(mod_fn(s0 * s1, 3), 0) is a reshape divisibility guard
    guard = FakeGuard(sympy.Eq(mod_fn(s0 * s1, 3), 0))
    shape_env = FakeShapeEnv([guard])

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

  def test_symbol_bounds_with_upper_bounds_immutability_and_updates(self):
    s0, s1 = sympy.symbols("s0 s1", integer=True)
    sb = symbol_bounds.SymbolBounds({
        s0: value_ranges.ValueRanges(2, 8),
        s1: value_ranges.ValueRanges(4, 16),
    })

    updated = sb.with_upper_bounds({"s0": 32})

    # Verify original instance is immutable
    self.assertEqual(int(sb.symbol_ranges[s0].lower), 2)
    self.assertEqual(int(sb.symbol_ranges[s0].upper), 8)
    self.assertEqual(int(sb.symbol_ranges[s1].lower), 4)
    self.assertEqual(int(sb.symbol_ranges[s1].upper), 16)

    # Verify updated instance reflects new upper bound while preserving lower bound
    self.assertEqual(int(updated.symbol_ranges[s0].lower), 2)
    self.assertEqual(int(updated.symbol_ranges[s0].upper), 32)
    self.assertEqual(int(updated.symbol_ranges[s1].lower), 4)
    self.assertEqual(int(updated.symbol_ranges[s1].upper), 16)

  def test_symbol_bounds_get_symint_bounds_and_get_upper_bound(self):
    captured_symint = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_symint
      for arg in example_inputs:
        if isinstance(arg, torch.Tensor) and isinstance(
            arg.shape[0], torch.SymInt
        ):
          captured_symint = arg.shape[0]
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)
    f(t)

    self.assertIsNotNone(captured_symint)
    sym_expr = captured_symint.node.expr
    sb = symbol_bounds.SymbolBounds({sym_expr: value_ranges.ValueRanges(2, 8)})

    # 1. Atomic symbol lookup
    self.assertEqual(sb.get_symint_bounds(captured_symint), (2, 8))
    self.assertEqual(sb.get_upper_bound(captured_symint), 8)

    # 2. Compound expression evaluated via bound_sympy: 3 * s0 + 1 -> [7, 25]
    compound_symint = captured_symint * 3 + 1
    self.assertEqual(sb.get_symint_bounds(compound_symint), (7, 25))
    self.assertEqual(sb.get_upper_bound(compound_symint), 25)

    # 3. Updating upper bound via with_upper_bounds updates compound evaluation
    sb_bucket = sb.with_upper_bounds({str(sym_expr): 10})
    self.assertEqual(sb_bucket.get_symint_bounds(compound_symint), (7, 31))
    self.assertEqual(sb_bucket.get_upper_bound(compound_symint), 31)

    # 4. get_upper_bound on concrete int
    self.assertEqual(sb.get_upper_bound(42), 42)


if __name__ == "__main__":
  absltest.main()
