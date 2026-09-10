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
import torch
from torch._dynamo.backends.common import aot_autograd
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

  def test_get_upper_bound_int(self):
    self.assertEqual(symbol_bounds.get_upper_bound(42), 42)
    self.assertEqual(symbol_bounds.get_upper_bound(0), 0)
    self.assertEqual(symbol_bounds.get_upper_bound(128), 128)

  def test_get_upper_bound_invalid_type(self):
    with self.assertRaises(ValueError):
      symbol_bounds.get_upper_bound("not_a_valid_type")

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


if __name__ == "__main__":
  absltest.main()
