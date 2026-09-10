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

"""Tests for shape guard constraint validation and fallback in torch_tpu."""

from __future__ import annotations

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class SymConstraintTest(seed_test_utils.RepeatableTest):

  def test_single_symbol_reshape_invalid_bound_raises_constraint_violation(
      self,
  ):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(3, -1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(12, dtype=torch.float32, device="tpu").reshape(2, 6)
    # Total elements = 2 * s. Reshape to (3, -1) requires (2 * s) % 3 == 0.
    # An upper bound of max=10 violates this guard since (2 * 10) % 3 != 0.
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=10)

    with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed) as ctx:
      compiled(x1)
    self.assertIn("ConstraintViolationError", str(ctx.exception))

  def test_joint_multi_symbol_reshape_invalid_bound_raises_constraint_violation(
      self,
  ):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(3, -1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    # Initial shape [6, 6] (total elements 36, divisible by 3).
    # Hints: s0=6, s1=6.
    # Individually with hints:
    #   s0_ub * s1_hint = 8 * 6 = 48 (divisible by 3) -> passes individually
    #   s0_hint * s1_ub = 6 * 8 = 48 (divisible by 3) -> passes individually
    # Jointly at upper bounds:
    #   s0_ub * s1_ub = 8 * 8 = 64 (64 % 3 != 0) -> fails jointly
    x1 = torch.arange(36, dtype=torch.float32, device="tpu").reshape(6, 6)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=8)

    with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed) as ctx:
      compiled(x1)
    self.assertIn("ConstraintViolationError", str(ctx.exception))

  def test_joint_multi_symbol_reshape_selected_bound_fallback(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(3, -1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    # Initial shape [6, 6] (total elements 36, divisible by 3).
    # Hints: s0=6, s1=6.
    # Individually with hints: candidate 128 passes individually (128 * 6 = 768 % 3 == 0).
    # Jointly at candidate bounds: 128 * 128 = 16384 (16384 % 3 != 0).
    # Because bounds were compiler-selected (not user-provided), validate_bounds falls
    # back both symbols to 2 * lower_bound (12 * 12 = 144 % 3 == 0), allowing compilation to succeed.
    x1 = torch.arange(36, dtype=torch.float32, device="tpu").reshape(6, 6)
    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(x1, 1)

    out1 = compiled(x1)
    expected = (
        torch.arange(36, dtype=torch.float32, device="tpu")
        .reshape(6, 6)
        .reshape(3, -1)
    )
    utils.assert_close(out1, expected)


if __name__ == "__main__":
  absltest.main()
