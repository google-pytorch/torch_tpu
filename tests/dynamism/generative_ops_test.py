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

"""Unit tests for generative operations with dynamic shapes in torch.compile."""

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class GenerativeOpsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  # =========================================================================
  # torch.arange Tests
  # =========================================================================

  def test_arange_dynamic_end(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.arange(s0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.arange(4, device=self.device)
    utils.assert_close(out1, expected)

  def test_arange_dynamic_start_and_end_static_length(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.arange(s0, s0 + 5, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.arange(4, 9, device=self.device)
    utils.assert_close(out1, expected)

  def test_arange_dynamic_start_and_end_dynamic_length(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.arange(s0, s0 * 2, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.arange(4, 8, device=self.device)
    utils.assert_close(out1, expected)

  def test_arange_with_step(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.arange(0, s0, 2, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(8, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0)

    out1 = compiled(x1)
    expected = torch.arange(0, 8, 2, device=self.device)
    utils.assert_close(out1, expected)

  def test_arange_plus_symint(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(1, device=x.device) + s0

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(1, 1024, device=self.device)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=2048)

    out1 = compiled(x1)
    expected = torch.tensor([1024], device=self.device)
    utils.assert_close(out1, expected)

  def test_arange_bounds(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(s0, s0 + 1, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(1, 1024, device=self.device)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=2048)

    out1 = compiled(x1)
    expected = torch.tensor([1024], device=self.device)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.ones Tests
  # =========================================================================

  def test_ones_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.ones(s0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.ones(4, device=self.device)
    utils.assert_close(out1, expected)

  def test_ones_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.ones((s0, 8), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.ones((4, 8), device=self.device)
    utils.assert_close(out1, expected)

  def test_ones_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.ones((s0, s1), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.ones((4, 6), device=self.device)
    utils.assert_close(out1, expected)

  def test_ones_with_dtype(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.ones(s0, dtype=torch.int32, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.ones(4, dtype=torch.int32, device=self.device)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.zeros Tests
  # =========================================================================

  def test_zeros_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.zeros(s0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.zeros(4, device=self.device)
    utils.assert_close(out1, expected)

  def test_zeros_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.zeros((s0, 16), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.zeros((4, 16), device=self.device)
    utils.assert_close(out1, expected)

  def test_zeros_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.zeros((s0, s1), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.zeros((4, 6), device=self.device)
    utils.assert_close(out1, expected)

  def test_zeros_with_dtype(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.zeros((s0, 8), dtype=torch.int64, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.zeros((4, 8), dtype=torch.int64, device=self.device)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.full Tests
  # =========================================================================

  def test_full_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.full((s0,), 3.14, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.full((4,), 3.14, device=self.device)
    utils.assert_close(out1, expected)

  def test_full_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.full((s0, 8), 7, dtype=torch.int32, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.full((4, 8), 7, dtype=torch.int32, device=self.device)
    utils.assert_close(out1, expected)

  def test_full_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.full((s0, s1), 42.0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = torch.full((4, 6), 42.0, device=self.device)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.empty Tests
  # =========================================================================

  def test_empty_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.empty(s0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4,))

  def test_empty_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.empty((s0, 8), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 8))

  def test_empty_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.empty((s0, s1), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 6))

  # =========================================================================
  # torch.Tensor.new_empty Tests
  # =========================================================================

  def test_new_empty_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_empty(s0)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4,))

  def test_new_empty_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_empty((s0, 8))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 8))

  def test_new_empty_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return x.new_empty((s0, s1))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 6))

  # =========================================================================
  # torch.Tensor.new_zeros Tests
  # =========================================================================

  def test_new_zeros_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_zeros(s0)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_zeros(4)
    utils.assert_close(out1, expected)

  def test_new_zeros_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_zeros((s0, 16))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_zeros((4, 16))
    utils.assert_close(out1, expected)

  def test_new_zeros_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return x.new_zeros((s0, s1))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_zeros((4, 6))
    utils.assert_close(out1, expected)

  def test_new_zeros_with_dtype(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_zeros((s0, 8), dtype=torch.int64)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_zeros((4, 8), dtype=torch.int64)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.Tensor.new_ones Tests
  # =========================================================================

  def test_new_ones_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_ones(s0)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_ones(4)
    utils.assert_close(out1, expected)

  def test_new_ones_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_ones((s0, 8))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_ones((4, 8))
    utils.assert_close(out1, expected)

  def test_new_ones_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return x.new_ones((s0, s1))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_ones((4, 6))
    utils.assert_close(out1, expected)

  def test_new_ones_with_dtype(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_ones(s0, dtype=torch.int32)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_ones(4, dtype=torch.int32)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.Tensor.new_full Tests
  # =========================================================================

  def test_new_full_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_full((s0,), 3.14)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_full((4,), 3.14)
    utils.assert_close(out1, expected)

  def test_new_full_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return x.new_full((s0, 8), 7, dtype=torch.int32)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_full((4, 8), 7, dtype=torch.int32)
    utils.assert_close(out1, expected)

  def test_new_full_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return x.new_full((s0, s1), 42.0)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = x1.new_full((4, 6), 42.0)
    utils.assert_close(out1, expected)

  # =========================================================================
  # torch.rand and torch.randn Tests
  # =========================================================================

  def test_rand_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.rand(s0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4,))
    self.assertTrue((out1 >= 0.0).all().item())
    self.assertTrue((out1 <= 1.0).all().item())

  def test_rand_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.rand((s0, 8), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 8))
    self.assertTrue((out1 >= 0.0).all().item())
    self.assertTrue((out1 <= 1.0).all().item())

  def test_rand_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.rand((s0, s1), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 6))
    self.assertTrue((out1 >= 0.0).all().item())
    self.assertTrue((out1 <= 1.0).all().item())

  def test_randn_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.randn(s0, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4,))

  def test_randn_2d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.randn((s0, 8), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 8))

  def test_randn_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.randn((s0, s1), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 6))

  # =========================================================================
  # torch.randint Tests
  # =========================================================================

  def test_randint_1d_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.randint(10, (s0,), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4,))
    self.assertTrue((out1 >= 0).all().item())
    self.assertTrue((out1 < 10).all().item())

  def test_randint_high_dynamic_size(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.randint(10, (s0, 8), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 8))
    self.assertTrue((out1 >= 0).all().item())
    self.assertTrue((out1 < 10).all().item())

  def test_randint_low_high_dynamic_size(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        return torch.randint(5, 15, (s0, 8), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 8))
    self.assertTrue((out1 >= 5).all().item())
    self.assertTrue((out1 < 15).all().item())

  def test_randint_multi_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        return torch.randint(5, 15, (s0, s1), device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(4, 6, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1.shape, (4, 6))
    self.assertTrue((out1 >= 5).all().item())
    self.assertTrue((out1 < 15).all().item())

  # =========================================================================
  # Multiple / Downstream Generative Ops Tests
  # =========================================================================

  def test_multiple_generative_ops_in_graph(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        ones = torch.ones((s0, s1), device=x.device)
        zeros = torch.zeros((s0, s1), device=x.device)
        full = torch.full((s0, s1), 3.0, device=x.device)
        return x + ones + zeros + full

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, 8, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = x1 + 1.0 + 0.0 + 3.0
    utils.assert_close(out1, expected)

  def test_multiple_new_generative_ops_in_graph(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[0]
        s1 = x.shape[1]
        ones = x.new_ones((s0, s1))
        zeros = x.new_zeros((s0, s1))
        full = x.new_full((s0, s1), 3.0)
        return x + ones + zeros + full

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, 8, device=self.device)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=16)

    out1 = compiled(x1)
    expected = x1 + 1.0 + 0.0 + 3.0
    utils.assert_close(out1, expected)


if __name__ == "__main__":
  absltest.main()
