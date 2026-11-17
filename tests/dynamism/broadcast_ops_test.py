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

"""Tests for dynamic broadcast-like operations through torch.compile()."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo import testing as dynamo_testing
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class DynamicRepeatTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    torch._dynamo.reset()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_repeat_unit_dim_dynamic_count(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x.repeat(y.shape[0], 1, 1)

    model = Model()
    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(model, backend=compile_counter)

    for batch_size, seq_len in [(3, 6), (7, 10)]:
      x_cpu = torch.arange(seq_len * 4, dtype=torch.float32).reshape(
          1, seq_len, 4
      )
      y_cpu = torch.ones((batch_size, 4), dtype=torch.float32)
      expected = model(x_cpu, y_cpu)

      x_tpu = x_cpu.to(self.device)
      y_tpu = y_cpu.to(self.device)
      torch._dynamo.mark_dynamic(x_tpu, 1, min=2, max=16)
      torch._dynamo.mark_dynamic(y_tpu, 0, min=2, max=16)

      out = compiled(x_tpu, y_tpu)
      self.assertEqual(out.shape, expected.shape)
      utils.assert_close(out, expected)

    self.assertEqual(compile_counter.frame_count, 1)

  def test_repeat_static_block_dim_dynamic_count(self):
    class Model(torch.nn.Module):

      def forward(self, pos_table, query):
        return pos_table.repeat(query.shape[0] // 12, 1, 1)

    model = Model()
    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(model, backend=compile_counter)

    for batch_size, seq_len in [(2, 6), (5, 10)]:
      pos_cpu = torch.arange(12 * seq_len * 4, dtype=torch.float32).reshape(
          12, seq_len, 4
      )
      query_cpu = torch.ones((12 * batch_size, seq_len, 4), dtype=torch.float32)
      expected = model(pos_cpu, query_cpu)

      pos_tpu = pos_cpu.to(self.device)
      query_tpu = query_cpu.to(self.device)
      torch._dynamo.mark_dynamic(pos_tpu, 1, min=2, max=16)
      torch._dynamo.mark_dynamic(query_tpu, 0, min=12, max=96)
      torch._dynamo.mark_dynamic(query_tpu, 1, min=2, max=16)

      out = compiled(pos_tpu, query_tpu)
      self.assertEqual(out.shape, expected.shape)
      utils.assert_close(out, expected)

    self.assertEqual(compile_counter.frame_count, 1)

  def test_repeat_prepended_leading_dim(self):
    class Model(torch.nn.Module):

      def forward(self, values, globals_ref):
        return values.repeat(globals_ref.shape[0], 1)

    model = Model()
    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(model, backend=compile_counter)

    for num_globals, batch_size in [(3, 5), (6, 9)]:
      values_cpu = torch.arange(batch_size, dtype=torch.int32)
      globals_cpu = torch.ones((num_globals,), dtype=torch.int32)
      expected = model(values_cpu, globals_cpu)

      values_tpu = values_cpu.to(self.device)
      globals_tpu = globals_cpu.to(self.device)
      torch._dynamo.mark_dynamic(values_tpu, 0, min=2, max=16)
      torch._dynamo.mark_dynamic(globals_tpu, 0, min=2, max=16)

      out = compiled(values_tpu, globals_tpu)
      self.assertEqual(out.shape, expected.shape)
      utils.assert_close(out, expected)

    self.assertEqual(compile_counter.frame_count, 1)

  def test_repeat_scalar_and_zero_dim(self):
    class ScalarRepeatModel(torch.nn.Module):

      def forward(self, scalar_val, ref):
        return scalar_val.repeat(ref.shape[0], 4)

    scalar_model = ScalarRepeatModel()
    compiled_scalar = torch.compile(scalar_model, backend="tpu")
    scalar_cpu = torch.tensor(3.5, dtype=torch.float32)
    ref_cpu = torch.ones((5,), dtype=torch.float32)
    ref_tpu = ref_cpu.to(self.device)
    torch._dynamo.mark_dynamic(ref_tpu, 0, min=2, max=16)
    out_scalar = compiled_scalar(scalar_cpu.to(self.device), ref_tpu)
    utils.assert_close(out_scalar, scalar_model(scalar_cpu, ref_cpu))

    class ZeroRepeatModel(torch.nn.Module):

      def forward(self, x, ref):
        return x.repeat(ref.shape[0], 0)

    zero_model = ZeroRepeatModel()
    compiled_zero = torch.compile(zero_model, backend="tpu")
    x_cpu = torch.ones((2, 3), dtype=torch.float32)
    ref_tpu_2 = torch.ones((4,), dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(ref_tpu_2, 0, min=2, max=16)
    out_zero = compiled_zero(x_cpu.to(self.device), ref_tpu_2)
    self.assertEqual(out_zero.shape, (8, 0))

    class ZeroInputDimRepeatModel(torch.nn.Module):

      def forward(self, empty_in, ref):
        return empty_in.repeat(ref.shape[0], 1) + torch.zeros_like(empty_in)

    zero_in_model = ZeroInputDimRepeatModel()
    compiled_zero_in = torch.compile(zero_in_model, backend="tpu")
    empty_tpu = torch.empty((0, 3), dtype=torch.float32, device=self.device)
    ref_tpu_3 = torch.ones((4,), dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(ref_tpu_3, 0, min=2, max=16)
    out_zero_in = compiled_zero_in(empty_tpu, ref_tpu_3)
    self.assertEqual(out_zero_in.shape, (0, 3))

  @parameterized.named_parameters(
      dict(testcase_name="unit_rows", block_rows=1),
      dict(testcase_name="block_rows", block_rows=4),
  )
  def test_repeat_dynamic_count_backward(self, block_rows):
    def fn(x, weight_blocks):
      repeated = x.repeat(weight_blocks.shape[0], 1)
      return (repeated * weight_blocks.reshape(-1, x.shape[1])).sum()

    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(fn, backend=compile_counter)
    for repeat_factor in [3, 6]:
      x_cpu = torch.randn(
          (block_rows, 5), dtype=torch.float32, requires_grad=True
      )
      weight_cpu = torch.randn(
          (repeat_factor, block_rows, 5), dtype=torch.float32
      )
      fn(x_cpu, weight_cpu).backward()

      x_tpu = x_cpu.detach().to(self.device).requires_grad_(True)
      weight_tpu = weight_cpu.to(self.device)
      torch._dynamo.mark_dynamic(weight_tpu, 0, min=2, max=16)
      compiled(x_tpu, weight_tpu).backward()

      utils.assert_close(x_tpu.grad, x_cpu.grad)

    self.assertEqual(compile_counter.frame_count, 1)

  def test_repeat_static_counts_on_dynamic_input(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.repeat(1, 2) + 1.0

    model = Model()
    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(model, backend=compile_counter)

    for rows in [3, 7]:
      x_cpu = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
      expected = model(x_cpu)

      x_tpu = x_cpu.to(self.device)
      torch._dynamo.mark_dynamic(x_tpu, 0, min=2, max=16)

      out = compiled(x_tpu)
      self.assertEqual(out.shape, expected.shape)
      utils.assert_close(out, expected)

    self.assertEqual(compile_counter.frame_count, 1)

  def test_repeat_dynamic_input_dim_symbolic_count(self):
    def fn(x, y):
      return x.repeat(y.shape[0], 1)

    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(fn, backend=compile_counter)
    for rows, repeat_factor in [(3, 2), (7, 5)]:
      x_cpu = torch.randn((rows, 5), dtype=torch.float32, requires_grad=True)
      y_cpu = torch.ones((repeat_factor, 5), dtype=torch.float32)
      grad_out = torch.randn((rows * repeat_factor, 5), dtype=torch.float32)
      expected = fn(x_cpu, y_cpu)
      expected.backward(grad_out)

      x_tpu = x_cpu.detach().to(self.device).requires_grad_(True)
      y_tpu = y_cpu.to(self.device)
      torch._dynamo.mark_dynamic(x_tpu, 0, min=2, max=16)
      torch._dynamo.mark_dynamic(y_tpu, 0, min=2, max=16)

      out = compiled(x_tpu, y_tpu)
      self.assertEqual(out.shape, expected.shape)
      utils.assert_close(out, expected)
      out.backward(grad_out.to(self.device))
      utils.assert_close(x_tpu.grad, x_cpu.grad)

    self.assertEqual(compile_counter.frame_count, 1)

  def test_repeat_dynamic_input_dim_static_count(self):
    def fn(x):
      return x.repeat(2, 1)

    compile_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(fn, backend=compile_counter)
    for rows in [3, 7]:
      x_cpu = torch.randn((rows, 5), dtype=torch.float32, requires_grad=True)
      grad_out = torch.randn((rows * 2, 5), dtype=torch.float32)
      expected = fn(x_cpu)
      expected.backward(grad_out)

      x_tpu = x_cpu.detach().to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(x_tpu, 0, min=2, max=16)

      out = compiled(x_tpu)
      self.assertEqual(out.shape, expected.shape)
      utils.assert_close(out, expected)
      out.backward(grad_out.to(self.device))
      utils.assert_close(x_tpu.grad, x_cpu.grad)

    self.assertEqual(compile_counter.frame_count, 1)


if __name__ == "__main__":
  absltest.main()
