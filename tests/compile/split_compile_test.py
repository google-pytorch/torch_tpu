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

import gc
import unittest.mock
from absl.testing import absltest
import torch
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.distributed import collective_ops
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class SplitCompileTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    # Dynamic shape buffers across partitions and host DMA alignment require
    # libtpu >= 0.0.44.
    if not utils.libtpu_at_least((0, 0, 44)):
      self.skipTest(
          "Dynamic shape buffer relaxation and host DMA require libtpu >="
          " 0.0.44"
      )

  def test_split_graph_with_dynamic_tensor(self):
    def f(x):
      y = x * 2
      z = y + 3
      return z + 4

    # Monkeypatch COLLECTIVE_OPS to force a split on 'mul' (x * 2)
    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      compiled_f = torch.compile(
          f, backend="tpu", options={"bounded_dynamism": True}
      )

      x = torch.ones((2, 2), device="tpu")
      torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=8)

      res = compiled_f(x)

    expected = torch.full((2, 2), 9.0, device="cpu")
    utils.assert_close(res.cpu(), expected)

  def test_split_graph_with_direct_symint_usage(self):
    def f(x):
      s1 = x.shape[0]
      s2 = x.shape[1]
      y = x * 2
      z = y.reshape(s1, 1, s2, 1)
      return z + 3

    # Monkeypatch COLLECTIVE_OPS to force a split on 'mul' (x * 2)
    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      compiled_f = torch.compile(
          f, backend="tpu", options={"bounded_dynamism": True}
      )

      x = torch.ones((8, 6), device="tpu")
      torch._dynamo.mark_dynamic(x, 0, min=2, max=16)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=16)

      res = compiled_f(x)

    expected = torch.full((8, 1, 6, 1), 5.0, device="cpu")
    utils.assert_close(res.cpu(), expected)

  def test_split_graph_with_embedded_constants(self):
    def f(x):
      y = x * 2
      c = torch.tensor(5.0)
      return y + c

    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      compiled_f = torch.compile(f, backend="tpu")

      x = torch.ones((2, 2), device="tpu")
      res = compiled_f(x)

    expected = torch.full((2, 2), 7.0, device="cpu")
    utils.assert_close(res.cpu(), expected)

  def test_split_graph_with_mutated_returned_placeholder(self):
    """Verifies that an FX graph returning in-place mutated input placeholders is correctly compiled by SplitCompiler and returns updated tensors."""
    graph = torch.fx.Graph()
    p_weight = graph.placeholder("weight")
    p_grad = graph.placeholder("grad")
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    base_compiler = compiler.StaticCompiler()
    split_comp = split_compiler.SplitCompiler(base_compiler)

    weight = torch.ones((2, 2), device="tpu")
    grad = torch.full((2, 2), 2.0, device="tpu")
    executable = split_comp(gm, (weight, grad))
    res_add, res_weight = executable(weight, grad)

    expected = torch.full((2, 2), 3.0, device="cpu")
    utils.assert_close(res_add.cpu(), expected)
    utils.assert_close(res_weight.cpu(), expected)

  def test_compiling_get_attr_constant_does_not_copy_it(self):
    """Verifies that compiling a graph does not duplicate `get_attr` constants."""
    # 128 MiB parameter: 1024 * 32768 * 4 bytes = 134,217,728 bytes.
    rows, cols = 1024, 32768

    root = torch.nn.Module()
    root.weight = torch.zeros((rows, cols), dtype=torch.float32).to("tpu")

    graph = torch.fx.Graph()
    p_x = graph.placeholder("x")
    c = graph.get_attr("weight")
    out = graph.call_function(torch.matmul, (p_x, c))
    graph.output(out)
    gm = torch.fx.GraphModule(root, graph)

    base_compiler = compiler.StaticCompiler()
    split_comp = split_compiler.SplitCompiler(base_compiler)
    x = torch.zeros((1, rows), dtype=torch.float32, device="tpu")

    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    mem_before = torch.accelerator.memory_allocated()

    split_comp(gm, (x,))

    torch.accelerator.synchronize()
    peak_mem_increase = torch.accelerator.max_memory_allocated() - mem_before

    # Assert that the peak memory increase is minimal, i.e. no large tensor
    # copies were made.
    self.assertLess(peak_mem_increase, 5 * 1024 * 1024)  # < 5 MiB


if __name__ == "__main__":
  absltest.main()
