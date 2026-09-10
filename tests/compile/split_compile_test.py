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

    base_compiler = compiler.StaticCompiler(debug=True)
    split_comp = split_compiler.SplitCompiler(base_compiler)

    weight = torch.ones((2, 2), device="tpu")
    grad = torch.full((2, 2), 2.0, device="tpu")
    executable = split_comp(gm, (weight, grad))
    res_add, res_weight = executable(weight, grad)

    expected = torch.full((2, 2), 3.0, device="cpu")
    utils.assert_close(res_add.cpu(), expected)
    utils.assert_close(res_weight.cpu(), expected)


class _DummyCompiledExecutable(split_compiler.CompiledArtifact):

  def __init__(self, gm):
    self.gm = gm
    self._updates_default_generator_state = False
    self.mlir_text = "dummy_mlir"
    self.graph_module_debug_str = gm.print_readable(print_output=False)

  def updates_default_generator_state(self) -> bool:
    return self._updates_default_generator_state

  def __reduce__(self):
    return (_DummyCompiledExecutable, (self.gm,))

  def __call__(self, *args, **kwargs):
    return self.gm(*args, **kwargs)


class _DummyBaseCompiler(compiler.Compiler):

  def __init__(self):
    super().__init__(debug=True)

  def __call__(self, gm, args, is_fwd=True, **kwargs):
    return _DummyCompiledExecutable(gm)


class SplitCompileHermeticTest(seed_test_utils.RepeatableTest):

  def test_single_partition_with_lifted_parameters_no_submod_split(self):
    """Verifies that an FX graph with lifted parameters as placeholders (matching AOT Autograd in DeepSeek-V2) without collectives is NOT partitioned into submod_0."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    w1 = graph.placeholder("w1")
    w2 = graph.placeholder("w2")
    bias = graph.placeholder("bias")
    h1 = graph.call_function(torch.matmul, (x, w1))
    h2 = graph.call_function(torch.matmul, (h1, w2))
    out = graph.call_function(torch.add, (h2, bias))
    graph.output(out)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    x_tensor = torch.ones(2, 4)
    w1_tensor = torch.ones(4, 4)
    w2_tensor = torch.ones(4, 4)
    bias_tensor = torch.ones(4)
    inputs = (x_tensor, w1_tensor, w2_tensor, bias_tensor)

    split_comp = split_compiler.SplitCompiler(_DummyBaseCompiler())
    executable = split_comp(gm, inputs)

    # In DeepSeek-V2, the forward graph has no collectives (single partition).
    # CL 967263385 removed the `num_partitions <= 1` bypass, forcing the graph
    # through split_module which partitioned it into 'submod_0' and hoisted all
    # parameter placeholders into call_module arguments.
    # The fix must bypass split_module and not create submod_0.
    self.assertIsInstance(executable, split_compiler._SplitCompiledExecutable)
    self.assertEqual(len(executable.compiled_executables), 1)
    # Verify execution produces correct output
    res = executable(*inputs)
    expected = (
        torch.matmul(torch.matmul(x_tensor, w1_tensor), w2_tensor) + bias_tensor
    )
    utils.assert_close(res, expected)


if __name__ == "__main__":
  absltest.main()
