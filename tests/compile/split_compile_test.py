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

    # Force a split on 'mul' (x * 2). `split_compiler` snapshots
    # `COLLECTIVE_OPS` into `_COLLECTIVE_OPS` at import time, so the snapshot is
    # what has to be patched.
    orig_ops = split_compiler._COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(split_compiler, "_COLLECTIVE_OPS", new_ops):
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

    # Force a split on 'mul' (x * 2). `split_compiler` snapshots
    # `COLLECTIVE_OPS` into `_COLLECTIVE_OPS` at import time, so the snapshot is
    # what has to be patched.
    orig_ops = split_compiler._COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(split_compiler, "_COLLECTIVE_OPS", new_ops):
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

    # Force a split on 'mul' (x * 2). `split_compiler` snapshots
    # `COLLECTIVE_OPS` into `_COLLECTIVE_OPS` at import time, so the snapshot is
    # what has to be patched.
    orig_ops = split_compiler._COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(split_compiler, "_COLLECTIVE_OPS", new_ops):
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

  def test_single_partition_fast_runner_direct_and_remapped_outputs(self):
    """Verifies _SplitCompiledExecutable._fast_runner handles unused placeholders, None, passthroughs, and list/tuple remapping."""
    base_compiler = compiler.StaticCompiler()
    split_comp = split_compiler.SplitCompiler(base_compiler)

    # 1. Graph with unused input placeholder (`unused`), passthrough placeholder
    # (`x`), literal `None`, and multiple computed outputs in a tuple.
    graph_tuple = torch.fx.Graph()
    p_x = graph_tuple.placeholder("x")
    p_unused = graph_tuple.placeholder("unused")
    p_y = graph_tuple.placeholder("y")
    del p_unused
    sum_node = graph_tuple.call_function(torch.add, (p_x, p_y))
    diff_node = graph_tuple.call_function(torch.sub, (p_x, p_y))
    graph_tuple.output((diff_node, None, p_x, sum_node))
    gm_tuple = torch.fx.GraphModule(torch.nn.Module(), graph_tuple)

    x = torch.full((2, 2), 5.0, device="tpu")
    unused = torch.full((2, 2), 99.0, device="tpu")
    y = torch.full((2, 2), 2.0, device="tpu")

    exe_tuple = split_comp(gm_tuple, (x, unused, y))
    self.assertIsNotNone(exe_tuple._fast_runner)
    out_diff, out_none, out_pass, out_sum = exe_tuple(x, unused, y)
    self.assertIsNone(out_none)
    utils.assert_close(out_diff.cpu(), torch.full((2, 2), 3.0, device="cpu"))
    utils.assert_close(out_pass.cpu(), torch.full((2, 2), 5.0, device="cpu"))
    utils.assert_close(out_sum.cpu(), torch.full((2, 2), 7.0, device="cpu"))

    # 2. Graph returning a list output (`_fast_remap_list_runner`).
    graph_list = torch.fx.Graph()
    lx = graph_list.placeholder("x")
    ly = graph_list.placeholder("y")
    prod_node = graph_list.call_function(torch.mul, (lx, ly))
    graph_list.output([prod_node, None, lx])
    gm_list = torch.fx.GraphModule(torch.nn.Module(), graph_list)

    exe_list = split_comp(gm_list, (x, y))
    self.assertIsNotNone(exe_list._fast_runner)
    list_res = exe_list([x, y])
    self.assertIsInstance(list_res, list)
    self.assertLen(list_res, 3)
    self.assertIsNone(list_res[1])
    utils.assert_close(
        list_res[0].cpu(), torch.full((2, 2), 10.0, device="cpu")
    )
    utils.assert_close(list_res[2].cpu(), torch.full((2, 2), 5.0, device="cpu"))

  def test_single_partition_fast_runner_identity_output_with_unused_placeholder(
      self,
  ):
    """Verifies the identity-output fast path stays correct when unused placeholders are filtered out."""
    base_compiler = compiler.StaticCompiler()
    split_comp = split_compiler.SplitCompiler(base_compiler)

    # The outer graph returns exactly the submodule's outputs in order (an
    # identity mapping), while `unused` is dropped from the submodule's
    # arguments, exercising the identity-plus-argument-filtering closure.
    graph = torch.fx.Graph()
    p_x = graph.placeholder("x")
    p_unused = graph.placeholder("unused")
    p_y = graph.placeholder("y")
    del p_unused
    sum_node = graph.call_function(torch.add, (p_x, p_y))
    diff_node = graph.call_function(torch.sub, (p_x, p_y))
    graph.output((sum_node, diff_node))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    x = torch.full((2, 2), 5.0, device="tpu")
    unused = torch.full((2, 2), 99.0, device="tpu")
    y = torch.full((2, 2), 2.0, device="tpu")

    executable = split_comp(gm, (x, unused, y))
    self.assertIsNotNone(executable._fast_runner)

    expected_sum = torch.full((2, 2), 7.0, device="cpu")
    expected_diff = torch.full((2, 2), 3.0, device="cpu")

    # Positional arguments and the AOTAutograd boxed calling convention
    # (`_boxed_call = True`) must produce identical results.
    positional_sum, positional_diff = executable(x, unused, y)
    utils.assert_close(positional_sum.cpu(), expected_sum)
    utils.assert_close(positional_diff.cpu(), expected_diff)

    boxed_sum, boxed_diff = executable([x, unused, y])
    utils.assert_close(boxed_sum.cpu(), expected_sum)
    utils.assert_close(boxed_diff.cpu(), expected_diff)

  def test_single_partition_fast_runner_unwraps_singleton_output(self):
    """Verifies the fast path unwraps a submodule's singleton output tuple into a bare tensor."""
    base_compiler = compiler.StaticCompiler()
    split_comp = split_compiler.SplitCompiler(base_compiler)

    # A graph whose output is a bare node (not a tuple) makes `compile_submod`
    # wrap the submodule output in a singleton tuple, so the fast runner must
    # unwrap it to match the parent graph's schema.
    graph = torch.fx.Graph()
    p_x = graph.placeholder("x")
    p_y = graph.placeholder("y")
    graph.output(graph.call_function(torch.add, (p_x, p_y)))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    x = torch.full((2, 2), 5.0, device="tpu")
    y = torch.full((2, 2), 2.0, device="tpu")

    executable = split_comp(gm, (x, y))
    self.assertIsNotNone(executable._fast_runner)

    result = executable(x, y)
    self.assertIsInstance(result, torch.Tensor)
    utils.assert_close(result.cpu(), torch.full((2, 2), 7.0, device="cpu"))

  def test_multi_partition_graph_falls_back_to_split_graph_module(self):
    """Verifies graphs split across partitions bypass the fast runner and still compute correctly."""
    base_compiler = compiler.StaticCompiler()

    graph = torch.fx.Graph()
    p_x = graph.placeholder("x")
    mul_node = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, 2))
    add_node = graph.call_function(torch.ops.aten.add.Tensor, (mul_node, 3))
    graph.output((add_node,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    x = torch.ones((2, 2), device="tpu")

    # Forcing `mul` to be treated as a collective splits the graph into
    # multiple partitions with a host-dispatched node in the outer graph, which
    # `_try_build_fast_runner` must reject. `split_compiler` snapshots
    # `COLLECTIVE_OPS` into `_COLLECTIVE_OPS` at import time, so the snapshot is
    # what has to be patched.
    orig_ops = split_compiler._COLLECTIVE_OPS
    with unittest.mock.patch.object(
        split_compiler,
        "_COLLECTIVE_OPS",
        orig_ops + (torch.ops.aten.mul,),
    ):
      split_comp = split_compiler.SplitCompiler(base_compiler)
      executable = split_comp(gm, (x,))
      self.assertIsNone(executable._fast_runner)
      (result,) = executable(x)

    utils.assert_close(result.cpu(), torch.full((2, 2), 5.0, device="cpu"))


if __name__ == "__main__":
  absltest.main()
