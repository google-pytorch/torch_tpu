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

import collections
import gc
import pickle
import unittest.mock
from absl.testing import absltest
from absl.testing import parameterized
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

  def test_split_graph_with_dynamic_tensor(self):
    def f(x):
      y = x * 2
      z = y + 3
      return z + 4

    # Monkeypatch COLLECTIVE_OPS to force a split on 'mul' (x * 2)
    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      compiled_f = torch.compile(f, backend="tpu")

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
      compiled_f = torch.compile(f, backend="tpu")

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

  def test_split_graph_without_value_metadata(self):
    """Verifies that nodes of a graph without fake values are compiled for TPU."""

    def f(x):
      y = torch.ops.aten.sin(x)
      z = torch.ops.aten.mul(y, 2.0)
      return torch.ops.aten.add(z, 3.0)

    # Monkeypatch COLLECTIVE_OPS to force a split on 'mul'.
    orig_ops = collective_ops.COLLECTIVE_OPS
    new_ops = orig_ops + (torch.ops.aten.mul,)

    x = torch.ones((2, 2), device="tpu")
    with unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops):
      split_comp = split_compiler.SplitCompiler(compiler.StaticCompiler())
      executable = split_comp(torch.fx.symbolic_trace(f), (x,))

    self.assertLen(executable.compiled_executables, 3)
    utils.assert_close(executable(x).cpu(), f(x).cpu())

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

  def test_compiled_forward_does_not_copy_saved_weights(self):
    """Verifies that a compiled forward holds no copies of the weights."""
    # Two 16 MiB weights: 4096 * 1024 * 4 bytes each.
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(4096, 1024, bias=False),
    ).to("tpu")
    x = torch.randn(8, 1024, device="tpu", requires_grad=True)
    compiled = torch.compile(
        model, backend="tpu", options={"serializable": False}
    )
    # Compiles the forward and backward and allocates the gradients.
    compiled(x).sum().backward()

    gc.collect()
    torch.accelerator.synchronize()
    mem_before = torch.accelerator.memory_allocated()
    y = compiled(x)
    torch.accelerator.synchronize()
    mem_held = torch.accelerator.memory_allocated() - mem_before
    y.sum().backward()

    # Backward needs both weights. The activations it needs and `y` take
    # less than 1 MiB, and copies of the weights would add 32 MiB.
    self.assertLess(mem_held, 5 * 1024 * 1024)  # < 5 MiB

  def test_compiled_forward_copies_buffer_it_updates(self):
    """Verifies backward after a forward that reads and then updates a buffer.

    The forward returns the buffer for backward, and AOTAutograd copies the
    new value into it after the forward runs. Unlike the weights in the test
    above, the buffer has to be returned as a copy.
    """

    class ScaleThenUpdate(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(4))

      def forward(self, x):
        # Backward needs `scale + 1`. Eager autograd saves the sum, so the
        # update below is allowed, but the min-cut partitioner saves `scale`
        # itself and recomputes the sum in backward.
        y = x * (self.scale + 1)
        self.scale.add_(1)
        return y

    model = ScaleThenUpdate()
    tpu_model = ScaleThenUpdate().to("tpu")
    x = torch.randn(2, 4, requires_grad=True)
    tpu_x = x.detach().to("tpu").requires_grad_()
    compiled = torch.compile(
        tpu_model, backend="tpu", options={"serializable": False}
    )

    model(x).sum().backward()
    compiled(tpu_x).sum().backward()

    utils.assert_close(tpu_x.grad.cpu(), x.grad)
    utils.assert_close(tpu_model.scale.cpu(), model.scale)

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

  def test_serializing_executable_does_not_retain_sunk_constants(self):
    """Verifies a compiled executable does not serialize the constants sunk into its submodules."""
    const_shape = (4096, 64, 128)

    root = torch.nn.Module()
    root.const = (
        torch.arange(4096 * 64 * 128, dtype=torch.float32)
        .reshape(const_shape)
        .to("tpu")
    )

    graph = torch.fx.Graph()
    p_x = graph.placeholder("x")
    c = graph.get_attr("const")
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, (p_x, c)))
    gm = torch.fx.GraphModule(root, graph)

    x = torch.ones(const_shape, dtype=torch.float32, device="tpu")
    split_comp = split_compiler.SplitCompiler(compiler.StaticCompiler())
    executable = split_comp(gm, (x,))

    payload = pickle.dumps(executable)

    gc.collect()
    torch.accelerator.synchronize()
    mem_before = torch.accelerator.memory_allocated()

    restored = pickle.loads(payload)

    torch.accelerator.synchronize()
    mem_increase = torch.accelerator.memory_allocated() - mem_before
    self.assertLess(mem_increase, 5 * 1024 * 1024)  # < 5 MiB

    # We check that the constant is still present in the serialized executable
    # by ensuring the result is correct.
    utils.assert_close(
        restored(x)[0, 0, :4].cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0])
    )

  @parameterized.named_parameters(
      ("single_field", 1),
      ("two_fields", 2),
  )
  def test_split_compile_preserves_namedtuple_output(self, num_fields: int):
    x = torch.randn(4, 4, device="tpu")
    y = torch.randn(4, 4, device="tpu")
    graph = torch.fx.Graph()
    px = graph.placeholder("x")
    py_node = graph.placeholder("y")
    sum_node = graph.call_function(torch.ops.aten.add.Tensor, (px, py_node))
    if num_fields == 1:
      out_type = collections.namedtuple("Single", ["sum"])
      graph.output(out_type(sum_node))
    else:
      out_type = collections.namedtuple("Pair", ["sum", "diff"])
      diff_node = graph.call_function(torch.ops.aten.sub.Tensor, (px, py_node))
      graph.output(out_type(sum_node, diff_node))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    executable = split_compiler.SplitCompiler(compiler.StaticCompiler())(
        gm, (x, y)
    )
    out = executable(x, y)

    self.assertIs(type(out), out_type)
    utils.assert_close(out.sum.cpu(), (x + y).cpu())
    if num_fields == 2:
      utils.assert_close(out.diff.cpu(), (x - y).cpu())


class SplitCompileHermeticTest(seed_test_utils.RepeatableTest):

  def test_compiled_submodule_preserves_metadata(self):
    child = torch.fx.symbolic_trace(torch.nn.Identity())
    child.meta["compiler_annotation"] = (1, 3)

    parent_graph = torch.fx.Graph()
    parent_x = parent_graph.placeholder("x")
    parent_out = parent_graph.call_module("child", (parent_x,))
    parent_graph.output(parent_out)
    parent = torch.fx.GraphModule({"child": child}, parent_graph)

    compiled_modules = []

    def compiler_fn(gm, unused_args, module_name=None):
      del unused_args, module_name
      compiled_modules.append(gm)
      return gm

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    submod_compiler = split_compiler._SubmodCompiler(
        parent, compiler_fn, fake_mode
    )
    submod_compiler.run(torch.ones((2, 2)))

    self.assertLen(compiled_modules, 1)
    compiled_child = compiled_modules[0]
    self.assertEqual(compiled_child.meta["compiler_annotation"], (1, 3))
    self.assertIsNot(compiled_child.meta, child.meta)

  # TODO: b/576283069 - Remove once minimum OSS PyTorch is >= 2.12.0.
  def test_register_output_namedtuple_globals_without_codegen_repr_fn(self):
    out_type = collections.namedtuple("LocalPair", ["sum", "diff"])
    graph = torch.fx.Graph()
    px = graph.placeholder("x")
    py_node = graph.placeholder("y")
    sum_node = graph.call_function(torch.ops.aten.add.Tensor, (px, py_node))
    diff_node = graph.call_function(torch.ops.aten.sub.Tensor, (px, py_node))
    graph.output(out_type(sum_node, diff_node))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Simulate PyTorch 2.11's CodeGen.generate_output, which formats output_args
    # via repr() without registering namedtuple classes through repr_fn.
    with unittest.mock.patch.object(
        gm.graph._codegen,
        "generate_output",
        side_effect=lambda output_args, *, descs=None: f"return {repr(output_args)}",
    ):
      split_compiler._register_output_namedtuple_globals(gm)
      gm.recompile()

    x = torch.tensor([3.0, 5.0])
    y = torch.tensor([1.0, 2.0])
    out = gm(x, y)
    self.assertIs(type(out), out_type)
    utils.assert_close(out.sum, x + y)
    utils.assert_close(out.diff, x - y)


if __name__ == "__main__":
  absltest.main()
