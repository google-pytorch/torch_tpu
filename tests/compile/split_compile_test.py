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
from collections.abc import Callable, Iterator
import contextlib
import gc
import pickle
import re
from typing import Any
import unittest.mock
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.fx.experimental import proxy_tensor
from torch.utils import checkpoint
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.distributed import collective_ops
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

# Importing `offload_ops` registers `torch.ops.ao.offload` and
# `torch.ops.ao.reload`.
try:
  # pylint: disable=g-import-not-at-top
  from torch._functorch._activation_offloading import offload_ops

  _HAS_AO_OPS = offload_ops is not None
except (ImportError, AttributeError):
  _HAS_AO_OPS = False


class SplitCompileTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")

  @contextlib.contextmanager
  def _assert_splits_on_mul(self) -> Iterator[None]:
    """Treats aten.mul as a collective and asserts the graph was split on it."""
    new_ops = collective_ops.COLLECTIVE_OPS + (torch.ops.aten.mul,)
    with (
        unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops),
        self.assertLogs(logging.get_absl_logger(), level="INFO") as logs,
    ):
      yield
    num_partitions = [
        int(n)
        for n in re.findall(
            r"Split graph into (\d+) partitions", "\n".join(logs.output)
        )
    ]
    self.assertNotEmpty(num_partitions)
    self.assertGreater(max(num_partitions), 1)

  def test_split_graph_with_dynamic_tensor(self):
    def f(x):
      y = x * 2
      z = y + 3
      return z + 4

    # Force a split on 'mul' (x * 2).
    with self._assert_splits_on_mul():
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

    # Force a split on 'mul' (x * 2).
    with self._assert_splits_on_mul():
      compiled_f = torch.compile(f, backend="tpu")

      x = torch.ones((8, 6), device="tpu")
      torch._dynamo.mark_dynamic(x, 0, min=2, max=16)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=16)

      res = compiled_f(x)

    expected = torch.full((8, 1, 6, 1), 5.0, device="cpu")
    utils.assert_close(res.cpu(), expected)

  def test_split_graph_with_compound_symint_in_backward(self):
    """Verifies backward split when a partition only receives a compound SymInt.

    AOTAutograd saves `s0 // 2` from forward as a backward placeholder.
    Splitting backward on `mul` leaves the upstream partition (`(-y).sum()`
    grad) with only scalar `grad_out` and `s0 // 2`, while `s0` is only used
    downstream in `slice_backward`.
    """

    def f(x):
      y = x[: x.shape[0] // 2] * 2.0
      return (-y).sum()

    x_ref = (
        torch.arange(32, dtype=torch.float32).reshape(8, 4).requires_grad_(True)
    )
    out_ref = f(x_ref)
    out_ref.backward()

    with self._assert_splits_on_mul():
      compiled_f = torch.compile(f, backend="tpu")

      x = x_ref.detach().to("tpu").requires_grad_(True)
      torch._dynamo.mark_dynamic(x, 0, min=8, max=16)
      out = compiled_f(x)
      out.backward()

    utils.assert_close(out.detach().cpu(), out_ref.detach())
    utils.assert_close(x.grad.cpu(), x_ref.grad)

  def test_split_graph_with_embedded_constants(self):
    def f(x):
      y = x * 2
      c = torch.tensor(5.0)
      return y + c

    with self._assert_splits_on_mul():
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

  def test_checkpointed_spmd_safe_block_is_not_split(self):
    """Verifies a checkpointed `@spmd_safe` block stays a single partition.

    `checkpoint.checkpoint` emits an unannotated outer higher-order op wrapping
    `safe_block`'s child `GraphModule`. Treating `aten.mul` as a collective
    checks that `SplitCompiler` inspects the child subgraph and keeps both
    forward and backward unsplit (`[1, 1]` partitions).
    """

    @spmd_util.spmd_safe
    def safe_block(x):
      return x + (x * 2.0) * 3.0

    def f(x):
      return checkpoint.checkpoint(safe_block, x, use_reentrant=False)

    new_ops = collective_ops.COLLECTIVE_OPS + (torch.ops.aten.mul,)
    x = torch.ones((2, 2), device="tpu", requires_grad=True)
    with (
        unittest.mock.patch.object(collective_ops, "COLLECTIVE_OPS", new_ops),
        self.assertLogs(logging.get_absl_logger(), level="INFO") as logs,
    ):
      compiled_f = torch.compile(
          f, backend="tpu", options={"serializable": False}
      )
      compiled_f(x).sum().backward()

    num_partitions = [
        int(n)
        for n in re.findall(
            r"Split graph into (\d+) partitions", "\n".join(logs.output)
        )
    ]
    self.assertEqual(num_partitions, [1, 1])
    utils.assert_close(x.grad.cpu(), torch.full((2, 2), 7.0))

  def test_move_constants_to_cpu(self):
    const = torch.tensor([1.0, 2.0], device="tpu")
    gm = proxy_tensor.make_fx(lambda x: x + const)(
        torch.ones((2,), device="tpu")
    )
    buffers = dict(gm.named_buffers())
    self.assertLen(buffers, 1)
    const_name = next(iter(buffers))

    split_compiler._move_constants_to_cpu(gm)

    moved = getattr(gm, const_name)
    self.assertEqual(moved.device.type, "cpu")
    utils.assert_close(moved, torch.tensor([1.0, 2.0]))

  def test_move_constants_to_cpu_nested_param(self):
    child_param = torch.nn.Parameter(
        torch.tensor([2.0], device="tpu"), requires_grad=True
    )

    def branch(t):
      return t + child_param

    def fn(pred, x):
      return torch.ops.higher_order.cond(pred, branch, branch, (x,))

    with torch.no_grad():
      parent_gm = proxy_tensor.make_fx(fn)(
          torch.tensor(True, device="tpu"), torch.ones((1,), device="tpu")
      )

    child_params = dict(parent_gm.true_graph_0.named_parameters())
    self.assertLen(child_params, 1)
    param_name = next(iter(child_params))

    split_compiler._move_constants_to_cpu(parent_gm)

    moved_param = getattr(parent_gm.true_graph_0, param_name)
    self.assertIsInstance(moved_param, torch.nn.Parameter)
    self.assertEqual(moved_param.device.type, "cpu")
    self.assertTrue(moved_param.requires_grad)

  def test_move_constants_to_cpu_under_fake_mode(self):
    const = torch.tensor([1.0], device="tpu")
    gm = proxy_tensor.make_fx(lambda x: x + const)(
        torch.ones((1,), device="tpu")
    )
    buffers = dict(gm.named_buffers())
    self.assertLen(buffers, 1)
    const_name = next(iter(buffers))

    with torch._subclasses.fake_tensor.FakeTensorMode():
      split_compiler._move_constants_to_cpu(gm)

    moved = getattr(gm, const_name)
    self.assertEqual(moved.device.type, "cpu")
    self.assertNotIsInstance(moved, torch._subclasses.fake_tensor.FakeTensor)

  def test_pickle_graph_rejects_generator_input(self):
    gm = torch.fx.symbolic_trace(lambda x, gen: x * 2.0)

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    fake_x = fake_mode.from_tensor(torch.ones((2, 2), device="tpu"))
    gen = tpu_torch_compile.make_export_mode_device_generator(
        torch.Generator(device="tpu")
    )

    state = split_compiler._SnapshotState(
        gm, (fake_x, gen), is_fwd=True, module_name=None
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Compiler unit test.
        pickle.PicklingError, "generator"
    ):
      split_compiler._pickle_graph(state)

  def test_recompile_fn_pickle_preserves_cpu_placement(self):
    def fn(x):
      c = torch.tensor([1.0, 2.0, 3.0])
      return x * c.to(x.device) + (torch.ones(3) * c).to(x.device)

    x = torch.tensor([2.0, 3.0, 4.0], device="tpu")
    c = torch.tensor([1.0, 2.0, 3.0])
    with torch.no_grad():
      gm = proxy_tensor.make_fx(
          fn, tracing_mode="fake", _allow_non_fake_inputs=True
      )(x)
    fake_x = torch._subclasses.fake_tensor.FakeTensorMode().from_tensor(x)
    recompile_fn = split_compiler._RecompileFn(
        split_compiler.SplitCompiler(compiler.StaticCompiler()),
        split_compiler._SnapshotState(
            gm, (fake_x,), is_fwd=True, module_name=None
        ),
    )

    recompiled = pickle.loads(pickle.dumps(recompile_fn))()
    num_cpu_partitions = sum(
        isinstance(submod, torch.fx.GraphModule)
        for submod in recompiled._split_gm.children()
    )
    self.assertEqual(
        (len(recompiled.compiled_executables), num_cpu_partitions), (1, 1)
    )
    utils.assert_close(
        recompiled(x).cpu(), torch.tensor([2.0, 3.0, 4.0]) * c + c
    )

  @absltest.skipIf(
      not _HAS_AO_OPS,
      "`torch._functorch._activation_offloading.offload_ops` is not available "
      "in this PyTorch version",
  )
  def test_recompile_fn_pickle_preserves_pinned_host_input(self):
    def fn(pinned, device_x):
      return torch.ops.ao.reload(pinned, torch.device("tpu")) + device_x

    base_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pinned_x = torch.ops.ao.offload(base_cpu.to("tpu"))
    device_x = torch.full((2, 3), 100.0, device="tpu")

    with torch.no_grad():
      gm = proxy_tensor.make_fx(
          fn, tracing_mode="fake", _allow_non_fake_inputs=True
      )(pinned_x, device_x)

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    fake_pinned = fake_mode.from_tensor(torch.empty((2, 3), device="tpu"))
    fake_pinned._is_pinned_host = True
    fake_device_x = fake_mode.from_tensor(device_x)

    recompile_fn = split_compiler._RecompileFn(
        split_compiler.SplitCompiler(compiler.StaticCompiler()),
        split_compiler._SnapshotState(
            gm, (fake_pinned, fake_device_x), is_fwd=True, module_name=None
        ),
    )

    recompiled = pickle.loads(pickle.dumps(recompile_fn))()
    self.assertEqual(recompiled.fingerprint(), recompile_fn().fingerprint())
    utils.assert_close(recompiled(pinned_x, device_x).cpu(), base_cpu + 100.0)

  def test_recompile_fn_pickle_preserves_custom_layout_input(self):
    def fn(x):
      return x * 2.0 + 1.0

    layout = torch.tpu.TpuLayout(minor_to_major=[0, 1], tiles=[[8, 128]])
    cpu_x = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    with torch.tpu.LayoutContext(layout):
      x = cpu_x.to("tpu")

    with torch.no_grad():
      gm = proxy_tensor.make_fx(
          fn, tracing_mode="fake", _allow_non_fake_inputs=True
      )(x)

    fake_x = torch._subclasses.fake_tensor.FakeTensorMode().from_tensor(x)
    fake_x._tpu_device_layout = compiler.get_non_default_device_layout(x)
    self.assertIsNotNone(fake_x._tpu_device_layout)

    recompile_fn = split_compiler._RecompileFn(
        split_compiler.SplitCompiler(compiler.StaticCompiler()),
        split_compiler._SnapshotState(
            gm, (fake_x,), is_fwd=True, module_name=None
        ),
    )

    recompiled = pickle.loads(pickle.dumps(recompile_fn))()
    self.assertEqual(recompiled.fingerprint(), recompile_fn().fingerprint())
    utils.assert_close(recompiled(x).cpu(), cpu_x * 2.0 + 1.0)


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

  def _trace_dynamo(
      self, fn: Callable[..., Any], *args: Any
  ) -> torch.fx.GraphModule:
    """Returns the single pre-AOTAutograd FX graph that Dynamo traces for `fn`."""
    graphs = []

    def backend(
        gm: torch.fx.GraphModule, example_inputs: Any
    ) -> Callable[..., Any]:
      del example_inputs
      graphs.append(gm)
      return gm.forward

    torch.compile(fn, backend=backend)(*args)
    self.assertLen(graphs, 1)
    return graphs[0]

  def test_is_fully_spmd_safe_passthrough_and_tensor_indexing(self):
    """Verifies passthrough graphs return `False` and tensor indexing counts as an op."""
    self.assertFalse(
        split_compiler._is_fully_spmd_safe(torch.fx.symbolic_trace(lambda x: x))
    )

    @spmd_util.spmd_safe
    def safe_index(x):
      return x[0]

    gm = self._trace_dynamo(safe_index, torch.ones(2, 2))
    self.assertTrue(split_compiler._is_fully_spmd_safe(gm))

  def test_is_fully_spmd_safe_skips_subgraph_higher_order_op(self):
    """Verifies subgraph HOPs and tuple unpacks defer to the child subgraph.

    `checkpoint` emits an unannotated HOP plus tuple-unpacking `getitem`s in the
    outer graph; `_is_fully_spmd_safe` should skip those pass-through nodes and
    check the ops inside the child `GraphModule` instead.
    """

    @spmd_util.spmd_safe
    def safe_pair(x):
      return x * 2, x + 1

    @spmd_util.spmd_safe
    def safe_add(a, b):
      return a + b

    def safe_ckpt(x):
      a, b = checkpoint.checkpoint(safe_pair, x, use_reentrant=False)
      return safe_add(a, b)

    gm = self._trace_dynamo(safe_ckpt, torch.ones(2, 2, requires_grad=True))
    self.assertTrue(split_compiler._is_fully_spmd_safe(gm))

  # Dynamo resumes `torch.fx.traceback.annotate` after a graph break only on
  # PyTorch builds that have `_dynamo_annotate` (nightly 2.16.0a0 and later).
  @absltest.skipIf(
      not hasattr(torch.fx.traceback, "_dynamo_annotate"),
      "This PyTorch build cannot resume torch.fx.traceback.annotate after a"
      " graph break.",
  )
  def test_is_fully_spmd_safe_across_graph_break(self):
    """Verifies both subgraphs around a Dynamo graph break keep `spmd_safe` metadata."""

    @spmd_util.spmd_safe
    def fn(x):
      x = x * 2
      torch._dynamo.graph_break()  # pylint: disable=protected-access
      return x + 1

    graphs = []

    def backend(gm, example_inputs):
      del example_inputs
      graphs.append(gm)
      return gm.forward

    torch.compile(fn, backend=backend)(torch.ones(2, 2))
    self.assertLen(graphs, 2)
    for gm in graphs:
      self.assertTrue(split_compiler._is_fully_spmd_safe(gm))

  def test_clone_graph_module_clones_child_graph_modules(self):
    def fn(pred, x):
      return torch.cond(
          pred,
          lambda t: t + torch.tensor([1.0]),
          lambda t: t - torch.tensor([1.0]),
          (x,),
      )

    gm = proxy_tensor.make_fx(fn)(torch.tensor(True), torch.tensor([2.0]))
    gm.meta["tag"] = "parent"
    gm.true_graph_0.meta["tag"] = "child"

    child_buffers = dict(gm.true_graph_0.named_buffers())
    self.assertLen(child_buffers, 1)
    const_name, child_const = next(iter(child_buffers.items()))

    cloned = split_compiler._clone_graph_module(gm)

    self.assertIsNot(cloned, gm)
    self.assertIsNot(cloned.true_graph_0, gm.true_graph_0)
    self.assertIsNot(cloned.false_graph_0, gm.false_graph_0)
    self.assertIsNot(cloned.true_graph_0.graph, gm.true_graph_0.graph)
    self.assertIs(getattr(cloned.true_graph_0, const_name), child_const)
    self.assertEqual(cloned.meta, gm.meta)
    self.assertIsNot(cloned.meta, gm.meta)
    self.assertEqual(cloned.true_graph_0.meta, gm.true_graph_0.meta)
    self.assertIsNot(cloned.true_graph_0.meta, gm.true_graph_0.meta)

    setattr(cloned.true_graph_0, const_name, torch.tensor([2.0]))
    self.assertIs(getattr(gm.true_graph_0, const_name), child_const)

  def test_recompile_fn_rejects_unexpected_unpickled_payload(self):
    bad_payload = pickle.dumps("not_a_snapshot")
    recompile_fn = split_compiler._RecompileFn(
        split_compiler.SplitCompiler(compiler.StaticCompiler()), bad_payload
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Compiler unit test.
        pickle.UnpicklingError, r"Expected _SnapshotState, got str\."
    ):
      recompile_fn()


if __name__ == "__main__":
  absltest.main()
