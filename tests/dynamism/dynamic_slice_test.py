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

"""Tests for slice-like op lowering on bounded dynamic dimensions.

Covers HandleSliceLikeOpsPass (slice, select, slice_backward, constant_pad_nd),
the pre-grad clone of returned dynamic views, and TPU end-to-end checks.
"""

from __future__ import annotations

from collections.abc import Callable
import sys
from typing import Any, NamedTuple

from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
import torch.nn.functional as F
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile.dynamic import graph_transform_context
from torch_tpu._internal.compile.dynamic import graph_transformations
from torch_tpu._internal.compile.dynamic import sym_bounds_manager
from torch_tpu._internal.compile.dynamic import view_ops_passes
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

def _compile_bd(fn: Callable[..., Any]) -> Callable[..., Any]:
  return torch.compile(fn, backend="tpu", dynamic=None)


class _PassResult(NamedTuple):
  """Graph after HandleSliceLikeOpsPass and the guards the pass installed."""

  gm: torch.fx.GraphModule
  ctx: graph_transform_context.GraphTransformContext
  new_guards: list[str]


class _CaptureDone(Exception):
  """Raised from the backend once the pass ran, to skip guard building."""


def _capture_and_run_slice_pass(
    fn: Callable[..., Any],
    inputs: list[torch.Tensor],
    bucket_idx: int | None = None,
) -> _PassResult:
  """Captures the FX graph via aot_autograd and runs HandleSliceLikeOpsPass.

  The pass runs inside the backend compiler, while the ShapeEnv is still
  unfrozen exactly as in production, so every SymBool the pass evaluates is
  recorded as a new guard and reported in the result.
  """
  result: _PassResult | None = None

  def fw_compiler(
      graph_module: torch.fx.GraphModule, example_inputs: list[Any]
  ) -> torch.fx.GraphModule:
    nonlocal result
    shape_env = torch._guards.detect_fake_mode(example_inputs).shape_env
    num_guards = len(shape_env.guards)
    bm = sym_bounds_manager.SymBoundsManager(example_inputs)
    if bucket_idx is None:
      bounds = bm.get_fallback_symbol_bounds()
    else:
      bounds = bm.get_bucketed_symbol_bounds([], num_buckets=bucket_idx + 1)[
          bucket_idx
      ]
    ctx = graph_transform_context.GraphTransformContext(
        graph_module, example_inputs, bounds
    )
    placeholders = list(
        graph_module.graph.find_nodes(op="placeholder", sort=True)
    )
    graph_transformations.ScanInputsCreatePlaceholdersPass(ctx, placeholders)(
        graph_module
    )
    view_ops_passes.HandleSliceLikeOpsPass(ctx)(graph_module)
    new_guards = [str(g.expr) for g in shape_env.guards[num_guards:]]
    result = _PassResult(graph_module, ctx, new_guards)
    raise _CaptureDone()

  torch._dynamo.reset()
  compiled = torch.compile(
      fn, backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True
  )
  try:
    compiled(*inputs)
  except torch._dynamo.exc.BackendCompilerFailed as e:
    if not isinstance(e.inner_exception, _CaptureDone):
      raise
  if result is None:
    raise AssertionError("the backend compiler was never invoked")
  return result


def _capture_dynamo_graph(
    fn: Callable[..., Any],
    inputs: list[torch.Tensor],
) -> torch.fx.GraphModule:
  """Returns the Dynamo (pre-grad) graph of fn traced with dynamic shapes."""
  captured: list[torch.fx.GraphModule] = []

  def backend(
      graph_module: torch.fx.GraphModule, example_inputs: list[Any]
  ) -> Callable[..., Any]:
    del example_inputs
    captured.append(graph_module)
    return graph_module.forward

  torch._dynamo.reset()
  torch.compile(fn, backend=backend, dynamic=True)(*inputs)
  if len(captured) != 1:
    raise AssertionError(f"expected one Dynamo graph, got {len(captured)}")
  return captured[0]


def _dynamic_slice_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
  return list(
      gm.graph.find_nodes(
          op="call_function", target=torch.ops.tpu._dynamic_slice
      )
  )


def _transitive_placeholder_names(node: torch.fx.Node) -> set[str]:
  """Returns the names of all placeholders node transitively depends on."""
  names: set[str] = set()
  seen: set[torch.fx.Node] = set()
  stack = [node]
  while stack:
    curr = stack.pop()
    if curr in seen:
      continue
    seen.add(curr)
    if curr.op == "placeholder":
      names.add(curr.name)
    stack.extend(curr.all_input_nodes)
  return names


class HandleSliceLikeOpsPassTest(seed_test_utils.RepeatableTest):
  """CPU-only unit tests for HandleSliceLikeOpsPass on bounded dynamic dims."""

  def _assert_single_dynamic_slice(self, result: _PassResult) -> torch.fx.Node:
    self.assertEmpty(
        result.new_guards, f"pass installed guards: {result.new_guards}"
    )
    slice_nodes = _dynamic_slice_nodes(result.gm)
    self.assertLen(slice_nodes, 1)
    return slice_nodes[0]

  def test_pass_normalizes_negative_symint_start(self) -> None:
    s = torch.ones(2, 3, 6)
    x = torch.ones(2, 3, 8)
    torch._dynamo.mark_static(s)
    torch._dynamo.mark_dynamic(x, 2, min=2, max=16)

    result = _capture_and_run_slice_pass(
        lambda s_in, x_in: torch.cat([s_in, x_in], dim=-1)[
            Ellipsis, -x_in.shape[-1] :
        ],
        [s, x],
    )

    slice_node = self._assert_single_dynamic_slice(result)
    last_start = slice_node.args[1][-1]
    self.assertEqual(last_start.target, torch.ops.aten.scalar_tensor.default)
    self.assertEqual(last_start.args[0], 6)

  def test_pass_normalizes_negative_symint_select_index(self) -> None:
    s = torch.ones(2, 3, 6)
    x = torch.ones(2, 3, 8)
    torch._dynamo.mark_static(s)
    torch._dynamo.mark_dynamic(x, 2, min=2, max=16)

    result = _capture_and_run_slice_pass(
        lambda s_in, x_in: torch.cat([s_in, x_in], dim=-1).select(
            -1, -x_in.shape[-1]
        ),
        [s, x],
    )

    slice_node = self._assert_single_dynamic_slice(result)
    last_start = slice_node.args[1][-1]
    self.assertEqual(last_start.target, torch.ops.aten.scalar_tensor.default)
    self.assertEqual(last_start.args[0], 6)

  def test_pass_pads_operand_when_slice_window_exceeds_upper_bound(
      self,
  ) -> None:
    m = torch.ones(4, 12)
    q = torch.ones(4, 4)
    torch._dynamo.mark_dynamic(m, 1, min=8, max=16)
    torch._dynamo.mark_dynamic(q, 1, min=2, max=8)

    result = _capture_and_run_slice_pass(
        lambda m_in, q_in: m_in[:, -q_in.shape[-1] :] + q_in,
        [m, q],
    )

    slice_node = self._assert_single_dynamic_slice(result)
    sliced_input = slice_node.args[0]
    self.assertEqual(sliced_input.op, "call_function")
    self.assertIn(
        sliced_input.target,
        (torch.ops.aten.cat.default, torch.ops.aten.cat),
    )

  def test_pass_does_not_pad_when_bucket_capacity_exceeds_slice_clamp(
      self,
  ) -> None:
    buf = torch.ones(4, 200)
    x = torch.ones(4, 16)
    torch._dynamo.mark_static(buf)
    torch._dynamo.mark_dynamic(x, 1)

    result = _capture_and_run_slice_pass(
        lambda b_in, x_in: b_in[:, x_in.shape[1] : x_in.shape[1] + 1] * 2.0,
        [buf, x],
        bucket_idx=1,
    )

    slice_node = self._assert_single_dynamic_slice(result)
    sliced_input = slice_node.args[0]
    self.assertEqual(sliced_input.op, "placeholder")

  def test_pass_adds_no_guards_for_1d_tail_slice(self) -> None:
    # cache_position[-q_len:] with independent kv/q symbols: at head the pass
    # evaluated `out_size == in_size` through the ShapeEnv and installed a
    # guard on `kv_len - q_len`.
    pos = torch.arange(12)
    q = torch.ones(4, 4)
    torch._dynamo.mark_dynamic(pos, 0, min=8, max=16)
    torch._dynamo.mark_dynamic(q, 1, min=2, max=8)

    result = _capture_and_run_slice_pass(
        lambda p_in, q_in: p_in[-q_in.shape[-1] :] * 2,
        [pos, q],
    )

    self._assert_single_dynamic_slice(result)

  def test_pass_normalizes_multi_symbol_negative_start_from_hint(
      self,
  ) -> None:
    # start = a_len - b_len has a negative trace-time hint (4 - 12) although
    # its symbolic upper bound (8 - 2) is positive, so the sign decision must
    # follow the hint that PyTorch's slice meta guarded on, i.e. the lowered
    # start has to be x_len + a_len - b_len.
    x = torch.ones(2, 3, 12)
    a = torch.ones(1, 4)
    b = torch.ones(1, 12)
    torch._dynamo.mark_dynamic(x, 2, min=8, max=16)
    torch._dynamo.mark_dynamic(a, 1, min=2, max=8)
    torch._dynamo.mark_dynamic(b, 1, min=2, max=16)

    result = _capture_and_run_slice_pass(
        lambda x_in, a_in, b_in: x_in[
            Ellipsis, a_in.shape[-1] - b_in.shape[-1] :
        ]
        * 2.0,
        [x, a, b],
    )

    slice_node = self._assert_single_dynamic_slice(result)
    x_placeholder = next(
        node
        for node in result.gm.graph.find_nodes(op="placeholder")
        if isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].dim() == 3
    )
    x_len_size_placeholder = f"{x_placeholder.meta['val'].shape[-1]}_size"
    self.assertIn(
        x_len_size_placeholder,
        _transitive_placeholder_names(slice_node.args[1][-1]),
    )

  def test_pass_slice_backward_normalizes_negative_start(self) -> None:
    x = torch.ones(2, 8, 8)
    torch._dynamo.mark_static(x, 0)
    torch._dynamo.mark_dynamic(x, 1, min=2, max=16)
    torch._dynamo.mark_static(x, 2)
    g = torch.ones(2, 1, 4)
    torch._dynamo.mark_static(g)

    # Negative start on the static dim 2 normalizes against its static size
    # (input_sizes carries an FX sym_size node for the dynamic dim 1).
    result = _capture_and_run_slice_pass(
        lambda g_in, x_in: torch.ops.aten.slice_backward(
            g_in, x_in.shape, 2, -4, sys.maxsize, 1
        ),
        [g, x],
    )
    self.assertEmpty(result.new_guards)
    bwd_nodes = list(
        result.gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.slice_backward.default
        )
    )
    self.assertLen(bwd_nodes, 1)
    self.assertEqual(bwd_nodes[0].args[3], 4)

    # On the dynamic dim the normalized start is symbolic, which the
    # slice_backward lowering must reject instead of lowering it against the
    # upper bound.
    g_dyn = torch.ones(2, 4, 8)
    torch._dynamo.mark_static(g_dyn)
    with self.assertRaises(  # ASSERT_RAISES_OK=Compiler pass unit test.
        torch._dynamo.exc.BackendCompilerFailed
    ) as cm:
      _capture_and_run_slice_pass(
          lambda g_in, x_in: torch.ops.aten.slice_backward(
              g_in, x_in.shape, 1, -4, sys.maxsize, 1
          ),
          [g_dyn, x],
      )
    self.assertIsInstance(cm.exception.inner_exception, NotImplementedError)
    self.assertIn("dynamic start", str(cm.exception.inner_exception))

  def test_pass_slice_backward_bounds_window_by_runtime_range(self) -> None:
    # Gradient of a static [4, 300] input sliced by a dynamic end with range
    # [2, 200]: bucket 1's capacity (256) overshoots the range, so the window
    # must end at the runtime upper bound (200) and slice_backward must get a
    # static [4, 200] gradient (zeros past the runtime size) while the static
    # input sizes stay unchanged.
    g = torch.ones(4, 16)
    torch._dynamo.mark_static(g, 0)
    torch._dynamo.mark_dynamic(g, 1, min=2, max=200)

    result = _capture_and_run_slice_pass(
        lambda g_in: torch.ops.aten.slice_backward(
            g_in, [4, 300], 1, 0, g_in.shape[1], 1
        ),
        [g],
        bucket_idx=1,
    )

    self.assertEmpty(result.new_guards)
    bwd_nodes = list(
        result.gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.slice_backward.default
        )
    )
    self.assertLen(bwd_nodes, 1)
    grad, input_sizes, dim, start, end, step = bwd_nodes[0].args
    self.assertEqual(list(input_sizes), [4, 300])
    self.assertEqual((dim, start, end, step), (1, 0, 200, 1))
    self.assertEqual(grad.target, torch.ops.aten.slice.Tensor)
    self.assertEqual(tuple(grad.meta["val"].shape), (4, 200))
    self.assertEqual(grad.args[0].target, torch.ops.aten.where.self)

  def test_pass_rejects_negative_pads_on_dynamic_tensor(self) -> None:
    # The C++ constant_pad_nd builder sizes its blocks from the static shape,
    # so negative pads on a bounded operand must fail with a clear message
    # instead of an opaque compile failure.
    x = torch.ones(2, 3, 8)
    torch._dynamo.mark_dynamic(x, 2, min=2, max=16)
    with self.assertRaises(  # ASSERT_RAISES_OK=Compiler pass unit test.
        torch._dynamo.exc.BackendCompilerFailed
    ) as cm:
      _capture_and_run_slice_pass(lambda x_in: F.pad(x_in, (-1, 2)), [x])
    self.assertIsInstance(cm.exception.inner_exception, NotImplementedError)
    self.assertIn("negative or dynamic pads", str(cm.exception.inner_exception))


class ReplaceDynamicOutputBroadcastOpsPreGradPassTest(
    seed_test_utils.RepeatableTest
):
  """CPU-only unit tests for the clone of returned dynamic slice views."""

  def _run_pass(
      self, fn: Callable[..., Any], inputs: list[torch.Tensor]
  ) -> tuple[torch.fx.GraphModule, list[torch.fx.Node]]:
    gm = _capture_dynamo_graph(fn, inputs)
    view_ops_passes.ReplaceDynamicOutputBroadcastOpsPreGradPass()(gm)
    clones = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.clone.default
        )
    )
    return gm, clones

  def test_returned_negative_dynamic_start_view_is_cloned_before_output(
      self,
  ) -> None:
    gm, clones = self._run_pass(
        lambda x, y: x[Ellipsis, -y.shape[-1] :],
        [torch.ones(2, 8), torch.ones(2, 4)],
    )
    self.assertLen(clones, 1)
    output_node = next(iter(gm.graph.find_nodes(op="output")))
    self.assertIs(clones[0].next, output_node)
    self.assertIn(clones[0], output_node.all_input_nodes)

  def test_returned_dynamic_end_view_is_cloned(self) -> None:
    # AOTAutograd skips view replay for views with symbolic arguments and
    # falls back to as_strided with the compiled output's metadata, so even
    # an offset-0 view has to be returned as a copy.
    _, clones = self._run_pass(
        lambda x, y: x[:, : y.shape[-1]],
        [torch.ones(4, 8), torch.ones(4, 3)],
    )
    self.assertLen(clones, 1)

  def test_returned_narrow_with_dynamic_start_is_cloned(self) -> None:
    _, clones = self._run_pass(
        lambda x, y: x.narrow(1, y.shape[-1], 2),
        [torch.ones(4, 8), torch.ones(4, 3)],
    )
    self.assertLen(clones, 1)

  def test_returned_static_view_of_dynamic_tensor_is_not_cloned(
      self,
  ) -> None:
    _, clones = self._run_pass(lambda x: x[:, 2:], [torch.ones(4, 8)])
    self.assertEmpty(clones)

  def test_returned_advanced_indexing_is_not_cloned(self) -> None:
    _, clones = self._run_pass(
        lambda x, idx: x[idx], [torch.ones(4, 8), torch.tensor([0, 2])]
    )
    self.assertEmpty(clones)

  def test_dynamic_view_that_is_not_returned_is_not_cloned(self) -> None:
    _, clones = self._run_pass(
        lambda x, y: x[Ellipsis, -y.shape[-1] :] * 2.0,
        [torch.ones(2, 8), torch.ones(2, 4)],
    )
    self.assertEmpty(clones)

  def test_clone_follows_in_graph_mutation_of_the_base(self) -> None:
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      v = x[Ellipsis, -y.shape[-1] :]
      x.add_(1.0)
      return v

    gm, clones = self._run_pass(fn, [torch.ones(2, 8), torch.ones(2, 4)])
    self.assertLen(clones, 1)
    order = list(gm.graph.nodes)
    mutations = [
        n
        for n in order
        if n.op in ("call_method", "call_function") and "add_" in str(n.target)
    ]
    self.assertLen(mutations, 1)
    self.assertGreater(order.index(clones[0]), order.index(mutations[0]))

  def test_expand_copy_rewrite_keeps_in_place_mutation(self) -> None:
    # FX dead-code elimination treats `call_method` nodes as pure and would
    # drop the unused `add_`; the pass must erase only the replaced expand.
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      e = x.expand(y.shape[-1], 8)
      x.add_(1.0)
      return e

    gm, clones = self._run_pass(fn, [torch.ones(1, 8), torch.ones(2, 4)])
    self.assertEmpty(clones)
    targets = [str(n.target) for n in gm.graph.nodes]
    self.assertIn("add_", targets)
    self.assertNotIn("expand", targets)
    self.assertLen(
        list(
            gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.expand_copy.default
            )
        ),
        1,
    )


class BoundedDynamicSliceTest(seed_test_utils.RepeatableTest):
  """End-to-end TPU tests for slice-like ops on bounded dynamic dimensions."""

  def setUp(self) -> None:
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    torch._dynamo.reset()
    self.device = torch.accelerator.current_accelerator()

  def test_cat_negative_dynamic_start_rolling_buffer(self) -> None:
    fn_cpu = lambda s, x: torch.cat([s, x], dim=-1)[Ellipsis, -x.shape[-1] :]
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      s_cpu = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
      s_tpu = s_cpu.to(self.device)
      for n in (8, 12, 10):
        x_cpu = (
            torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n) + 100
        )
        out_tpu = fn_tpu(s_tpu, x_cpu.to(self.device))
        with self.subTest(n=n):
          utils.assert_close(out_tpu.cpu(), fn_cpu(s_cpu, x_cpu))

  def test_cat_negative_static_start_rolling_buffer(self) -> None:
    fn_cpu = lambda s, x: torch.cat([s, x], dim=-1)[Ellipsis, -6:]
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      s_cpu = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
      s_tpu = s_cpu.to(self.device)
      for n in (8, 12, 10):
        x_cpu = (
            torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n) + 100
        )
        out_tpu = fn_tpu(s_tpu, x_cpu.to(self.device))
        with self.subTest(n=n):
          utils.assert_close(out_tpu.cpu(), fn_cpu(s_cpu, x_cpu))

  def test_pad_nonzero_static_start(self) -> None:
    fn_cpu = lambda x: F.pad(x, (2, 2))[Ellipsis, 2:]
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      for n in (8, 12, 10):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        out_tpu = fn_tpu(x_cpu.to(self.device))
        with self.subTest(n=n):
          utils.assert_close(out_tpu.cpu(), fn_cpu(x_cpu))

  def test_static_start_with_dynamic_and_negative_end(self) -> None:
    fn_open = lambda x: x[Ellipsis, 3:]
    fn_neg_end = lambda x: x[Ellipsis, 1:-1]
    compiled_open = _compile_bd(fn_open)
    compiled_neg_end = _compile_bd(fn_neg_end)

    with torch.no_grad():
      for n in (8, 12, 10):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        x_tpu = x_cpu.to(self.device)
        with self.subTest(case="open_end", n=n):
          utils.assert_close(compiled_open(x_tpu).cpu(), fn_open(x_cpu))
        with self.subTest(case="neg_end", n=n):
          utils.assert_close(compiled_neg_end(x_tpu).cpu(), fn_neg_end(x_cpu))

  def test_dynamic_symint_start_with_static_length(self) -> None:
    fn_tail = lambda x: x[Ellipsis, x.shape[-1] - 3 :]
    fn_cat_skip = lambda s, x: torch.cat([x, s], dim=-1)[
        Ellipsis, x.shape[-1] :
    ]
    compiled_tail = _compile_bd(fn_tail)
    compiled_cat_skip = _compile_bd(fn_cat_skip)

    with torch.no_grad():
      s_cpu = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6) - 50
      s_tpu = s_cpu.to(self.device)
      for n in (8, 12, 10):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        x_tpu = x_cpu.to(self.device)
        with self.subTest(case="tail_3", n=n):
          utils.assert_close(compiled_tail(x_tpu).cpu(), fn_tail(x_cpu))
        with self.subTest(case="cat_skip_x", n=n):
          utils.assert_close(
              compiled_cat_skip(s_tpu, x_tpu).cpu(), fn_cat_skip(s_cpu, x_cpu)
          )

  def test_slice_clamped_beyond_logical_size(self) -> None:
    fn_over_end = lambda x: x[Ellipsis, 2:1000]
    fn_over_neg_start = lambda x: x[Ellipsis, -1000:]
    compiled_over_end = _compile_bd(fn_over_end)
    compiled_over_neg_start = _compile_bd(fn_over_neg_start)

    with torch.no_grad():
      for n in (8, 12, 10):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        x_tpu = x_cpu.to(self.device)
        with self.subTest(case="over_end", n=n):
          utils.assert_close(compiled_over_end(x_tpu).cpu(), fn_over_end(x_cpu))
        with self.subTest(case="over_neg_start", n=n):
          utils.assert_close(
              compiled_over_neg_start(x_tpu).cpu(), fn_over_neg_start(x_cpu)
          )

  def test_middle_dim_3d_and_4d_slicing(self) -> None:
    fn_3d_pos = lambda x: x[:, 1:, :]
    fn_3d_neg = lambda x: x[:, -4:, :]
    fn_4d_mid = lambda x: x[:, :, 2:-1, :]
    compiled_3d_pos = _compile_bd(fn_3d_pos)
    compiled_3d_neg = _compile_bd(fn_3d_neg)
    compiled_4d_mid = _compile_bd(fn_4d_mid)

    with torch.no_grad():
      for n in (8, 12, 10):
        x3_cpu = torch.arange(2 * n * 6, dtype=torch.float32).reshape(2, n, 6)
        x3_tpu = x3_cpu.to(self.device)
        x4_cpu = torch.arange(2 * 3 * n * 4, dtype=torch.float32).reshape(
            2, 3, n, 4
        )
        x4_tpu = x4_cpu.to(self.device)
        with self.subTest(case="3d_pos", n=n):
          utils.assert_close(compiled_3d_pos(x3_tpu).cpu(), fn_3d_pos(x3_cpu))
        with self.subTest(case="3d_neg", n=n):
          utils.assert_close(compiled_3d_neg(x3_tpu).cpu(), fn_3d_neg(x3_cpu))
        with self.subTest(case="4d_mid", n=n):
          utils.assert_close(compiled_4d_mid(x4_tpu).cpu(), fn_4d_mid(x4_cpu))

  def test_multiple_dynamic_dims_slicing(self) -> None:
    fn_cpu = lambda x: x[1:, 2:]
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      for n, m in ((8, 11), (12, 15), (10, 13)):
        x_cpu = torch.arange(n * m, dtype=torch.float32).reshape(n, m)
        out_tpu = fn_tpu(x_cpu.to(self.device))
        with self.subTest(n=n, m=m):
          utils.assert_close(out_tpu.cpu(), fn_cpu(x_cpu))

  def test_sliding_window_two_independent_symbols(self) -> None:
    fn_cpu = lambda m, q: m[:, -q.shape[-1] :] + q
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      for kv_len, q_len in ((16, 3), (20, 5), (18, 7), (16, 16)):
        m_cpu = torch.arange(4 * kv_len, dtype=torch.float32).reshape(4, kv_len)
        q_cpu = torch.arange(4 * q_len, dtype=torch.float32).reshape(4, q_len)
        out_tpu = fn_tpu(m_cpu.to(self.device), q_cpu.to(self.device))
        with self.subTest(kv_len=kv_len, q_len=q_len):
          utils.assert_close(out_tpu.cpu(), fn_cpu(m_cpu, q_cpu))

  def test_negative_dynamic_end_and_empty_slice(self) -> None:
    fn_neg_end = lambda x, k: x[Ellipsis, : -k.shape[-1]] * 2.0
    fn_empty = lambda x: x[Ellipsis, 5:3] * 2.0
    compiled_neg_end = _compile_bd(fn_neg_end)
    compiled_empty = _compile_bd(fn_empty)

    with torch.no_grad():
      for n, m in ((8, 2), (12, 3), (10, 4)):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        k_cpu = torch.ones(1, m)
        x_tpu = x_cpu.to(self.device)
        with self.subTest(case="neg_dynamic_end", n=n, m=m):
          utils.assert_close(
              compiled_neg_end(x_tpu, k_cpu.to(self.device)).cpu(),
              fn_neg_end(x_cpu, k_cpu),
          )
        with self.subTest(case="empty", n=n):
          out_tpu = compiled_empty(x_tpu).cpu()
          self.assertEqual(out_tpu.shape, fn_empty(x_cpu).shape)

  def test_constant_pad_two_dims_with_value(self) -> None:
    fn_cpu = lambda x: F.pad(x, (1, 2, 0, 3), value=-1.0)
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      for n in (8, 12, 10):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        out_tpu = fn_tpu(x_cpu.to(self.device))
        with self.subTest(n=n):
          utils.assert_close(out_tpu.cpu(), fn_cpu(x_cpu))

  def test_returned_dynamic_end_view_of_static_input(self) -> None:
    # x[2:n] is returned directly: without the pre-grad clone AOTAutograd
    # rebuilds the alias with as_strided from the compiled output's metadata
    # (storage offset 0), which returns rows 0..n-3 instead of 2..n-1.
    # The slice is on the outermost dim: for x[:, :n] of a static (4, 16)
    # base the view's contiguity depends on n == 16, so tracing adds the
    # guard Ne(n, 16) that conflicts with the bound 16 the slice implies
    # (pre-existing limitation of the fallback bound validation).
    fn_cpu = lambda x, k: x[2 : k.shape[-1]]
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      x_cpu = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
      x_tpu = x_cpu.to(self.device)
      for m in (5, 9, 7):
        k_cpu = torch.ones(1, m)
        out_tpu = fn_tpu(x_tpu, k_cpu.to(self.device))
        with self.subTest(m=m):
          utils.assert_close(out_tpu.cpu(), fn_cpu(x_cpu, k_cpu))

  def test_returned_dynamic_view_reflects_base_mutation(self) -> None:
    def fn(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
      v = x[-k.shape[-1] :]
      x.add_(1.0)
      return v

    fn_tpu = _compile_bd(fn)

    with torch.no_grad():
      for m in (3, 5, 4):
        x_cpu = torch.arange(16 * 2, dtype=torch.float32).reshape(16, 2)
        k_cpu = torch.ones(1, m)
        x_tpu = x_cpu.to(self.device)
        out_tpu = fn_tpu(x_tpu, k_cpu.to(self.device))
        with self.subTest(m=m):
          utils.assert_close(out_tpu.cpu(), fn(x_cpu, k_cpu))
          utils.assert_close(x_tpu.cpu(), x_cpu)

  def test_multi_symbol_negative_start_follows_runtime_sign(self) -> None:
    # start = a_len - b_len is negative at runtime while its symbolic upper
    # bound is positive, so the lowering must normalize it against x_len.
    fn_cpu = lambda x, a, b: x[Ellipsis, a.shape[-1] - b.shape[-1] :] * 2.0
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      for x_len, a_len, b_len in ((12, 4, 12), (16, 6, 14), (10, 2, 8)):
        x_cpu = torch.arange(2 * 3 * x_len, dtype=torch.float32).reshape(
            2, 3, x_len
        )
        a_cpu = torch.ones(1, a_len)
        b_cpu = torch.ones(1, b_len)
        out_tpu = fn_tpu(
            x_cpu.to(self.device), a_cpu.to(self.device), b_cpu.to(self.device)
        )
        with self.subTest(x_len=x_len, a_len=a_len, b_len=b_len):
          utils.assert_close(out_tpu.cpu(), fn_cpu(x_cpu, a_cpu, b_cpu))

  def test_select_negative_dynamic_index(self) -> None:
    fn_cpu = lambda s, x: torch.cat([s, x], dim=-1).select(-1, -x.shape[-1])
    fn_tpu = _compile_bd(fn_cpu)

    with torch.no_grad():
      s_cpu = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
      s_tpu = s_cpu.to(self.device)
      for n in (8, 12, 10):
        x_cpu = (
            torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n) + 100
        )
        out_tpu = fn_tpu(s_tpu, x_cpu.to(self.device))
        with self.subTest(n=n):
          utils.assert_close(out_tpu.cpu(), fn_cpu(s_cpu, x_cpu))

  def test_slice_backward_static_start_on_bounded_dim(self) -> None:
    fn_open = _compile_bd(lambda x: (x[Ellipsis, 2:] * 2.0).sum())
    fn_interior = _compile_bd(lambda x: (x[Ellipsis, 1:-1] * 3.0).sum())

    for n in (8, 12, 10):
      x1_cpu = torch.arange(
          2 * 3 * n, dtype=torch.float32, requires_grad=True
      ).reshape(2, 3, n)
      x1_cpu.retain_grad()
      (x1_cpu[Ellipsis, 2:] * 2.0).sum().backward()

      x1_tpu = x1_cpu.detach().to(self.device).requires_grad_(True)
      fn_open(x1_tpu).backward()
      with self.subTest(case="open_end_bwd", n=n):
        self.assertIsNotNone(x1_tpu.grad)
        self.assertIsNotNone(x1_cpu.grad)
        utils.assert_close(x1_tpu.grad.cpu(), x1_cpu.grad)

      x2_cpu = torch.arange(
          2 * 3 * n, dtype=torch.float32, requires_grad=True
      ).reshape(2, 3, n)
      x2_cpu.retain_grad()
      (x2_cpu[Ellipsis, 1:-1] * 3.0).sum().backward()

      x2_tpu = x2_cpu.detach().to(self.device).requires_grad_(True)
      fn_interior(x2_tpu).backward()
      with self.subTest(case="interior_bwd", n=n):
        self.assertIsNotNone(x2_tpu.grad)
        self.assertIsNotNone(x2_cpu.grad)
        utils.assert_close(x2_tpu.grad.cpu(), x2_cpu.grad)


if __name__ == "__main__":
  absltest.main()
