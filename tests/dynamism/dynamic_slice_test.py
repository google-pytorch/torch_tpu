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

from collections.abc import Callable, Sequence, Set
import sys
from typing import Any, NamedTuple

from absl.testing import absltest
import torch
from torch._dynamo import testing as dynamo_testing
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
    inputs: Sequence[torch.Tensor],
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


def _dynamic_slice_nodes(gm: torch.fx.GraphModule) -> Sequence[torch.fx.Node]:
  return list(
      gm.graph.find_nodes(
          op="call_function", target=torch.ops.tpu._dynamic_slice
      )
  )


def _transitive_placeholder_names(node: torch.fx.Node) -> Set[str]:
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
            g_in, [4, 300], dim=1, start=0, end=g_in.shape[1], step=1
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

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_pass_adds_no_guards_for_strided_slice(self) -> None:
    for hint in (10, 11):
      x = torch.ones(2, hint)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=16)
      for fn in (
          lambda x_in: x_in[:, ::2],
          lambda x_in: x_in[:, 1::2],
      ):
        result = _capture_and_run_slice_pass(fn, [x])
        slice_node = self._assert_single_dynamic_slice(result)
        self.assertEqual(slice_node.args[0].target, torch.ops.aten.view.default)
        squeeze_nodes = list(
            result.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.squeeze.dim
            )
        )
        self.assertLen(squeeze_nodes, 1)

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_pass_adds_no_guards_for_dynamic_start_strided_slice(self) -> None:
    x = torch.ones(2, 12)
    k = torch.ones(1, 3)
    torch._dynamo.mark_dynamic(x, 1, min=8, max=16)
    torch._dynamo.mark_dynamic(k, 1, min=1, max=6)
    result = _capture_and_run_slice_pass(
        lambda x_in, k_in: x_in[:, k_in.shape[-1] :: 2], [x, k]
    )
    slice_node = self._assert_single_dynamic_slice(result)
    self.assertEqual(slice_node.args[0].target, torch.ops.aten.view.default)

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_pass_empty_strided_slice_and_rejects_dynamic_step(self) -> None:
    x = torch.ones(2, 10)
    torch._dynamo.mark_dynamic(x, 1, min=6, max=16)
    empty_res = _capture_and_run_slice_pass(lambda x_in: x_in[:, 5:3:2], [x])
    slice_node = self._assert_single_dynamic_slice(empty_res)
    self.assertEqual(slice_node.args[2][-1], 0)

    k = torch.ones(1, 2)
    torch._dynamo.mark_dynamic(k, 1, min=1, max=4)
    with self.assertRaises(  # ASSERT_RAISES_OK=Compiler pass unit test.
        torch._dynamo.exc.BackendCompilerFailed
    ) as cm:
      _capture_and_run_slice_pass(
          lambda x_in, k_in: torch.ops.aten.slice.Tensor(
              x_in, 1, 0, x_in.shape[1], k_in.shape[-1]
          ),
          [x, k],
      )
    self.assertIsInstance(cm.exception.inner_exception, NotImplementedError)
    self.assertIn("step", str(cm.exception.inner_exception).lower())

  @absltest.skip("TODO(b/582377587): Enable with slice_scatter lowering.")
  def test_pass_lowers_slice_scatter_unit_and_strided(self) -> None:
    x = torch.ones(2, 12)
    k = torch.ones(1, 3)
    torch._dynamo.mark_dynamic(x, 1, min=6, max=16)
    torch._dynamo.mark_dynamic(k, 1, min=1, max=6)

    # Unit start=0 scatter lowers to where without index_select.
    s_unit = torch.ones(2, 3)
    torch._dynamo.mark_dynamic(s_unit, 1, min=1, max=6)
    res_unit = _capture_and_run_slice_pass(
        lambda x_in, src_in: torch.ops.aten.slice_scatter.default(
            x_in, src_in, 1, 0, src_in.shape[1], 1
        ),
        [x, s_unit],
    )
    self.assertEmpty(res_unit.new_guards)
    self.assertEmpty(
        list(
            res_unit.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_select.default
            )
        )
    )
    self.assertLen(
        list(
            res_unit.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.where.self
            )
        ),
        1,
    )

    # Strided 0 <= start < step lowers via phase folding without guards/gather.
    s_phase = torch.ones(2, 5)
    torch._dynamo.mark_dynamic(s_phase, 1, min=2, max=8)
    res_phase = _capture_and_run_slice_pass(
        lambda x_in, src_in: torch.ops.aten.slice_scatter.default(
            x_in, src_in, 1, 1, x_in.shape[1], 2
        ),
        [x, s_phase],
    )
    self.assertEmpty(res_phase.new_guards)
    self.assertEmpty(
        list(
            res_phase.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_select.default
            )
        )
    )
    self.assertLen(
        list(
            res_phase.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.where.self
            )
        ),
        1,
    )

    # Dynamic start + step=2 lowers cleanly without guards.
    s_dyn = torch.ones(2, 4)
    torch._dynamo.mark_dynamic(s_dyn, 1, min=1, max=8)
    res_dyn = _capture_and_run_slice_pass(
        lambda x_in, src_in, k_in: torch.ops.aten.slice_scatter.default(
            x_in,
            src_in,
            1,
            k_in.shape[1],
            k_in.shape[1] + src_in.shape[1] * 2,
            2,
        ),
        [x, s_dyn, k],
    )
    self.assertEmpty(res_dyn.new_guards)
    self.assertLen(
        list(
            res_dyn.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_select.default
            )
        ),
        1,
    )

  @absltest.skip("TODO(b/582377587): Enable with strided slice_backward.")
  def test_pass_slice_backward_strided_step_uses_scatter_and_rejects_dynamic(
      self,
  ) -> None:
    x = torch.ones(2, 12)
    g = torch.ones(2, 6)
    torch._dynamo.mark_dynamic(x, 1, min=6, max=16)
    torch._dynamo.mark_dynamic(g, 1, min=3, max=8)

    result = _capture_and_run_slice_pass(
        lambda g_in, x_in: torch.ops.aten.slice_backward.default(
            g_in, x_in.shape, 1, 1, x_in.shape[1], 2
        ),
        [g, x],
    )
    self.assertEmpty(result.new_guards)
    self.assertEmpty(
        list(
            result.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.slice_backward.default
            )
        )
    )
    self.assertEmpty(
        list(
            result.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_select.default
            )
        )
    )
    self.assertLen(
        list(
            result.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.where.self
            )
        ),
        1,
    )

    k = torch.ones(1, 2)
    torch._dynamo.mark_dynamic(k, 1, min=1, max=4)
    with self.assertRaises(  # ASSERT_RAISES_OK=Compiler pass unit test.
        torch._dynamo.exc.BackendCompilerFailed
    ) as cm:
      _capture_and_run_slice_pass(
          lambda g_in, x_in, k_in: torch.ops.aten.slice_backward.default(
              g_in, x_in.shape, 1, 0, x_in.shape[1], k_in.shape[-1]
          ),
          [g, x, k],
      )
    self.assertIsInstance(cm.exception.inner_exception, NotImplementedError)
    self.assertIn("step", str(cm.exception.inner_exception).lower())

  @absltest.skip("TODO(b/582377587): Enable with select_scatter lowering.")
  def test_pass_lowers_select_scatter_and_select_backward(self) -> None:
    x = torch.ones(2, 12, 4)
    src = torch.ones(2, 4)
    k = torch.ones(3)
    torch._dynamo.mark_dynamic(x, 1, min=6, max=16)
    torch._dynamo.mark_dynamic(k, 0, min=1, max=5)

    res_scat = _capture_and_run_slice_pass(
        lambda x_in, src_in: torch.ops.aten.select_scatter.default(
            x_in, src_in, 1, -1
        ),
        [x, src],
    )
    self.assertEmpty(res_scat.new_guards)
    self.assertEmpty(
        list(
            res_scat.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.select_scatter.default
            )
        )
    )
    self.assertEmpty(
        list(
            res_scat.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.index_select.default
            )
        )
    )
    self.assertLen(
        list(
            res_scat.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.where.self
            )
        ),
        1,
    )

    for fn in (
        lambda g_in, x_in, _k: torch.ops.aten.select_backward.default(
            g_in, x_in.shape, 1, 0
        ),
        lambda g_in, x_in, _k: torch.ops.aten.select_backward.default(
            g_in, x_in.shape, 1, -1
        ),
        lambda g_in, x_in, k_in: torch.ops.aten.select_backward.default(
            g_in, x_in.shape, 1, k_in.shape[0]
        ),
    ):
      res_bwd = _capture_and_run_slice_pass(fn, [src, x, k])
      self.assertEmpty(res_bwd.new_guards)
      self.assertEmpty(
          list(
              res_bwd.gm.graph.find_nodes(
                  op="call_function",
                  target=torch.ops.aten.select_backward.default,
              )
          )
      )
      self.assertLen(
          list(
              res_bwd.gm.graph.find_nodes(
                  op="call_function", target=torch.ops.aten.where.self
              )
          ),
          1,
      )


class ReplaceDynamicOutputBroadcastOpsPreGradPassTest(
    seed_test_utils.RepeatableTest
):
  """CPU-only unit tests for the clone of returned dynamic slice views."""

  def _run_pass(
      self, fn: Callable[..., Any], inputs: Sequence[torch.Tensor]
  ) -> tuple[torch.fx.GraphModule, Sequence[torch.fx.Node]]:
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

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_strided_slice_even_and_odd_no_recompile(self) -> None:
    def gptj_rotary(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      return x[Ellipsis, ::2], x[Ellipsis, 1::2]

    cases = [
        ("step2_returned", lambda x: x[Ellipsis, ::2]),
        ("step2_mul", lambda x: x[Ellipsis, ::2] * 2.0),
        ("offset1_step2_mul", lambda x: x[Ellipsis, 1::2] * 2.0),
        ("interior_step2_mul", lambda x: x[Ellipsis, 1:-1:2] * 2.0),
        ("gptj_rotary_pair", gptj_rotary),
        (
            "gptj_rotary_interleave",
            lambda x: torch.cat([-x[Ellipsis, 1::2], x[Ellipsis, ::2]], dim=-1),
        ),
    ]
    for name, fn in cases:
      with self.subTest(case=name):
        torch._dynamo.reset()
        counter = dynamo_testing.CompileCounterWithBackend("tpu")
        fn_tpu = torch.compile(fn, backend=counter, dynamic=None)
        with torch.no_grad():
          for n in (10, 11, 12, 13, 15, 16):
            x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(
                2, 3, n
            )
            x_tpu = x_cpu.to(self.device)
            torch._dynamo.mark_dynamic(x_tpu, 2, min=6, max=32)
            out_tpu = fn_tpu(x_tpu)
            out_cpu = fn(x_cpu)
            if isinstance(out_cpu, tuple):
              self.assertIsInstance(out_tpu, tuple)
              self.assertLen(out_tpu, len(out_cpu))
              for t_tpu, t_cpu in zip(out_tpu, out_cpu, strict=True):
                utils.assert_close(t_tpu.cpu(), t_cpu)
            else:
              utils.assert_close(out_tpu.cpu(), out_cpu)
        self.assertEqual(counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_strided_slice_dynamic_start_no_recompile(self) -> None:
    fn = lambda x, k: x[:, k.shape[-1] :: 2] * 2.0
    torch._dynamo.reset()
    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    fn_tpu = torch.compile(fn, backend=counter, dynamic=None)

    with torch.no_grad():
      for n, start_k in ((12, 2), (13, 3), (15, 4), (16, 5), (11, 2)):
        x_cpu = torch.arange(4 * n, dtype=torch.float32).reshape(4, n)
        k_cpu = torch.ones(1, start_k)
        x_tpu = x_cpu.to(self.device)
        k_tpu = k_cpu.to(self.device)
        torch._dynamo.mark_dynamic(x_tpu, 1, min=6, max=24)
        torch._dynamo.mark_dynamic(k_tpu, 1, min=2, max=6)
        out_tpu = fn_tpu(x_tpu, k_tpu)
        utils.assert_close(out_tpu.cpu(), fn(x_cpu, k_cpu))
    self.assertEqual(counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with strided dynamic view support.")
  def test_dynamic_strided_view_as_graph_input_no_recompile(self) -> None:
    fn = lambda v: v + 1.0
    cases = [
        ("slice_mid_step2", (2, 10, 4), 1, lambda b: b[:, ::2, :]),
        (
            "transpose_slice_step2",
            (2, 8, 10, 4),
            2,
            lambda b: b.transpose(1, 2)[:, :, ::2, :],
        ),
        (
            "transpose_slice_dyn_step2",
            (2, 4, 10),
            2,
            lambda b: b.transpose(1, 2)[:, ::2, :],
        ),
    ]
    for name, base_shape, dyn_dim, view_fn in cases:
      with self.subTest(case=name):
        torch._dynamo.reset()
        counter = dynamo_testing.CompileCounterWithBackend("tpu")
        fn_tpu = torch.compile(fn, backend=counter, dynamic=None)
        compilations: list[int] = []
        with torch.no_grad():
          for n in (10, 11, 12, 13, 15, 16):
            shape = list(base_shape)
            shape[dyn_dim] = n
            base_cpu = torch.arange(
                torch.Size(shape).numel(), dtype=torch.float32
            ).reshape(shape)
            view_cpu = view_fn(base_cpu)
            view_tpu = view_fn(base_cpu.to(self.device))
            before = counter.frame_count
            out_tpu = fn_tpu(view_tpu)
            compilations.append(counter.frame_count - before)
            utils.assert_close(out_tpu.cpu(), fn(view_cpu))
        # Call 0 compiles static, call 1 compiles dynamic, calls 2..5 must not
        # recompile across even or odd lengths.
        self.assertEqual(compilations, [1, 1, 0, 0, 0, 0])

  @absltest.skip("TODO(b/582377587): Enable with slice_scatter lowering.")
  def test_inplace_strided_slice_mutation_even_and_odd_no_recompile(
      self,
  ) -> None:
    def mut_even(x: torch.Tensor) -> torch.Tensor:
      x[Ellipsis, ::2].add_(1.0)
      return x

    def mut_odd(x: torch.Tensor) -> torch.Tensor:
      x[Ellipsis, 1::2].mul_(2.0)
      return x

    def mut_interior(x: torch.Tensor) -> torch.Tensor:
      x[Ellipsis, 1:-1:2].add_(3.0)
      return x

    for name, fn in (
        ("inplace_even", mut_even),
        ("inplace_odd", mut_odd),
        ("inplace_interior", mut_interior),
    ):
      with self.subTest(case=name):
        torch._dynamo.reset()
        counter = dynamo_testing.CompileCounterWithBackend("tpu")
        fn_tpu = torch.compile(fn, backend=counter, dynamic=None)
        with torch.no_grad():
          for n in (10, 11, 12, 13, 15, 16):
            x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(
                2, 3, n
            )
            x_tpu = x_cpu.clone().to(self.device)
            torch._dynamo.mark_dynamic(x_tpu, 2, min=6, max=32)
            out_tpu = fn_tpu(x_tpu)
            out_cpu = fn(x_cpu.clone())
            utils.assert_close(out_tpu.cpu(), out_cpu)
            utils.assert_close(x_tpu.cpu(), out_cpu)
        self.assertEqual(counter.frame_count, 1)

    # Dynamic unit-stride prefix mutation and dynamic-start strided scatter.
    def mut_dyn_prefix(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
      x[Ellipsis, : k.shape[-1]].add_(5.0)
      return x

    def scatter_dyn_start(
        x: torch.Tensor, src: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
      start = k.shape[-1] - 1
      end = start + src.shape[-1] * 2
      return torch.slice_scatter(x, src, dim=2, start=start, end=end, step=2)

    torch._dynamo.reset()
    prefix_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled_prefix = torch.compile(
        mut_dyn_prefix, backend=prefix_counter, dynamic=None
    )
    torch._dynamo.reset()
    scatter_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled_scatter = torch.compile(
        scatter_dyn_start, backend=scatter_counter, dynamic=None
    )
    with torch.no_grad():
      for n, k_len, src_len in (
          (12, 3, 3),
          (13, 4, 4),
          (15, 5, 5),
          (16, 3, 6),
          (11, 2, 2),
      ):
        x_cpu = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
        # Keep k_len >= 2 so Dynamo doesn't specialize k_len == 1, while
        # start = k_len - 1 exercises both odd (1, 3) and even (2, 4) starts.
        k_cpu = torch.ones(1, k_len)
        src_cpu = (
            torch.arange(2 * 3 * src_len, dtype=torch.float32).reshape(
                2, 3, src_len
            )
            + 1000.0
        )

        x_prefix_tpu = x_cpu.clone().to(self.device)
        k_tpu = k_cpu.to(self.device)
        torch._dynamo.mark_dynamic(x_prefix_tpu, 2, min=6, max=24)
        torch._dynamo.mark_dynamic(k_tpu, 1, min=2, max=8)
        out_prefix_tpu = compiled_prefix(x_prefix_tpu, k_tpu)
        out_prefix_cpu = mut_dyn_prefix(x_cpu.clone(), k_cpu)
        utils.assert_close(out_prefix_tpu.cpu(), out_prefix_cpu)
        utils.assert_close(x_prefix_tpu.cpu(), out_prefix_cpu)

        x_scat_tpu = x_cpu.to(self.device)
        src_tpu = src_cpu.to(self.device)
        torch._dynamo.mark_dynamic(x_scat_tpu, 2, min=6, max=24)
        torch._dynamo.mark_dynamic(src_tpu, 2, min=2, max=10)
        torch._dynamo.mark_dynamic(k_tpu, 1, min=2, max=8)
        out_scat_tpu = compiled_scatter(x_scat_tpu, src_tpu, k_tpu)
        out_scat_cpu = scatter_dyn_start(x_cpu, src_cpu, k_cpu)
        utils.assert_close(out_scat_tpu.cpu(), out_scat_cpu)
    self.assertEqual(prefix_counter.frame_count, 1)
    self.assertEqual(scatter_counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with slice_scatter lowering.")
  def test_inplace_strided_slice_mutation_with_dynamic_outer_dim(
      self,
  ) -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
      x[:, ::2].add_(1.0)
      return x

    torch._dynamo.reset()
    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    fn_tpu = torch.compile(fn, backend=counter, dynamic=None)
    with torch.no_grad():
      for b, n in ((2, 10), (3, 11), (4, 12), (5, 13), (3, 15)):
        x_cpu = torch.arange(b * n, dtype=torch.float32).reshape(b, n)
        x_tpu = x_cpu.clone().to(self.device)
        torch._dynamo.mark_dynamic(x_tpu, 0, min=2, max=8)
        torch._dynamo.mark_dynamic(x_tpu, 1, min=6, max=24)
        out_tpu = fn_tpu(x_tpu)
        out_cpu = fn(x_cpu.clone())
        utils.assert_close(out_tpu.cpu(), out_cpu)
        utils.assert_close(x_tpu.cpu(), out_cpu)
    self.assertEqual(counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with strided slice_backward.")
  def test_strided_slice_backward_even_and_odd_no_recompile(self) -> None:
    def pair_loss(x: torch.Tensor) -> torch.Tensor:
      return (x[Ellipsis, ::2] * 2.0).sum() + (x[Ellipsis, 1::2] * 3.0).sum()

    torch._dynamo.reset()
    pair_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    pair_tpu = torch.compile(pair_loss, backend=pair_counter, dynamic=None)
    for n in (10, 11, 12, 13, 15, 16):
      x_base = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)
      x_cpu = x_base.clone().requires_grad_(True)
      pair_loss(x_cpu).backward()
      x_tpu = x_base.clone().to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(x_tpu, 2, min=6, max=32)
      pair_tpu(x_tpu).backward()
      utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)
    self.assertEqual(pair_counter.frame_count, 1)

    def interior_strided_loss(x: torch.Tensor) -> torch.Tensor:
      return (x[:, 1:-1:2] * 1.5).sum()

    torch._dynamo.reset()
    interior_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    interior_tpu = torch.compile(
        interior_strided_loss, backend=interior_counter, dynamic=None
    )
    for n in (10, 11, 12, 13, 15, 16):
      x_base = torch.arange(2 * n, dtype=torch.float32).reshape(2, n)
      x_cpu = x_base.clone().requires_grad_(True)
      interior_strided_loss(x_cpu).backward()
      x_tpu = x_base.clone().to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(x_tpu, 1, min=6, max=32)
      interior_tpu(x_tpu).backward()
      utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)
    self.assertEqual(interior_counter.frame_count, 1)

    def scatter_loss(x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
      out = torch.slice_scatter(x, src, dim=1, start=1, end=x.shape[1], step=2)
      return (out * 2.5).sum()

    torch._dynamo.reset()
    scatter_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    scatter_tpu = torch.compile(
        scatter_loss, backend=scatter_counter, dynamic=None
    )
    for n in (10, 11, 12, 13, 15, 16):
      src_len = n // 2
      x_base = torch.arange(2 * n, dtype=torch.float32).reshape(2, n)
      src_base = (
          torch.arange(2 * src_len, dtype=torch.float32).reshape(2, src_len)
          + 100.0
      )
      x_cpu = x_base.clone().requires_grad_(True)
      src_cpu = src_base.clone().requires_grad_(True)
      scatter_loss(x_cpu, src_cpu).backward()

      x_tpu = x_base.clone().to(self.device).requires_grad_(True)
      src_tpu = src_base.clone().to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(x_tpu, 1, min=6, max=32)
      torch._dynamo.mark_dynamic(src_tpu, 1, min=3, max=16)
      scatter_tpu(x_tpu, src_tpu).backward()
      utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)
      utils.assert_close(src_tpu.grad.cpu(), src_cpu.grad)
    self.assertEqual(scatter_counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with select_scatter lowering.")
  def test_select_scatter_and_select_backward_no_recompile(self) -> None:
    def select_mut_loss(x: torch.Tensor) -> torch.Tensor:
      y = x.clone()
      y.select(1, -1).mul_(3.0)
      return y.sum()

    torch._dynamo.reset()
    mut_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    mut_tpu = torch.compile(select_mut_loss, backend=mut_counter, dynamic=None)
    for n in (6, 9, 10, 13):
      x_base = torch.arange(2 * n * 4, dtype=torch.float32).reshape(2, n, 4)
      x_cpu = x_base.clone().requires_grad_(True)
      select_mut_loss(x_cpu).backward()
      x_tpu = x_base.clone().to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(x_tpu, 1, min=4, max=24)
      mut_tpu(x_tpu).backward()
      utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)
    self.assertEqual(mut_counter.frame_count, 1)

    def select_dyn_idx_bwd(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
      return (
          x[:, 0, :].sum()
          + x[:, -1, :].sum() * 2.0
          + x[:, k.shape[0], :].sum() * 4.0
      )

    torch._dynamo.reset()
    bwd_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    bwd_tpu = torch.compile(
        select_dyn_idx_bwd, backend=bwd_counter, dynamic=None
    )
    for n, idx in ((6, 2), (9, 4), (10, 3), (13, 5)):
      x_base = torch.arange(2 * n * 4, dtype=torch.float32).reshape(2, n, 4)
      k_cpu = torch.ones(idx)
      x_cpu = x_base.clone().requires_grad_(True)
      select_dyn_idx_bwd(x_cpu, k_cpu).backward()

      x_tpu = x_base.clone().to(self.device).requires_grad_(True)
      k_tpu = k_cpu.to(self.device)
      torch._dynamo.mark_dynamic(x_tpu, 1, min=4, max=24)
      torch._dynamo.mark_dynamic(k_tpu, 0, min=1, max=8)
      bwd_tpu(x_tpu, k_tpu).backward()
      utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)
    self.assertEqual(bwd_counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_non_power_of_two_step3_unaligned_bucket_no_recompile(self) -> None:
    """Verifies step=3 across n % 3 in {0, 1, 2} on unaligned bucket max=16."""
    # max=16 is not divisible by 3 (16 -> aligned 18), exercising bucket tail
    # padding on forward slice and post-scatter tail trimming on write-back.
    fwd_fn = lambda x: (
        x[Ellipsis, ::3] * 2.0,
        x[Ellipsis, 1::3] + 1.0,
        x[Ellipsis, 2::3] - 3.0,
    )
    torch._dynamo.reset()
    fwd_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    fwd_tpu = torch.compile(fwd_fn, backend=fwd_counter, dynamic=None)

    def mut_step3(x: torch.Tensor) -> torch.Tensor:
      x[Ellipsis, ::3].add_(10.0)
      x[Ellipsis, 2::3].mul_(2.0)
      return x

    torch._dynamo.reset()
    mut_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    mut_tpu = torch.compile(mut_step3, backend=mut_counter, dynamic=None)

    def bwd_step3(x: torch.Tensor) -> torch.Tensor:
      return (
          (x[Ellipsis, ::3] * 2.0).sum()
          + (x[Ellipsis, 1::3] * 3.0).sum()
          + (x[Ellipsis, 2::3] * 4.0).sum()
      )

    torch._dynamo.reset()
    bwd_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    bwd_tpu = torch.compile(bwd_step3, backend=bwd_counter, dynamic=None)

    for n in (9, 10, 11, 12, 13, 14):
      x_base = torch.arange(2 * 3 * n, dtype=torch.float32).reshape(2, 3, n)

      with torch.no_grad():
        x_fwd_tpu = x_base.to(self.device)
        torch._dynamo.mark_dynamic(x_fwd_tpu, 2, min=6, max=16)
        out_fwd_tpu = fwd_tpu(x_fwd_tpu)
        out_fwd_cpu = fwd_fn(x_base)
        for t_tpu, t_cpu in zip(out_fwd_tpu, out_fwd_cpu, strict=True):
          utils.assert_close(t_tpu.cpu(), t_cpu)

        x_mut_tpu = x_base.clone().to(self.device)
        torch._dynamo.mark_dynamic(x_mut_tpu, 2, min=6, max=16)
        out_mut_tpu = mut_tpu(x_mut_tpu)
        out_mut_cpu = mut_step3(x_base.clone())
        utils.assert_close(out_mut_tpu.cpu(), out_mut_cpu)
        utils.assert_close(x_mut_tpu.cpu(), out_mut_cpu)

      x_bwd_cpu = x_base.clone().requires_grad_(True)
      bwd_step3(x_bwd_cpu).backward()
      x_bwd_tpu = x_base.clone().to(self.device).requires_grad_(True)
      torch._dynamo.mark_dynamic(x_bwd_tpu, 2, min=6, max=16)
      bwd_tpu(x_bwd_tpu).backward()
      utils.assert_close(x_bwd_tpu.grad.cpu(), x_bwd_cpu.grad)

    self.assertEqual(fwd_counter.frame_count, 1)
    self.assertEqual(mut_counter.frame_count, 1)
    self.assertEqual(bwd_counter.frame_count, 1)

    # Also verify step=3 dynamic view passed as a graph input (n % 3 = 0, 1, 2).
    torch._dynamo.reset()
    in_counter = dynamo_testing.CompileCounterWithBackend("tpu")
    in_tpu = torch.compile(lambda v: v + 5.0, backend=in_counter, dynamic=None)
    in_compiles: list[int] = []
    with torch.no_grad():
      for n in (8, 11, 12, 13, 14, 15):
        b_cpu = torch.arange(2 * n * 4, dtype=torch.float32).reshape(2, n, 4)
        v_cpu = b_cpu[:, 1::3, :]
        v_tpu = b_cpu.to(self.device)[:, 1::3, :]
        before = in_counter.frame_count
        out_in_tpu = in_tpu(v_tpu)
        in_compiles.append(in_counter.frame_count - before)
        utils.assert_close(out_in_tpu.cpu(), v_cpu + 5.0)
    self.assertEqual(in_compiles, [1, 1, 0, 0, 0, 0])


if __name__ == "__main__":
  absltest.main()
