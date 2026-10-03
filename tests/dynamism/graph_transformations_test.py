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

"""Unit tests for graph transformations."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
from unittest import mock
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import dynamic_compiler
from torch_tpu._internal.compile.dynamic import graph_transform_context
from torch_tpu._internal.compile.dynamic import graph_transformations
from torch_tpu._internal.compile.dynamic import sym_bounds_manager
from torch_tpu._internal.compile.dynamic import view_ops_passes
from torch_tpu._internal.utils import utils
from tests import seed_test_utils


class GraphTransformationsTest(seed_test_utils.RepeatableTest):

  def test_handle_dynamic_input_tensor_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    # Apply the passes
    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    graph_transformations.HandleDynamicInputTensorPass(
        captured_sm, placeholders
    )(captured_gm)

    # Verify that a new placeholder was added for the dynamic dimension
    new_placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    self.assertLen(new_placeholders, len(placeholders) + 1)

    # Verify that pad_real_to_static was inserted
    pad_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu._pad_real_to_static,
        )
    )
    self.assertLen(pad_nodes, 1)

  def test_handle_generative_ops_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return torch.arange(0, x.shape[0])

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )

    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    graph_transformations.HandleGenerativeOpsPass(captured_sm)(captured_gm)

    # Verify that dynamic_arange was inserted for arange
    dynamic_arange_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu._dynamic_arange,
        )
    )
    self.assertLen(dynamic_arange_nodes, 1)

    dynamic_arange_node = dynamic_arange_nodes[0]

    # Verify the arguments passed to dynamic_arange
    self.assertEqual(
        dynamic_arange_node.args[0].target, torch.ops.aten.scalar_tensor.default
    )
    self.assertEqual(dynamic_arange_node.args[1].op, "placeholder")
    self.assertEqual(
        dynamic_arange_node.args[2].target, torch.ops.aten.scalar_tensor.default
    )
    self.assertEqual(dynamic_arange_node.args[3], 8)  # max_length
    self.assertEqual(dynamic_arange_node.args[4], torch.int64)  # expected_dtype

  def test_scan_inputs_create_placeholders_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return torch.arange(0, x.shape[0])

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )

    # Find a SymInt placeholder
    symint_ph = None
    for node in placeholders:
      if "val" in node.meta and isinstance(node.meta["val"], torch.SymInt):
        symint_ph = node
        break

    self.assertIsNotNone(symint_ph)

    # Apply the pass
    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    # Verify that a new placeholder was added
    new_placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    self.assertLen(new_placeholders, len(placeholders) + 1)

    # Verify that it is in _symint_to_placeholder
    sym_str = str(symint_ph.meta["val"])
    self.assertIn(sym_str, captured_sm._symint_to_placeholder)  # pylint: disable=protected-access

  def test_handle_dynamic_input_tensor_pass_reuse_placeholder(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, y):
      return x + y

    x = torch.ones(4)
    y = torch.ones(4)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(y, 0, min=2, max=8)

    f(x, y)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )

    # Verify that we actually have the same SymInt for both inputs.
    tensor_phs = [
        node
        for node in placeholders
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor)
    ]
    self.assertLen(tensor_phs, 2)

    x_sym = tensor_phs[0].meta["val"].shape[0]
    y_sym = tensor_phs[1].meta["val"].shape[0]
    self.assertEqual(x_sym, y_sym)

    # Apply the passes
    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    graph_transformations.HandleDynamicInputTensorPass(
        captured_sm, placeholders
    )(captured_gm)

    # Verify that only ONE new placeholder was added
    # (for the shared dynamic dimension)
    new_placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    self.assertLen(new_placeholders, len(placeholders) + 1)

    # Verify that both pad_real_to_static nodes use the SAME placeholder
    pad_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu._pad_real_to_static,
        )
    )
    self.assertLen(pad_nodes, 2)

    size_input_0 = pad_nodes[0].args[1][0]
    size_input_1 = pad_nodes[1].args[1][0]
    self.assertEqual(size_input_0, size_input_1)
    self.assertEqual(size_input_0.op, "placeholder")
    self.assertTrue(size_input_0.name.endswith("_size"))

  def test_handle_reshape_like_ops_pass_dynamic_input_trailing_static_dim(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x.view(x.shape[0], 4, 2)

    t = torch.ones(4, 8)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=16)

    f(t)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    view_ops_passes.HandleReshapeLikeOpsPass(captured_sm)(captured_gm)

    dynamic_reshape_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu._dynamic_reshape,
        )
    )
    self.assertLen(dynamic_reshape_nodes, 1)

  def test_handle_reshape_like_ops_pass_view_copy(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return torch.ops.aten.view_copy.default(x, [x.shape[0], 4, 2])

    t = torch.ones(4, 8)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=16)

    f(t)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    view_ops_passes.HandleReshapeLikeOpsPass(captured_sm)(captured_gm)

    dynamic_reshape_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu._dynamic_reshape,
        )
    )
    self.assertLen(dynamic_reshape_nodes, 1)

  def test_can_use_pad_real_to_static_libtpu_version(self):
    t = torch.ones(4, dtype=torch.float32)
    with mock.patch.object(utils, "libtpu_version", return_value=(0, 0, 48)):
      self.assertFalse(dynamic_compiler._can_use_pad_real_to_static([t]))  # pylint: disable=protected-access
    with mock.patch.object(utils, "libtpu_version", return_value=(0, 0, 49)):
      self.assertTrue(dynamic_compiler._can_use_pad_real_to_static([t]))  # pylint: disable=protected-access
    with mock.patch.object(utils, "libtpu_version", return_value=None):
      self.assertTrue(dynamic_compiler._can_use_pad_real_to_static([t]))  # pylint: disable=protected-access

  def _capture_dynamic_stride_view_graph(
      self, make_view: Callable[[int], torch.Tensor]
  ) -> tuple[torch.fx.GraphModule, list[Any], int]:
    """Captures the graph compiled for a view input with dynamic strides.

    Compiles `x + 1` for views of bases with two sequence lengths. The sizes
    of the views are the same, so the second compilation sees an input with
    static sizes and dynamic strides, like a KV-cache view across decode steps.

    Args:
      make_view: Builds the view input for a given sequence length.

    Returns:
      The graph module and example inputs of the second compilation, and the
      index of the view's placeholder (Dynamo lifts the stride symbols as
      leading SymInt placeholders).
    """
    captured: list[tuple[torch.fx.GraphModule, list[Any]]] = []

    def fw_compiler(graph_module, example_inputs):
      captured.append((graph_module, list(example_inputs)))
      return graph_module

    torch._dynamo.reset()

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler))
    def f(x):
      return x + 1

    for seq_len in (16, 20):
      f(make_view(seq_len))

    self.assertLen(captured, 2)
    graph_module, example_inputs = captured[-1]
    placeholders = list(
        graph_module.graph.find_nodes(op="placeholder", sort=True)
    )
    tensor_indices = [
        idx
        for idx, node in enumerate(placeholders)
        if isinstance(node.meta.get("val"), torch.Tensor)
    ]
    self.assertLen(tensor_indices, 1)
    view_val = placeholders[tensor_indices[0]].meta["val"]
    self.assertTrue(any(isinstance(s, torch.SymInt) for s in view_val.stride()))
    return graph_module, example_inputs, tensor_indices[0]

  def _check_input_dynamic_view_pass_metadata(
      self,
      make_view: Callable[[int], torch.Tensor],
      expected_view_ops: list[torch._ops.OpOverload],
  ) -> None:
    """Checks that every inserted view node carries metadata and compiles."""
    graph_module, example_inputs, view_index = (
        self._capture_dynamic_stride_view_graph(make_view)
    )
    view_arg_indices, example_inputs = (
        graph_transformations.apply_pre_dynamism_transformations(
            graph_module, example_inputs
        )
    )
    self.assertEqual(view_arg_indices, {view_index})

    view_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target
        in (torch.ops.aten.slice.Tensor, torch.ops.aten.permute.default)
    ]
    self.assertEqual([node.target for node in view_nodes], expected_view_ops)
    for node in view_nodes:
      with self.subTest(node=node.name):
        self.assertIn("val", node.meta)
        expected_val = node.target(
            *torch.fx.node.map_arg(node.args, lambda arg: arg.meta["val"])
        )
        self.assertEqual(list(node.meta["val"].shape), list(expected_val.shape))
        self.assertEqual(
            list(node.meta["val"].stride()), list(expected_val.stride())
        )

    # The later passes read the metadata of the inserted nodes.
    bm = sym_bounds_manager.SymBoundsManager(example_inputs)
    graph_transformations.apply_dynamism_transformations(
        graph_module, example_inputs, bm.get_fallback_symbol_bounds()
    )

  def test_input_dynamic_view_pass_transposed_slice_input(self):
    """`k.transpose(1, 2)[:, :4]` decomposes into a slice and a permute."""
    self._check_input_dynamic_view_pass_metadata(
        lambda seq_len: torch.ones(2, 3, seq_len, 8).transpose(1, 2)[:, :4],
        [torch.ops.aten.slice.Tensor, torch.ops.aten.permute.default],
    )

  def test_input_dynamic_view_pass_two_dim_slice_input(self):
    """`hidden[:, 1:3, 2:6]` decomposes into two slices."""
    self._check_input_dynamic_view_pass_metadata(
        lambda seq_len: torch.ones(2, seq_len, 8)[:, 1:3, 2:6],
        [torch.ops.aten.slice.Tensor, torch.ops.aten.slice.Tensor],
    )


if __name__ == "__main__":
  absltest.main()
