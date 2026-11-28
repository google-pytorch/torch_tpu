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

"""Unit tests for GraphTransformContext."""

from __future__ import annotations
import copy
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import graph_transform_context
from torch_tpu._internal.compile.dynamic import graph_transformations
from torch_tpu._internal.compile.dynamic import output_sym_shape
from torch_tpu._internal.compile.dynamic import sym_bounds_manager
from torch_tpu._internal.compile.dynamic import sym_utils
from tests import seed_test_utils


class GraphTransformContextTest(seed_test_utils.RepeatableTest):

  def test_has_dynamic_shape(self):
    captured_gm = None

    def fw_compiler(graph_module, example_inputs):
      del example_inputs
      nonlocal captured_gm
      captured_gm = graph_module
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, y):
      return x + 1, y + 1

    t1 = torch.ones(4)
    t2 = torch.ones(4)
    torch._dynamo.mark_dynamic(t1, 0, min=2, max=8)

    f(t1, t2)

    self.assertIsNotNone(captured_gm)
    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    # Dynamic placeholder vs static placeholder
    self.assertTrue(sym_utils.has_dynamic_shape(placeholders[1]))
    self.assertFalse(sym_utils.has_dynamic_shape(placeholders[2]))

  def test_symbol_manager_extraction(self):
    captured_sm = None
    captured_output_sym_shapes = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_sm, captured_output_sym_shapes
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      captured_output_sym_shapes = output_sym_shape.create_output_sym_shapes(
          graph_module, example_inputs
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_sm)
    self.assertEqual([s.is_dynamic for s in captured_output_sym_shapes], [True])
    self.assertEqual(
        captured_sm.get_symint_bounds(captured_sm.example_inputs[1].shape[0]),
        (2, 8),
    )
    self.assertEqual(
        captured_sm.get_upper_bound(captured_sm.example_inputs[1].shape[0]), 8
    )

  def test_symbol_manager_static_shape(self):
    captured_sm = None
    captured_output_sym_shapes = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_sm, captured_output_sym_shapes
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_sm = graph_transform_context.GraphTransformContext(
          graph_module, example_inputs, bm.get_fallback_symbol_bounds()
      )
      captured_output_sym_shapes = output_sym_shape.create_output_sym_shapes(
          graph_module, example_inputs
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler))
    def f(x):
      return x + 1

    t = torch.ones(2, 3)

    f(t)

    self.assertIsNotNone(captured_sm)
    self.assertEqual(
        [s.is_dynamic for s in captured_output_sym_shapes], [False]
    )
    self.assertEqual(
        captured_sm.get_upper_bound(captured_sm.example_inputs[0].shape[0]), 2
    )
    self.assertEqual(
        captured_sm.get_upper_bound(captured_sm.example_inputs[0].shape[1]), 3
    )

  def test_validate_bounds_with_multi_symbol_sum_guard(self):
    captured_bm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_bm
      captured_bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      captured_bm.get_fallback_symbol_bounds()
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, y):
      assert x.shape[0] + y.shape[0] == 64
      return x * 2, y * 3

    t1 = torch.ones(20)
    t2 = torch.ones(44)
    torch._dynamo.mark_dynamic(t1, 0)
    torch._dynamo.mark_dynamic(t2, 0)

    f(t1, t2)

    self.assertIsNotNone(captured_bm)

  def test_handle_symint_usages_rewrites_group_norm_shape_args(self):
    captured = {}

    def run_pass(graph_module, example_inputs, key):
      gm_copy = copy.deepcopy(graph_module)
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      ctx = graph_transform_context.GraphTransformContext(
          gm_copy, example_inputs, bm.get_fallback_symbol_bounds()
      )
      placeholders = list(gm_copy.graph.find_nodes(op="placeholder", sort=True))
      graph_transformations.ScanInputsCreatePlaceholdersPass(ctx, placeholders)(
          gm_copy
      )
      graph_transformations.HandleSymIntUsagesPass(ctx)(gm_copy)
      captured[key] = gm_copy
      return graph_module

    backend = aot_autograd(
        fw_compiler=lambda gm, ex: run_pass(gm, ex, "fw"),
        bw_compiler=lambda gm, ex: run_pass(gm, ex, "bw"),
    )

    @torch.compile(backend=backend, dynamic=True)
    def f(x, w, b):
      return torch.nn.functional.group_norm(x, 2, w, b)

    x = torch.randn(4, 6, 4, 5, requires_grad=True)
    w = torch.randn(6, requires_grad=True)
    b = torch.randn(6, requires_grad=True)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(x, 2, min=2, max=6)
    torch._dynamo.mark_dynamic(x, 3, min=2, max=7)

    out = f(x, w, b)
    out.sum().backward()

    fw_nodes = [
        n
        for n in captured["fw"].graph.nodes
        if n.target == torch.ops.aten.native_group_norm.default
    ]
    self.assertLen(fw_nodes, 1)
    # (N, C, HxW) at args 3, 4, 5 are rewritten to static upper bounds:
    # N -> 8 (explicit max), C -> 12 (fallback 2*6 under dynamic=True),
    # HxW -> 42 (6 * 7).
    self.assertEqual(fw_nodes[0].args[3], 8)
    self.assertEqual(fw_nodes[0].args[4], 12)
    self.assertEqual(fw_nodes[0].args[5], 42)

    bw_nodes = [
        n
        for n in captured["bw"].graph.nodes
        if n.target == torch.ops.aten.native_group_norm_backward.default
    ]
    self.assertLen(bw_nodes, 1)
    # (N, C, HxW) at args 5, 6, 7 are rewritten to static upper bounds (8, 12, 42).
    self.assertEqual(bw_nodes[0].args[5], 8)
    self.assertEqual(bw_nodes[0].args[6], 12)
    self.assertEqual(bw_nodes[0].args[7], 42)

  def test_handle_symint_usages_rewrites_group_norm_shape_kwargs(self):
    captured_gm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm
      gm_copy = copy.deepcopy(graph_module)
      bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      ctx = graph_transform_context.GraphTransformContext(
          gm_copy, example_inputs, bm.get_fallback_symbol_bounds()
      )
      placeholders = list(gm_copy.graph.find_nodes(op="placeholder", sort=True))
      graph_transformations.ScanInputsCreatePlaceholdersPass(ctx, placeholders)(
          gm_copy
      )
      for node in gm_copy.graph.nodes:
        if node.target == torch.ops.aten.native_group_norm.default:
          n_arg, c_arg, hxw_arg, group_arg, eps_arg = node.args[3:8]
          node.args = node.args[:3]
          node.kwargs = {
              "N": n_arg,
              "C": c_arg,
              "HxW": hxw_arg,
              "group": group_arg,
              "eps": eps_arg,
          }
      graph_transformations.HandleSymIntUsagesPass(ctx)(gm_copy)
      captured_gm = gm_copy
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, w, b):
      return torch.nn.functional.group_norm(x, 2, w, b)

    x = torch.randn(4, 6, 4, 5)
    w = torch.randn(6)
    b = torch.randn(6)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(x, 2, min=2, max=6)
    torch._dynamo.mark_dynamic(x, 3, min=2, max=7)

    f(x, w, b)

    self.assertIsNotNone(captured_gm)
    fw_nodes = [
        n
        for n in captured_gm.graph.nodes
        if n.target == torch.ops.aten.native_group_norm.default
    ]
    self.assertLen(fw_nodes, 1)
    self.assertEqual(fw_nodes[0].kwargs["N"], 8)
    self.assertEqual(fw_nodes[0].kwargs["C"], 12)
    self.assertEqual(fw_nodes[0].kwargs["HxW"], 42)
    self.assertEqual(fw_nodes[0].kwargs["group"], 2)


if __name__ == "__main__":
  absltest.main()
