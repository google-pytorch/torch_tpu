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
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import graph_transformations
from torch_tpu._internal.compile.dynamic import sym_shape_manager


class GraphTransformationsTest(absltest.TestCase):

  def test_handle_dynamic_input_tensor_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
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

    # Verify that set_dimension_logical_size was inserted
    set_dim_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu.set_dimension_logical_size,
        )
    )
    self.assertLen(set_dim_nodes, 1)

  def test_handle_generative_ops_pass(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
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
            target=torch.ops.tpu.dynamic_arange,
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
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
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

    # Verify that it is in symint_to_placeholder
    sym_str = str(symint_ph.meta["val"])
    self.assertIn(sym_str, captured_sm.symint_to_placeholder)

  def test_handle_dynamic_input_tensor_pass_reuse_placeholder(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
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

    # Verify that both set_dimension_logical_size nodes use the SAME placeholder
    set_dim_nodes = list(
        captured_gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.tpu.set_dimension_logical_size,
        )
    )
    self.assertLen(set_dim_nodes, 2)

    size_input_0 = set_dim_nodes[0].args[2]
    size_input_1 = set_dim_nodes[1].args[2]
    self.assertEqual(size_input_0, size_input_1)
    self.assertEqual(size_input_0.op, "placeholder")
    self.assertTrue(size_input_0.name.endswith("_size"))

  def test_duplicate_symint_placeholder(self):
    captured_gm = None
    captured_sm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm, captured_sm
      captured_gm = graph_module
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
      )
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, s0, s1):
      return x + s0 + s1

    x = torch.ones(4)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
    s = x.shape[0]

    f(x, s, s)

    self.assertIsNotNone(captured_gm)
    self.assertIsNotNone(captured_sm)

    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )

    # Apply ScanInputsCreatePlaceholdersPass.
    # If it doesn't handle duplicates, it might insert duplicate placeholders,
    # causing recompile() or lint() to crash with SyntaxError.
    graph_transformations.ScanInputsCreatePlaceholdersPass(
        captured_sm, placeholders
    )(captured_gm)

    captured_gm.recompile()
    captured_gm.graph.lint()


if __name__ == "__main__":
  absltest.main()
