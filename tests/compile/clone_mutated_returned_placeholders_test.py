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

"""Tests for clone_mutated_returned_placeholders FX pass."""

from absl.testing import absltest
import torch
import torch._dynamo.source
from torch import fx
from torch_tpu._internal.compile.fx_passes import clone_mutated_returned_placeholders
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class CloneMutatedReturnedPlaceholdersTest(seed_test_utils.RepeatableTest):

  def test_clones_mutated_returned_placeholder(self):
    """Verifies that placeholders returned in the output and used in compute are cloned."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    # In-place addition mutating p_weight
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    # The output node returns the mutated placeholder directly
    graph.output((add_node, p_weight))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    # Check that a clone node was created right after p_weight
    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    clone_node = clone_nodes[0]
    self.assertEqual(clone_node.args, (p_weight,))
    self.assertEqual(clone_node.meta["val"].shape, p_weight.meta["val"].shape)

    # p_weight's only user should be the clone node
    self.assertEqual(list(p_weight.users.keys()), [clone_node])

    # Downstream compute and output should now use the clone node
    self.assertIn(clone_node, add_node.args)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertIn(clone_node, output_node.args[0])
    self.assertNotIn(p_weight, output_node.args[0])

    # Verify that the generated code was recompiled to invoke clone
    self.assertIn("clone", gm.code)

  def test_apply_on_graph_directly(self):
    """Verifies that apply() works directly on a raw torch.fx.Graph instance."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))

    # Apply directly on graph, where gm is None
    clone_mutated_returned_placeholders.apply(graph)

    clone_nodes = [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    self.assertEqual(list(p_weight.users.keys()), [clone_nodes[0]])

  def test_recompiles_graph_module(self):
    """Verifies that apply() triggers gm.recompile(), updating gm.code."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_grad = graph.placeholder("grad")
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))
    gm = GraphModule(torch.nn.Module(), graph)

    self.assertNotIn("clone", gm.code)
    clone_mutated_returned_placeholders.apply(gm)
    self.assertIn("clone", gm.code)

  def test_passthrough_placeholder_not_cloned(self):
    """Verifies that placeholders only returned without compute uses are not cloned."""
    graph = Graph()
    p_in = graph.placeholder("in_tensor")
    p_in.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    graph.output((p_in,))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_compute_placeholder_not_returned_not_cloned(self):
    """Verifies that compute placeholders not present in the output are not cloned."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_y = graph.placeholder("y")
    p_y.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.add.Tensor, (p_x, p_y))
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_multiple_mutated_returned_placeholders(self):
    """Verifies multiple placeholders are all cloned."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_m = graph.placeholder("exp_avg")
    p_m.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_v = graph.placeholder("exp_avg_sq")
    p_v.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_step = graph.placeholder("step")
    p_step.meta["val"] = torch.empty((), dtype=torch.int32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    # In-place ops mutating weight, m, v, step
    graph.call_function(torch.ops.aten.add_.Tensor, (p_step, 1))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_m, p_grad))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_v, p_grad))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_weight, p_m))

    graph.output((p_weight, p_m, p_v, p_step))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 4)
    for p in (p_weight, p_m, p_v, p_step):
      self.assertLen(p.users, 1)
      cloned_user = list(p.users.keys())[0]
      self.assertEqual(cloned_user.target, torch.ops.aten.clone.default)

  def test_symint_placeholder_not_cloned(self):
    """Verifies that SymInt placeholders in compute/output are not cloned."""
    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    symint_s0 = shape_env.create_symintnode(
        shape_env.create_symbol(4, torch._dynamo.source.ConstantSource("s0")),
        hint=4,
    )

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode(
        shape_env=shape_env
    )
    with fake_mode:
      fake_tensor = torch.empty((symint_s0, 4), dtype=torch.float32)

    graph = Graph()
    p_s0 = graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    p_x = graph.placeholder("x")
    p_x.meta["val"] = fake_tensor

    # p_s0 is used in compute (slice) and also returned in output
    slice_node = graph.call_function(
        torch.ops.aten.slice.Tensor, (p_x, 0, 0, p_s0)
    )
    graph.output((slice_node, p_s0))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)


if __name__ == "__main__":
  absltest.main()
