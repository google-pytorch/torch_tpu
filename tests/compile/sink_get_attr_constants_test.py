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

"""Unit tests for sink_get_attr_constants pass."""

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import sink_get_attr_constants
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class SinkGetAttrConstantsTest(seed_test_utils.RepeatableTest):

  def test_sink_single_constant_attribute(self):
    const_tensor = torch.tensor(5.0)

    # Submodule initially has placeholder 'x' and placeholder 'c'
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    # Parent module has get_attr 'c_attr' and passes it to submod_0
    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c = parent_graph.get_attr("c_attr")
    call_submod = parent_graph.call_module("submod_0", (p_x, p_c))
    parent_graph.output(call_submod)

    parent_root = torch.nn.Module()
    parent_root.c_attr = const_tensor
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    # Verify submodule placeholders and get_attr nodes
    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 1)
    self.assertEqual(submod_placeholders[0].name, "x")

    submod_get_attrs = [n for n in submod.graph.nodes if n.op == "get_attr"]
    self.assertLen(submod_get_attrs, 1)
    self.assertEqual(submod_get_attrs[0].target, "c_attr")
    self.assertTrue(hasattr(submod, "c_attr"))
    self.assertEqual(submod.c_attr, const_tensor)

    # Verify parent call_site arguments
    self.assertEqual(call_submod.args, (p_x,))

    # Verify dead get_attr removed from parent
    parent_get_attrs = [n for n in parent_gm.graph.nodes if n.op == "get_attr"]
    self.assertEmpty(parent_get_attrs)

  def test_sink_constant_to_multiple_submodules(self):
    const_tensor = torch.tensor(3.0)

    # Submodule 0
    sub_graph0 = Graph()
    sub_x0 = sub_graph0.placeholder("x")
    sub_c0 = sub_graph0.placeholder("c")
    sub_out0 = sub_graph0.call_function(
        torch.ops.aten.mul.Tensor, (sub_x0, sub_c0)
    )
    sub_graph0.output(sub_out0)
    submod0 = GraphModule(torch.nn.Module(), sub_graph0)

    # Submodule 1
    sub_graph1 = Graph()
    sub_y1 = sub_graph1.placeholder("y")
    sub_c1 = sub_graph1.placeholder("c")
    sub_out1 = sub_graph1.call_function(
        torch.ops.aten.add.Tensor, (sub_y1, sub_c1)
    )
    sub_graph1.output(sub_out1)
    submod1 = GraphModule(torch.nn.Module(), sub_graph1)

    # Parent module passes c_attr to both submodules
    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c = parent_graph.get_attr("c_attr")
    call_submod0 = parent_graph.call_module("submod_0", (p_x, p_c))
    call_submod1 = parent_graph.call_module("submod_1", (call_submod0, p_c))
    parent_graph.output(call_submod1)

    parent_root = torch.nn.Module()
    parent_root.c_attr = const_tensor
    parent_root.submod_0 = submod0
    parent_root.submod_1 = submod1
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    # Both submodules should have c_attr sunk into them
    self.assertLen([n for n in submod0.graph.nodes if n.op == "placeholder"], 1)
    self.assertLen([n for n in submod0.graph.nodes if n.op == "get_attr"], 1)
    self.assertEqual(call_submod0.args, (p_x,))

    self.assertLen([n for n in submod1.graph.nodes if n.op == "placeholder"], 1)
    self.assertLen([n for n in submod1.graph.nodes if n.op == "get_attr"], 1)
    self.assertEqual(call_submod1.args, (call_submod0,))

  def test_non_tensor_get_attr_is_not_sunk(self):
    # A get_attr referencing a non-tensor attribute must be left untouched,
    # since _assign_attr(..., CONSTANT) only accepts tensors/script objects.
    non_tensor_attr = torch.nn.Module()

    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c = parent_graph.get_attr("non_tensor_attr")
    call_submod = parent_graph.call_module("submod_0", (p_x, p_c))
    parent_graph.output(call_submod)

    parent_root = torch.nn.Module()
    parent_root.non_tensor_attr = non_tensor_attr
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    # Submodule placeholders are unchanged and no get_attr was added.
    self.assertLen([n for n in submod.graph.nodes if n.op == "placeholder"], 2)
    self.assertEmpty([n for n in submod.graph.nodes if n.op == "get_attr"])

    # The call site still passes the non-tensor get_attr through.
    self.assertEqual(call_submod.args, (p_x, p_c))
    parent_get_attrs = [n for n in parent_gm.graph.nodes if n.op == "get_attr"]
    self.assertLen(parent_get_attrs, 1)

  def test_shared_submodule_updates_all_call_sites(self):
    const_tensor = torch.tensor(5.0)

    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    # Two call sites into the SAME submodule, both passing the same constant.
    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c = parent_graph.get_attr("c_attr")
    call_a = parent_graph.call_module("submod_0", (p_x, p_c))
    call_b = parent_graph.call_module("submod_0", (call_a, p_c))
    parent_graph.output(call_b)

    parent_root = torch.nn.Module()
    parent_root.c_attr = const_tensor
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    # The constant is sunk once into the shared submodule.
    self.assertLen([n for n in submod.graph.nodes if n.op == "placeholder"], 1)
    self.assertLen([n for n in submod.graph.nodes if n.op == "get_attr"], 1)

    # BOTH call sites must have the constant argument removed, otherwise the
    # first call site would pass 2 args to a submodule that now takes 1.
    self.assertEqual(call_a.args, (p_x,))
    self.assertEqual(call_b.args, (call_a,))

  def test_constant_with_no_users_deletes_attribute(self):
    const_tensor = torch.tensor(5.0)

    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c = parent_graph.get_attr("c_attr")
    parent_graph.get_attr("unused_attr")
    call_submod = parent_graph.call_module("submod_0", (p_x, p_c))
    parent_graph.output(call_submod)

    parent_root = torch.nn.Module()
    parent_root.c_attr = const_tensor
    parent_root.unused_attr = torch.tensor(10.0)
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    self.assertEmpty([n for n in parent_gm.graph.nodes if n.op == "get_attr"])
    self.assertFalse(hasattr(parent_gm, "c_attr"))
    self.assertFalse(hasattr(parent_gm, "unused_attr"))
    self.assertEqual(parent_gm(torch.tensor(1.0)), torch.tensor(6.0))

  def test_constant_with_users_keeps_attribute(self):
    const_tensor = torch.tensor(5.0)

    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c = parent_graph.get_attr("c_attr")
    call_submod = parent_graph.call_module("submod_0", (p_x, p_c))
    out = parent_graph.call_function(
        torch.ops.aten.add.Tensor, (call_submod, p_c)
    )
    parent_graph.output(out)

    parent_root = torch.nn.Module()
    parent_root.c_attr = const_tensor
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    self.assertLen([n for n in parent_gm.graph.nodes if n.op == "get_attr"], 1)
    self.assertTrue(hasattr(parent_gm, "c_attr"))
    self.assertEqual(parent_gm(torch.tensor(1.0)), torch.tensor(11.0))

  def test_live_targets_preserves_attribute_when_duplicate_get_attr_exists(
      self,
  ):
    const_tensor = torch.tensor(5.0)

    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    # Two distinct get_attr nodes targeting the same attribute "c_attr":
    # p_c1 is sunk into submod_0 (becoming dead), while p_c2 remains live.
    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_c1 = parent_graph.get_attr("c_attr")
    p_c2 = parent_graph.get_attr("c_attr")
    call_submod = parent_graph.call_module("submod_0", (p_x, p_c1))
    out = parent_graph.call_function(
        torch.ops.aten.add.Tensor, (call_submod, p_c2)
    )
    parent_graph.output(out)

    parent_root = torch.nn.Module()
    parent_root.c_attr = const_tensor
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    sink_get_attr_constants.apply(parent_gm)

    parent_get_attrs = [n for n in parent_gm.graph.nodes if n.op == "get_attr"]
    self.assertEqual(parent_get_attrs, [p_c2])
    self.assertTrue(hasattr(parent_gm, "c_attr"))
    self.assertEqual(parent_gm(torch.tensor(1.0)), torch.tensor(11.0))


if __name__ == "__main__":
  absltest.main()
