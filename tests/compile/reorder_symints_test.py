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

from absl.testing import absltest
import torch
import torch._dynamo.source
from torch import fx
from torch_tpu._internal.compile.fx_passes import reorder_symints
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class ReorderSymintsTest(seed_test_utils.RepeatableTest):

  def test_reorder_symints(self):
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

    # Submodule initially has Tensor 'x' before SymInt 's0'
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_x.meta["val"] = fake_tensor
    sub_s0 = sub_graph.placeholder("s0")
    sub_s0.meta["val"] = symint_s0
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_x))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_x.meta["val"] = fake_tensor
    p_s0 = parent_graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    call_submod = parent_graph.call_module("submod_0", (p_x, p_s0))
    parent_graph.output(call_submod)

    parent_gm = GraphModule({"submod_0": submod}, parent_graph)

    reorder_symints.apply(parent_gm)

    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 2)
    self.assertEqual(submod_placeholders[0].name, "s0")
    self.assertEqual(submod_placeholders[0].meta["val"], symint_s0)
    self.assertEqual(submod_placeholders[1].name, "x")
    self.assertEqual(call_submod.args, (p_s0, p_x))

  def test_no_op_when_already_in_order(self):
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

    # Submodule already has SymInt 's0' before Tensor 'x'
    sub_graph = Graph()
    sub_s0 = sub_graph.placeholder("s0")
    sub_s0.meta["val"] = symint_s0
    sub_x = sub_graph.placeholder("x")
    sub_x.meta["val"] = fake_tensor
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_x))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_s0 = parent_graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    p_x = parent_graph.placeholder("x")
    p_x.meta["val"] = fake_tensor
    call_submod = parent_graph.call_module("submod_0", (p_s0, p_x))
    parent_graph.output(call_submod)

    parent_gm = GraphModule({"submod_0": submod}, parent_graph)

    reorder_symints.apply(parent_gm)

    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 2)
    self.assertEqual(submod_placeholders[0].name, "s0")
    self.assertEqual(submod_placeholders[1].name, "x")
    self.assertEqual(call_submod.args, (p_s0, p_x))

  def test_multiple_symints_and_tensors(self):
    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    symint_s0 = shape_env.create_symintnode(
        shape_env.create_symbol(4, torch._dynamo.source.ConstantSource("s0")),
        hint=4,
    )
    symint_s1 = shape_env.create_symintnode(
        shape_env.create_symbol(2, torch._dynamo.source.ConstantSource("s1")),
        hint=2,
    )

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode(
        shape_env=shape_env
    )
    with fake_mode:
      fake_tensor_x = torch.empty((symint_s0, 4), dtype=torch.float32)
      fake_tensor_y = torch.empty((symint_s1, 2), dtype=torch.float32)

    # Interleaved inputs: (x, s0, y, s1)
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_x.meta["val"] = fake_tensor_x
    sub_s0 = sub_graph.placeholder("s0")
    sub_s0.meta["val"] = symint_s0
    sub_y = sub_graph.placeholder("y")
    sub_y.meta["val"] = fake_tensor_y
    sub_s1 = sub_graph.placeholder("s1")
    sub_s1.meta["val"] = symint_s1
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_x))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    p_x.meta["val"] = fake_tensor_x
    p_s0 = parent_graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    p_y = parent_graph.placeholder("y")
    p_y.meta["val"] = fake_tensor_y
    p_s1 = parent_graph.placeholder("s1")
    p_s1.meta["val"] = symint_s1
    call_submod = parent_graph.call_module("submod_0", (p_x, p_s0, p_y, p_s1))
    parent_graph.output(call_submod)

    parent_gm = GraphModule({"submod_0": submod}, parent_graph)

    reorder_symints.apply(parent_gm)

    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 4)
    self.assertEqual(
        [n.name for n in submod_placeholders], ["s0", "s1", "x", "y"]
    )
    self.assertEqual(call_submod.args, (p_s0, p_s1, p_x, p_y))


if __name__ == "__main__":
  absltest.main()
