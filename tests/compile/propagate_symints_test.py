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
from torch import fx
from torch_tpu._internal.compile.fx_passes import propagate_symints
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class PropagateSymintsTest(seed_test_utils.RepeatableTest):

  def test_propagate_symints(self):
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

    sub_graph = Graph()
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
    call_submod = parent_graph.call_module("submod_0", (p_x,))
    parent_graph.output(call_submod)

    parent_gm = GraphModule({"submod_0": submod}, parent_graph)

    propagate_symints.apply(parent_gm)

    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 2)
    self.assertEqual(submod_placeholders[0].name, f"sym_{str(symint_s0)}")
    self.assertEqual(submod_placeholders[0].meta["val"], symint_s0)
    self.assertEqual(submod_placeholders[1].name, "x")
    self.assertEqual(call_submod.args, (p_s0, p_x))

  def test_no_op_when_already_propagated(self):
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

    propagate_symints.apply(parent_gm)

    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 2)
    self.assertEqual(call_submod.args, (p_s0, p_x))

  def test_propagate_symints_from_stride(self):
    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    symint_s0 = shape_env.create_symintnode(
        shape_env.create_symbol(
            1024, torch._dynamo.source.ConstantSource("s0")
        ),
        hint=1024,
    )

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode(
        shape_env=shape_env
    )
    with fake_mode:
      base_tensor = torch.empty((16, 8, symint_s0, 8), dtype=torch.float32)
      view_tensor = base_tensor[:, :, :127, :]

    # view_tensor has static shape (16, 8, 127, 8) and dynamic stride (8*s0, s0, 8, 1)
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_x.meta["val"] = view_tensor
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_x))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = Graph()
    p_s0 = parent_graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    p_x = parent_graph.placeholder("x")
    p_x.meta["val"] = view_tensor
    call_submod = parent_graph.call_module("submod_0", (p_x,))
    parent_graph.output(call_submod)

    parent_gm = GraphModule({"submod_0": submod}, parent_graph)

    propagate_symints.apply(parent_gm)

    submod_placeholders = [
        n for n in submod.graph.nodes if n.op == "placeholder"
    ]
    self.assertLen(submod_placeholders, 2)
    self.assertEqual(submod_placeholders[0].name, f"sym_{str(symint_s0)}")
    self.assertEqual(submod_placeholders[0].meta["val"], symint_s0)
    self.assertEqual(submod_placeholders[1].name, "x")
    self.assertEqual(call_submod.args, (p_s0, p_x))


if __name__ == "__main__":
  absltest.main()
