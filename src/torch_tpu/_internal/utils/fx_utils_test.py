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
from torch_tpu._internal.utils import fx_utils
from tests import seed_test_utils


class _ModuleWithNonTensorAttr(torch.nn.Module):

  def __init__(self, non_tensor_attr: int):
    super().__init__()
    self.non_tensor_attr = non_tensor_attr


class FxUtilsTest(seed_test_utils.RepeatableTest):

  def test_iter_tensor_get_attrs_basic_and_deduplication(self):
    graph = torch.fx.Graph()
    x_node = graph.placeholder("x")
    first_const = graph.get_attr("const_tensor")
    duplicate_const = graph.get_attr("const_tensor")
    param_node = graph.get_attr("param_attr")
    nested_node = graph.get_attr("sub.nested_tensor")
    graph.get_attr("non_tensor_attr")
    sum_node = graph.call_function(
        torch.ops.aten.add.Tensor, (x_node, first_const)
    )
    sum_node = graph.call_function(
        torch.ops.aten.add.Tensor, (sum_node, duplicate_const)
    )
    sum_node = graph.call_function(
        torch.ops.aten.add.Tensor, (sum_node, param_node)
    )
    out = graph.call_function(
        torch.ops.aten.add.Tensor, (sum_node, nested_node)
    )
    graph.output(out)

    const_tensor = torch.tensor([1.0, 2.0])
    param_attr = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    nested_tensor = torch.tensor([5.0, 6.0])

    sub_mod = torch.nn.Module()
    sub_mod.register_buffer("nested_tensor", nested_tensor)

    root = _ModuleWithNonTensorAttr(non_tensor_attr=42)
    root.register_buffer("const_tensor", const_tensor)
    root.register_parameter("param_attr", param_attr)
    root.add_module("sub", sub_mod)
    gm = torch.fx.GraphModule(root, graph)

    results = list(fx_utils.iter_tensor_get_attrs(gm))

    self.assertLen(results, 3)
    self.assertEqual(
        [(mod, name) for mod, name, _ in results],
        [
            (gm, "const_tensor"),
            (gm, "param_attr"),
            (gm, "sub.nested_tensor"),
        ],
    )
    self.assertIs(results[0][2], const_tensor)
    self.assertIs(results[1][2], param_attr)
    self.assertIs(results[2][2], nested_tensor)

  def test_iter_tensor_get_attrs_skip_targets(self):
    graph = torch.fx.Graph()
    x_node = graph.placeholder("x")
    keep_node = graph.get_attr("keep_attr")
    skip_node = graph.get_attr("skip_attr")
    sum_node = graph.call_function(
        torch.ops.aten.add.Tensor, (x_node, keep_node)
    )
    graph.output(
        graph.call_function(torch.ops.aten.add.Tensor, (sum_node, skip_node))
    )

    keep_tensor = torch.tensor([1.0, 2.0])
    skip_tensor = torch.tensor([3.0, 4.0])
    root = torch.nn.Module()
    root.register_buffer("keep_attr", keep_tensor)
    root.register_buffer("skip_attr", skip_tensor)
    gm = torch.fx.GraphModule(root, graph)

    results = list(
        fx_utils.iter_tensor_get_attrs(gm, skip_targets={"skip_attr"})
    )

    self.assertLen(results, 1)
    self.assertIs(results[0][0], gm)
    self.assertEqual(results[0][1], "keep_attr")
    self.assertIs(results[0][2], keep_tensor)

  def test_iter_tensor_get_attrs_accepts_owned_and_detached_graph(self):
    graph = torch.fx.Graph()
    x_node = graph.placeholder("x")
    const_node = graph.get_attr("const_tensor")
    graph.output(
        graph.call_function(torch.ops.aten.add.Tensor, (x_node, const_node))
    )
    const_tensor = torch.tensor([1.0, 2.0])
    root = torch.nn.Module()
    root.register_buffer("const_tensor", const_tensor)
    gm = torch.fx.GraphModule(root, graph)

    owned_results = list(fx_utils.iter_tensor_get_attrs(gm.graph))
    self.assertLen(owned_results, 1)
    self.assertIs(owned_results[0][0], gm)
    self.assertEqual(owned_results[0][1], "const_tensor")
    self.assertIs(owned_results[0][2], const_tensor)

    detached_graph = torch.fx.Graph()
    detached_graph.get_attr("const_tensor")
    self.assertEmpty(list(fx_utils.iter_tensor_get_attrs(detached_graph)))

  def test_iter_tensor_get_attrs_nested_graph_modules(self):
    child_graph = torch.fx.Graph()
    child_x = child_graph.placeholder("x")
    child_const_node = child_graph.get_attr("child_tensor")
    child_graph.output(
        child_graph.call_function(
            torch.ops.aten.add.Tensor, (child_x, child_const_node)
        )
    )
    child_tensor = torch.tensor([1.0, 2.0])
    child_root = torch.nn.Module()
    child_root.register_buffer("child_tensor", child_tensor)
    child_gm = torch.fx.GraphModule(child_root, child_graph)

    parent_graph = torch.fx.Graph()
    parent_x = parent_graph.placeholder("x")
    parent_const_node = parent_graph.get_attr("parent_tensor")
    child_out = parent_graph.call_module("child", (parent_x,))
    parent_graph.output(
        parent_graph.call_function(
            torch.ops.aten.add.Tensor, (child_out, parent_const_node)
        )
    )
    parent_tensor = torch.tensor([3.0, 4.0])
    parent_root = torch.nn.Module()
    parent_root.register_buffer("parent_tensor", parent_tensor)
    parent_root.add_module("child", child_gm)
    parent_gm = torch.fx.GraphModule(parent_root, parent_graph)

    results = list(fx_utils.iter_tensor_get_attrs(parent_gm))

    self.assertLen(results, 2)
    self.assertIs(results[0][0], parent_gm)
    self.assertEqual(results[0][1], "parent_tensor")
    self.assertIs(results[0][2], parent_tensor)
    self.assertIs(results[1][0], child_gm)
    self.assertEqual(results[1][1], "child_tensor")
    self.assertIs(results[1][2], child_tensor)


if __name__ == "__main__":
  absltest.main()
