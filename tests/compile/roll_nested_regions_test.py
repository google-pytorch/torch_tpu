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

"""Unit tests for roll_nested_regions FX pass."""

from absl.testing import absltest
import torch
import torch.fx
from torch_tpu._internal.compile.fx_passes import roll_nested_regions
from tests import seed_test_utils

Graph = torch.fx.Graph
GraphModule = torch.fx.GraphModule


class RollNestedRegionsTest(seed_test_utils.RepeatableTest):

  def test_isomorphic_subgraphs_rolled(self):
    # Create two isomorphic subgraphs (submod_0, submod_1) doing y = x + w
    def make_submod():
      sub_g = Graph()
      sx = sub_g.placeholder("x")
      sw = sub_g.placeholder("w")
      sout = sub_g.call_function(torch.ops.aten.add.Tensor, (sx, sw))
      sub_g.output(sout)
      return GraphModule(torch.nn.Module(), sub_g)

    submod0 = make_submod()
    submod1 = make_submod()

    parent_g = Graph()
    px = parent_g.placeholder("x")
    pw0 = parent_g.placeholder("w0")
    pw1 = parent_g.placeholder("w1")

    hop = torch.ops.higher_order.invoke_subgraph
    call0 = parent_g.call_function(hop, (submod0, "sub_id_0", px, pw0))
    call1 = parent_g.call_function(hop, (submod1, "sub_id_1", call0, pw1))
    parent_g.output(call1)

    parent_gm = GraphModule(torch.nn.Module(), parent_g)
    parent_gm.submod_0 = submod0
    parent_gm.submod_1 = submod1

    roll_nested_regions.apply(parent_gm)

    # Verify that a scan call was generated
    call_functions = [
        str(n.target) for n in parent_gm.graph.nodes if n.op == "call_function"
    ]
    self.assertTrue(any("scan" in fn for fn in call_functions))

  def test_subgraph_with_get_attr(self):
    # Subgraphs with internal get_attr constant tensors in addition to scanned param
    def make_submod_with_attr(val):
      sub_g = Graph()
      sx = sub_g.placeholder("x")
      sw = sub_g.placeholder("w")
      c = sub_g.get_attr("const_weight")
      tmp = sub_g.call_function(torch.ops.aten.add.Tensor, (sx, sw))
      sout = sub_g.call_function(torch.ops.aten.add.Tensor, (tmp, c))
      sub_g.output(sout)
      root = torch.nn.Module()
      root.const_weight = torch.tensor(val)
      return GraphModule(root, sub_g)

    submod0 = make_submod_with_attr(1.0)
    submod1 = make_submod_with_attr(2.0)

    parent_g = Graph()
    px = parent_g.placeholder("x")
    pw0 = parent_g.placeholder("w0")
    pw1 = parent_g.placeholder("w1")

    hop = torch.ops.higher_order.invoke_subgraph
    call0 = parent_g.call_function(hop, (submod0, "sub_id_0", px, pw0))
    call1 = parent_g.call_function(hop, (submod1, "sub_id_1", call0, pw1))
    parent_g.output(call1)

    parent_gm = GraphModule(torch.nn.Module(), parent_g)
    parent_gm.submod_0 = submod0
    parent_gm.submod_1 = submod1

    roll_nested_regions.apply(parent_gm)

    call_functions = [
        str(n.target) for n in parent_gm.graph.nodes if n.op == "call_function"
    ]
    self.assertTrue(any("scan" in fn for fn in call_functions))

  def test_differing_subgraphs_not_rolled(self):
    # Submod 0 does add, Submod 1 does mul
    sub_g0 = Graph()
    sx0 = sub_g0.placeholder("x")
    sw0 = sub_g0.placeholder("w")
    sout0 = sub_g0.call_function(torch.ops.aten.add.Tensor, (sx0, sw0))
    sub_g0.output(sout0)
    submod0 = GraphModule(torch.nn.Module(), sub_g0)

    sub_g1 = Graph()
    sx1 = sub_g1.placeholder("x")
    sw1 = sub_g1.placeholder("w")
    sout1 = sub_g1.call_function(torch.ops.aten.mul.Tensor, (sx1, sw1))
    sub_g1.output(sout1)
    submod1 = GraphModule(torch.nn.Module(), sub_g1)

    parent_g = Graph()
    px = parent_g.placeholder("x")
    pw0 = parent_g.placeholder("w0")
    pw1 = parent_g.placeholder("w1")

    hop = torch.ops.higher_order.invoke_subgraph
    call0 = parent_g.call_function(hop, (submod0, "sub_id_0", px, pw0))
    call1 = parent_g.call_function(hop, (submod1, "sub_id_1", call0, pw1))
    parent_g.output(call1)

    parent_gm = GraphModule(torch.nn.Module(), parent_g)
    parent_gm.submod_0 = submod0
    parent_gm.submod_1 = submod1

    roll_nested_regions.apply(parent_gm)

    call_functions = [
        str(n.target) for n in parent_gm.graph.nodes if n.op == "call_function"
    ]
    # No scan should be generated because subgraphs are not isomorphic
    self.assertFalse(any("scan" in fn for fn in call_functions))


if __name__ == "__main__":
  absltest.main()
