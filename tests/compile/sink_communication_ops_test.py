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

"""Unit tests for sink_communication_ops FX pass."""

from absl.testing import absltest
import torch
import torch.fx
from torch_tpu._internal.compile.fx_passes import sink_communication_ops
from tests import seed_test_utils

Graph = torch.fx.Graph
GraphModule = torch.fx.GraphModule


class SinkCommunicationOpsTest(seed_test_utils.RepeatableTest):

  def _get_c10d_ops(self):
    """Returns functional c10d collective and wait ops."""
    if hasattr(torch.ops, "_c10d_functional") and hasattr(
        torch.ops._c10d_functional, "all_gather_into_tensor"
    ):
      ag_op = torch.ops._c10d_functional.all_gather_into_tensor.default
      wait_op = torch.ops._c10d_functional.wait_tensor.default
    else:
      ag_op = torch.ops.c10d_functional.all_gather_into_tensor.default
      wait_op = torch.ops.c10d_functional.wait_tensor.default
    return ag_op, wait_op

  def test_raise_comms(self):
    ag_op, wait_op = self._get_c10d_ops()

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")

    # Initial order: heavy compute before comm start
    c1 = graph.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
    c2 = graph.call_function(torch.ops.aten.add.Tensor, (c1, 3.0))
    ag = graph.call_function(ag_op, (w, "0", 2))
    wt = graph.call_function(wait_op, (ag,))
    out = graph.call_function(torch.ops.aten.add.Tensor, (c2, wt))
    graph.output(out)

    sink_communication_ops.raise_comms(graph)

    node_order = [n.name for n in graph.nodes if n.op == "call_function"]
    # Collective start (ag) should be raised before independent compute (c1, c2)
    self.assertEqual(node_order[0], ag.name)

  def test_sink_waits(self):
    ag_op, wait_op = self._get_c10d_ops()

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")

    ag = graph.call_function(ag_op, (w, "0", 2))
    wt = graph.call_function(wait_op, (ag,))
    # Independent compute ops placed after wait initially
    c1 = graph.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
    c2 = graph.call_function(torch.ops.aten.add.Tensor, (c1, 3.0))
    out = graph.call_function(torch.ops.aten.add.Tensor, (c2, wt))
    graph.output(out)

    sink_communication_ops.sink_waits(graph)

    node_order = [n.name for n in graph.nodes if n.op == "call_function"]
    # wt should be sunk after independent compute (c1, c2) and immediately before out
    self.assertIn(c1.name, node_order)
    self.assertIn(c2.name, node_order)
    c2_idx = node_order.index(c2.name)
    wt_idx = node_order.index(wt.name)
    self.assertGreater(wt_idx, c2_idx)

  def test_reorder_communication_preserving_peak_memory(self):
    ag_op, wait_op = self._get_c10d_ops()

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")

    # Initial order: c1 -> c2 -> ag -> wt -> view -> out
    c1 = graph.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
    c2 = graph.call_function(torch.ops.aten.add.Tensor, (c1, 3.0))
    ag = graph.call_function(ag_op, (w, "0", 2))
    wt = graph.call_function(wait_op, (ag,))
    vw = graph.call_function(torch.ops.aten.view.default, (wt, [-1]))
    out = graph.call_function(torch.ops.aten.add.Tensor, (c2, vw))
    graph.output(out)

    sink_communication_ops.reorder_communication_preserving_peak_memory(graph)

    node_order = [n.name for n in graph.nodes if n.op == "call_function"]
    # ag should be raised first, then compute (c1, c2), then wt -> vw -> out
    self.assertEqual(node_order[0], ag.name)
    ag_idx = node_order.index(ag.name)
    c1_idx = node_order.index(c1.name)
    c2_idx = node_order.index(c2.name)
    wt_idx = node_order.index(wt.name)
    vw_idx = node_order.index(vw.name)
    out_idx = node_order.index(out.name)

    self.assertLess(ag_idx, c1_idx)
    self.assertLess(c1_idx, wt_idx)
    self.assertLess(c2_idx, wt_idx)
    self.assertLess(wt_idx, vw_idx)
    self.assertLess(vw_idx, out_idx)

  def test_sink_comm_into_subgraphs(self):
    ag_op, wait_op = self._get_c10d_ops()

    # Submodule graph: takes x and gathered w, performs matmul
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_w = sub_graph.placeholder("w_gathered")
    sub_out = sub_graph.call_function(torch.ops.aten.mm.default, (sub_x, sub_w))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    # Parent graph: invokes subgraph HOP
    parent_graph = Graph()
    px = parent_graph.placeholder("x")
    pw = parent_graph.placeholder("w_sharded")

    ag = parent_graph.call_function(ag_op, (pw, "0", 2))
    wt = parent_graph.call_function(wait_op, (ag,))
    vw = parent_graph.call_function(torch.ops.aten.view.default, (wt, [4, 4]))

    hop = torch.ops.higher_order.invoke_subgraph
    call_sub = parent_graph.call_function(
        hop,
        (submod, "sub_id_0", px, vw),
    )
    parent_graph.output(call_sub)

    parent_gm = GraphModule(torch.nn.Module(), parent_graph)
    parent_gm.submod_0 = submod

    sink_communication_ops.apply(parent_gm)

    # Verify that collective nodes (ag, wt, vw) were removed from root graph
    call_functions = [
        n.name for n in parent_gm.graph.nodes if n.op == "call_function"
    ]
    self.assertNotIn(ag.name, call_functions)
    self.assertNotIn(wt.name, call_functions)
    self.assertNotIn(vw.name, call_functions)

    # The hop should now take raw input pw
    hop_node = [n for n in parent_gm.graph.nodes if n.op == "call_function"][0]
    self.assertIn(pw, hop_node.args)

    # A new submodule should be registered with ingested comms
    self.assertTrue(hasattr(parent_gm, "subgraph_ingested_comm_0"))
    ingested_submod = getattr(parent_gm, "subgraph_ingested_comm_0")
    sub_ops = [
        str(n.target)
        for n in ingested_submod.graph.nodes
        if n.op == "call_function"
    ]
    self.assertTrue(any("all_gather" in op for op in sub_ops))
    self.assertTrue(any("wait_tensor" in op for op in sub_ops))
    self.assertTrue(any("view" in op for op in sub_ops))


if __name__ == "__main__":
  absltest.main()
