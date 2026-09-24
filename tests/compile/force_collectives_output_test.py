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

"""Unit tests for the force_collectives_output pass."""

import operator

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import force_collectives_output
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph

# pylint: disable=protected-access
_ALL_REDUCE = torch.ops._c10d_functional.all_reduce.default
_ALL_GATHER_COALESCED = (
    torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
)
_WAIT = torch.ops._c10d_functional.wait_tensor.default
# pylint: enable=protected-access


def _all_reduce(graph: Graph, x: fx.Node) -> fx.Node:
  return graph.call_function(_ALL_REDUCE, (x, "sum", "group"))


def _all_gather_coalesced(graph: Graph, xs: list[fx.Node]) -> fx.Node:
  return graph.call_function(_ALL_GATHER_COALESCED, (xs, 1, "group"))


def _wait(graph: Graph, x: fx.Node) -> fx.Node:
  return graph.call_function(_WAIT, (x,))


def _outputs(gm: GraphModule):
  """Returns the arguments of the output node of `gm`."""
  output_node = gm.graph.find_nodes(op="output")[0]
  return output_node.args[0]


def _wrap_in_parent(submod: GraphModule) -> tuple[GraphModule, fx.Node]:
  """Returns a parent GraphModule calling `submod` and the call_module node."""
  parent_graph = Graph()
  call_args = tuple(
      parent_graph.placeholder(placeholder.name)
      for placeholder in submod.graph.find_nodes(op="placeholder")
  )
  call_submod = parent_graph.call_module("submod_0", call_args)
  parent_graph.output(call_submod)

  parent_root = torch.nn.Module()
  parent_root.submod_0 = submod
  return GraphModule(parent_root, parent_graph), call_submod


class ForceCollectivesOutputTest(seed_test_utils.RepeatableTest):

  def test_chained_collectives_do_not_return_intermediates(self):
    # The intermediate wait_tensor is consumed by the second collective, so it
    # already reaches an output and cannot be eliminated as dead code. Making
    # it an extra output would keep its buffer alive for the caller and bloat
    # memory.
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    collective0 = _all_reduce(sub_graph, x)
    wait0 = _wait(sub_graph, collective0)
    collective1 = _all_reduce(sub_graph, wait0)
    wait1 = _wait(sub_graph, collective1)
    sub_graph.output((wait1,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (wait1,))

  def test_collective_feeding_a_compute_op_is_not_returned(self):
    # The collective result is consumed by an op that is already an output, so
    # neither the collective nor its wait_tensor need to be returned.
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    collective = _all_reduce(sub_graph, x)
    wait = _wait(sub_graph, collective)
    add = sub_graph.call_function(torch.ops.aten.add.Tensor, (wait, x))
    sub_graph.output((add,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (add,))

  def test_coalesced_collective_consumed_by_getitem_is_not_returned(self):
    # Coalesced collectives return a list, so the wait_tensor consumes a
    # getitem rather than the collective itself. The collective is still alive
    # through the getitem and must not be returned.
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    collective = _all_gather_coalesced(sub_graph, [x])
    getitem = sub_graph.call_function(operator.getitem, (collective, 0))
    wait = _wait(sub_graph, getitem)
    sub_graph.output((wait,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (wait,))

  def test_dead_collective_is_returned(self):
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    add = sub_graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    collective = _all_reduce(sub_graph, x)
    wait = _wait(sub_graph, collective)
    sub_graph.output((add,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (add, wait))

  def test_dead_collective_chain_only_returns_the_last_wait(self):
    # Returning the last wait_tensor keeps the whole chain alive, so the
    # intermediate results don't need to be returned as well.
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    add = sub_graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    collective0 = _all_reduce(sub_graph, x)
    wait0 = _wait(sub_graph, collective0)
    collective1 = _all_reduce(sub_graph, wait0)
    wait1 = _wait(sub_graph, collective1)
    sub_graph.output((add,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (add, wait1))

  def test_dead_collective_without_wait_is_returned(self):
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    add = sub_graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    collective = _all_reduce(sub_graph, x)
    sub_graph.output((add,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (add, collective))

  def test_dead_wait_of_returned_collective_is_returned(self):
    # The collective is already an output but its wait_tensor is dead. The
    # wait_tensor must still run, so it has to be returned.
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    collective = _all_reduce(sub_graph, x)
    wait = _wait(sub_graph, collective)
    sub_graph.output((collective,))
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, _ = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (collective, wait))

  def test_single_non_tuple_output_is_wrapped_and_unpacked_by_getitem(self):
    sub_graph = Graph()
    x = sub_graph.placeholder("x")
    add = sub_graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    collective = _all_reduce(sub_graph, x)
    wait = _wait(sub_graph, collective)
    sub_graph.output(add)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    parent_gm, call_submod = _wrap_in_parent(submod)
    force_collectives_output.apply(parent_gm)

    self.assertEqual(_outputs(submod), (add, wait))

    # The parent used to consume a single tensor, so a getitem must be
    # inserted to unpack the newly returned tuple.
    getitem_nodes = [
        n for n in parent_gm.graph.nodes if n.target == operator.getitem
    ]
    self.assertLen(getitem_nodes, 1)
    self.assertEqual(getitem_nodes[0].args, (call_submod, 0))
    self.assertEqual(_outputs(parent_gm), getitem_nodes[0])


if __name__ == "__main__":
  absltest.main()
