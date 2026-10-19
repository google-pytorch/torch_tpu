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

"""Unit tests for process_group_utils."""

import operator

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
from torch.fx.experimental import proxy_tensor
from torch.testing._internal.distributed import fake_pg
from torch_tpu._internal.distributed import process_group_utils
from tests import seed_test_utils

FakeStore = fake_pg.FakeStore
make_fx = proxy_tensor.make_fx
ProcessGroupId = process_group_utils.ProcessGroupId


class ProcessGroupUtilsTest(seed_test_utils.RepeatableTest):

  def setUp(self) -> None:
    super().setUp()
    self.world_size = 4
    if not dist.is_initialized():
      dist.init_process_group(
          backend="fake",
          store=FakeStore(),
          rank=0,
          world_size=self.world_size,
      )

  def tearDown(self) -> None:
    if dist.is_initialized():
      dist.destroy_process_group()
    super().tearDown()

  def test_extract_process_group_id_traced_string(self) -> None:

    class DistributedModule(torch.nn.Module):

      def forward(self, x):
        reduced = fc.all_reduce(x, "sum", group=dist.group.WORLD)
        return fc.wait_tensor(reduced)

    module = DistributedModule()
    example_input = torch.ones(2, 2)
    gm = make_fx(module)(example_input)

    all_reduce_nodes = [
        n
        for n in gm.graph.nodes
        if process_group_utils._get_collective_op(n)
        == torch.ops._c10d_functional.all_reduce
    ]
    self.assertLen(all_reduce_nodes, 1)
    node = all_reduce_nodes[0]
    self.assertIsInstance(node.args[2], str)

    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(
        pg_id, ProcessGroupId(range(self.world_size), self.world_size)
    )

  def test_extract_process_group_id_process_group(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", dist.group.WORLD),
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(
        pg_id, ProcessGroupId(range(self.world_size), self.world_size)
    )

  def test_extract_process_group_id_int(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", 0),
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(
        pg_id, ProcessGroupId(range(self.world_size), self.world_size)
    )

  def test_get_collective_op_valid(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node_packet = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", [0, 1]),
    )
    node_overload = graph.call_function(
        torch.ops._c10d_functional.all_reduce.default,
        args=(x, "sum", "0"),
    )
    self.assertEqual(
        process_group_utils._get_collective_op(node_packet),
        torch.ops._c10d_functional.all_reduce,
    )
    self.assertEqual(
        process_group_utils._get_collective_op(node_overload),
        torch.ops._c10d_functional.all_reduce,
    )

  def test_get_collective_op_non_collective(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node_add = graph.call_function(operator.add, args=(x, x))
    self.assertIsNone(process_group_utils._get_collective_op(x))
    self.assertIsNone(process_group_utils._get_collective_op(node_add))

  def test_extract_process_group_id_from_kwargs_group_name(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum"),
        kwargs={"group_name": [0, 1]},
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(pg_id, ProcessGroupId((0, 1), self.world_size))

  def test_extract_process_group_id_from_kwargs_group(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum"),
        kwargs={"group": range(4)},
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(pg_id, ProcessGroupId((0, 1, 2, 3), self.world_size))

  def test_extract_process_group_id_positional_all_reduce(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    # In all_reduce schema: (Tensor self, str reduceOp, str group_name)
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", [0, 1]),
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(pg_id, ProcessGroupId((0, 1), self.world_size))

  def test_extract_process_group_id_positional_reduce_scatter_tensor(
      self,
  ) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    # In reduce_scatter_tensor schema:
    # (Tensor input, str reduceOp, int group_size, str group_name)
    node = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor,
        args=(x, "sum", 2, [0, 1]),
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(pg_id, ProcessGroupId((0, 1), self.world_size))

  def test_extract_process_group_id_experimental_dim_collectives(self) -> None:
    # The tpu:: dim collectives are recognized as collectives, and their group
    # is found from the positional group_name argument, which sits at a
    # different index in each op's schema.
    group_args = (1, self.world_size, dist.group.WORLD.group_name)
    for op, op_args in (
        (torch.ops.tpu._all_gather_dim, group_args),
        (torch.ops.tpu._reduce_scatter_dim, ("sum", *group_args)),
    ):
      with self.subTest(op=op):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(op, args=(x, *op_args))
        self.assertEqual(process_group_utils._get_collective_op(node), op)
        pg_id = process_group_utils._extract_process_group_id_from_node(node)
        self.assertEqual(
            pg_id, ProcessGroupId(range(self.world_size), self.world_size)
        )

  def test_extract_process_group_id_str(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", "0"),
    )
    pg_id = process_group_utils._extract_process_group_id_from_node(node)
    self.assertEqual(
        pg_id, ProcessGroupId(range(self.world_size), self.world_size)
    )

  def test_get_num_collectives_per_pg_empty_graph(self) -> None:
    class EmptyModule(torch.nn.Module):

      def forward(self, x):
        return x

    traced = torch.fx.symbolic_trace(EmptyModule())
    counts = process_group_utils.get_num_collectives_per_pg(traced)
    self.assertEqual(counts, {})

  def test_get_num_collectives_per_pg_multiple_collectives(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node1 = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", [0, 1]),
    )
    node2 = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(node1, "sum", [0, 1]),
    )
    node3 = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(node2, "sum", [2, 3]),
    )
    graph.output(node3)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    counts = process_group_utils.get_num_collectives_per_pg(gm)
    pg1 = ProcessGroupId((0, 1), self.world_size)
    pg2 = ProcessGroupId((2, 3), self.world_size)
    self.assertEqual(counts, {pg1: 2, pg2: 1})

  def test_get_num_collectives_per_pg_traced_model(self) -> None:
    class MultiCollectiveModel(torch.nn.Module):

      def forward(self, x):
        x1 = fc.all_reduce(x, "sum", group=dist.group.WORLD)
        x1_w = fc.wait_tensor(x1)
        x2 = fc.all_reduce(x1_w, "sum", group=dist.group.WORLD)
        return fc.wait_tensor(x2)

    gm = make_fx(MultiCollectiveModel())(torch.ones(2, 2))
    counts = process_group_utils.get_num_collectives_per_pg(gm)
    global_pg = ProcessGroupId(
        range(self.world_size), world_size=self.world_size
    )
    self.assertEqual(counts, {global_pg: 2})


class ProcessGroupIdTest(seed_test_utils.RepeatableTest):
  """Unit tests for ProcessGroupId."""

  def test_valid_process_group_id(self) -> None:
    pg = ProcessGroupId([0, 1, 2])
    self.assertEqual(pg.ranks, (0, 1, 2))
    self.assertIsNone(pg.world_size)

  def test_valid_with_world_size(self) -> None:
    pg = ProcessGroupId([0, 2, 3], world_size=4)
    self.assertEqual(pg.ranks, (0, 2, 3))
    self.assertEqual(pg.world_size, 4)

  def test_hash_and_equality(self) -> None:
    pg1 = ProcessGroupId([0, 1])
    pg2 = ProcessGroupId([0, 1])
    pg3 = ProcessGroupId([0, 2])
    self.assertEqual(pg1, pg2)
    self.assertNotEqual(pg1, pg3)
    self.assertEqual(hash(pg1), hash(pg2))
    self.assertEqual({pg1: "val"}[pg2], "val")
    # Contiguous groups take an O(1) equality path.
    self.assertEqual(ProcessGroupId(range(4)), ProcessGroupId([0, 1, 2, 3]))
    self.assertNotEqual(ProcessGroupId(range(4)), ProcessGroupId(range(1, 5)))
    self.assertNotEqual(ProcessGroupId(range(4)), ProcessGroupId([0, 1, 2, 4]))

  def test_iteration_len_indexing(self) -> None:
    pg = ProcessGroupId([0, 1, 3])
    self.assertLen(pg, 3)
    self.assertEqual(list(pg), [0, 1, 3])
    self.assertEqual(pg[0], 0)
    self.assertEqual(pg[1], 1)
    self.assertEqual(pg[2], 3)

  def test_contains(self) -> None:
    pg = ProcessGroupId([1, 4, 7])
    self.assertIn(4, pg)
    self.assertNotIn(5, pg)
    self.assertNotIn(0, pg)
    self.assertNotIn(8, pg)

  def test_is_contiguous(self) -> None:
    self.assertTrue(ProcessGroupId(range(3, 9)).is_contiguous())
    self.assertFalse(ProcessGroupId([0, 2]).is_contiguous())

  @parameterized.parameters(([3, 4, 5, 6],), ([0, 2, 5],), ([7],))
  def test_to_wire_and_from_wire(self, ranks: list[int]) -> None:
    pg = ProcessGroupId(ranks)
    self.assertEqual(ProcessGroupId.from_wire(pg.to_wire()), pg)

  def test_to_wire_contiguous_is_constant_size(self) -> None:
    self.assertEqual(
        ProcessGroupId(range(4096)).to_wire(), {"start": 0, "stop": 4096}
    )

  def test_from_wire_shares_instances(self) -> None:
    wire = ProcessGroupId(range(16)).to_wire()
    self.assertIs(
        ProcessGroupId.from_wire(wire), ProcessGroupId.from_wire(wire)
    )


if __name__ == "__main__":
  absltest.main()
