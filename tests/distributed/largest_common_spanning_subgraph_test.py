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

"""Tests `largest_common_spanning_subgraph` with input validation turned on.

The BUILD target sets `TORCH_TPU_ENABLE_DEBUG_CHECKS=1`, which the `*_raise`
tests rely on.
"""

import graphlib
import itertools
import operator
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch.fx.passes import split_module
from torch_tpu._internal.distributed import largest_common_spanning_subgraph as lcs
from tests import seed_test_utils


class _Model(torch.nn.Module):
  """Traces to a graph with every FX op kind."""

  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(4, 4)
    self.register_buffer("scale", torch.ones(4))

  def forward(self, x, y):
    h = torch.sin(x) + y.cos()
    return self.linear(h * self.scale).relu()


def _traced_model() -> tuple[fx.Graph, dict[str, fx.Node]]:
  graph = fx.symbolic_trace(_Model()).graph
  return graph, {node.name: node for node in graph.nodes}


def _submod_shape(submod: fx.GraphModule) -> list[tuple[object, ...]]:
  """Returns each node's op, target, and args, with nodes replaced by indices.

  Placeholder names and device indices are left out since they may differ
  between ranks.
  """
  index = {node: i for i, node in enumerate(submod.graph.nodes)}

  def leaf(value: object) -> object:
    if isinstance(value, fx.Node):
      return index[value]
    if isinstance(value, torch.device):
      return value.type
    return type(value), value

  return [
      (
          node.op,
          None if node.op == "placeholder" else node.target,
          fx.node.map_aggregate((node.args, node.kwargs), leaf),
      )
      for node in submod.graph.nodes
  ]


def _assert_valid_split_modules(
    graphs: list[fx.Graph],
    partitions: list[dict[str, int]],
) -> None:
  """Checks that `split_module` builds the same submodules on every graph.

  Also checks that every graph can run the shared submodules in ID order.
  """
  num_shared = max(partitions[0].values(), default=-1) + 1
  shapes_per_graph = []
  for graph, partition_map in zip(graphs, partitions, strict=True):
    if set(partition_map.values()) != set(range(num_shared)):
      raise AssertionError(
          f"{partition_map} does not use IDs 0 to {num_shared - 1}."
      )
    unshared_ids = itertools.count(num_shared)
    full_map = {
        node.name: (
            partition_map[node.name]
            if node.name in partition_map
            else next(unshared_ids)
        )
        for node in graph.nodes
        if node.op not in ("placeholder", "get_attr", "output")
    }
    gm = fx.GraphModule(graph.owning_module or torch.nn.Module(), graph)
    split_gm = split_module.split_module(
        gm, gm, lambda node, full_map=full_map: full_map[node.name]
    )
    submods = [
        split_gm.graph.find_nodes(
            op="call_module", target=f"submod_{partition_id}"
        )[0]
        for partition_id in range(num_shared)
    ]

    # Running the shared submodules in ID order adds an edge from each one to
    # the next, which must not create a cycle.
    sorter = graphlib.TopologicalSorter(
        {node: node.all_input_nodes for node in split_gm.graph.nodes}
    )
    for previous, current in itertools.pairwise(submods):
      sorter.add(current, previous)
    sorter.prepare()  # Raises `graphlib.CycleError` otherwise.

    shapes_per_graph.append(
        [_submod_shape(getattr(split_gm, submod.target)) for submod in submods]
    )

  for shapes in shapes_per_graph[1:]:
    for partition_id, (shape, reference) in enumerate(
        zip(shapes, shapes_per_graph[0])
    ):
      if shape != reference:
        raise AssertionError(
            f"submod_{partition_id} differs across graphs:"
            f" {shape} != {reference}"
        )


def _shared_partitions(
    graphs: list[fx.Graph],
    anchors: list[tuple[fx.Node, ...]],
) -> list[dict[str, int]]:
  partitions = lcs.largest_common_spanning_subgraph(graphs, anchors)
  _assert_valid_split_modules(graphs, partitions)
  return partitions


class LargestCommonSpanningSubgraphTest(seed_test_utils.RepeatableTest):

  def test_is_partitionable(self):
    graph, _ = _traced_model()

    self.assertEqual(
        {node.op: lcs.is_partitionable(node) for node in graph.nodes},
        {
            "placeholder": False,
            "get_attr": False,
            "call_function": True,
            "call_method": True,
            "call_module": True,
            "output": False,
        },
    )

  def test_identical_graphs_span_all_compute_nodes(self):
    (g0, n0), (g1, n1) = _traced_model(), _traced_model()

    partitions = _shared_partitions([g0, g1], [(n0["add"], n1["add"])])

    # The `scale` buffer is a `get_attr` input, so it stays outside.
    expected = {"sin": 0, "cos": 0, "add": 0, "mul": 0, "linear": 0, "relu": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_node_name_shifts_do_not_break_matching(self):
    def build(suffix):
      g = fx.Graph()
      x = g.placeholder("x" + suffix)
      s = g.call_function(torch.sin, (x,), name="sin" + suffix)
      g.output(g.call_function(torch.cos, (s,)))
      return g, s

    (g0, s0), (g1, s1) = build(""), build("_1")

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    self.assertEqual(partitions, [{"sin": 0, "cos": 0}, {"sin_1": 0, "cos": 0}])

  @parameterized.named_parameters(
      ("same", (4,), torch.float32, "meta", {"sin": 0, "add": 0}),
      ("different_input_shapes", (1,), torch.float32, "meta", {"sin": 0}),
      ("different_output_dtypes", (4,), torch.bfloat16, "meta", {"sin": 0}),
      ("different_output_device_types", (4,), torch.float32, "cpu", {"sin": 0}),
  )
  def test_shapes_dtypes_and_device_types_must_match(
      self, y_shape1, dtype1, device1, expected
  ):
    def build(y_shape, dtype, device):
      g = fx.Graph()
      x, y = g.placeholder("x"), g.placeholder("y")
      s = g.call_function(torch.sin, (x,))
      a = g.call_function(torch.add, (s, y))
      x.meta["val"] = torch.empty(4, device="meta")
      s.meta["val"] = torch.empty(4, device="meta")
      y.meta["val"] = torch.empty(y_shape, device="meta")
      a.meta["val"] = torch.empty(4, dtype=dtype, device=device)
      g.output(a)
      return g, s

    g0, s0 = build((4,), torch.float32, "meta")
    g1, s1 = build(y_shape1, dtype1, device1)

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    self.assertEqual(partitions, [expected, expected])

  @parameterized.named_parameters(
      ("equal", 1, 1, True),
      ("different_values", 1, 2, False),
      ("int_and_float", 1, 1.0, False),
      ("int_and_bool", 1, True, False),
      ("tuple_and_list", (1, 2), [1, 2], False),
      ("equal_slices", slice(0, 2), slice(0, 2), True),
      ("different_slices", slice(0, 2), slice(0, 3), False),
      ("device_index", torch.device("cpu", 0), torch.device("cpu"), True),
      ("device_type", torch.device("cpu"), torch.device("meta"), False),
  )
  def test_constant_args_compare_by_type_and_value(
      self, const0, const1, matched
  ):
    def build(const):
      g = fx.Graph()
      a = g.call_function(torch.add, (g.placeholder("x"), const))
      r = g.call_function(torch.relu, (a,))
      g.output(r)
      return g, r

    (g0, r0), (g1, r1) = build(const0), build(const1)

    partitions = _shared_partitions([g0, g1], [(r0, r1)])

    expected = {"add": 0, "relu": 0} if matched else {"relu": 0}
    self.assertEqual(partitions, [expected, expected])

  @parameterized.named_parameters(
      ("kwarg", torch.add, lambda x, s: ((x,), {"other": s})),
      ("list", torch.cat, lambda x, s: (([x, s],), {})),
      ("repeated", torch.add, lambda x, s: ((s, s), {})),
      ("slice", operator.getitem, lambda x, s: ((x, slice(None, s)), {})),
  )
  def test_node_inputs_anywhere_in_args_are_followed(self, target, make_args):
    def build():
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      user = g.call_function(target, *make_args(x, s))
      g.output(user)
      return g, user

    (g0, u0), (g1, u1) = build(), build()

    partitions = _shared_partitions([g0, g1], [(u0, u1)])

    expected = {"sin": 0, u0.name: 0}
    self.assertEqual(partitions, [expected, expected])

  def test_kwargs_order_must_match(self):
    g0 = fx.Graph()
    x0, y0 = g0.placeholder("x"), g0.placeholder("y")
    a0 = g0.call_function(torch.add, (), {"input": x0, "other": y0})
    r0 = g0.call_function(torch.relu, (a0,))
    g0.output(r0)

    g1 = fx.Graph()
    x1, y1 = g1.placeholder("x"), g1.placeholder("y")
    a1 = g1.call_function(torch.add, (), {"other": y1, "input": x1})
    r1 = g1.call_function(torch.relu, (a1,))
    g1.output(r1)

    partitions = _shared_partitions([g0, g1], [(r0, r1)])

    # `split_module` orders submodule placeholders by `kwargs` insertion order,
    # so reversed `kwargs` keys must not match.
    self.assertEqual(partitions, [{"relu": 0}, {"relu": 0}])

  def test_divergent_node_stops_traversal(self):
    def build(before_op, after_op):
      g = fx.Graph()
      b = g.call_function(before_op, (g.placeholder("x"),))
      a = g.call_function(torch.add, (b, 1))
      r = g.call_function(torch.relu, (a,))
      m = g.call_function(after_op, (r, 2))
      g.output(g.call_function(torch.cos, (m,)))
      return g, r

    g0, r0 = build(torch.sin, torch.mul)
    g1, r1 = build(torch.tanh, torch.sub)

    partitions = _shared_partitions([g0, g1], [(r0, r1)])

    expected = {"add": 0, "relu": 0}
    self.assertEqual(partitions, [expected, expected])

  @parameterized.named_parameters(
      ("without_anchor", ["sin"], {"sin": 0}),
      ("anchor_after_sin", ["sin", "add"], {"sin": 0, "add": 1, "cos": 1}),
      ("anchor_before_sin", ["add", "sin"], {"sin": 0, "add": 0, "cos": 0}),
  )
  def test_ambiguous_users_are_paired_only_through_an_anchor(
      self, anchor_names, expected
  ):
    def build():
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      add_a = g.call_function(torch.add, (s, 1))
      add_b = g.call_function(torch.add, (s, 1))
      cos = g.call_function(torch.cos, (add_a,))
      g.output((cos, g.call_function(torch.tanh, (add_b,))))
      return g, {"sin": s, "add": add_a}

    (g0, n0), (g1, n1) = build(), build()

    # Growth from `sin` cannot tell its two `add` users apart, so only an
    # anchor pairs one. Growth from that `add` reaches `sin` through its input
    # and takes it in, unless `sin` already started its own partition.
    anchors = [(n0[name], n1[name]) for name in anchor_names]
    partitions = _shared_partitions([g0, g1], anchors)

    self.assertEqual(partitions, [expected, expected])

  def test_users_reading_in_different_slots_are_told_apart(self):
    def build():
      g = fx.Graph()
      x, y = g.placeholder("x"), g.placeholder("y")
      s = g.call_function(torch.sin, (x,))
      g.output((
          g.call_function(torch.sub, (s, y)),
          g.call_function(torch.sub, (y, s)),
      ))
      return g, s

    (g0, s0), (g1, s1) = build(), build()

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    # The `sub` users read `sin` in different slots, so neither is ambiguous.
    expected = {"sin": 0, "sub": 0, "sub_1": 0}
    self.assertEqual(partitions, [expected, expected])

  @parameterized.named_parameters(
      ("one_graph", [torch.cos], {"sin": 0, "relu": 0, "cos": 0}),
      (
          "three_graphs",
          [torch.cos, torch.cos, torch.tanh],
          {"sin": 0, "relu": 0},
      ),
  )
  def test_partitions_are_common_to_all_graphs(self, last_ops, expected):
    def build(last_op):
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      r = g.call_function(torch.relu, (s,))
      g.output(g.call_function(last_op, (r,)))
      return g, r

    graphs, anchor = zip(*(build(op) for op in last_ops))

    partitions = _shared_partitions(list(graphs), [anchor])

    self.assertEqual(partitions, [expected] * len(last_ops))

  def test_no_anchors_give_no_partitions(self):
    (g0, _), (g1, _) = _traced_model(), _traced_model()

    partitions = lcs.largest_common_spanning_subgraph([g0, g1], [])

    self.assertEqual(partitions, [{}, {}])

  def test_validation_skipped_without_debug_checks(self):
    (g0, n0), (g1, n1) = _traced_model(), _traced_model()
    anchor = (n0["add"], n1["add"])

    # Duplicate anchors raise in `_validate_inputs`, which is skipped here.
    with mock.patch.object(lcs, "_debug_checks_enabled", return_value=False):
      partitions = _shared_partitions([g0, g1], [anchor, anchor])

    expected = {"sin": 0, "cos": 0, "add": 0, "mul": 0, "linear": 0, "relu": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_partitions_are_the_same_without_debug_checks(self):
    (g0, n0), (g1, n1) = _traced_model(), _traced_model()

    with mock.patch.object(lcs, "_debug_checks_enabled", return_value=False):
      partitions = _shared_partitions([g0, g1], [(n0["add"], n1["add"])])

    expected = {"sin": 0, "cos": 0, "add": 0, "mul": 0, "linear": 0, "relu": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_unconditional_checks_raise_without_debug_checks(self):
    (g0, n0), (g1, n1) = _traced_model(), _traced_model()

    with mock.patch.object(lcs, "_debug_checks_enabled", return_value=False):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python input validation.
          ValueError, "non-empty"
      ):
        lcs.largest_common_spanning_subgraph([], [])
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python input validation.
          ValueError, "do not match"
      ):
        lcs.largest_common_spanning_subgraph([g0, g1], [(n0["sin"], n1["cos"])])

  def test_empty_graphs_raise(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python input validation.
        ValueError, "non-empty"
    ):
      lcs.largest_common_spanning_subgraph([], [])

  @parameterized.parameters("placeholder", "get_attr", "output")
  def test_anchors_on_unpartitioned_nodes_raise(self, op):
    (g0, _), (g1, _) = _traced_model(), _traced_model()
    anchor = (g0.find_nodes(op=op)[0], g1.find_nodes(op=op)[0])

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python input validation.
        ValueError, "does not partition"
    ):
      lcs.largest_common_spanning_subgraph([g0, g1], [anchor])

  @parameterized.named_parameters(
      ("wrong_length", lambda a, b: [(a,)], "has length 1, expected 2"),
      ("other_graph", lambda a, b: [(a, a)], "does not belong to graphs"),
      ("shared_node", lambda a, b: [(a, b), (a, b)], "must not share nodes"),
  )
  def test_malformed_anchors_raise(self, make_anchors, message):
    (g0, n0), (g1, n1) = _traced_model(), _traced_model()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python input validation.
        ValueError, message
    ):
      lcs.largest_common_spanning_subgraph(
          [g0, g1], make_anchors(n0["add"], n1["add"])
      )

  def test_anchors_whose_nodes_do_not_match_raise(self):
    (g0, n0), (g1, n1) = _traced_model(), _traced_model()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python input validation.
        ValueError, "do not match"
    ):
      lcs.largest_common_spanning_subgraph([g0, g1], [(n0["sin"], n1["cos"])])

  def test_partitions_are_not_cut_at_unrelated_graph_specific_nodes(self):
    def build(with_extra_op):
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      extra = [g.call_function(torch.tanh, (x,))] if with_extra_op else []
      c = g.call_function(torch.cos, (s,))
      g.output((c, *extra))
      return g, s

    g0, s0 = build(with_extra_op=True)
    g1, s1 = build(with_extra_op=False)

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    expected = {"sin": 0, "cos": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_node_with_graph_specific_user_stops_downstream_growth(self):
    def build(with_extra_user):
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      c = g.call_function(torch.cos, (s,))
      r = g.call_function(torch.relu, (c,))
      extra = [g.call_function(torch.tanh, (c,))] if with_extra_user else []
      g.output((r, *extra))
      return g, s

    g0, s0 = build(with_extra_user=True)
    g1, s1 = build(with_extra_user=False)

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    # `{sin, cos}` returns `cos` on both graphs. Adding `relu` would make `cos`
    # an output only on `g0`, where `tanh` still reads it.
    expected = {"sin": 0, "cos": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_independent_partitions_need_not_keep_node_order(self):
    g0 = fx.Graph()
    x0, y0 = g0.placeholder("x"), g0.placeholder("y")
    s0 = g0.call_function(torch.sin, (x0,))
    c0 = g0.call_function(torch.cos, (y0,))
    g0.output((s0, c0))

    g1 = fx.Graph()
    x1, y1 = g1.placeholder("x"), g1.placeholder("y")
    c1 = g1.call_function(torch.cos, (y1,))
    s1 = g1.call_function(torch.sin, (x1,))
    g1.output((s1, c1))

    partitions = _shared_partitions([g0, g1], [(s0, s1), (c0, c1)])

    # Either partition can run first.
    sin_first, cos_first = {"sin": 0, "cos": 1}, {"sin": 1, "cos": 0}
    self.assertIn(partitions, ([sin_first] * 2, [cos_first] * 2))

  def test_partition_ids_stay_dense_after_a_skipped_anchor(self):
    def build(add_reads_sin_twice):
      g = fx.Graph()
      x, y = g.placeholder("x"), g.placeholder("y")
      s = g.call_function(torch.sin, (x,))
      a = g.call_function(torch.add, (s, s if add_reads_sin_twice else y))
      c = g.call_function(torch.cos, (a,))
      g.output((s, c))
      return g, s, a, c

    g0, s0, a0, c0 = build(add_reads_sin_twice=True)
    g1, s1, a1, c1 = build(add_reads_sin_twice=False)

    partitions = _shared_partitions([g0, g1], [(s0, s1), (a0, a1), (c0, c1)])

    # `add` reads different inputs on each graph, so its anchor is skipped.
    expected = {"sin": 0, "cos": 1}
    self.assertEqual(partitions, [expected, expected])

  @parameterized.named_parameters(
      ("shared_input", torch.add, "sx", "sy"),
      ("repeated_input", torch.addcmul, "sxx", "sxy"),
  )
  def test_partition_stops_where_input_reads_diverge(self, op, reads0, reads1):
    def build(reads):
      g = fx.Graph()
      x, y = g.placeholder("x"), g.placeholder("y")
      s = g.call_function(torch.sin, (x,))
      nodes = {"s": s, "x": x, "y": y}
      g.output(g.call_function(op, tuple(nodes[name] for name in reads)))
      return g, s

    (g0, s0), (g1, s1) = build(reads0), build(reads1)

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    # Adding the op would wire the submodule inputs differently on each graph.
    self.assertEqual(partitions, [{"sin": 0}, {"sin": 0}])

  def test_node_reading_members_in_swapped_slots_stays_out(self):
    def build(swap):
      g = fx.Graph()
      x, y = g.placeholder("x"), g.placeholder("y")
      s = g.call_function(torch.sin, (x,))
      c = g.call_function(torch.cos, (y,))
      a = g.call_function(torch.add, (s, c))
      # Differs per graph, so it stays out. It reads `sin`, `cos`, and `add`
      # before `addcmul` does, which fixes the order of the partition's outputs.
      t = g.call_function(torch.stack if swap else torch.cat, ([s, c, a],))
      m = g.call_function(torch.addcmul, (c, s, a) if swap else (s, c, a))
      g.output((t, m))
      return g, a

    (g0, a0), (g1, a1) = build(swap=False), build(swap=True)

    partitions = _shared_partitions([g0, g1], [(a0, a1)])

    # `addcmul` pairs up through `add` but reads `sin` and `cos` in swapped
    # slots, so its submodule would compute different values per graph.
    expected = {"sin": 0, "cos": 0, "add": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_divergent_bypass_path_prevents_cycle(self):
    def build(scale_op):
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      a = g.call_function(torch.add, (s, g.call_function(scale_op, (s, 2))))
      g.output(a)
      return g, s

    g0, s0 = build(torch.mul)
    g1, s1 = build(torch.div)

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    # Putting `sin` and `add` together would leave `mul` or `div` both reading
    # from and feeding into the partition.
    self.assertEqual(partitions, [{"sin": 0}, {"sin": 0}])

  def test_asymmetric_diamond_reaches_join_after_longer_branch(self):
    def build():
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      c = g.call_function(torch.cos, (s,))
      n = g.call_function(torch.neg, (g.call_function(torch.tanh, (s,)),))
      g.output(g.call_function(torch.add, (c, n)))
      return g, s

    (g0, s0), (g1, s1) = build(), build()

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    # `add` can only join after `neg`, or `neg` would both read from and feed
    # the partition.
    expected = {"sin": 0, "cos": 0, "tanh": 0, "neg": 0, "add": 0}
    self.assertEqual(partitions, [expected, expected])

  def test_internal_node_order_must_match(self):
    g0 = fx.Graph()
    x0, y0 = g0.placeholder("x"), g0.placeholder("y")
    s0 = g0.call_function(torch.sin, (x0,))
    c0 = g0.call_function(torch.cos, (y0,))
    a0 = g0.call_function(torch.add, (s0, c0))
    g0.output(a0)

    g1 = fx.Graph()
    x1, y1 = g1.placeholder("x"), g1.placeholder("y")
    c1 = g1.call_function(torch.cos, (y1,))
    s1 = g1.call_function(torch.sin, (x1,))
    a1 = g1.call_function(torch.add, (s1, c1))
    g1.output(a1)

    partitions = _shared_partitions([g0, g1], [(a0, a1)])

    # `sin` and `cos` run in a different order on each graph, so only one of
    # them can join `add`.
    with_sin, with_cos = {"sin": 0, "add": 0}, {"cos": 0, "add": 0}
    self.assertIn(partitions, ([with_sin] * 2, [with_cos] * 2))

  def test_swapped_external_output_read_order_stops_growth(self):
    g0 = fx.Graph()
    s0 = g0.call_function(torch.sin, (g0.placeholder("x"),))
    c0 = g0.call_function(torch.cos, (s0,))
    u0 = g0.call_function(torch.sum, (s0,))
    v0 = g0.call_function(torch.relu, (c0,))
    g0.output((u0, v0))

    g1 = fx.Graph()
    s1 = g1.call_function(torch.sin, (g1.placeholder("x"),))
    c1 = g1.call_function(torch.cos, (s1,))
    v1 = g1.call_function(torch.neg, (c1,))
    u1 = g1.call_function(torch.tanh, (s1,))
    g1.output((v1, u1))

    partitions = _shared_partitions([g0, g1], [(s0, s1)])

    # `sin` is read first outside `{sin, cos}` on `g0` and `cos` is on `g1`, so
    # the submodule would return them in a different order.
    self.assertEqual(partitions, [{"sin": 0}, {"sin": 0}])

  def test_split_module_orders_outputs_by_first_outside_read(self):
    g = fx.Graph()
    x = g.placeholder("x")
    s = g.call_function(torch.sin, (x,))
    c = g.call_function(torch.cos, (x,))
    t = g.call_function(torch.tanh, (x,))
    # Outside `{sin, cos, tanh}`, `sub` reads `cos` and then `sin`. The graph
    # output reads `tanh` after that, even though it's in the first slot.
    g.output((t, g.call_function(torch.sub, (c, s))))
    gm = fx.GraphModule(torch.nn.Module(), g)
    members = {"sin", "cos", "tanh"}

    split_gm = split_module.split_module(
        gm, gm, lambda node: 0 if node.name in members else 1
    )

    (output,) = split_gm.submod_0.graph.find_nodes(op="output")
    self.assertEqual(
        [node.target for node in output.args[0]],
        [torch.cos, torch.sin, torch.tanh],
    )

  @parameterized.named_parameters(
      ("one_anchor", 1, {"sin": 0}),
      ("two_anchors", 2, {"sin": 0, "cos": 1}),
  )
  def test_outputs_read_in_swapped_slots_stop_growth(
      self, num_anchors, expected
  ):
    def build(return_sin_first):
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      c = g.call_function(torch.cos, (s,))
      g.output((s, c) if return_sin_first else (c, s))
      return g, s, c

    g0, s0, c0 = build(return_sin_first=True)
    g1, s1, c1 = build(return_sin_first=False)

    # With an anchor on `cos` too, growth from `sin` still cannot take it in.
    anchors = [(s0, s1), (c0, c1)][:num_anchors]
    partitions = _shared_partitions([g0, g1], anchors)

    self.assertEqual(partitions, [expected, expected])

  @parameterized.named_parameters(
      ("sin_anchor_first", True),
      ("sin_anchor_last", False),
  )
  def test_anchor_nodes_pair_only_with_their_anchor(self, sin_first):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    s0 = g0.call_function(torch.sin, (x0,))
    u0 = g0.call_function(torch.cos, (x0,))
    g0.output((u0, g0.call_function(torch.cos, (s0,))))

    g1 = fx.Graph()
    s1 = g1.call_function(torch.sin, (g1.placeholder("x"),))
    v1 = g1.call_function(torch.cos, (s1,))
    g1.output(v1)

    # `v1` belongs to the other anchor, so growth from `sin` must not pair it
    # with the `cos` that reads `sin` on `g0`, whichever anchor comes first.
    anchors = [(s0, s1), (u0, v1)] if sin_first else [(u0, v1), (s0, s1)]
    partitions = _shared_partitions([g0, g1], anchors)

    expected = {"sin": 0, "cos": 1}
    self.assertEqual(partitions, [expected, expected])

  def test_partitions_run_in_an_order_every_graph_supports(self):
    def build(relu_reads_cos):
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      c = g.call_function(torch.cos, (x,))
      r = g.call_function(torch.relu, (c if relu_reads_cos else x,))
      g.output((g.call_function(torch.add, (s, r)), c))
      return g, s, c

    g0, s0, c0 = build(relu_reads_cos=False)
    g1, s1, c1 = build(relu_reads_cos=True)

    partitions = _shared_partitions([g0, g1], [(s0, s1), (c0, c1)])

    # On `g1`, `cos` feeds `add` through `relu`, so `cos` must run first even
    # though its anchor comes second.
    expected = {"cos": 0, "sin": 1, "add": 1}
    self.assertEqual(partitions, [expected, expected])

  def test_anchor_that_cannot_share_a_run_order_is_skipped(self):
    def build(sin_first):
      g = fx.Graph()
      x = g.placeholder("x")
      if sin_first:
        s = g.call_function(torch.sin, (x,))
        c = g.call_function(torch.cos, (g.call_function(torch.relu, (s,)),))
      else:
        c = g.call_function(torch.cos, (x,))
        s = g.call_function(torch.sin, (g.call_function(torch.relu, (c,)),))
      g.output((s, c))
      return g, s, c

    g0, s0, c0 = build(sin_first=True)
    g1, s1, c1 = build(sin_first=False)

    partitions = _shared_partitions([g0, g1], [(s0, s1), (c0, c1)])

    # `sin` feeds `cos` on `g0` and `cos` feeds `sin` on `g1`, so the later
    # anchor, `cos`, is left out.
    self.assertEqual(partitions, [{"sin": 0}, {"sin": 0}])

  def test_graph_object_passed_twice_counts_as_two_graphs(self):
    g = fx.Graph()
    first = g.call_function(torch.sin, (g.placeholder("x"),))
    relu = g.call_function(torch.relu, (first,))
    second = g.call_function(torch.sin, (relu,))
    g.output(second)

    partitions = _shared_partitions([g, g], [(second, first)])

    # `relu` runs before the partition on the first copy and after it on the
    # second, which is only a cycle if both copies share one `relu` vertex.
    self.assertEqual(partitions, [{"sin_1": 0}, {"sin": 0}])

  def test_run_order_conflict_across_three_graphs_skips_last_anchor(self):
    ops = (torch.sin, torch.cos, torch.tanh)

    def build(first, second, third):
      # `first` feeds `second` through `relu`, and `third` only reads `x`.
      g = fx.Graph()
      x = g.placeholder("x")
      nodes = {first: g.call_function(first, (x,))}
      relu = g.call_function(torch.relu, (nodes[first],))
      nodes[second] = g.call_function(second, (relu,))
      nodes[third] = g.call_function(third, (x,))
      g.output(tuple(nodes[op] for op in ops))
      return g, nodes

    g0, n0 = build(torch.sin, torch.cos, torch.tanh)
    g1, n1 = build(torch.cos, torch.tanh, torch.sin)
    g2, n2 = build(torch.tanh, torch.sin, torch.cos)

    anchors = [(n0[op], n1[op], n2[op]) for op in ops]
    partitions = _shared_partitions([g0, g1, g2], anchors)

    # Any two graphs share a run order, but all three need `sin` before `cos`
    # before `tanh` before `sin`, so the last anchor, `tanh`, is left out.
    expected = {"sin": 0, "cos": 1}
    self.assertEqual(partitions, [expected] * 3)


if __name__ == "__main__":
  absltest.main()
