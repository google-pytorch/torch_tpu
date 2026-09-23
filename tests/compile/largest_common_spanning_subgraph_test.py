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

"""Unit tests for largest_common_spanning_subgraph."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch_tpu._internal.compile import largest_common_spanning_subgraph as lcs
from tests import seed_test_utils


def _find_node_by_target(graph: fx.Graph, target: object) -> fx.Node:
  for node in graph.nodes:
    if node.target == target:
      return node
  raise AssertionError(f"Target {target!r} not found in graph.")


def _build_twin_adds_graph() -> tuple[fx.Graph, fx.Node, fx.Node]:
  """Builds `sin` feeding twin `add(., 1)`s that feed `cos` and `tanh`.

  Returns:
    The graph, the `sin` node and the `add` twin that feeds `cos`.
  """
  g = fx.Graph()
  x = g.placeholder("x")
  anchor = g.call_function(torch.sin, (x,))
  branch_a = g.call_function(torch.add, (anchor, 1))
  branch_b = g.call_function(torch.add, (anchor, 1))
  out_a = g.call_function(torch.cos, (branch_a,))
  out_b = g.call_function(torch.tanh, (branch_b,))
  g.output((out_a, out_b))
  return g, anchor, branch_a


def _only_op(graph: fx.Graph) -> fx.Node:
  (node,) = [n for n in graph.nodes if n.op not in ("placeholder", "output")]
  return node


class LargestCommonSpanningSubgraphTest(seed_test_utils.RepeatableTest):

  def test_identical_graphs_span_everything(self):
    def fn(x, y):
      a = torch.sin(x)
      b = torch.cos(y)
      c = torch.add(a, b)
      d = torch.relu(c)
      return torch.neg(d)

    g0 = fx.symbolic_trace(fn).graph
    g1 = fx.symbolic_trace(fn).graph
    anchor = (
        _find_node_by_target(g0, torch.add),
        _find_node_by_target(g1, torch.add),
    )

    segments = lcs.largest_common_spanning_subgraph([g0, g1], [anchor])

    expected = {"sin": 0, "cos": 0, "add": 0, "relu": 0, "neg": 0}
    self.assertEqual(segments, [expected, expected])

  def test_node_name_shifts_do_not_break_matching(self):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    s0 = g0.call_function(torch.sin, (x0,))
    c0 = g0.call_function(torch.cos, (s0,))
    g0.output(c0)

    g1 = fx.Graph()
    x1 = g1.placeholder("x_rank1")
    transient = g1.call_function(torch.sin, (x1,))
    g1.erase_node(transient)
    s1 = g1.call_function(torch.sin, (x1,))
    c1 = g1.call_function(torch.cos, (s1,))
    g1.output(c1)

    self.assertNotEqual(s0.name, s1.name)
    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(s0, s1)])
    self.assertEqual(segments, [{"sin": 0, "cos": 0}, {"sin_1": 0, "cos": 0}])

  @parameterized.named_parameters(
      ("output_shape", None, torch.empty(8), None, torch.empty(6)),
      (
          "dtype",
          None,
          torch.empty(8),
          None,
          torch.empty(8, dtype=torch.bfloat16),
      ),
      (
          "input_shape",
          torch.empty(4, 8),
          torch.empty(8),
          torch.empty(2, 8),
          torch.empty(8),
      ),
  )
  def test_shape_differences_do_not_prevent_alignment(
      self,
      x0_val: torch.Tensor | None,
      sum0_val: torch.Tensor,
      x1_val: torch.Tensor | None,
      sum1_val: torch.Tensor,
  ):
    graphs, sums = [], []
    for x_val, sum_val in ((x0_val, sum0_val), (x1_val, sum1_val)):
      g = fx.Graph()
      x = g.placeholder("x")
      x.meta["val"] = x_val
      s = g.call_function(torch.sum, (x, 0))
      s.meta["val"] = sum_val
      g.output(s)
      graphs.append(g)
      sums.append(s)

    segments = lcs.largest_common_spanning_subgraph(graphs, [tuple(sums)])
    self.assertEqual(segments, [{s.name: 0} for s in sums])

  @parameterized.named_parameters(
      ("equal", 1, 1, True),
      ("different_values", 1, 2, False),
      ("int_and_float", 1, 1.0, False),
      ("int_and_bool", 1, True, False),
  )
  def test_constants_compare_by_type_and_value(self, const0, const1, aligned):
    graphs = []
    for const in (const0, const1):
      g = fx.Graph()
      g.output(g.call_function(torch.add, (g.placeholder("x"), const)))
      graphs.append(g)

    segments = lcs.largest_common_spanning_subgraph(
        graphs, [tuple(_only_op(g) for g in graphs)]
    )

    expected = {"add": 0} if aligned else {}
    self.assertEqual(segments, [expected, expected])

  def test_device_index_is_not_compared(self):
    graphs = []
    for device in (torch.device("cpu", 0), torch.device("cpu")):
      g = fx.Graph()
      g.output(g.call_function(torch.zeros, ((2,),), {"device": device}))
      graphs.append(g)

    segments = lcs.largest_common_spanning_subgraph(
        graphs, [tuple(_only_op(g) for g in graphs)]
    )

    self.assertEqual(segments, [{"zeros": 0}, {"zeros": 0}])

  def test_node_inputs_in_kwargs_are_followed(self):
    def build():
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      a = g.call_function(torch.add, (x,), {"other": s})
      g.output(a)
      return g, a

    (g0, a0), (g1, a1) = build(), build()
    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(a0, a1)])

    expected = {"sin": 0, "add": 0}
    self.assertEqual(segments, [expected, expected])

  def test_node_inputs_nested_in_lists_are_followed(self):
    def build():
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      c = g.call_function(torch.cos, (x,))
      cat = g.call_function(torch.cat, ([s, c],))
      g.output(cat)
      return g, cat

    (g0, cat0), (g1, cat1) = build(), build()
    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(cat0, cat1)])

    expected = {"sin": 0, "cos": 0, "cat": 0}
    self.assertEqual(segments, [expected, expected])

  def test_user_reading_a_common_node_twice_is_matched(self):
    def build():
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      g.output(g.call_function(torch.add, (s, s)))
      return g, s

    (g0, s0), (g1, s1) = build(), build()
    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(s0, s1)])

    expected = {"sin": 0, "add": 0}
    self.assertEqual(segments, [expected, expected])

  def test_divergent_node_stops_traversal(self):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    pre_div0 = g0.call_function(torch.sin, (x0,))
    pre_common0 = g0.call_function(torch.add, (pre_div0, 1))
    anchor0 = g0.call_function(torch.relu, (pre_common0,))
    post_div0 = g0.call_function(torch.mul, (anchor0, 2))
    tail0 = g0.call_function(torch.cos, (post_div0,))
    g0.output(tail0)

    g1 = fx.Graph()
    x1 = g1.placeholder("x")
    pre_div1 = g1.call_function(torch.tanh, (x1,))
    pre_common1 = g1.call_function(torch.add, (pre_div1, 1))
    anchor1 = g1.call_function(torch.relu, (pre_common1,))
    post_div1 = g1.call_function(torch.sub, (anchor1, 2))
    tail1 = g1.call_function(torch.cos, (post_div1,))
    g1.output(tail1)

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(anchor0, anchor1)]
    )

    expected = {"add": 0, "relu": 0}
    self.assertEqual(segments, [expected, expected])

  def test_anchors_on_placeholders_or_outputs_are_ignored(self):
    def build():
      g = fx.Graph()
      x = g.placeholder("x")
      n = g.call_function(torch.relu, (x,))
      out = g.output(n)
      return g, x, n, out

    g0, x0, n0, out0 = build()
    g1, x1, n1, out1 = build()

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(x0, x1), (out0, out1), (n0, n1)]
    )

    self.assertEqual(segments, [{"relu": 0}, {"relu": 0}])

  def test_anchor_with_different_ops_is_ignored(self):
    def build(last_op):
      g = fx.Graph()
      s = g.call_function(torch.sin, (g.placeholder("x"),))
      last = g.call_function(last_op, (s,))
      g.output(last)
      return g, s, last

    g0, s0, last0 = build(torch.tanh)
    g1, s1, last1 = build(torch.cos)

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(last0, last1), (s0, s1)]
    )

    # The first anchor should be skipped; the second should still grow the
    # region.
    self.assertEqual(segments, [{"sin": 0}, {"sin": 0}])

  def test_anchors_in_one_region_give_one_segment(self):
    def fn(x):
      a = torch.sin(x)
      b = torch.relu(a)
      c = torch.cos(b)
      return c

    g0 = fx.symbolic_trace(fn).graph
    g1 = fx.symbolic_trace(fn).graph

    anchor_a = (
        _find_node_by_target(g0, torch.sin),
        _find_node_by_target(g1, torch.sin),
    )
    anchor_c = (
        _find_node_by_target(g0, torch.cos),
        _find_node_by_target(g1, torch.cos),
    )

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [anchor_a, anchor_c]
    )

    expected = {"sin": 0, "relu": 0, "cos": 0}
    self.assertEqual(segments, [expected, expected])

  def test_regions_from_different_anchors_join(self):
    g0, anchor0, branch_a0 = _build_twin_adds_graph()
    g1, anchor1, branch_a1 = _build_twin_adds_graph()

    # From `sin` alone the twin `add`s are ambiguous (see
    # test_ambiguous_neighbor_is_not_matched). An anchor on one twin should
    # reach `sin` through its input, so the regions join.
    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(anchor0, anchor1), (branch_a0, branch_a1)]
    )

    # The other twin sits between `add` and `cos`, so they should be two
    # segments.
    expected = {"sin": 0, "add": 0, "cos": 1}
    self.assertEqual(segments, [expected, expected])

  def test_two_anchors_stay_separate_across_divergence(self):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    a0 = g0.call_function(torch.sin, (x0,))
    mid0 = g0.call_function(torch.mul, (a0, 2))
    b0 = g0.call_function(torch.cos, (mid0,))
    g0.output(b0)

    g1 = fx.Graph()
    x1 = g1.placeholder("x")
    a1 = g1.call_function(torch.sin, (x1,))
    mid1 = g1.call_function(torch.div, (a1, 2))
    b1 = g1.call_function(torch.cos, (mid1,))
    g1.output(b1)

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(a0, a1), (b0, b1)]
    )

    expected = {"sin": 0, "cos": 1}
    self.assertEqual(segments, [expected, expected])

  def test_three_graphs_intersect(self):
    def build(last_op):
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      r = g.call_function(torch.relu, (s,))
      last = g.call_function(last_op, (r,))
      g.output(last)
      return g, r

    g0, r0 = build(torch.cos)
    g1, r1 = build(torch.cos)
    g2, r2 = build(torch.tanh)

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1, g2], [(r0, r1, r2)]
    )

    expected = {"sin": 0, "relu": 0}
    self.assertEqual(segments, [expected, expected, expected])

  def test_ambiguous_neighbor_is_not_matched(self):
    g0, anchor0, _ = _build_twin_adds_graph()
    g1, anchor1, _ = _build_twin_adds_graph()

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(anchor0, anchor1)]
    )

    self.assertEqual(segments, [{"sin": 0}, {"sin": 0}])

  def test_diamond_reached_twice_is_stable(self):
    def fn(x):
      root = torch.sin(x)
      left = torch.cos(root)
      right = torch.tanh(root)
      join = torch.add(left, right)
      return join

    g0 = fx.symbolic_trace(fn).graph
    g1 = fx.symbolic_trace(fn).graph
    anchor = (
        _find_node_by_target(g0, torch.add),
        _find_node_by_target(g1, torch.add),
    )

    segments = lcs.largest_common_spanning_subgraph([g0, g1], [anchor])

    expected = {"sin": 0, "cos": 0, "tanh": 0, "add": 0}
    self.assertEqual(segments, [expected, expected])

  def test_empty_graphs_raise(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        ValueError, "non-empty"
    ):
      lcs.largest_common_spanning_subgraph([], [])

  def test_invalid_anchors_raise(self):
    g0, g1 = fx.Graph(), fx.Graph()
    n0 = g0.call_function(torch.sin, (g0.placeholder("x"),))
    g0.output(n0)
    g1.output(g1.call_function(torch.sin, (g1.placeholder("x"),)))

    # 1. An anchor needs one node per graph.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        ValueError, "has length 1, expected 2"
    ):
      lcs.largest_common_spanning_subgraph([g0, g1], [(n0,)])

    # 2. Each anchor node must belong to its graph.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing input validation.
        ValueError, "does not belong to graphs"
    ):
      lcs.largest_common_spanning_subgraph([g0, g1], [(n0, n0)])

  def test_segments_are_cut_at_rank_specific_nodes(self):
    def build(with_extra_op):
      g = fx.Graph()
      x = g.placeholder("x")
      s = g.call_function(torch.sin, (x,))
      c = g.call_function(torch.cos, (s,))
      extra = [g.call_function(torch.tanh, (x,))] if with_extra_op else []
      r = g.call_function(torch.relu, (c,))
      n = g.call_function(torch.neg, (r,))
      g.output((n, *extra))
      return g, c

    g0, c0 = build(with_extra_op=True)
    g1, c1 = build(with_extra_op=False)

    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(c0, c1)])

    expected = {"sin": 0, "cos": 0, "relu": 1, "neg": 1}
    self.assertEqual(segments, [expected, expected])

  def test_node_with_rank_specific_user_is_dropped(self):
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

    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(s0, s1)])

    # cos would be a partition output on g0 only.
    expected = {"sin": 0, "relu": 1}
    self.assertEqual(segments, [expected, expected])

  def test_node_with_differently_aligned_inputs_is_dropped(self):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    s0 = g0.call_function(torch.sin, (x0,))
    a0 = g0.call_function(torch.add, (s0, s0))
    g0.output(a0)

    g1 = fx.Graph()
    x1 = g1.placeholder("x")
    s1 = g1.call_function(torch.sin, (x1,))
    c1 = g1.call_function(torch.cos, (x1,))
    a1 = g1.call_function(torch.add, (s1, c1))
    g1.output(a1)

    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(a0, a1)])

    self.assertEqual(segments, [{"sin": 0}, {"sin": 0}])

  def test_differently_ordered_nodes_give_no_segments(self):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    y0 = g0.placeholder("y")
    s0 = g0.call_function(torch.sin, (x0,))
    c0 = g0.call_function(torch.cos, (y0,))
    g0.output((s0, c0))

    g1 = fx.Graph()
    x1 = g1.placeholder("x")
    y1 = g1.placeholder("y")
    c1 = g1.call_function(torch.cos, (y1,))
    s1 = g1.call_function(torch.sin, (x1,))
    g1.output((s1, c1))

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(s0, s1), (c0, c1)]
    )

    # No segments at all: as separate segments, `sin` and `cos` would run in a
    # different order on the two graphs.
    self.assertEqual(segments, [{}, {}])

  def test_segment_ids_stay_dense_after_a_dropped_segment(self):
    def build(shared_input, with_extra_ops):
      g = fx.Graph()
      x = g.placeholder("x")
      y = g.placeholder("y")
      a = g.call_function(torch.sin, (x,))
      extra = [g.call_function(torch.tanh, (x,))] if with_extra_ops else []
      b = g.call_function(torch.add, (x, x if shared_input else y))
      extra += [g.call_function(torch.neg, (x,))] if with_extra_ops else []
      c = g.call_function(torch.cos, (b,))
      g.output((a, c, *extra))
      return g, a, b

    g0, a0, b0 = build(shared_input=True, with_extra_ops=True)
    g1, a1, b1 = build(shared_input=False, with_extra_ops=False)

    segments = lcs.largest_common_spanning_subgraph(
        [g0, g1], [(a0, a1), (b0, b1)]
    )

    # The extra ops on g0 cut `sin`, `add` and `cos` into three segments, and
    # `add` reads its inputs differently, so its segment should be dropped and
    # `cos` should be segment 1, not 2.
    expected = {"sin": 0, "cos": 1}
    self.assertEqual(segments, [expected, expected])

  def test_segment_with_different_input_sharing_is_dropped(self):
    g0 = fx.Graph()
    x0 = g0.placeholder("x")
    a0 = g0.call_function(torch.add, (x0, x0))
    g0.output(a0)

    g1 = fx.Graph()
    x1 = g1.placeholder("x")
    y1 = g1.placeholder("y")
    a1 = g1.call_function(torch.add, (x1, y1))
    g1.output(a1)

    segments = lcs.largest_common_spanning_subgraph([g0, g1], [(a0, a1)])

    # The partition would have one input on g0 but two on g1.
    self.assertEqual(segments, [{}, {}])


if __name__ == "__main__":
  absltest.main()
