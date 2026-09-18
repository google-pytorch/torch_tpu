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

"""Tests for fuse_mean_reduction FX pass."""

from unittest import mock

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import fuse_mean_reduction
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class FuseMeanReductionTest(seed_test_utils.RepeatableTest):

  def test_mean_dim_single_axis(self):
    """Verifies mean(x, dim=1, keepdim=False) -> mul(sum(x, [1], False), 1.0/8)."""
    shape = (4, 8, 16)
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty(shape, dtype=torch.float32)

    mean_node = graph.call_function(torch.ops.aten.mean.dim, (p_x, 1, False))
    mean_node.meta["val"] = torch.empty((4, 16), dtype=torch.float32)
    graph.output((mean_node,))
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_mean_reduction.apply(gm)

    mean_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mean
    ]
    sum_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.sum
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]

    self.assertEmpty(mean_nodes)
    self.assertLen(sum_nodes, 1)
    self.assertLen(mul_nodes, 1)

    sum_node = sum_nodes[0]
    mul_node = mul_nodes[0]

    self.assertEqual(sum_node.target, torch.ops.aten.sum.dim_IntList)
    self.assertEqual(sum_node.args[0], p_x)
    self.assertEqual(sum_node.args[1], [1])
    self.assertFalse(sum_node.args[2])

    self.assertEqual(mul_node.target, torch.ops.aten.mul.Scalar)
    self.assertEqual(mul_node.args[0], sum_node)
    self.assertAlmostEqual(mul_node.args[1], 1.0 / 8.0)

    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertEqual(out_node.args[0][0], mul_node)

  def test_mean_dim_negative_axis_keepdim(self):
    """Verifies mean(x, dim=-1, keepdim=True) decomposes with reciprocal 1.0/16."""
    shape = (4, 8, 16)
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty(shape, dtype=torch.float32)

    mean_node = graph.call_function(torch.ops.aten.mean.dim, (p_x, -1, True))
    mean_node.meta["val"] = torch.empty((4, 8, 1), dtype=torch.float32)
    graph.output((mean_node,))
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_mean_reduction.apply(gm)

    sum_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.sum
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]

    self.assertLen(sum_nodes, 1)
    self.assertLen(mul_nodes, 1)

    sum_node = sum_nodes[0]
    mul_node = mul_nodes[0]

    self.assertEqual(sum_node.target, torch.ops.aten.sum.dim_IntList)
    self.assertEqual(sum_node.args[0], p_x)
    self.assertEqual(sum_node.args[1], [-1])
    self.assertTrue(sum_node.args[2])

    self.assertEqual(mul_node.target, torch.ops.aten.mul.Scalar)
    self.assertEqual(mul_node.args[0], sum_node)
    self.assertAlmostEqual(mul_node.args[1], 1.0 / 16.0)

  def test_mean_dim_multiple_axes(self):
    """Verifies reduction across multiple axes: (0, 2)."""
    shape = (4, 8, 16)
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty(shape, dtype=torch.float32)

    mean_node = graph.call_function(
        torch.ops.aten.mean.dim, (p_x, [0, 2], False)
    )
    mean_node.meta["val"] = torch.empty((8,), dtype=torch.float32)
    graph.output((mean_node,))
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_mean_reduction.apply(gm)

    sum_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.sum
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]

    self.assertLen(sum_nodes, 1)
    self.assertLen(mul_nodes, 1)

    sum_node = sum_nodes[0]
    mul_node = mul_nodes[0]

    self.assertEqual(sum_node.target, torch.ops.aten.sum.dim_IntList)
    self.assertEqual(sum_node.args[0], p_x)
    self.assertEqual(sum_node.args[1], [0, 2])
    self.assertAlmostEqual(mul_node.args[1], 1.0 / (4.0 * 16.0))

  def test_mean_default_all_axes(self):
    """Verifies mean.default reduces all dimensions with reciprocal 1.0/512."""
    shape = (4, 8, 16)
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty(shape, dtype=torch.float32)

    mean_node = graph.call_function(torch.ops.aten.mean.default, (p_x,))
    mean_node.meta["val"] = torch.empty((), dtype=torch.float32)
    graph.output((mean_node,))
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_mean_reduction.apply(gm)

    mean_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mean
    ]
    sum_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.sum
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]

    self.assertEmpty(mean_nodes)
    self.assertLen(sum_nodes, 1)
    self.assertLen(mul_nodes, 1)

    sum_node = sum_nodes[0]
    mul_node = mul_nodes[0]

    self.assertEqual(sum_node.target, torch.ops.aten.sum.default)
    self.assertEqual(sum_node.args[0], p_x)
    self.assertEqual(mul_node.target, torch.ops.aten.mul.Scalar)
    self.assertEqual(mul_node.args[0], sum_node)
    self.assertAlmostEqual(mul_node.args[1], 1.0 / (4.0 * 8.0 * 16.0))

  def test_mean_dim_with_dtype_kwarg(self):
    """Verifies dtype kwarg is forwarded to sum."""
    shape = (4, 8)
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty(shape, dtype=torch.float32)

    mean_node = graph.call_function(
        torch.ops.aten.mean.dim,
        (p_x, [1], True),
        kwargs={"dtype": torch.float64},
    )
    mean_node.meta["val"] = torch.empty((4, 1), dtype=torch.float64)
    graph.output((mean_node,))
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_mean_reduction.apply(gm)

    sum_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.sum
    ]
    self.assertLen(sum_nodes, 1)
    self.assertEqual(sum_nodes[0].kwargs.get("dtype"), torch.float64)

  def test_mean_dynamic_dimension_untouched(self):
    """Verifies that dynamic reduction dimensions are NOT rewritten."""
    graph = Graph()
    p_x = graph.placeholder("x")
    mock_val = mock.MagicMock()
    # Dim 0 is a SymInt, Dim 1 is static 8
    mock_sym = mock.MagicMock(spec=torch.SymInt)
    mock_val.shape = (mock_sym, 8)
    p_x.meta["val"] = mock_val

    # Reducing dynamic dim 0 should be skipped
    mean_node = graph.call_function(torch.ops.aten.mean.dim, (p_x, [0], False))
    graph.output((mean_node,))
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_mean_reduction.apply(gm)

    mean_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mean
    ]
    self.assertLen(mean_nodes, 1)

  def test_numerical_correctness(self):
    """Verifies transformed module produces same numerical results as eager mean."""

    class MeanModule(torch.nn.Module):

      def forward(self, x):
        m1 = torch.mean(x, dim=1, keepdim=True)
        m2 = torch.mean(x, dim=(0, 2), keepdim=False)
        m3 = torch.mean(x)
        return m1, m2, m3

    x = torch.randn(4, 8, 16, dtype=torch.float32)
    mod = MeanModule()
    expected = mod(x)

    traced_gm = fx.symbolic_trace(mod)
    # Populate fake tensor meta
    for node in traced_gm.graph.nodes:
      if node.op == "placeholder":
        node.meta["val"] = torch.empty_like(x)
    # Propagate shapes with ShapeProp
    fx.passes.shape_prop.ShapeProp(traced_gm).propagate(x)

    fuse_mean_reduction.apply(traced_gm)

    actual = traced_gm(x)
    for exp, act in zip(expected, actual):
      test_utils.assert_close(act, exp, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
