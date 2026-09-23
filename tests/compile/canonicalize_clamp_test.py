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

"""Tests for canonicalize_clamp FX pass."""

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import canonicalize_clamp
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class CanonicalizeClampTest(seed_test_utils.RepeatableTest):

  def test_clamp_min_zero_canonicalized_to_relu(self):
    graph = Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.clamp_min.default, (x, 0))
    graph.output((node,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0].target, torch.ops.aten.relu.default)
    self.assertEqual(nodes[0].args, (x,))

  def test_clamp_zero_none_canonicalized_to_relu(self):
    graph = Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.clamp.default, (x, 0, None))
    graph.output((node,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0].target, torch.ops.aten.relu.default)
    self.assertEqual(nodes[0].args, (x,))

  def test_clamp_min_nonzero_not_canonicalized(self):
    graph = Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.clamp_min.default, (x, 1.5))
    graph.output((node,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0].target, torch.ops.aten.clamp_min.default)

  def test_clamp_with_max_not_canonicalized(self):
    graph = Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.clamp.default, (x, 0, 6.0))
    graph.output((node,))
    gm = GraphModule(torch.nn.Module(), graph)

    canonicalize_clamp.apply(gm)

    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0].target, torch.ops.aten.clamp.default)


if __name__ == "__main__":
  absltest.main()
