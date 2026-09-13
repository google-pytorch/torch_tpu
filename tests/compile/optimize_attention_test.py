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

"""Unit tests for optimize_attention pass."""

import operator
from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import optimize_attention
from tests import seed_test_utils


class OptimizeAttentionTest(seed_test_utils.RepeatableTest):

  def test_eliminate_empty_mask(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 32

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    empty_mask = graph.call_function(torch.ops.aten.empty.memory_format, ([0],))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, empty_mask, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    # Mask should now be None
    self.assertIsNone(node.args[3])
    self.assertFalse(node.args[5])

    # The empty mask allocation should have been eliminated
    empty_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.empty.memory_format
    ]
    self.assertEmpty(empty_nodes)

  def test_eliminate_all_zero_additive_mask(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 32

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    zero_mask = graph.call_function(
        torch.ops.aten.zeros.default, ([batch, heads, seq_len, seq_len],)
    )
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, zero_mask, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertIsNone(sdpa_nodes[0].args[3])

    zero_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.zeros.default
    ]
    self.assertEmpty(zero_nodes)

  def test_eliminate_all_true_boolean_mask(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 32

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("m")

    # m is passed as all True boolean mask
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.ones((batch, heads, seq_len, seq_len), dtype=torch.bool),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertIsNone(sdpa_nodes[0].args[3])

  def test_preserve_non_trivial_mask(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 32

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")
    m = graph.placeholder("m")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, m, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    non_trivial_mask = torch.ones(
        (batch, heads, seq_len, seq_len), dtype=torch.bool
    )
    non_trivial_mask[0, 0, :, 0] = False

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        non_trivial_mask,
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertIsNotNone(sdpa_nodes[0].args[3])

  def test_fold_query_scale_multiplication_with_explicit_scale(self):
    batch, heads, seq_len, head_dim = 1, 6, 197, 64
    scale_factor = 64**-0.5  # 0.125

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    # timm/ViT pattern: q = q * scale, sdpa(q, k, v, scale=1.0)
    q_scaled = graph.call_function(torch.ops.aten.mul.Scalar, (q, scale_factor))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q_scaled, k, v, None, 0.0, False),
        kwargs={"scale": 1.0},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    # Query input should now be original q
    self.assertIs(node.args[0], q)
    # Scale should be folded to 0.125
    self.assertAlmostEqual(node.kwargs["scale"], 0.125, places=5)

    # The mul node should have been eliminated
    mul_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.mul.Scalar
    ]
    self.assertEmpty(mul_nodes)

  def test_fold_query_scale_multiplication_with_default_scale(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 64

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    q_scaled = graph.call_function(torch.ops.aten.mul.Scalar, (q, 0.5))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q_scaled, k, v, None, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    self.assertIs(node.args[0], q)
    # Default scale is 1/sqrt(64) = 0.125; with 0.5 pre-scale -> 0.0625
    self.assertAlmostEqual(node.kwargs["scale"], 0.0625, places=5)

    mul_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.mul.Scalar
    ]
    self.assertEmpty(mul_nodes)

  def test_fold_query_scale_division(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 64

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    q_scaled = graph.call_function(torch.ops.aten.div.Scalar, (q, 8.0))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q_scaled, k, v, None, 0.0, False),
        kwargs={"scale": 1.0},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    self.assertIs(node.args[0], q)
    self.assertAlmostEqual(node.kwargs["scale"], 0.125, places=5)

    div_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.div.Scalar
    ]
    self.assertEmpty(div_nodes)

  def test_fold_both_query_and_key_scale(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 64

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    q_scaled = graph.call_function(torch.ops.aten.mul.Scalar, (q, 0.5))
    k_scaled = graph.call_function(torch.ops.aten.mul.Scalar, (k, 0.25))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q_scaled, k_scaled, v, None, 0.0, False),
        kwargs={"scale": 1.0},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    self.assertIs(node.args[0], q)
    self.assertIs(node.args[1], k)
    # 0.5 * 0.25 * 1.0 = 0.125
    self.assertAlmostEqual(node.kwargs["scale"], 0.125, places=5)

    mul_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.mul.Scalar
    ]
    self.assertEmpty(mul_nodes)

  def test_fold_chained_multiplications(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 64

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    q1 = graph.call_function(torch.ops.aten.mul.Scalar, (q, 0.5))
    q2 = graph.call_function(torch.ops.aten.mul.Scalar, (q1, 0.25))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q2, k, v, None, 0.0, False),
        kwargs={"scale": 1.0},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    self.assertIs(node.args[0], q)
    self.assertAlmostEqual(node.kwargs["scale"], 0.125, places=5)

    mul_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.mul.Scalar
    ]
    self.assertEmpty(mul_nodes)

  def test_canonicalize_scale_node(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 64

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    c_64 = graph.call_function(torch.ops.aten.scalar_tensor.default, (64.0,))
    c_sqrt = graph.call_function(torch.ops.aten.sqrt.default, (c_64,))
    c_recip = graph.call_function(torch.ops.aten.reciprocal.default, (c_sqrt,))

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, None, 0.0, False),
        kwargs={"scale": c_recip},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
    ]

    optimize_attention.apply(gm, example_inputs)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    node = sdpa_nodes[0]
    # Scale should be canonicalized to float
    self.assertIsInstance(node.kwargs["scale"], float)
    self.assertAlmostEqual(node.kwargs["scale"], 0.125, places=5)

    # Subgraph nodes should be eliminated
    recip_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.reciprocal.default
    ]
    self.assertEmpty(recip_nodes)

  def test_overrideable_fwd_and_bwd(self):
    batch, heads, seq_len, head_dim = 1, 4, 16, 64

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    zero_mask = graph.call_function(
        torch.ops.aten.zeros.default, ([batch, heads, seq_len, seq_len],)
    )
    q_scaled = graph.call_function(torch.ops.aten.mul.Scalar, (q, 0.125))

    # _scaled_dot_product_fused_attention_overrideable
    fwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        (q_scaled, k, v, zero_mask, 0.0, False, False),
        kwargs={"scale": 1.0},
    )
    out = graph.call_function(operator.getitem, (fwd, 0))
    logsumexp = graph.call_function(operator.getitem, (fwd, 1))

    # backward
    grad_out = graph.placeholder("grad_out")
    dummy_seed = graph.placeholder("seed")
    dummy_offset = graph.placeholder("offset")
    cum_seq_q = graph.placeholder("cum_seq_q")
    cum_seq_k = graph.placeholder("cum_seq_k")

    bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        (
            grad_out,
            q,
            k,
            v,
            zero_mask,
            [True, True, True, False],
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            seq_len,
            seq_len,
            0.0,
            False,
            dummy_seed,
            dummy_offset,
        ),
        kwargs={"scale": 1.0},
    )
    graph.output((out, bwd))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.randn(batch, heads, seq_len, head_dim),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(batch + 1, dtype=torch.int32),
        torch.zeros(batch + 1, dtype=torch.int32),
    ]

    optimize_attention.apply(gm, example_inputs)

    # In fwd, mask should be None, q should be unscaled, scale = 0.125
    fwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ]
    self.assertLen(fwd_nodes, 1)
    fwd_node = fwd_nodes[0]
    self.assertIs(fwd_node.args[0], q)
    self.assertIsNone(fwd_node.args[3])
    self.assertAlmostEqual(fwd_node.kwargs["scale"], 0.125, places=5)

    # In bwd, zero_mask should be replaced by empty([0])
    bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default
    ]
    self.assertLen(bwd_nodes, 1)
    bwd_node = bwd_nodes[0]
    self.assertIsNotNone(bwd_node.args[4])
    self.assertEqual(
        bwd_node.args[4].target, torch.ops.aten.empty.memory_format
    )


if __name__ == "__main__":
  absltest.main()
