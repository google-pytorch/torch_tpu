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

"""Unit tests for fold_gqa pass."""

import operator
from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import fold_gqa
from tests import seed_test_utils


class FoldGqaTest(seed_test_utils.RepeatableTest):

  def test_fold_gqa_in_scaled_dot_product_attention(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    # Simulate repeat_kv(k, 8)
    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 2, 8, 64, 32])
    )
    k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (k_exp, [1, 16, 64, 32])
    )

    # Simulate repeat_kv(v, 8)
    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 2, 8, 64, 32])
    )
    v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (v_exp, [1, 16, 64, 32])
    )

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_reshaped, v_reshaped, None, 0.0, True),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    sdpa_node = sdpa_nodes[0]

    # Verify key and value were replaced with unexpanded originals
    self.assertEqual(sdpa_node.args[1], k_orig)
    self.assertEqual(sdpa_node.args[2], v_orig)

    # Verify dead repeat_kv chain was removed
    expand_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default
    ]
    self.assertEmpty(expand_nodes)
    unsqueeze_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.unsqueeze.default
    ]
    self.assertEmpty(unsqueeze_nodes)

  def test_fold_gqa_with_view_and_unsafe_view(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    # Key uses view
    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 4, 4, 128, 64])
    )
    k_view = graph.call_function(
        torch.ops.aten.view.default, (k_exp, [1, 16, 128, 64])
    )

    # Value uses _unsafe_view
    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 4, 4, 128, 64])
    )
    v_view = graph.call_function(
        torch.ops.aten._unsafe_view.default, (v_exp, [1, 16, 128, 64])
    )

    sdpa = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        (q, k_view, v_view, None, 0.0, False, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ]
    self.assertLen(sdpa_nodes, 1)
    sdpa_node = sdpa_nodes[0]
    self.assertEqual(sdpa_node.args[1], k_orig)
    self.assertEqual(sdpa_node.args[2], v_orig)

  def test_non_gqa_sdpa_untouched(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k, v, None, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[1], k)
    self.assertEqual(sdpa_nodes[0].args[2], v)

  def test_mismatched_groups_untouched(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    # Key group size 8
    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 2, 8, 64, 32])
    )
    k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (k_exp, [1, 16, 64, 32])
    )

    # Value group size 4 (mismatched)
    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 4, 4, 64, 32])
    )
    v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (v_exp, [1, 16, 64, 32])
    )

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_reshaped, v_reshaped, None, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    # Should NOT be folded because of mismatch
    self.assertEqual(sdpa_nodes[0].args[1], k_reshaped)
    self.assertEqual(sdpa_nodes[0].args[2], v_reshaped)

  def test_fold_gqa_in_scaled_dot_product_attention_backward(self):
    graph = fx.Graph()
    grad_out = graph.placeholder("grad_out")
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    # Forward expansion for k and v
    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 2, 8, 64, 32])
    )
    k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (k_exp, [1, 16, 64, 32])
    )

    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 2, 8, 64, 32])
    )
    v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (v_exp, [1, 16, 64, 32])
    )

    logsumexp = graph.placeholder("logsumexp")
    fwd_out = graph.placeholder("fwd_out")

    # Call backward op with expanded k and v
    sdpa_bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        (
            grad_out,
            q,
            k_reshaped,
            v_reshaped,
            fwd_out,
            logsumexp,
            0.0,
            True,
            0.125,
        ),
    )

    grad_q = graph.call_function(operator.getitem, (sdpa_bwd, 0))
    grad_k = graph.call_function(operator.getitem, (sdpa_bwd, 1))
    grad_v = graph.call_function(operator.getitem, (sdpa_bwd, 2))

    # Backward autograd reduction: reshape(1, 2, 8, 64, 32) -> sum(dim=2)
    grad_k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (grad_k, [1, 2, 8, 64, 32])
    )
    grad_k_sum = graph.call_function(
        torch.ops.aten.sum.dim_IntList, (grad_k_reshaped, [2])
    )

    grad_v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (grad_v, [1, 2, 8, 64, 32])
    )
    grad_v_sum = graph.call_function(
        torch.ops.aten.sum.dim_IntList, (grad_v_reshaped, [2])
    )

    # Some consumer of the gradients
    add_k = graph.call_function(torch.ops.aten.add.Tensor, (grad_k_sum, 1.0))
    add_v = graph.call_function(torch.ops.aten.add.Tensor, (grad_v_sum, 1.0))

    graph.output((grad_q, add_k, add_v))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default
    ]
    self.assertLen(bwd_nodes, 1)
    bwd_node = bwd_nodes[0]

    # Verify key and value were replaced with unexpanded originals in backward args
    self.assertEqual(bwd_node.args[2], k_orig)
    self.assertEqual(bwd_node.args[3], v_orig)

    # Verify dead forward repeat_kv chain was removed
    expand_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default
    ]
    self.assertEmpty(expand_nodes)

    # Verify backward reshape and sum were removed
    sum_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList
    ]
    self.assertEmpty(sum_nodes)

    # Verify add uses grad_k / grad_v (getitem nodes) directly
    self.assertEqual(add_k.args[0], grad_k)
    self.assertEqual(add_v.args[0], grad_v)

  def test_fold_gqa_backward_keepdim_squeeze(self):
    graph = fx.Graph()
    grad_out = graph.placeholder("grad_out")
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 2, 8, 64, 32])
    )
    k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (k_exp, [1, 16, 64, 32])
    )

    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 2, 8, 64, 32])
    )
    v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (v_exp, [1, 16, 64, 32])
    )

    logsumexp = graph.placeholder("logsumexp")
    fwd_out = graph.placeholder("fwd_out")

    sdpa_bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        (
            grad_out,
            q,
            k_reshaped,
            v_reshaped,
            fwd_out,
            logsumexp,
            0.0,
            True,
            0.125,
        ),
    )

    grad_k = graph.call_function(operator.getitem, (sdpa_bwd, 1))

    # Backward autograd reduction with keepdim=True and squeeze
    grad_k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (grad_k, [1, 2, 8, 64, 32])
    )
    grad_k_sum = graph.call_function(
        torch.ops.aten.sum.dim_IntList, (grad_k_reshaped, [2], True)
    )
    grad_k_sq = graph.call_function(torch.ops.aten.squeeze.dim, (grad_k_sum, 2))
    add_k = graph.call_function(torch.ops.aten.add.Tensor, (grad_k_sq, 1.0))

    graph.output(add_k)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    squeeze_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.squeeze.dim
    ]
    self.assertEmpty(squeeze_nodes)
    sum_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList
    ]
    self.assertEmpty(sum_nodes)
    self.assertEqual(add_k.args[0], grad_k)

  def test_fold_gqa_with_chained_clone_contiguous(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    # Key: orig -> unsq -> exp -> reshape -> clone -> contiguous
    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 2, 8, 64, 32])
    )
    k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (k_exp, [1, 16, 64, 32])
    )
    k_clone = graph.call_function(torch.ops.aten.clone.default, (k_reshaped,))
    k_contig = graph.call_function(
        torch.ops.aten.contiguous.default, (k_clone,)
    )

    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 2, 8, 64, 32])
    )
    v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (v_exp, [1, 16, 64, 32])
    )

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_contig, v_reshaped, None, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[1], k_orig)
    self.assertEqual(sdpa_nodes[0].args[2], v_orig)

  def test_fold_gqa_backward_keepdim_as_kwarg(self):
    graph = fx.Graph()
    grad_out = graph.placeholder("grad_out")
    q = graph.placeholder("q")
    k_orig = graph.placeholder("k_orig")
    v_orig = graph.placeholder("v_orig")

    k_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (k_orig, 2))
    k_exp = graph.call_function(
        torch.ops.aten.expand.default, (k_unsq, [1, 2, 8, 64, 32])
    )
    k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (k_exp, [1, 16, 64, 32])
    )

    v_unsq = graph.call_function(torch.ops.aten.unsqueeze.default, (v_orig, 2))
    v_exp = graph.call_function(
        torch.ops.aten.expand.default, (v_unsq, [1, 2, 8, 64, 32])
    )
    v_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (v_exp, [1, 16, 64, 32])
    )

    logsumexp = graph.placeholder("logsumexp")
    fwd_out = graph.placeholder("fwd_out")

    sdpa_bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        (
            grad_out,
            q,
            k_reshaped,
            v_reshaped,
            fwd_out,
            logsumexp,
            0.0,
            True,
            0.125,
        ),
    )

    grad_k = graph.call_function(operator.getitem, (sdpa_bwd, 1))

    # Backward autograd reduction passing keepdim=True as a kwarg
    grad_k_reshaped = graph.call_function(
        torch.ops.aten.reshape.default, (grad_k, [1, 2, 8, 64, 32])
    )
    grad_k_sum = graph.call_function(
        torch.ops.aten.sum.dim_IntList,
        (grad_k_reshaped, [2]),
        {"keepdim": True},
    )
    grad_k_sq = graph.call_function(torch.ops.aten.squeeze.dim, (grad_k_sum, 2))
    add_k = graph.call_function(torch.ops.aten.add.Tensor, (grad_k_sq, 1.0))

    graph.output(add_k)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    fold_gqa.apply(gm)

    squeeze_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.squeeze.dim
    ]
    self.assertEmpty(squeeze_nodes)
    sum_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList
    ]
    self.assertEmpty(sum_nodes)
    self.assertEqual(add_k.args[0], grad_k)


if __name__ == "__main__":
  absltest.main()
