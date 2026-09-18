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

"""Unit tests for eliminate_sdpa_contiguous pass."""

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import eliminate_sdpa_contiguous
from tests import seed_test_utils


class EliminateSdpaContiguousTest(seed_test_utils.RepeatableTest):

  def test_eliminate_clone_in_sdpa_forward(self):
    graph = fx.Graph()
    q_proj = graph.placeholder("q_proj")
    k_proj = graph.placeholder("k_proj")
    v_proj = graph.placeholder("v_proj")

    # Simulate: proj.transpose(1, 2).contiguous() -> clone
    q_trans = graph.call_function(torch.ops.aten.transpose.int, (q_proj, 1, 2))
    q_clone = graph.call_function(torch.ops.aten.clone.default, (q_trans,))

    k_trans = graph.call_function(torch.ops.aten.transpose.int, (k_proj, 1, 2))
    k_clone = graph.call_function(torch.ops.aten.clone.default, (k_trans,))

    v_trans = graph.call_function(torch.ops.aten.transpose.int, (v_proj, 1, 2))
    v_clone = graph.call_function(torch.ops.aten.clone.default, (v_trans,))

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q_clone, k_clone, v_clone, None, 0.0, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    sdpa_node = sdpa_nodes[0]

    # Verify query, key, value bypassed clone and point to transposed nodes
    self.assertEqual(sdpa_node.args[0], q_trans)
    self.assertEqual(sdpa_node.args[1], k_trans)
    self.assertEqual(sdpa_node.args[2], v_trans)

    # Verify dead clone nodes were erased
    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_eliminate_clone_with_kwargs(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    q_clone = graph.call_function(torch.ops.aten.clone.default, (q,))
    k_clone = graph.call_function(torch.ops.aten.clone.default, (k,))
    v_clone = graph.call_function(torch.ops.aten.clone.default, (v,))

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (),
        kwargs={"query": q_clone, "key": k_clone, "value": v_clone},
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    sdpa_node = sdpa_nodes[0]

    self.assertEqual(sdpa_node.kwargs["query"], q)
    self.assertEqual(sdpa_node.kwargs["key"], k)
    self.assertEqual(sdpa_node.kwargs["value"], v)

    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_eliminate_chained_clone_contiguous(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    k_clone = graph.call_function(torch.ops.aten.clone.default, (k,))
    k_contig = graph.call_function(
        torch.ops.aten.contiguous.default
        if hasattr(torch.ops.aten, "contiguous")
        else torch.ops.aten.clone.default,
        (k_clone,),
    )

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_contig, v),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[1], k)

    # Verify both chained operations were removed
    remaining_clones = [
        n
        for n in gm.graph.nodes
        if n.target
        in (
            torch.ops.aten.clone.default,
            getattr(torch.ops.aten, "contiguous", None),
        )
    ]
    self.assertEmpty(remaining_clones)

  def test_eliminate_to_copy_same_dtype_device(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    k.meta["val"] = torch.empty((2, 4, 16, 64), dtype=torch.float32)
    k_to_copy = graph.call_function(
        torch.ops.aten._to_copy.default,
        (k,),
        kwargs={"dtype": torch.float32},
    )
    k_to_copy.meta["val"] = torch.empty((2, 4, 16, 64), dtype=torch.float32)

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_to_copy, v),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[1], k)

    to_copy_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default
    ]
    self.assertEmpty(to_copy_nodes)

  def test_preserve_to_copy_different_dtype(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    k.meta["val"] = torch.empty((2, 4, 16, 64), dtype=torch.float32)
    k_cast = graph.call_function(
        torch.ops.aten._to_copy.default,
        (k,),
        kwargs={"dtype": torch.bfloat16},
    )
    k_cast.meta["val"] = torch.empty((2, 4, 16, 64), dtype=torch.bfloat16)

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_cast, v),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    # _to_copy should NOT be bypassed because it performs a dtype conversion
    self.assertEqual(sdpa_nodes[0].args[1], k_cast)

    to_copy_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default
    ]
    self.assertLen(to_copy_nodes, 1)

  def test_eliminate_copy_default(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    k_empty = graph.call_function(
        torch.ops.aten.empty.memory_format, ([2, 4, 16, 64],)
    )
    k_copy = graph.call_function(
        torch.ops.aten.copy.default,
        (k_empty, k),
    )

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_copy, v),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[1], k)

    copy_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.copy.default
    ]
    self.assertEmpty(copy_nodes)

  def test_eliminate_clone_in_other_sdpa_overloads(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    q_clone = graph.call_function(torch.ops.aten.clone.default, (q,))
    k_clone = graph.call_function(torch.ops.aten.clone.default, (k,))
    v_clone = graph.call_function(torch.ops.aten.clone.default, (v,))

    sdpa = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        (q_clone, k_clone, v_clone, None, 0.0, False, False),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[0], q)
    self.assertEqual(sdpa_nodes[0].args[1], k)
    self.assertEqual(sdpa_nodes[0].args[2], v)

    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_eliminate_clone_in_sdpa_backward(self):
    graph = fx.Graph()
    grad_out = graph.placeholder("grad_out")
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    grad_clone = graph.call_function(torch.ops.aten.clone.default, (grad_out,))
    q_clone = graph.call_function(torch.ops.aten.clone.default, (q,))
    k_clone = graph.call_function(torch.ops.aten.clone.default, (k,))
    v_clone = graph.call_function(torch.ops.aten.clone.default, (v,))

    sdpa_bwd = graph.call_function(
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        (grad_clone, q_clone, k_clone, v_clone, None),
    )
    graph.output(sdpa_bwd)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    bwd_nodes = [
        n
        for n in gm.graph.nodes
        if n.target
        == torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default
    ]
    self.assertLen(bwd_nodes, 1)
    self.assertEqual(bwd_nodes[0].args[0], grad_out)
    self.assertEqual(bwd_nodes[0].args[1], q)
    self.assertEqual(bwd_nodes[0].args[2], k)
    self.assertEqual(bwd_nodes[0].args[3], v)

    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_shared_clone_key_and_value(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    kv = graph.placeholder("kv")

    kv_clone = graph.call_function(torch.ops.aten.clone.default, (kv,))

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, kv_clone, kv_clone),
    )
    graph.output(sdpa)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    self.assertEqual(sdpa_nodes[0].args[1], kv)
    self.assertEqual(sdpa_nodes[0].args[2], kv)

    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_clone_with_other_pure_user_bypassed_for_sdpa_but_retained(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    k_clone = graph.call_function(torch.ops.aten.clone.default, (k,))

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_clone, v),
    )
    other = graph.call_function(torch.ops.aten.sin.default, (k_clone,))
    graph.output((sdpa, other))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    # SDPA receives k directly
    self.assertEqual(sdpa_nodes[0].args[1], k)

    # k_clone is still needed by the sin operation, so it is retained
    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    self.assertEqual(clone_nodes[0], k_clone)

  def test_clone_with_mutating_source_not_bypassed(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    v = graph.placeholder("v")

    k_clone = graph.call_function(torch.ops.aten.clone.default, (k,))
    # In-place mutation on k
    mutate = graph.call_function(torch.ops.aten.add_.Tensor, (k, k))

    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (q, k_clone, v),
    )
    graph.output((sdpa, mutate))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    eliminate_sdpa_contiguous.apply(gm)

    sdpa_nodes = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    self.assertLen(sdpa_nodes, 1)
    # Because k is mutated in place, clone should NOT be bypassed
    self.assertEqual(sdpa_nodes[0].args[1], k_clone)

    clone_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)


if __name__ == "__main__":
  absltest.main()
