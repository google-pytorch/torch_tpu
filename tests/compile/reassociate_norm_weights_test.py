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

"""Tests for reassociate_norm_weights FX pass."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch.fx.experimental import proxy_tensor
from torch_tpu._internal.compile.fx_passes import reassociate_norm_weights
from torch_tpu._internal.utils import test_utils as utils
from tests import env_test_utils
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph
make_fx = proxy_tensor.make_fx


class ReassociateNormWeightsTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("rank1", (256,)),
      ("rank2", (4, 256)),
      ("rank3", (2, 512, 256)),
      ("rank4", (1, 8, 512, 256)),
  )
  def test_reassociate_rhs_one_plus_weight(self, x_shape):
    """Verifies norm_x * (1.0 + weight) is rewritten to norm_x + (norm_x * weight)."""
    hidden_dim = x_shape[-1]
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty(x_shape, dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((hidden_dim,), dtype=torch.float32)

    # 1.0 + w
    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    one_plus_w.meta["val"] = torch.empty((hidden_dim,), dtype=torch.float32)

    # norm_x * (1.0 + w)
    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, one_plus_w))
    out.meta["val"] = torch.empty(x_shape, dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # Verify that standalone 1.0 + w add node was eliminated
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]

    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)

    new_mul = mul_nodes[0]
    new_add = add_nodes[0]

    # mul should compute norm_x * w
    self.assertEqual(new_mul.args[0], p_x)
    self.assertEqual(new_mul.args[1], p_w)

    # add should compute norm_x + (norm_x * w)
    self.assertEqual(new_add.args[0], p_x)
    self.assertEqual(new_add.args[1], new_mul)

    # output consumes new_add
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertEqual(output_node.args[0][0], new_add)

  @parameterized.named_parameters(
      ("rhs_1_plus_w", False, False),
      ("rhs_w_plus_1", False, True),
      ("lhs_1_plus_w", True, False),
      ("lhs_w_plus_1", True, True),
  )
  def test_reassociate_operand_permutations(self, lhs_add, reverse_add):
    """Verifies all operand order permutations for (1.0 + w) * norm_x."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 128), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((128,), dtype=torch.float32)

    add_args = (p_w, 1.0) if reverse_add else (1.0, p_w)
    add_node = graph.call_function(torch.ops.aten.add.Tensor, add_args)
    add_node.meta["val"] = torch.empty((128,), dtype=torch.float32)

    mul_args = (add_node, p_x) if lhs_add else (p_x, add_node)
    out = graph.call_function(torch.ops.aten.mul.Tensor, mul_args)
    out.meta["val"] = torch.empty((2, 128), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], p_w)
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_reassociate_with_weight_cast(self):
    """Verifies (1.0 + weight.float()) * norm_x where weight is bfloat16 and norm_x is float32."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.bfloat16)

    w_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_w,),
        {"dtype": torch.float32},
    )
    w_f32.meta["val"] = torch.empty((64,), dtype=torch.float32)

    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, w_f32))
    one_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, one_plus_w))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # In float32 mode, new_mul takes (norm_x, w_f32)
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], w_f32)
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_reassociate_with_outer_to_copy(self):
    """Verifies _to_copy(1.0 + weight.float(), dtype=bfloat16) * norm_x where norm_x and weight are bfloat16."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.bfloat16)

    w_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_w,),
        {"dtype": torch.float32},
    )
    w_f32.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_f32 = graph.call_function(torch.ops.aten.add.Tensor, (1.0, w_f32))
    add_f32.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_bf16 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (add_f32,),
        {"dtype": torch.bfloat16},
    )
    add_bf16.meta["val"] = torch.empty((64,), dtype=torch.bfloat16)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_bf16))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # All intermediate float32 conversions and adds should be eliminated!
    # new_mul should directly consume p_x (bfloat16) and p_w (bfloat16).
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    copy_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
    ]

    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEmpty(copy_nodes)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], p_w)
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_reassociate_multiple_norm_layers(self):
    """Verifies multiple RMSNorm layers in the same graph are all reassociated."""
    graph = Graph()
    p_x1 = graph.placeholder("x1")
    p_x1.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w1 = graph.placeholder("w1")
    p_w1.meta["val"] = torch.empty((64,), dtype=torch.float32)

    p_x2 = graph.placeholder("x2")
    p_x2.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w2 = graph.placeholder("w2")
    p_w2.meta["val"] = torch.empty((64,), dtype=torch.float32)

    # Layer 1
    add1 = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w1))
    add1.meta["val"] = torch.empty((64,), dtype=torch.float32)
    out1 = graph.call_function(torch.ops.aten.mul.Tensor, (p_x1, add1))
    out1.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    # Layer 2
    add2 = graph.call_function(torch.ops.aten.add.Tensor, (p_w2, 1.0))
    add2.meta["val"] = torch.empty((64,), dtype=torch.float32)
    out2 = graph.call_function(torch.ops.aten.mul.Tensor, (add2, p_x2))
    out2.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out1, out2))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]

    self.assertLen(mul_nodes, 2)
    self.assertLen(add_nodes, 2)

    self.assertEqual(mul_nodes[0].args[0], p_x1)
    self.assertEqual(mul_nodes[0].args[1], p_w1)
    self.assertEqual(add_nodes[0].args[0], p_x1)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

    self.assertEqual(mul_nodes[1].args[0], p_x2)
    self.assertEqual(mul_nodes[1].args[1], p_w2)
    self.assertEqual(add_nodes[1].args[0], p_x2)
    self.assertEqual(add_nodes[1].args[1], mul_nodes[1])

  def test_non_matching_constant_not_rewritten(self):
    """Verifies (2.0 + weight) * norm_x is NOT rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (2.0, p_w))
    add_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # Node structure should be unchanged
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], add_node)

  def test_2d_weight_not_rewritten(self):
    """Verifies 2D weight (1.0 + W_2D) * norm_x is NOT rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    add_node.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes, 1)
    self.assertEqual(mul_nodes[0].args[1], add_node)

  def test_integer_one_scalar(self):
    """Verifies (1 + w) * norm_x with integer 1 is rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (1, p_w))
    add_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], p_w)
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_scalar_tensor_one(self):
    """Verifies scalar_tensor(1.0) + w is rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    cst_one = graph.call_function(torch.ops.aten.scalar_tensor.default, (1.0,))
    cst_one.meta["val"] = torch.tensor(1.0, dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (cst_one, p_w))
    add_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], p_w)

  def test_subtraction_not_rewritten(self):
    """Verifies (1.0 - w) * norm_x is NOT rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    sub_node = graph.call_function(torch.ops.aten.sub.Tensor, (1.0, p_w))
    sub_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, sub_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes, 1)
    self.assertEqual(mul_nodes[0].args[1], sub_node)

  def test_weight_depends_on_norm_x_not_rewritten(self):
    """Verifies (1.0 + (norm_x + w)) * norm_x is NOT rewritten due to cycle/dependency."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((64,), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    x_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (p_x, p_w))
    x_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (1.0, x_plus_w))
    add_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((64,), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes, 1)
    self.assertEqual(mul_nodes[0].args[1], add_node)

  def test_reassociate_differing_dtypes_inserts_cast(self):
    """Verifies when norm_x is bfloat16 and w is float32 without prior cast, a cast is inserted."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    add_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    copy_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
    ]

    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertLen(copy_nodes, 1)
    # copy_node casts p_w to bfloat16
    self.assertEqual(copy_nodes[0].args[0], p_w)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], copy_nodes[0])
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_numerical_correctness_eager_parity_make_fx(self):
    """Verifies ATen graph rewrite and numerical parity with eager PyTorch using make_fx."""

    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
      variance = x.pow(2).mean(-1, keepdim=True)
      norm_x = x * torch.rsqrt(variance + 1e-6)
      return norm_x * (1.0 + weight)

    dim = 128
    x = torch.randn(4, 32, dim, dtype=torch.float32)
    w = torch.randn(dim, dtype=torch.float32)

    eager_out = rms_norm(x, w)

    gm = make_fx(rms_norm)(x, w)
    reassociate_norm_weights.apply(gm)
    rewritten_out = gm(x, w)

    # Verify numerical equivalence
    utils.assert_close(rewritten_out, eager_out, rtol=1e-5, atol=1e-5)

    # Verify structural rewrite: the standalone (1.0 + w) was eliminated
    # and replaced with distributive norm_x + (norm_x * w)
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    # The graph should have add (for variance epsilon) and add (for final norm_x + norm_x * w)
    # Neither should be adding 1.0 to w.
    for add_node in add_nodes:
      self.assertNotIn(1.0, add_node.args)
      self.assertNotIn(1, add_node.args)

  def test_gemma_rmsnorm_bfloat16_eager_parity(self):
    """Verifies Gemma-style bfloat16 RMSNorm with float32 upcasting is rewritten to pure bfloat16 FMA."""

    def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
      variance = x.float().pow(2).mean(-1, keepdim=True)
      norm_x = x * torch.rsqrt(variance + 1e-6).type_as(x)
      return norm_x * (1.0 + weight.float()).type_as(x)

    dim = 64
    x = torch.randn(2, 16, dim, dtype=torch.bfloat16)
    w = torch.randn(dim, dtype=torch.bfloat16)

    eager_out = gemma_rms_norm(x, w)

    gm = make_fx(gemma_rms_norm)(x, w)
    reassociate_norm_weights.apply(gm)
    rewritten_out = gm(x, w)

    # In eager, (1.0 + w.float()).bfloat16() suffers from low-mantissa truncation
    # when adding 1.0 to small weight values in bfloat16, whereas distributive FMA
    # preserves full relative precision of w. Tolerances reflect bfloat16 precision.
    utils.assert_close(rewritten_out, eager_out, rtol=0.05, atol=0.05)

    # Verify that the final multiplication and addition operate directly on bfloat16
    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    final_add = (
        out_node.args[0]
        if isinstance(out_node.args[0], torch.fx.Node)
        else out_node.args[0][0]
    )
    self.assertEqual(
        getattr(final_add.target, "overloadpacket", final_add.target),
        torch.ops.aten.add,
    )
    final_mul = final_add.args[1]
    self.assertEqual(
        getattr(final_mul.target, "overloadpacket", final_mul.target),
        torch.ops.aten.mul,
    )

  def test_shared_one_plus_weight_multiple_users(self):
    """Verifies (1.0 + w) shared across two RMSNorm mul nodes is rewritten for both."""
    graph = Graph()
    p_x1 = graph.placeholder("norm_x1")
    p_x1.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_x2 = graph.placeholder("norm_x2")
    p_x2.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    one_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out1 = graph.call_function(torch.ops.aten.mul.Tensor, (p_x1, one_plus_w))
    out1.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    out2 = graph.call_function(torch.ops.aten.mul.Tensor, (p_x2, one_plus_w))
    out2.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out1, out2))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 2)
    self.assertLen(add_nodes, 2)
    self.assertEqual(mul_nodes[0].args[0], p_x1)
    self.assertEqual(mul_nodes[0].args[1], p_w)
    self.assertEqual(add_nodes[0].args[0], p_x1)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])
    self.assertEqual(mul_nodes[1].args[0], p_x2)
    self.assertEqual(mul_nodes[1].args[1], p_w)
    self.assertEqual(add_nodes[1].args[0], p_x2)
    self.assertEqual(add_nodes[1].args[1], mul_nodes[1])

  def test_shared_one_plus_weight_retained_if_used_by_non_norm_op(self):
    """Verifies (1.0 + w) is retained when consumed by an external non-norm user."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    one_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out_norm = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, one_plus_w))
    out_norm.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out_norm, one_plus_w))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # mul(p_x, one_plus_w) was rewritten to p_x + p_x * p_w
    # But one_plus_w is retained because output still consumes it
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 2)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertEqual(output_node.args[0][0], add_nodes[1])
    self.assertEqual(output_node.args[0][1], one_plus_w)

  def test_alpha_scaling_not_rewritten(self):
    """Verifies add with alpha != 1 (e.g. 1.0 + 2.0 * w) is NOT rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    add_node = graph.call_function(
        torch.ops.aten.add.Tensor, (1.0, p_w), {"alpha": 2.0}
    )
    add_node.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes, 1)
    self.assertEqual(mul_nodes[0].args[1], add_node)

  def test_scalar_full_and_ones_nodes(self):
    """Verifies full and ones constant scalar representations are matched and rewritten."""
    # Test aten.full([], 1.0)
    graph1 = Graph()
    p_x = graph1.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph1.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    full_node = graph1.call_function(torch.ops.aten.full.default, ([], 1.0))
    full_node.meta["val"] = torch.tensor(1.0, dtype=torch.float32)

    add1 = graph1.call_function(torch.ops.aten.add.Tensor, (full_node, p_w))
    add1.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out1 = graph1.call_function(torch.ops.aten.mul.Tensor, (p_x, add1))
    out1.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph1.output((out1,))
    gm1 = GraphModule(torch.nn.Module(), graph1)
    reassociate_norm_weights.apply(gm1)

    mul_nodes1 = [
        n
        for n in gm1.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes1, 1)
    self.assertEqual(mul_nodes1[0].args[0], p_x)
    self.assertEqual(mul_nodes1[0].args[1], p_w)

    # Test aten.ones([])
    graph2 = Graph()
    p_x2 = graph2.placeholder("norm_x")
    p_x2.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w2 = graph2.placeholder("w")
    p_w2.meta["val"] = torch.empty((64,), dtype=torch.float32)

    ones_node = graph2.call_function(torch.ops.aten.ones.default, ([],))
    ones_node.meta["val"] = torch.tensor(1.0, dtype=torch.float32)

    add2 = graph2.call_function(torch.ops.aten.add.Tensor, (ones_node, p_w2))
    add2.meta["val"] = torch.empty((64,), dtype=torch.float32)

    out2 = graph2.call_function(torch.ops.aten.mul.Tensor, (p_x2, add2))
    out2.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph2.output((out2,))
    gm2 = GraphModule(torch.nn.Module(), graph2)
    reassociate_norm_weights.apply(gm2)

    mul_nodes2 = [
        n
        for n in gm2.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(mul_nodes2, 1)
    self.assertEqual(mul_nodes2[0].args[0], p_x2)
    self.assertEqual(mul_nodes2[0].args[1], p_w2)

  def test_rank0_scalar_weight(self):
    """Verifies rank-0 scalar weight parameter (1.0 + w_0d) * norm_x is rewritten."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((), dtype=torch.float32)

    add_node = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    add_node.meta["val"] = torch.empty((), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, add_node))
    out.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], p_w)
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_gemma3_rmsnorm_eliminates_detached_weight_to_copy_graph(self):
    """Verifies synthetic Gemma3RMSNorm graph eliminates detached _to_copy on weight."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.bfloat16)

    # x.float()
    x_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_x,),
        {"dtype": torch.float32},
    )
    x_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    # norm_f32
    norm_f32 = graph.call_function(torch.ops.aten.mul.Tensor, (x_f32, 0.5))
    norm_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    # weight.float()
    w_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_w,),
        {"dtype": torch.float32},
    )
    w_f32.meta["val"] = torch.empty((64,), dtype=torch.float32)

    # 1.0 + weight.float()
    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, w_f32))
    one_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    # norm_f32 * (1.0 + weight.float())
    mul_f32 = graph.call_function(
        torch.ops.aten.mul.Tensor, (norm_f32, one_plus_w)
    )
    mul_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    # output.type_as(x) -> _to_copy back to bfloat16
    out_bf16 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (mul_f32,),
        {"dtype": torch.bfloat16},
    )
    out_bf16.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)

    graph.output((out_bf16,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # 1. Check weight detached _to_copy is completely eliminated
    copy_nodes_on_weight = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
        and len(n.args) >= 1
        and n.args[0] == p_w
    ]
    self.assertEmpty(
        copy_nodes_on_weight,
        "Expected 0 detached _to_copy nodes on weight parameter.",
    )

    # 2. Check mul and add nodes structure
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    # mul_nodes[0] is norm_f32, mul_nodes[1] is new distributive mul
    self.assertLen(mul_nodes, 2)
    self.assertLen(add_nodes, 1)

    new_mul = mul_nodes[1]
    new_add = add_nodes[0]

    # new_mul should directly multiply effective_norm_x with p_w
    self.assertEqual(new_mul.args[1], p_w)
    effective_norm_x = new_mul.args[0]
    self.assertEqual(
        getattr(
            effective_norm_x.target, "overloadpacket", effective_norm_x.target
        ),
        torch.ops.aten._to_copy,
    )
    self.assertEqual(effective_norm_x.args[0], norm_f32)

    # new_add computes effective_norm_x + new_mul
    self.assertEqual(new_add.args[0], effective_norm_x)
    self.assertEqual(new_add.args[1], new_mul)

    # output consumes new_add directly
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertEqual(output_node.args[0][0], new_add)

  def test_gemma3_rmsnorm_eager_parity_make_fx(self):
    """Verifies Gemma3RMSNorm with float upcasting eliminates weight copy and preserves parity."""

    def gemma3_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
      variance = x.float().pow(2).mean(-1, keepdim=True)
      norm_x = x.float() * torch.rsqrt(variance + 1e-6)
      output = norm_x * (1.0 + weight.float())
      return output.type_as(x)

    dim = 64
    x = torch.randn(2, 16, dim, dtype=torch.bfloat16)
    w = torch.randn(dim, dtype=torch.bfloat16) * 0.1

    eager_out = gemma3_rms_norm(x, w)

    gm = make_fx(gemma3_rms_norm)(x, w)
    reassociate_norm_weights.apply(gm)
    rewritten_out = gm(x, w)

    # Numerical parity
    utils.assert_close(rewritten_out, eager_out, rtol=0.05, atol=0.05)

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    self.assertLen(placeholders, 2)
    p_w = placeholders[1]

    # Verify zero detached _to_copy on weight parameter
    copy_nodes_on_weight = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
        and len(n.args) >= 1
        and n.args[0] == p_w
    ]
    self.assertEmpty(
        copy_nodes_on_weight,
        "Expected 0 detached _to_copy nodes on weight parameter.",
    )

    # Verify final output is produced directly by aten.add
    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    final_op = (
        out_node.args[0]
        if isinstance(out_node.args[0], torch.fx.Node)
        else out_node.args[0][0]
    )
    self.assertEqual(
        getattr(final_op.target, "overloadpacket", final_op.target),
        torch.ops.aten.add,
    )
    # The second argument of add is mul(effective_norm_x, weight)
    final_mul = final_op.args[1]
    self.assertEqual(
        getattr(final_mul.target, "overloadpacket", final_mul.target),
        torch.ops.aten.mul,
    )
    self.assertEqual(final_mul.args[1], p_w)

  def test_gemma3_rmsnorm_reversed_operands_make_fx(self):
    """Verifies (weight.float() + 1.0) * output operand ordering preserves parity and eliminates copy."""

    def gemma3_rms_norm_reversed(
        x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
      variance = x.float().pow(2).mean(-1, keepdim=True)
      norm_x = x.float() * torch.rsqrt(variance + 1e-6)
      output = (weight.float() + 1.0) * norm_x
      return output.type_as(x)

    dim = 64
    x = torch.randn(2, 16, dim, dtype=torch.bfloat16)
    w = torch.randn(dim, dtype=torch.bfloat16) * 0.1

    eager_out = gemma3_rms_norm_reversed(x, w)

    gm = make_fx(gemma3_rms_norm_reversed)(x, w)
    reassociate_norm_weights.apply(gm)
    rewritten_out = gm(x, w)

    utils.assert_close(rewritten_out, eager_out, rtol=0.05, atol=0.05)

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    self.assertLen(placeholders, 2)
    p_w = placeholders[1]

    copy_nodes_on_weight = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
        and len(n.args) >= 1
        and n.args[0] == p_w
    ]
    self.assertEmpty(
        copy_nodes_on_weight,
        "Expected 0 detached _to_copy nodes on weight parameter.",
    )

  def test_gemma3_rmsnorm_multiple_layers_make_fx(self):
    """Verifies multiple sequential Gemma3RMSNorm layers all eliminate weight copies."""

    def gemma3_two_layer_rms_norm(
        x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
      variance1 = x.float().pow(2).mean(-1, keepdim=True)
      norm1 = x.float() * torch.rsqrt(variance1 + 1e-6)
      h1 = (norm1 * (1.0 + w1.float())).type_as(x)

      variance2 = h1.float().pow(2).mean(-1, keepdim=True)
      norm2 = h1.float() * torch.rsqrt(variance2 + 1e-6)
      h2 = (norm2 * (1.0 + w2.float())).type_as(x)
      return h2

    dim = 64
    x = torch.randn(2, 16, dim, dtype=torch.bfloat16)
    w1 = torch.randn(dim, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(dim, dtype=torch.bfloat16) * 0.1

    eager_out = gemma3_two_layer_rms_norm(x, w1, w2)

    gm = make_fx(gemma3_two_layer_rms_norm)(x, w1, w2)
    reassociate_norm_weights.apply(gm)
    rewritten_out = gm(x, w1, w2)

    utils.assert_close(rewritten_out, eager_out, rtol=0.05, atol=0.05)

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    self.assertLen(placeholders, 3)
    p_w1, p_w2 = placeholders[1], placeholders[2]

    for wp in (p_w1, p_w2):
      copy_nodes = [
          n
          for n in gm.graph.nodes
          if n.op == "call_function"
          and getattr(n.target, "overloadpacket", n.target)
          == torch.ops.aten._to_copy
          and len(n.args) >= 1
          and n.args[0] == wp
      ]
      self.assertEmpty(
          copy_nodes,
          f"Expected 0 detached _to_copy nodes on weight parameter {wp.name}.",
      )

  def test_gemma3_rmsnorm_float16_eager_parity_make_fx(self):
    """Verifies float16 Gemma3RMSNorm eliminates weight copies and preserves parity."""

    def gemma3_rms_norm_fp16(
        x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
      variance = x.float().pow(2).mean(-1, keepdim=True)
      norm_x = x.float() * torch.rsqrt(variance + 1e-6)
      output = norm_x * (1.0 + weight.float())
      return output.type_as(x)

    dim = 64
    x = torch.randn(2, 16, dim, dtype=torch.float16)
    w = torch.randn(dim, dtype=torch.float16) * 0.1

    eager_out = gemma3_rms_norm_fp16(x, w)

    gm = make_fx(gemma3_rms_norm_fp16)(x, w)
    reassociate_norm_weights.apply(gm)
    rewritten_out = gm(x, w)

    utils.assert_close(rewritten_out, eager_out, rtol=0.05, atol=0.05)

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    self.assertLen(placeholders, 2)
    p_w = placeholders[1]

    copy_nodes_on_weight = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
        and len(n.args) >= 1
        and n.args[0] == p_w
    ]
    self.assertEmpty(
        copy_nodes_on_weight,
        "Expected 0 detached _to_copy nodes on float16 weight parameter.",
    )

  def test_gemma3_rmsnorm_with_already_bf16_norm_x(self):
    """Verifies when norm_x is already bfloat16, no extra _to_copy is added on norm_x."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.bfloat16)

    # norm_x is already bfloat16 (e.g. from an earlier layer or norm function)
    # in ATen IR, multiplying norm_x (bf16) with (1.0 + w.float()) (f32) inserts _to_copy(norm_x, f32)
    norm_x_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_x,),
        {"dtype": torch.float32},
    )
    norm_x_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    w_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_w,),
        {"dtype": torch.float32},
    )
    w_f32.meta["val"] = torch.empty((64,), dtype=torch.float32)

    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, w_f32))
    one_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    mul_f32 = graph.call_function(
        torch.ops.aten.mul.Tensor, (norm_x_f32, one_plus_w)
    )
    mul_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    out_bf16 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (mul_f32,),
        {"dtype": torch.bfloat16},
    )
    out_bf16.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)

    graph.output((out_bf16,))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # Resulting graph should have 0 _to_copy nodes entirely!
    copy_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target)
        == torch.ops.aten._to_copy
    ]
    self.assertEmpty(
        copy_nodes,
        "Expected 0 _to_copy nodes when norm_x is already bfloat16.",
    )

    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    self.assertLen(mul_nodes, 1)
    self.assertLen(add_nodes, 1)
    self.assertEqual(mul_nodes[0].args[0], p_x)
    self.assertEqual(mul_nodes[0].args[1], p_w)
    self.assertEqual(add_nodes[0].args[0], p_x)
    self.assertEqual(add_nodes[0].args[1], mul_nodes[0])

  def test_gemma3_rmsnorm_mixed_consumers_retains_fp32_fma(self):
    """Verifies when mul has both a bf16-cast user and an fp32 user, fp32 FMA is preserved."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty((2, 64), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((64,), dtype=torch.bfloat16)

    w_f32 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (p_w,),
        {"dtype": torch.float32},
    )
    w_f32.meta["val"] = torch.empty((64,), dtype=torch.float32)

    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, w_f32))
    one_plus_w.meta["val"] = torch.empty((64,), dtype=torch.float32)

    mul_f32 = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, one_plus_w))
    mul_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    # Consumer 1 casts to bfloat16
    out_bf16 = graph.call_function(
        torch.ops.aten._to_copy.default,
        (mul_f32,),
        {"dtype": torch.bfloat16},
    )
    out_bf16.meta["val"] = torch.empty((2, 64), dtype=torch.bfloat16)

    # Consumer 2 uses mul_f32 directly in float32
    out_f32 = graph.call_function(
        torch.ops.aten.sin.default,
        (mul_f32,),
    )
    out_f32.meta["val"] = torch.empty((2, 64), dtype=torch.float32)

    graph.output((out_bf16, out_f32))
    gm = GraphModule(torch.nn.Module(), graph)

    reassociate_norm_weights.apply(gm)

    # mul_f32 is rewritten to p_x + p_x * w_f32 in fp32, and out_bf16 casts new_add to bf16
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.add
    ]
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", n.target) == torch.ops.aten.mul
    ]
    self.assertLen(add_nodes, 1)
    self.assertLen(mul_nodes, 1)
    new_add = add_nodes[0]
    self.assertEqual(out_bf16.args[0], new_add)
    self.assertEqual(out_f32.args[0], new_add)

  def test_disabled_by_env_var(self):
    """Verifies graph is untouched when TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS=0."""
    graph = Graph()
    p_x = graph.placeholder("norm_x")
    p_x.meta["val"] = torch.empty((4, 256), dtype=torch.float32)
    p_w = graph.placeholder("w")
    p_w.meta["val"] = torch.empty((256,), dtype=torch.float32)

    one_plus_w = graph.call_function(torch.ops.aten.add.Tensor, (1.0, p_w))
    one_plus_w.meta["val"] = torch.empty((256,), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, one_plus_w))
    out.meta["val"] = torch.empty((4, 256), dtype=torch.float32)

    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    orig_nodes = list(gm.graph.nodes)
    with env_test_utils.scoped_env(
        {"TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS": "0"}
    ):
      reassociate_norm_weights.apply(gm)

    self.assertEqual(list(gm.graph.nodes), orig_nodes)


if __name__ == "__main__":
  absltest.main()
