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

"""Tests for fuse_rope FX pass."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch.fx.experimental import proxy_tensor
from torch_tpu._internal.compile.fx_passes import fuse_rope
from tests import seed_test_utils

assert_close = torch.testing.assert_close

GraphModule = fx.GraphModule
Graph = fx.Graph
make_fx = proxy_tensor.make_fx


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Standard Hugging Face rotate_half."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
  """Standard Hugging Face apply_rotary_pos_emb."""
  return (x * cos) + (_rotate_half(x) * sin)


def _make_rope_cos_sin(
    shape: tuple[int, ...] | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Constructs RoPE cos and sin with duplicated half-frequency channels."""
  half_shape = list(shape)
  half_shape[-1] = half_shape[-1] // 2
  half_cos = torch.randn(half_shape, dtype=torch.float32)
  half_sin = torch.randn(half_shape, dtype=torch.float32)
  cos = torch.cat([half_cos, half_cos], dim=-1)
  sin = torch.cat([half_sin, half_sin], dim=-1)
  return cos, sin


class FuseRoPETest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("rank2", (16, 64)),
      ("rank3", (2, 16, 64)),
      ("rank4", (2, 4, 16, 64)),
  )
  def test_fuse_rope_make_fx_traced(self, x_shape):
    """Verifies Hugging Face RoPE traced by make_fx is properly fused."""
    x = torch.randn(x_shape, dtype=torch.float32)
    # cos and sin typically broadcast over batch / heads
    cos_shape = list(x_shape)
    for i in range(len(cos_shape) - 2):
      cos_shape[i] = 1
    cos, sin = _make_rope_cos_sin(cos_shape)

    # Compute expected reference output
    expected = _apply_rotary_pos_emb(x, cos, sin)

    # Trace with make_fx
    gm = make_fx(_apply_rotary_pos_emb)(x, cos, sin)

    # Apply fuse_rope pass
    fuse_rope.apply(gm)

    # 1. Verify rotate_half is eliminated: no neg node and no rotate_half cat
    neg_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and (
            n.target == torch.ops.aten.neg.default
            or getattr(n.target, "overloadpacket", None) == torch.ops.aten.neg
        )
    ]
    self.assertEmpty(neg_nodes)

    # 2. Verify we have 1 cat node (the final output cat([y1, y2], -1))
    cat_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and (
            n.target == torch.ops.aten.cat.default
            or getattr(n.target, "overloadpacket", None) == torch.ops.aten.cat
        )
    ]
    self.assertLen(cat_nodes, 1)

    # 3. Verify sub node exists for y1 = (x1 * c1) - (x2 * s1)
    sub_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and (
            n.target
            in (
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.sub.default,
            )
            or getattr(n.target, "overloadpacket", None) == torch.ops.aten.sub
        )
    ]
    self.assertLen(sub_nodes, 1)

    # 4. Verify add node exists for y2 = (x2 * c1) + (x1 * s1)
    add_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and (
            n.target
            in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add.default,
            )
            or getattr(n.target, "overloadpacket", None) == torch.ops.aten.add
        )
    ]
    self.assertLen(add_nodes, 1)

    # 5. Verify 4 mul nodes: x1*c1, x2*s1, x2*c1, x1*s1
    mul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and (
            n.target
            in (
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul.default,
            )
            or getattr(n.target, "overloadpacket", None) == torch.ops.aten.mul
        )
    ]
    self.assertLen(mul_nodes, 4)

    # 6. Verify numerical output matches reference
    actual = gm(x, cos, sin)
    assert_close(actual, expected)

  @parameterized.named_parameters(
      ("standard", False, False, False),
      ("reversed_add", True, False, False),
      ("reversed_mul_x", False, True, False),
      ("reversed_mul_rot", False, False, True),
      ("reversed_all", True, True, True),
  )
  def test_fuse_rope_permutations(
      self, reverse_add, reverse_mul_x, reverse_mul_rot
  ):
    """Verifies robustness to commutative operand orderings in add and mul."""
    shape = (2, 4, 8, 32)
    hidden_dim = shape[-1]
    half_dim = hidden_dim // 2

    x = torch.randn(shape, dtype=torch.float32)
    cos, sin = _make_rope_cos_sin(shape)
    expected = _apply_rotary_pos_emb(x, cos, sin)

    graph = Graph()
    p_x = graph.placeholder("x")
    p_cos = graph.placeholder("cos")
    p_sin = graph.placeholder("sin")

    # Slices on x
    x1 = graph.call_function(
        torch.ops.aten.slice.Tensor, (p_x, -1, 0, half_dim, 1)
    )
    x2 = graph.call_function(
        torch.ops.aten.slice.Tensor, (p_x, -1, half_dim, hidden_dim, 1)
    )
    neg_x2 = graph.call_function(torch.ops.aten.neg.default, (x2,))
    rot_x = graph.call_function(torch.ops.aten.cat.default, ([neg_x2, x1], -1))

    # mul terms with configured operand ordering
    term_x_args = (p_cos, p_x) if reverse_mul_x else (p_x, p_cos)
    term_x = graph.call_function(torch.ops.aten.mul.Tensor, term_x_args)

    term_rot_args = (p_sin, rot_x) if reverse_mul_rot else (rot_x, p_sin)
    term_rot = graph.call_function(torch.ops.aten.mul.Tensor, term_rot_args)

    add_args = (term_rot, term_x) if reverse_add else (term_x, term_rot)
    out = graph.call_function(torch.ops.aten.add.Tensor, add_args)

    graph.output(out)
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_rope.apply(gm)

    # Verify rotate_half was eliminated and rewritten into fused cat
    cat_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.cat.default
    ]
    self.assertLen(cat_nodes, 1)

    actual = gm(x, cos, sin)
    assert_close(actual, expected)

  def test_fuse_rope_with_slice_copy(self):
    """Verifies matching and rewriting when slice_copy.Tensor is used."""
    shape = (2, 8, 32)
    hidden_dim = shape[-1]
    half_dim = hidden_dim // 2

    x = torch.randn(shape, dtype=torch.float32)
    cos, sin = _make_rope_cos_sin(shape)
    expected = _apply_rotary_pos_emb(x, cos, sin)

    graph = Graph()
    p_x = graph.placeholder("x")
    p_cos = graph.placeholder("cos")
    p_sin = graph.placeholder("sin")

    x1 = graph.call_function(
        torch.ops.aten.slice_copy.Tensor, (p_x, -1, 0, half_dim, 1)
    )
    x2 = graph.call_function(
        torch.ops.aten.slice_copy.Tensor, (p_x, -1, half_dim, hidden_dim, 1)
    )
    neg_x2 = graph.call_function(torch.ops.aten.neg.default, (x2,))
    rot_x = graph.call_function(torch.ops.aten.cat.default, ([neg_x2, x1], -1))

    term_x = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, p_cos))
    term_rot = graph.call_function(torch.ops.aten.mul.Tensor, (rot_x, p_sin))
    out = graph.call_function(torch.ops.aten.add.Tensor, (term_x, term_rot))

    graph.output(out)
    gm = GraphModule(torch.nn.Module(), graph)

    fuse_rope.apply(gm)

    # Slices created for cos and sin should also use slice_copy.Tensor
    slice_copy_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.slice_copy.Tensor
    ]
    # x1, x2, c1, s1 = 4 slice_copy nodes
    self.assertLen(slice_copy_nodes, 4)

    actual = gm(x, cos, sin)
    assert_close(actual, expected)

  def test_fuse_rope_q_and_k_shares_cos_sin(self):
    """Verifies that q and k sharing cos/sin reuses the sliced cos/sin nodes."""
    shape = (2, 4, 16, 64)
    x_q = torch.randn(shape, dtype=torch.float32)
    x_k = torch.randn(shape, dtype=torch.float32)
    cos, sin = _make_rope_cos_sin((1, 1, 16, 64))

    def apply_both(q, k, cos, sin):
      q_out = _apply_rotary_pos_emb(q, cos, sin)
      k_out = _apply_rotary_pos_emb(k, cos, sin)
      return q_out, k_out

    expected_q, expected_k = apply_both(x_q, x_k, cos, sin)

    gm = make_fx(apply_both)(x_q, x_k, cos, sin)
    fuse_rope.apply(gm)

    # Verify rotate_half neg nodes are all gone
    neg_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", None) == torch.ops.aten.neg
    ]
    self.assertEmpty(neg_nodes)

    # Slices on cos and sin should be shared between q and k
    # 2 slices for q (x1, x2), 2 slices for k (x1, x2), 1 slice for c1, 1 slice for s1 = 6 total
    slice_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", None)
        in (
            torch.ops.aten.slice,
            torch.ops.aten.slice_copy,
        )
    ]
    self.assertLen(slice_nodes, 6)

    # Exactly 2 cat nodes (one for q_out, one for k_out)
    cat_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "overloadpacket", None) == torch.ops.aten.cat
    ]
    self.assertLen(cat_nodes, 2)

    actual_q, actual_k = gm(x_q, x_k, cos, sin)
    assert_close(actual_q, expected_q)
    assert_close(actual_k, expected_k)

  def test_disabled_by_env_var(self):
    """Verifies that pass does not modify the graph when disabled."""
    shape = (2, 8, 32)
    x = torch.randn(shape, dtype=torch.float32)
    cos, sin = _make_rope_cos_sin(shape)

    gm = make_fx(_apply_rotary_pos_emb)(x, cos, sin)
    num_nodes_before = len(list(gm.graph.nodes))

    with mock.patch.object(
        fuse_rope, "_is_fuse_rope_enabled", return_value=False
    ):
      fuse_rope.apply(gm)

    num_nodes_after = len(list(gm.graph.nodes))
    self.assertEqual(num_nodes_before, num_nodes_after)

  def test_negative_unequal_slices(self):
    """Verifies that slices not splitting hidden_dim equally are not fused."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_cos = graph.placeholder("cos")
    p_sin = graph.placeholder("sin")

    # Unequal slices: 0..20 and 20..64
    x1 = graph.call_function(torch.ops.aten.slice.Tensor, (p_x, -1, 0, 20, 1))
    x2 = graph.call_function(torch.ops.aten.slice.Tensor, (p_x, -1, 20, 64, 1))
    neg_x2 = graph.call_function(torch.ops.aten.neg.default, (x2,))
    rot_x = graph.call_function(torch.ops.aten.cat.default, ([neg_x2, x1], -1))

    term_x = graph.call_function(torch.ops.aten.mul.Tensor, (p_x, p_cos))
    term_rot = graph.call_function(torch.ops.aten.mul.Tensor, (rot_x, p_sin))
    out = graph.call_function(torch.ops.aten.add.Tensor, (term_x, term_rot))

    graph.output(out)
    gm = GraphModule(torch.nn.Module(), graph)
    num_nodes_before = len(list(gm.graph.nodes))

    fuse_rope.apply(gm)
    num_nodes_after = len(list(gm.graph.nodes))
    self.assertEqual(num_nodes_before, num_nodes_after)


if __name__ == "__main__":
  absltest.main()
