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

"""Unit tests for Variant 3 memoized RoPE pass."""

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import fuse_rope
from torch_tpu._internal.utils.test_utils import assert_close
from tests import seed_test_utils


def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def standard_rope(q, cos, sin):
  return (q * cos) + (rotate_half(q) * sin)


class FuseRopeVariant3Test(seed_test_utils.RepeatableTest):

  def test_fuse_rope_v3_memoization_across_q_and_k(self):
    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")

    half_dim = 32
    full_dim = 64

    # Build RoPE for q
    q1 = graph.call_function(torch.ops.aten.slice.Tensor, (q, -1, 0, half_dim))
    q2 = graph.call_function(
        torch.ops.aten.slice.Tensor, (q, -1, half_dim, full_dim)
    )
    rot_q = graph.call_function(
        torch.ops.aten.cat.default,
        ([graph.call_function(torch.ops.aten.neg.default, (q2,)), q1], -1),
    )
    q_out = graph.call_function(
        torch.ops.aten.add.Tensor,
        (
            graph.call_function(torch.ops.aten.mul.Tensor, (q, cos)),
            graph.call_function(torch.ops.aten.mul.Tensor, (rot_q, sin)),
        ),
    )

    # Build RoPE for k sharing the same cos and sin
    k1 = graph.call_function(torch.ops.aten.slice.Tensor, (k, -1, 0, half_dim))
    k2 = graph.call_function(
        torch.ops.aten.slice.Tensor, (k, -1, half_dim, full_dim)
    )
    rot_k = graph.call_function(
        torch.ops.aten.cat.default,
        ([graph.call_function(torch.ops.aten.neg.default, (k2,)), k1], -1),
    )
    k_out = graph.call_function(
        torch.ops.aten.add.Tensor,
        (
            graph.call_function(torch.ops.aten.mul.Tensor, (k, cos)),
            graph.call_function(torch.ops.aten.mul.Tensor, (rot_k, sin)),
        ),
    )

    out = graph.call_function(torch.ops.aten.add.Tensor, (q_out, k_out))
    graph.output(out)

    gm = fx.GraphModule(torch.nn.Module(), graph)
    fuse_rope.apply(gm)

    # In Variant 3, cos and sin slices (c and s) are memoized!
    # q requires 2 slices (q1, q2)
    # k requires 2 slices (k1, k2)
    # cos/sin requires ONLY 2 slices total (c, s) shared by both q and k!
    # Total slice ops = 2 + 2 + 2 = 6! (Without memoization it would be 8).
    slice_nodes = [
        n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor
    ]
    self.assertLen(slice_nodes, 6)

  def test_fuse_rope_v3_numerical_equivalence(self):
    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
    half_dim = head_dim // 2

    q_data = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_data = torch.randn(batch_size, num_heads, seq_len, head_dim)
    c_half = torch.randn(batch_size, 1, seq_len, half_dim)
    s_half = torch.randn(batch_size, 1, seq_len, half_dim)
    cos_data = torch.cat([c_half, c_half], dim=-1)
    sin_data = torch.cat([s_half, s_half], dim=-1)

    expected_q = standard_rope(q_data, cos_data, sin_data)
    expected_k = standard_rope(k_data, cos_data, sin_data)

    graph = fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")

    q1 = graph.call_function(torch.ops.aten.slice.Tensor, (q, -1, 0, half_dim))
    q2 = graph.call_function(
        torch.ops.aten.slice.Tensor, (q, -1, half_dim, head_dim)
    )
    rot_q = graph.call_function(
        torch.ops.aten.cat.default,
        ([graph.call_function(torch.ops.aten.neg.default, (q2,)), q1], -1),
    )
    q_out = graph.call_function(
        torch.ops.aten.add.Tensor,
        (
            graph.call_function(torch.ops.aten.mul.Tensor, (q, cos)),
            graph.call_function(torch.ops.aten.mul.Tensor, (rot_q, sin)),
        ),
    )

    k1 = graph.call_function(torch.ops.aten.slice.Tensor, (k, -1, 0, half_dim))
    k2 = graph.call_function(
        torch.ops.aten.slice.Tensor, (k, -1, half_dim, head_dim)
    )
    rot_k = graph.call_function(
        torch.ops.aten.cat.default,
        ([graph.call_function(torch.ops.aten.neg.default, (k2,)), k1], -1),
    )
    k_out = graph.call_function(
        torch.ops.aten.add.Tensor,
        (
            graph.call_function(torch.ops.aten.mul.Tensor, (k, cos)),
            graph.call_function(torch.ops.aten.mul.Tensor, (rot_k, sin)),
        ),
    )

    graph.output((q_out, k_out))

    gm = fx.GraphModule(torch.nn.Module(), graph)
    fuse_rope.apply(gm)

    actual_q, actual_k = gm(q_data, k_data, cos_data, sin_data)
    assert_close(actual_q, expected_q, rtol=1e-5, atol=1e-5)
    assert_close(actual_k, expected_k, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
