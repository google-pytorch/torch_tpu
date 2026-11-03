# Copyright 2026 The TorchTPU Authors
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

"""Unit tests for DeepSeek-V4-Flash functional fallbacks."""

from absl.testing import absltest
import torch

try:
  from .fallbacks import (
      act_quant_fallback,
      apply_rotary_emb_functional,
      fp4_act_quant_fallback,
      fp4_gemm_fallback,
      fp8_gemm_fallback,
      hadamard_transform_fallback,
      hc_split_sinkhorn_fallback,
      mla_sparse_attn_fallback,
      update_kv_cache_functional,
  )
except ImportError:
  from fallbacks import (
      act_quant_fallback,
      apply_rotary_emb_functional,
      fp4_act_quant_fallback,
      fp4_gemm_fallback,
      fp8_gemm_fallback,
      hadamard_transform_fallback,
      hc_split_sinkhorn_fallback,
      mla_sparse_attn_fallback,
      update_kv_cache_functional,
  )


class FallbacksTest(absltest.TestCase):

  def test_hadamard_orthogonality_and_shape(self):
    x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    out = hadamard_transform_fallback(x)
    self.assertEqual(out.shape, x.shape)
    self.assertEqual(out.dtype, x.dtype)

    # Property of normalized Walsh-Hadamard: H(H(x)) = x
    reconstructed = hadamard_transform_fallback(out)
    self.assertTrue(
        torch.allclose(x.float(), reconstructed.float(), atol=1e-2),
        "Hadamard inversion failed to reconstruct original tensor.",
    )

  def test_quantization_fp8_and_fp4_scales(self):
    x = torch.randn(2, 64, 256, dtype=torch.bfloat16)
    q_fp8, scales_fp8 = act_quant_fallback(x, block_size=128)
    self.assertEqual(q_fp8.shape, x.shape)
    self.assertEqual(scales_fp8.shape[0], x.numel() // 128)

    q_fp4, scales_fp4 = fp4_act_quant_fallback(x, block_size=32)
    self.assertEqual(q_fp4.shape, x.shape)
    self.assertEqual(scales_fp4.shape[0], x.numel() // 32)

  def test_gemm_fp8_and_fp4(self):
    # a: (batch, M, K), b: (N, K)
    a = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    a_scale = torch.ones(2, 16, 1, dtype=torch.float32)
    b = torch.randn(32, 64, dtype=torch.bfloat16)
    b_scale = torch.ones(32, 1, dtype=torch.float32)

    out_fp8 = fp8_gemm_fallback(a, a_scale, b, b_scale)
    self.assertEqual(out_fp8.shape, (2, 16, 32))

    out_fp4 = fp4_gemm_fallback(a, a_scale, b, b_scale)
    self.assertEqual(out_fp4.shape, (2, 16, 32))

  def test_mla_attention_shape(self):
    bsz, seqlen, n_heads, head_dim = 2, 8, 4, 64
    q = torch.randn(bsz, seqlen, n_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(bsz, seqlen, head_dim, dtype=torch.bfloat16)
    scale = head_dim**-0.5

    out = mla_sparse_attn_fallback(q, kv, sm_scale=scale, causal=True)
    self.assertEqual(out.shape, q.shape)

  def test_sinkhorn_balancing_and_shapes(self):
    b, s, hc_mult = 2, 4, 4
    mix_hc = (2 + hc_mult) * hc_mult  # 24
    mixes = torch.randn(b, s, mix_hc, dtype=torch.bfloat16)
    hc_scale = torch.ones(3, dtype=torch.float32)
    hc_base = torch.zeros(mix_hc, dtype=torch.float32)

    pre, post, comb = hc_split_sinkhorn_fallback(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=10
    )
    self.assertEqual(pre.shape, (b, s, hc_mult))
    self.assertEqual(post.shape, (b, s, hc_mult))
    self.assertEqual(comb.shape, (b, s, hc_mult, hc_mult))

    self.assertTrue(torch.all(comb > 0), "Sinkhorn weights must be strictly positive.")
    self.assertTrue(torch.all(pre > 0), "Pre-activations must be positive.")

  def test_rotary_functional_immutability(self):
    bsz, seqlen, n_heads, head_dim = 2, 4, 4, 32
    x = torch.randn(bsz, seqlen, n_heads, head_dim, dtype=torch.bfloat16)
    x_orig = x.clone()
    freqs = torch.randn(seqlen, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    rotated = apply_rotary_emb_functional(x, freqs_cis)
    self.assertEqual(rotated.shape, x.shape)
    # Ensure input tensor was not mutated in place
    self.assertTrue(
        torch.equal(x, x_orig),
        "Input tensor was mutated in place by apply_rotary_emb_functional.",
    )

  def test_functional_kv_cache_update_immutability(self):
    bsz, max_seqlen, dim = 2, 32, 64
    kv_cache = torch.zeros(bsz, max_seqlen, dim)
    kv_cache_orig = kv_cache.clone()

    new_kv = torch.ones(bsz, 4, dim)
    start_pos = 8

    updated_cache = update_kv_cache_functional(kv_cache, new_kv, start_pos)
    self.assertEqual(updated_cache.shape, kv_cache.shape)
    # Ensure original cache was not mutated
    self.assertTrue(
        torch.equal(kv_cache, kv_cache_orig),
        "Original cache was mutated in place by update_kv_cache_functional.",
    )
    # Verify values at start_pos:start_pos+4 are updated
    self.assertTrue(torch.all(updated_cache[:, start_pos : start_pos + 4] == 1.0))
    self.assertTrue(torch.all(updated_cache[:, :start_pos] == 0.0))


if __name__ == "__main__":
  absltest.main()
