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

"""Numerical tests for the flex_attention forward kernel on TPU.

The oracle is PyTorch's dense reference for the flex_attention HOP
(`math_attention`) on CPU; like the kernel it returns `lse` in log2 units.
"""

# pylint: disable=protected-access

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._higher_order_ops.flex_attention import math_attention
from torch.nn.attention import flex_attention as fa
from torch_tpu._internal.compile import flex_attention
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

_BLOCK = 512
# VMEM bytes of the forward's 512x512 tile for bf16 and head_dim 128.
_TILE_512_BYTES = 4980736
_TOLERANCES = {
    torch.float32: dict(atol=2e-4, rtol=2e-4),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}
_LSE_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}


def _identity(score, b, h, q_idx, kv_idx):
  del b, h, q_idx, kv_idx
  return score


# Block-aligned masks: no partially visible block.
def _block_causal(b, h, q_idx, kv_idx):
  del b, h
  return q_idx // _BLOCK >= kv_idx // _BLOCK


def _block_causal_empty_first_row(b, h, q_idx, kv_idx):
  return (q_idx >= _BLOCK) & _block_causal(b, h, q_idx, kv_idx)


# Element-level masks: partially visible blocks.
def _causal(b, h, q_idx, kv_idx):
  del b, h
  return q_idx >= kv_idx


def _sliding_window(b, h, q_idx, kv_idx):
  del b, h
  return (q_idx >= kv_idx) & (q_idx - kv_idx < 700)


def _documents(b, h, q_idx, kv_idx):
  del b, h
  return (q_idx // 768 == kv_idx // 768) & (q_idx >= kv_idx)


def _window_per_batch(b, h, q_idx, kv_idx):
  del h
  return (q_idx >= kv_idx) & (q_idx - kv_idx < 640 + 384 * b)


def _first_block_or_batch_1(b, h, q_idx, kv_idx):
  # Batch 0 sees only the first KV block; batch 1 sees every block.
  del h, q_idx
  return (b == 1) | (kv_idx < _BLOCK)


def _prefix_lm(b, h, q_idx, kv_idx):
  # The first 640 tokens attend bidirectionally; the rest causally.
  del b, h
  return (kv_idx < 640) | (q_idx >= kv_idx)


def _causal_offset(b, h, q_idx, kv_idx):
  # Queries are the last T of S keys (prefill after a cache of S - T).
  del b, h
  return q_idx + 1024 >= kv_idx


def _alibi(score, b, h, q_idx, kv_idx):
  del b
  return score - 0.1 * (h + 1) * (q_idx - kv_idx).abs()


def _rel_pos(score, b, h, q_idx, kv_idx):
  del b, h
  return score + (q_idx - kv_idx).to(torch.float32) * 0.01


def _softcap(score, b, h, q_idx, kv_idx):
  # Tanh softcap (as in Gemma 2). TorchTPU lowers tanh(score / 2) with sign and
  # expm1(-|score|); scores are ~N(0, 1), so they fall on both sides of the
  # |x| = 0.007 switch in the splice's expm1.
  del b, h, q_idx, kv_idx
  return 2 * torch.tanh(score / 2)


def _expm1_of_tiny_scores(score, b, h, q_idx, kv_idx):
  # ~score, but evaluates expm1 at |x| < 1e-3, where exp(x) - 1 loses most
  # digits: lowering expm1 as exp(x) - 1 is off by 3e-2 (f32 tolerance 2e-4).
  del b, h, q_idx, kv_idx
  return 1e4 * torch.expm1(score / 1e4)


def _block_mask(mask_mod, batch, seq, kv_seq=None):
  return fa.create_block_mask(
      mask_mod, batch, 1, seq, kv_seq or seq, device="cpu", BLOCK_SIZE=_BLOCK
  )


def _reference(q, k, v, block_mask, scale, score_mod):
  q, k, v = (t.detach().cpu().float() for t in (q, k, v))
  out, lse, _ = math_attention(
      q, k, v, score_mod, block_mask.as_tuple(), scale, {}
  )
  return out, lse


def _reference_grads(q, k, v, grad_out, block_mask, scale):
  """dQ, dK and dV on CPU in f32: autograd through PyTorch's math_attention."""
  q, k, v, grad_out = (t.detach().cpu().float() for t in (q, k, v, grad_out))
  for t in (q, k, v):
    t.requires_grad_(True)
  out, _, _ = math_attention(
      q, k, v, _identity, block_mask.as_tuple(), scale, {}
  )
  out.backward(grad_out)
  return q.grad, k.grad, v.grad


class FlexAttentionKilnTest(seed_test_utils.RepeatableTest):

  def _inputs(self, batch, q_heads, kv_heads, seq, dim, dtype):
    torch.manual_seed(0)
    device = torch.device("tpu")
    q = torch.randn(batch, q_heads, seq, dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, seq, dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_heads, seq, dim, dtype=dtype, device=device)
    return q, k, v

  def _check(
      self,
      q,
      k,
      v,
      mask_mod,
      *,
      score_mod=None,
      mask_batch=1,
      rows=slice(None),
      block_mask=None,
      kiln=False,
  ):
    """Runs the kernel and compares (out, lse) with the CPU reference.

    `kiln` calls the Kiln op directly, also where flex_attention_fwd would use
    flash attention.
    """
    scale = q.shape[-1] ** -0.5
    if block_mask is None:
      block_mask = _block_mask(mask_mod, mask_batch, q.shape[-2], k.shape[-2])
    if kiln:
      out, lse = torch.ops.tpu._flex_attention_fwd(
          q,
          k,
          v,
          *flex_attention.kernel_args(
              q, k, v, block_mask, scale=scale, score_mod=score_mod
          ),
      )
      lse = lse.reshape(q.shape[:3])
    else:
      out, lse = flex_attention.flex_attention_fwd(
          q,
          k,
          v,
          block_mask,
          scale=scale,
          score_mod=score_mod,
      )
    self.assertEqual(out.dtype, q.dtype)
    self.assertEqual(lse.dtype, torch.float32)
    ref_out, ref_lse = _reference(
        q, k, v, block_mask, scale, score_mod or _identity
    )
    out, lse = out.cpu().float(), lse.cpu()
    test_utils.assert_close(
        out[:, :, rows], ref_out[:, :, rows], **_TOLERANCES[q.dtype]
    )
    test_utils.assert_close(
        lse[:, :, rows], ref_lse[:, :, rows], **_LSE_TOLERANCES[q.dtype]
    )
    return out, lse

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      mask_mod=[fa.noop_mask, _block_causal],
  )
  def test_matches_reference(self, dtype, mask_mod):
    q, k, v = self._inputs(1, 2, 2, 2048, 128, dtype)
    self._check(q, k, v, mask_mod, kiln=True)

  def test_single_tile(self):
    # The first and last KV step of a row are the same grid step.
    q, k, v = self._inputs(1, 2, 2, _BLOCK, 128, torch.float32)
    self._check(q, k, v, fa.noop_mask, kiln=True)

  def test_batched_gqa(self):
    # The block mask is shared across the batch (mask_batch=1).
    q, k, v = self._inputs(2, 8, 2, 1024, 128, torch.float32)
    self._check(q, k, v, _block_causal)

  def test_fully_masked_rows_give_zero_and_neg_inf_lse(self):
    q, k, v = self._inputs(1, 1, 1, 1024, 128, torch.float32)
    # Query block 0 lists no KV block; the other rows must be unaffected.
    out, lse = self._check(
        q, k, v, _block_causal_empty_first_row, rows=slice(_BLOCK, None)
    )
    self.assertTrue((out[:, :, :_BLOCK] == 0).all())
    self.assertTrue((lse[:, :, :_BLOCK] == -torch.inf).all())

  @parameterized.parameters(_sliding_window, _documents, _prefix_lm)
  def test_mask_mod(self, mask_mod):
    q, k, v = self._inputs(1, 2, 2, 2048, 128, torch.float32)
    self._check(q, k, v, mask_mod)

  def test_mask_mod_uses_batch_with_gqa(self):
    q, k, v = self._inputs(2, 4, 2, 2048, 128, torch.float32)
    self._check(q, k, v, _window_per_batch, mask_batch=2)

  def test_mask_mod_applies_to_a_single_block_list(self):
    # Full and partial blocks in one list (separate_full_blocks=False).
    seq = 2048
    dense = _block_mask(_causal, 1, seq).to_dense()
    block_mask = fa.BlockMask.from_kv_blocks(
        dense.sum(-1).to(torch.int32),
        dense.argsort(dim=-1, descending=True, stable=True).to(torch.int32),
        BLOCK_SIZE=_BLOCK,
        mask_mod=_causal,
        seq_lengths=(seq, seq),
    )
    self.assertIsNone(block_mask.full_kv_num_blocks)
    q, k, v = self._inputs(1, 2, 2, seq, 128, torch.float32)
    self._check(q, k, v, _causal, block_mask=block_mask)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_score_mod(self, dtype):
    q, k, v = self._inputs(1, 2, 2, 1024, 128, dtype)
    self._check(q, k, v, _causal, score_mod=_alibi)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_softcap_score_mod(self, dtype):
    # tanh lowers to expm1 and sign, which the splice rewrites for Mosaic.
    q, k, v = self._inputs(1, 2, 2, 1024, 128, dtype)
    self._check(q, k, v, _causal, score_mod=_softcap)

  def test_expm1_score_mod_is_accurate_near_zero(self):
    # The splice's expm1 uses x + x^2/2 near 0 rather than exp(x) - 1.
    q, k, v = self._inputs(1, 2, 2, 1024, 128, torch.float32)
    self._check(q, k, v, _causal, score_mod=_expm1_of_tiny_scores)

  def test_merges_mask_blocks_into_the_tile_with_a_mask_mod(self):
    block_mask = _block_mask(_causal, 1, 4 * _BLOCK)
    num_blocks, indices, tile = flex_attention._block_tables(
        block_mask, True, (2 * _BLOCK, 2 * _BLOCK)
    )
    self.assertEqual(tile, (2 * _BLOCK, 2 * _BLOCK))
    self.assertEqual(num_blocks.tolist(), [[[1, 2]]])
    self.assertEqual(indices.tolist(), [[[[0, 1], [0, 1]]]])

  def test_merges_uniform_mask_blocks_without_a_mask_mod(self):
    block_mask = _block_mask(fa.noop_mask, 1, 4 * _BLOCK)
    num_blocks, _, tile = flex_attention._block_tables(
        block_mask, False, (2 * _BLOCK, 2 * _BLOCK)
    )
    self.assertEqual(tile, (2 * _BLOCK, 2 * _BLOCK))
    self.assertEqual(num_blocks.tolist(), [[[2, 2]]])

  def test_keeps_mixed_mask_blocks_without_a_mask_mod(self):
    block_mask = _block_mask(_block_causal, 1, 4 * _BLOCK)
    num_blocks, _, tile = flex_attention._block_tables(
        block_mask, False, (2 * _BLOCK, 2 * _BLOCK)
    )
    self.assertEqual(tile, (_BLOCK, _BLOCK))
    self.assertEqual(num_blocks.tolist(), [[[1, 2, 3, 4]]])

  def test_tables_by_kv_list_the_query_blocks_of_each_kv_block(self):
    block_mask = _block_mask(_causal, 1, 4 * _BLOCK)
    num_blocks, indices, tile = flex_attention._block_tables(
        block_mask, True, (2 * _BLOCK, 2 * _BLOCK), by_kv=True
    )
    self.assertEqual(tile, (2 * _BLOCK, 2 * _BLOCK))
    # KV tile 0 is seen by both query tiles, KV tile 1 by query tile 1 only.
    self.assertEqual(num_blocks.tolist(), [[[2, 1]]])
    self.assertEqual(indices.tolist(), [[[[0, 1], [1, 0]]]])

  @parameterized.parameters(
      # (backward, VMEM limit, expected tile); the limit is half the capacity.
      (False, _TILE_512_BYTES, (512, 512)),
      (False, _TILE_512_BYTES - 1, (512, 256)),
      # The backward needs twice the forward's VMEM.
      (True, 2 * _TILE_512_BYTES, (512, 512)),
      (True, 2 * _TILE_512_BYTES - 1, (512, 256)),
      (True, _TILE_512_BYTES, (256, 256)),
  )
  def test_kernel_tile_fits_vmem(self, backward, limit, tile):
    q = torch.zeros(1, 1, 512, 128, dtype=torch.bfloat16)
    block_mask = fa.create_block_mask(
        fa.noop_mask, 1, 1, 512, 512, device="cpu", BLOCK_SIZE=128
    )
    args_fn = (
        (lambda *a: flex_attention._bwd_kernel_args(*a)[1])
        if backward
        else flex_attention.kernel_args
    )
    with mock.patch.object(
        flex_attention._device_ops_backend,
        "_get_vmem_capacity_bytes",
        return_value=2 * limit,
    ):
      args = args_fn(q, q, q, block_mask)
    self.assertEqual(tuple(args[4:6]), tile)
    self.assertAlmostEqual(args[6], 128**-0.5)  # The default scale.

  def _check_grads(self, q, k, v, mask_mod, *, mask_batch=1):
    scale = q.shape[-1] ** -0.5
    block_mask = _block_mask(mask_mod, mask_batch, q.shape[-2], k.shape[-2])
    out, lse = flex_attention.flex_attention_fwd(
        q, k, v, block_mask, scale=scale
    )
    grad_out = torch.randn_like(out)
    grads = flex_attention.flex_attention_bwd(
        q, k, v, out, lse, grad_out, block_mask, scale=scale
    )
    refs = _reference_grads(q, k, v, grad_out, block_mask, scale)
    # Measured worst abs error over dQ, dK, dV: 2.5e-4 (f32), 5.7e-3 (bf16).
    # TODO(b/576669251): Tighten f32 once the forward lse is refined.
    tol = 5e-4 if q.dtype == torch.float32 else 2e-2
    for name, grad, ref in zip(("dq", "dk", "dv"), grads, refs, strict=True):
      with self.subTest(name):
        self.assertEqual(grad.dtype, q.dtype)
        test_utils.assert_close(grad.cpu().float(), ref, atol=tol, rtol=tol)
    return grads

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      mask_mod=[fa.noop_mask, _causal],
  )
  def test_grads_match_reference(self, dtype, mask_mod):
    # Causal: each query tile lists only the KV tiles up to the diagonal.
    q, k, v = self._inputs(1, 2, 2, 2048, 128, dtype)
    self._check_grads(q, k, v, mask_mod)

  @parameterized.parameters(_sliding_window, _documents, _prefix_lm)
  def test_grads_mask_mod(self, mask_mod):
    # The mask_mod is spliced into both backward kernels.
    q, k, v = self._inputs(1, 2, 2, 2048, 128, torch.float32)
    self._check_grads(q, k, v, mask_mod)

  def test_grads_batched_gqa(self):
    # dK / dV sum over the query heads of each KV head, dQ reads its query
    # head's KV head, and each batch uses its own block tables.
    q, k, v = self._inputs(2, 4, 2, 4096, 128, torch.float32)
    self._check_grads(q, k, v, _first_block_or_batch_1, mask_batch=2)

  def test_grads_single_tile(self):
    # The first and last KV step of a row are the same grid step.
    q, k, v = self._inputs(1, 2, 2, _BLOCK, 128, torch.float32)
    self._check_grads(q, k, v, _causal)

  def test_grads_with_fully_masked_rows(self):
    # 32 MiB of VMEM gives 512x512 f32 backward tiles, so query tile 0 lists
    # no KV tile: its dQ must be 0.
    q, k, v = self._inputs(1, 1, 1, 1024, 128, torch.float32)
    with mock.patch.object(
        flex_attention._device_ops_backend,
        "_get_vmem_capacity_bytes",
        return_value=32 << 20,
    ):
      dq, _, _ = self._check_grads(q, k, v, _block_causal_empty_first_row)
    self.assertTrue((dq[:, :, :_BLOCK] == 0).all())

  def test_rel_pos_score_mod(self):
    q, k, v = self._inputs(1, 2, 2, 1024, 128, torch.float32)
    self._check(q, k, v, _causal, score_mod=_rel_pos)

  def test_rectangular(self):
    q, _, _ = self._inputs(1, 2, 2, 3072, 128, torch.float32)
    _, k, v = self._inputs(1, 2, 2, 4096, 128, torch.float32)
    self._check(q, k, v, _causal_offset)

  def test_grads_rectangular(self):
    # dQ: query tile 0 skips the last KV tile; dK/dV: KV tile 3 skips query
    # tile 0.
    q, _, _ = self._inputs(1, 2, 2, 3072, 128, torch.float32)
    _, k, v = self._inputs(1, 2, 2, 4096, 128, torch.float32)
    self._check_grads(q, k, v, _causal_offset)

  def test_dense_uses_flash_attention(self):
    q, k, v = self._inputs(1, 2, 2, 2048, 128, torch.bfloat16)
    out, _ = self._check(q, k, v, fa.noop_mask)
    # Bit-exact with F.sdpa: dense attention ran flash attention.
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5
    )
    test_utils.assert_close(out, expected.cpu().float(), atol=0, rtol=0)
    # A score_mod keeps dense attention on the Kiln kernel.
    self._check(q, k, v, fa.noop_mask, score_mod=_alibi)


if __name__ == "__main__":
  absltest.main()
