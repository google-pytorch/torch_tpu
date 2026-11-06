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

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._higher_order_ops.flex_attention import math_attention
from torch.nn.attention import flex_attention as fa
from torch_tpu._internal.compile import flex_attention
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

_BLOCK = 512
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


def _alibi(score, b, h, q_idx, kv_idx):
  del b
  return score - 0.1 * (h + 1) * (q_idx - kv_idx).abs()


def _block_mask(mask_mod, batch, seq):
  return fa.create_block_mask(
      mask_mod, batch, 1, seq, seq, device="cpu", BLOCK_SIZE=_BLOCK
  )


def _reference(q, k, v, block_mask, scale, score_mod):
  q, k, v = (t.detach().cpu().float() for t in (q, k, v))
  out, lse, _ = math_attention(
      q, k, v, score_mod, block_mask.as_tuple(), scale, {}
  )
  return out, lse


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
  ):
    """Runs the kernel and compares (out, lse) with the CPU reference."""
    scale = q.shape[-1] ** -0.5
    if block_mask is None:
      block_mask = _block_mask(mask_mod, mask_batch, q.shape[-2])
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
    self._check(q, k, v, mask_mod)

  def test_single_tile(self):
    # The first and last KV step of a row are the same grid step.
    q, k, v = self._inputs(1, 2, 2, _BLOCK, 128, torch.float32)
    self._check(q, k, v, fa.noop_mask)

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

  @parameterized.parameters(_sliding_window, _documents)
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


if __name__ == "__main__":
  absltest.main()
