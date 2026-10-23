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

"""Numerical tests for the Kiln flex_attention forward kernel on TPU.

The oracle is PyTorch's dense reference for the flex_attention HOP
(`math_attention`) on CPU; like the kernel it returns `lse` in log2 units.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._higher_order_ops.flex_attention import math_attention
from torch.nn.attention import flex_attention as fa
from torch_tpu._internal.compile import flex_attention_kiln
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


# The kernel has no per-element mask yet, so only block-aligned masks apply.
def _block_causal(b, h, q_idx, kv_idx):
  del b, h
  return q_idx // _BLOCK >= kv_idx // _BLOCK


def _block_diagonal(b, h, q_idx, kv_idx):
  del b, h
  return q_idx // _BLOCK == kv_idx // _BLOCK


def _block_causal_empty_first_row(b, h, q_idx, kv_idx):
  return (q_idx >= _BLOCK) & _block_causal(b, h, q_idx, kv_idx)


def _block_mask(mask_mod, seq):
  return fa.create_block_mask(
      mask_mod, 1, 1, seq, seq, device="cpu", BLOCK_SIZE=_BLOCK
  )


def _reference(q, k, v, mask_mod, scale):
  q, k, v = (t.detach().cpu().float() for t in (q, k, v))
  identity = lambda score, b, h, q_idx, kv_idx: score
  out, lse, _ = math_attention(
      q,
      k,
      v,
      identity,
      _block_mask(mask_mod, q.shape[-2]).as_tuple(),
      scale,
      {},
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

  def _run(self, q, k, v, mask_mod):
    """Returns (out, lse, ref_out, ref_lse) on CPU."""
    scale = q.shape[-1] ** -0.5
    out, lse = flex_attention_kiln.flex_attention_fwd(
        q, k, v, _block_mask(mask_mod, q.shape[-2]), scale=scale
    )
    self.assertEqual(out.dtype, q.dtype)
    self.assertEqual(lse.dtype, torch.float32)
    ref_out, ref_lse = _reference(q, k, v, mask_mod, scale)
    return out.cpu().float(), lse.cpu(), ref_out, ref_lse

  def _check(self, q, k, v, mask_mod, rows=slice(None)):
    out, lse, ref_out, ref_lse = self._run(q, k, v, mask_mod)
    test_utils.assert_close(
        out[:, :, rows], ref_out[:, :, rows], **_TOLERANCES[q.dtype]
    )
    test_utils.assert_close(
        lse[:, :, rows], ref_lse[:, :, rows], **_LSE_TOLERANCES[q.dtype]
    )
    return out, lse

  @parameterized.product(
      dtype=[torch.float32, torch.bfloat16],
      seq=[512, 2048],
      mask_mod=[fa.noop_mask, _block_causal, _block_diagonal],
  )
  def test_matches_reference(self, dtype, seq, mask_mod):
    q, k, v = self._inputs(1, 2, 2, seq, 128, dtype)
    self._check(q, k, v, mask_mod)

  @parameterized.parameters(torch.float32, torch.bfloat16)
  def test_batched_gqa(self, dtype):
    q, k, v = self._inputs(2, 8, 2, 1024, 128, dtype)
    self._check(q, k, v, _block_causal)

  def test_fully_masked_rows_give_zero_and_neg_inf_lse(self):
    q, k, v = self._inputs(1, 1, 1, 1024, 128, torch.float32)
    # Query block 0 lists no KV block; the other rows must be unaffected.
    out, lse = self._check(
        q, k, v, _block_causal_empty_first_row, rows=slice(_BLOCK, None)
    )
    self.assertTrue((out[:, :, :_BLOCK] == 0).all())
    self.assertTrue((lse[:, :, :_BLOCK] == -torch.inf).all())

  # This is only for now, the kernel will support this later.


if __name__ == "__main__":
  absltest.main()
