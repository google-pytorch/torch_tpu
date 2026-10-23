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

"""flex_attention forward on TPU through the Kiln kernel."""

import math

import torch
from torch.nn.attention import flex_attention as fa


def _block_tables(
    block_mask: fa.BlockMask,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns the (kv_num_blocks, kv_indices) tables of the fully visible blocks.

  Partially visible blocks need a per-element mask, which the kernel does not
  have yet, so a mask that has any is rejected.
  """
  if block_mask.full_kv_num_blocks is None:
    # Built with separate_full_blocks=False: one list holds every block.
    return block_mask.kv_num_blocks, block_mask.kv_indices
  if block_mask.kv_num_blocks.any():
    raise NotImplementedError(
        "flex_attention on TPU does not support partially masked blocks yet"
    )
  return block_mask.full_kv_num_blocks, block_mask.full_kv_indices


def flex_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: fa.BlockMask,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Attends over the KV blocks listed in `block_mask`, each fully visible.

  There is no per-element mask yet, so only block masks whose blocks are all
  either fully visible or fully hidden are accepted.

  Args:
    query: [B, Hq, T, D] bf16 or f32 TPU tensor.
    key: [B, Hkv, S, D], with Hkv dividing Hq.
    value: [B, Hkv, S, Dv].
    block_mask: BlockMask whose block sizes divide T and S.
    scale: Softmax scale; defaults to 1 / sqrt(D).

  Returns:
    `(out, lse)`: out is [B, Hq, T, Dv] in query's dtype; lse is f32
    [B, Hq, T] in log2 units, matching the flex_attention HOP contract.
  """
  if scale is None:
    scale = 1.0 / math.sqrt(query.shape[-1])
  kv_num_blocks, kv_indices = _block_tables(block_mask)
  mask_batch, mask_heads, _ = kv_num_blocks.shape
  device = query.device
  out, lse = torch.ops.tpu._flex_attention_fwd(
      query,
      key,
      value,
      kv_num_blocks.to(torch.int32).reshape(-1).to(device),
      kv_indices.to(torch.int32).reshape(-1).to(device),
      mask_batch,
      mask_heads,
      block_mask.BLOCK_SIZE[0],
      block_mask.BLOCK_SIZE[1],
      float(scale),
  )
  return out, lse.reshape(query.shape[:3])
