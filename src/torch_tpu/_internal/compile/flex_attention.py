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

"""flex_attention forward on TPU."""

from collections.abc import Callable
import math

import torch
from torch.nn.attention import flex_attention as fa
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile


def mod_to_mlir(
    mod: Callable[..., torch.Tensor],
    *,
    q_block: int,
    kv_block: int,
    has_score: bool,
) -> str:
  """Lowers a `score_mod` / `mask_mod` to the StableHLO the kernel splices.

  `mod` runs once on TPU placeholders shaped as one full [q_block, kv_block]
  tile (score f32, then b / h / q_idx / kv_idx int32) with every op deferred,
  and the deferred graph is emitted as `@main(score?, b, h, q_idx, kv_idx)`.
  Tracing at the full tile shape keeps broadcasts out of the module, which
  SpliceStablehlo requires.
  """
  shape = [q_block, kv_block]
  dtypes = ([torch.float32] if has_score else []) + [torch.int32] * 4
  args = [tpu_torch_compile.placeholder(shape, dt, False) for dt in dtypes]
  with execution_mode.set_eager_mode(
      execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH  # pylint: disable=protected-access
  ):
    # Clone so an in-place op in `mod` cannot overwrite a placeholder.
    out = mod(*[a.clone() for a in args])
    name = "score_mod" if has_score else "mask_mod"
    expected = torch.float32 if has_score else torch.bool
    if out.dtype != expected:
      raise TypeError(f"expected {name} to return {expected}, got {out.dtype}")
    if list(out.shape) != shape:
      raise ValueError(
          f"expected {name} to return shape {shape}, got {list(out.shape)}"
      )
    module = tpu_torch_compile.build_mlir(
        result_tensors=[out], argument_tensors=args, use_stablehlo_bounds=False
    )
  return tpu_torch_compile.serialize_mlir_text(module)


def _block_tables(
    block_mask: fa.BlockMask,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns the (kv_num_blocks, kv_indices) of every listed block."""
  dense = block_mask.to_dense()
  return dense.sum(-1), dense.argsort(dim=-1, descending=True, stable=True)


def flex_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: fa.BlockMask,
    *,
    scale: float | None = None,
    score_mod: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Attends over the KV blocks listed in `block_mask`, applying the mods.

  Args:
    query: [B, Hq, T, D] bf16 or f32 TPU tensor.
    key: [B, Hkv, S, D], with Hkv dividing Hq.
    value: [B, Hkv, S, Dv].
    block_mask: BlockMask whose block sizes divide T and S. Its mask_mod is
      applied to every listed block, as in flex_attention.
    scale: Softmax scale; defaults to 1 / sqrt(D).
    score_mod: `(score, b, h, q_idx, kv_idx) -> score`, as in flex_attention.

  Returns:
    `(out, lse)`: out is [B, Hq, T, Dv] in query's dtype; lse is f32
    [B, Hq, T] in log2 units, matching the flex_attention HOP contract.
  """
  if scale is None:
    scale = 1.0 / math.sqrt(query.shape[-1])
  q_block, kv_block = block_mask.BLOCK_SIZE
  kv_num_blocks, kv_indices = _block_tables(block_mask)
  mask_batch, mask_heads, _ = kv_num_blocks.shape
  device = query.device
  # A noop mask_mod keeps every listed block whole; skip splicing it.
  mask_mod = block_mask.mask_mod
  if mask_mod is fa.noop_mask:
    mask_mod = None

  def mlir(mod, has_score):
    if mod is None:
      return ""
    return mod_to_mlir(
        mod, q_block=q_block, kv_block=kv_block, has_score=has_score
    )

  out, lse = torch.ops.tpu._flex_attention_fwd(
      query,
      key,
      value,
      kv_num_blocks.reshape(-1).to(device=device, dtype=torch.int32),
      kv_indices.reshape(-1).to(device=device, dtype=torch.int32),
      mask_batch,
      mask_heads,
      q_block,
      kv_block,
      float(scale),
      mlir(score_mod, has_score=True),
      mlir(mask_mod, has_score=False),
  )
  return out, lse.reshape(query.shape[:3])
