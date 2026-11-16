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
from torch_tpu._internal.device import _device_ops_backend


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


def _round_up(n: int, multiple: int) -> int:
  """Returns the smallest multiple of `multiple` that is >= `n`."""
  return (n + multiple - 1) // multiple * multiple


def _flash_tile(
    q_len: int, kv_len: int, head_dim: int, is_f32: bool, backward: bool
) -> tuple[int, int]:
  """Returns the largest (q, kv) tile that fits in VMEM.

  Port of CreateTiling (flash_attention_kernel_builder.cc), without a mask or
  dropout.
  """
  elem = 4 if is_f32 else 2

  def initial_tile(n):  # GetInitialTileSize
    return 128 if n <= 128 else 256 if n <= 256 else _round_up(n, 256)

  def vmem_bytes(qt, kt):  # EstimateTileVmemBytes
    linear = 2 * (qt * head_dim * (3 * elem + 4) + kt * head_dim * 2 * elem)
    total = linear + 2 * qt * 1024 + 2 * 4 * qt * kt
    return 2 * total if backward else total

  def halve(n):
    return max(256, _round_up(n // 2, 256))

  def kv_compute_tile(kt):  # GetKvComputeTileSize
    if kt < 1024:
      return kt
    for kc in (1024, 768, 512, 256):
      if kt % kc == 0:
        return kc
    return 128

  # Same limit as the kernel (flex_attention_aten_kernels.cc).
  limit = _device_ops_backend._get_vmem_capacity_bytes() // 2  # pylint: disable=protected-access
  qt, kt = initial_tile(q_len), initial_tile(kv_len)
  while vmem_bytes(qt, kt) > limit and (qt > 256 or kt > 256):
    if qt > kt and qt > 256:
      qt = halve(qt)
    elif kt > 256:
      kt = halve(kt)
    else:
      qt = halve(qt)
  return qt, kv_compute_tile(kt)


def _fit(tile: int, block: int, length: int) -> int:
  """Returns the largest multiple of `block` <= `tile` dividing `length`.

  Uses the same heuristic as F.sdpa, but can be improved in the future.
  """
  if length % block:
    return block
  tile = max(block, tile // block * block)
  while length % tile:
    tile -= block
  return tile


def _block_tables(
    block_mask: fa.BlockMask,
    has_mask_mod: bool,
    tile: tuple[int, int],
    by_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
  """Merges the mask's blocks into `tile` blocks for the kernel to visit.

  Returns (kv_num_blocks, kv_indices, tile), or with `by_kv` the query blocks
  of each KV block (q_num_blocks, q_indices, tile). Without a mask_mod, keeps
  the mask's block size if a merged block would mix listed and unlisted blocks.
  """
  dense = block_mask.to_dense()
  q_block, kv_block = block_mask.BLOCK_SIZE
  rq, rk = tile[0] // q_block, tile[1] // kv_block
  b, h, nq, nkv = dense.shape
  dense = dense.to(torch.int32)
  groups = dense.reshape(b, h, nq // rq, rq, nkv // rk, rk)
  merged = groups.amax(dim=(3, 5))
  if not has_mask_mod and not torch.equal(merged, groups.amin(dim=(3, 5))):
    merged, tile = dense, (q_block, kv_block)
  if by_kv:
    merged = merged.transpose(-2, -1)
  num_blocks = merged.sum(-1)
  indices = merged.argsort(dim=-1, descending=True, stable=True)
  return num_blocks, indices, tile


def _tile_and_mask_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: fa.BlockMask,
    backward: bool,
) -> tuple[tuple[int, int], Callable[..., torch.Tensor] | None]:
  """Returns the kernel tile and the mask_mod to splice (None for noop)."""
  q_len, kv_len = query.shape[-2], key.shape[-2]
  mask_q_block, mask_kv_block = block_mask.BLOCK_SIZE
  qt, kc = _flash_tile(
      q_len,
      kv_len,
      max(query.shape[-1], value.shape[-1]),
      query.dtype == torch.float32,
      backward,
  )
  tile = (_fit(qt, mask_q_block, q_len), _fit(kc, mask_kv_block, kv_len))
  # A noop mask_mod masks nothing; skip it.
  mask_mod = block_mask.mask_mod
  if mask_mod is fa.noop_mask:
    mask_mod = None
  return tile, mask_mod


def _flat_tables(
    num_blocks: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  return (
      num_blocks.reshape(-1).to(device=device, dtype=torch.int32),
      indices.reshape(-1).to(device=device, dtype=torch.int32),
  )


def kernel_args(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: fa.BlockMask,
    *,
    scale: float | None = None,
    score_mod: Callable[..., torch.Tensor] | None = None,
) -> tuple[object, ...]:
  """Returns the arguments of `_flex_attention_fwd` after query, key, value."""
  if scale is None:
    scale = 1.0 / math.sqrt(query.shape[-1])
  tile, mask_mod = _tile_and_mask_mod(
      query, key, value, block_mask, backward=False
  )
  kv_num_blocks, kv_indices, (q_block, kv_block) = _block_tables(
      block_mask, mask_mod is not None, tile
  )
  mask_batch, mask_heads, _ = kv_num_blocks.shape

  def mlir(mod, has_score):
    if mod is None:
      return ""
    return mod_to_mlir(
        mod, q_block=q_block, kv_block=kv_block, has_score=has_score
    )

  return (
      *_flat_tables(kv_num_blocks, kv_indices, query.device),
      mask_batch,
      mask_heads,
      q_block,
      kv_block,
      float(scale),
      mlir(score_mod, has_score=True),
      mlir(mask_mod, has_score=False),
  )


def _is_dense(
    block_mask: fa.BlockMask,
    score_mod: Callable[..., torch.Tensor] | None,
    kv_len: int,
) -> bool:
  """True if no score is masked or modified, from the block tables alone."""
  if score_mod is not None or block_mask.mask_mod is not fa.noop_mask:
    return False
  num_blocks = block_mask.kv_num_blocks
  if block_mask.full_kv_num_blocks is not None:
    num_blocks = num_blocks + block_mask.full_kv_num_blocks
  return bool(torch.all(num_blocks.eq(kv_len // block_mask.BLOCK_SIZE[1])))


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
  if _is_dense(block_mask, score_mod, key.shape[-2]):
    # Flash attention is faster here; its lse is also in log2 units.
    out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
        query, key, value, None, True, scale=scale
    )[:2]
    return out, lse.reshape(query.shape[:3])
  out, lse = torch.ops.tpu._flex_attention_fwd(
      query,
      key,
      value,
      *kernel_args(
          query,
          key,
          value,
          block_mask,
          scale=scale,
          score_mod=score_mod,
      ),
  )
  return out, lse.reshape(query.shape[:3])


def _dkv_kernel_args(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: fa.BlockMask,
    *,
    scale: float | None = None,
) -> tuple[object, ...]:
  """Returns the arguments of `_flex_attention_bwd_dkv` after di."""
  if scale is None:
    scale = 1.0 / math.sqrt(query.shape[-1])
  tile, mask_mod = _tile_and_mask_mod(
      query, key, value, block_mask, backward=True
  )
  q_num_blocks, q_indices, (q_block, kv_block) = _block_tables(
      block_mask, mask_mod is not None, tile, by_kv=True
  )
  mask_batch, mask_heads, _ = q_num_blocks.shape
  mask_mlir = ""
  if mask_mod is not None:
    mask_mlir = mod_to_mlir(
        mask_mod, q_block=q_block, kv_block=kv_block, has_score=False
    )
  return (
      *_flat_tables(q_num_blocks, q_indices, query.device),
      mask_batch,
      mask_heads,
      q_block,
      kv_block,
      float(scale),
      mask_mlir,
  )


def flex_attention_bwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    block_mask: fa.BlockMask,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns (dk, dv) of `flex_attention_fwd` for the gradient `grad_out`.

  `out` and `lse` are the forward's results; the other arguments are as in
  `flex_attention_fwd`. score_mod is not supported yet.
  """
  rows = (*query.shape[:2], 1, query.shape[2])
  di = (grad_out.float() * out.float()).sum(-1)
  return torch.ops.tpu._flex_attention_bwd_dkv(
      query,
      key,
      value,
      grad_out,
      lse.reshape(rows),
      di.reshape(rows),
      *_dkv_kernel_args(query, key, value, block_mask, scale=scale),
  )
