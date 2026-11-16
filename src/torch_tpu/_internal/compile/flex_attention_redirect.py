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

"""Runs PyTorch flex_attention on TPU tensors.

Upstream flex_attention rejects TPU tensors and has no TPU kernel for its
higher-order op. `patch()` lets TPU tensors through and redirects the common
case (no score_mod, no LSE requested) to scaled_dot_product_attention with an
additive bias built from the BlockMask, which TorchTPU lowers to its flash
attention kernel. Everything else falls back to upstream's eager math path.
"""

# pylint: disable=protected-access

import weakref

import torch
from torch._higher_order_ops.flex_attention import flex_attention_backward
from torch.nn.attention import flex_attention
import torch.nn.functional as F


def equal_square_tile_size(mask: torch.Tensor) -> int | None:
  """Returns d if `mask` is T // d identical d x d tiles along its diagonal.

  Such a [T, T] bool mask is T // d equal-length documents that share one
  within-document mask and never see each other.
  """
  size = mask.shape[0]
  # Entry i of the sub-diagonal is mask[i + 1, i]: False where token i + 1
  # can't see token i, i.e. where a new document starts.
  doc_starts = (~mask.diagonal(offset=-1)).nonzero()
  if doc_starts.numel() == 0:
    return None
  doc_len = int(doc_starts[0]) + 1
  if size % doc_len:
    return None
  # Accept only if the first document's mask, repeated along the diagonal,
  # reproduces the whole mask.
  first_doc = mask[:doc_len, :doc_len]
  num_docs = size // doc_len
  tiled = torch.block_diag(*[first_doc] * num_docs)
  return doc_len if torch.equal(tiled, mask) else None


def patch() -> None:
  """Patches PyTorch flex_attention to support TPU tensors."""
  original_validate = flex_attention._validate_device

  def _validate_device(query, key, value):
    if query.device.type != "tpu":
      original_validate(query, key, value)

  flex_attention._validate_device = _validate_device

  dispatch_key = torch._C.DispatchKey.CompositeExplicitAutograd
  original_bwd = flex_attention_backward.py_kernels.pop(dispatch_key)

  @flex_attention_backward.py_impl(dispatch_key)
  def _sdpa_dense_backward(query, key, value, *args, **kwargs):
    with query.device:
      return original_bwd(query, key, value, *args, **kwargs)

  original_flex_attention = flex_attention.flex_attention

  _doc_len_cache = weakref.WeakKeyDictionary()

  @torch.compiler.assume_constant_result
  def _get_doc_len(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      device: torch.device,
  ) -> int | None:
    """Returns the document length if `block_mask` is equal-length documents."""
    cache = _doc_len_cache.setdefault(block_mask, {})
    key = (q_len, kv_len)
    if key in cache:
      return cache[key]

    b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
    doc_len = None
    if (
        q_len == kv_len
        and b_mask == 1
        and h_mask == 1
        # Sliced BlockMasks carry an error-raising mask_mod stub.
        and block_mask.mask_mod
        not in (
            flex_attention.noop_mask,
            getattr(flex_attention, "_sliced_mask_mod_error", None),
        )
    ):
      with torch.no_grad():
        mask = flex_attention.create_mask(
            block_mask.mask_mod, 1, 1, q_len, kv_len, device=device
        )
      doc_len = equal_square_tile_size(mask[0, 0])
    return cache.setdefault(key, doc_len)

  _causal_mask_cache = weakref.WeakKeyDictionary()

  @torch.compiler.assume_constant_result
  def _is_causal_block_mask(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      device: torch.device,
  ) -> bool:
    """Returns True if `block_mask` is a lower-triangular causal mask."""
    cache = _causal_mask_cache.setdefault(block_mask, {})
    key = (q_len, kv_len)
    if key in cache:
      return cache[key]

    b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
    is_square = q_len > 0 and q_len == kv_len
    is_broadcastable = b_mask == 1 and h_mask == 1
    if not (is_square and is_broadcastable):
      return cache.setdefault(key, False)

    bool_mask = flex_attention.create_mask(
        block_mask.mask_mod, 1, 1, q_len, kv_len, device=device
    )
    expected = torch.ones(
        (q_len, kv_len), dtype=torch.bool, device=device
    ).tril_()

    is_causal = torch.equal(bool_mask[0, 0], expected)
    return cache.setdefault(key, is_causal)

  def _get_sdpa_attn_bias(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor:
    # Cached on the BlockMask. Under torch.compile Dynamo records the cache
    # write as a side effect, so the bias is built in-graph once and later
    # graphs read it as an input.
    cache = block_mask.__dict__.setdefault("_tpu_sdpa_mask_cache", {})
    key = (q_len, kv_len, dtype, device)
    if key not in cache:
      b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
      with torch.no_grad():
        bool_mask = flex_attention.create_mask(
            block_mask.mask_mod,
            b_mask,
            h_mask,
            q_len,
            kv_len,
            device=device,
        )
        attn_bias = torch.zeros(
            bool_mask.shape, dtype=dtype, device=device
        ).masked_fill_(~bool_mask, float("-inf"))
      cache[key] = attn_bias
    return cache[key]

  def _flex_attention_tpu(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      score_mod=None,
      block_mask=None,
      scale=None,
      enable_gqa: bool = False,
      return_lse: bool = False,
      kernel_options=None,
      *,
      return_aux=None,
  ):
    need_aux = return_lse or (
        return_aux is not None
        and (
            getattr(return_aux, "lse", False)
            or getattr(return_aux, "max_scores", False)
        )
    )
    if (
        query.device.type == "tpu"
        and score_mod is None
        and not need_aux
        and query.dtype in (torch.bfloat16, torch.float32)
    ):
      q_len = query.shape[-2]
      kv_len = key.shape[-2]
      # Equal-length packed documents (a block-diagonal mask with identical
      # tiles) attend per document, with the documents folded into the batch.
      num_docs = 1
      if block_mask is not None and query.shape[0] == 1:
        doc_len = _get_doc_len(block_mask, q_len, kv_len, query.device)
        if doc_len:
          num_docs = q_len // doc_len

      doc_q_len = q_len // num_docs
      doc_kv_len = kv_len // num_docs
      is_causal = (
          _is_causal_block_mask(block_mask, doc_q_len, doc_kv_len, query.device)
          if block_mask is not None
          else False
      )
      attn_bias = (
          _get_sdpa_attn_bias(
              block_mask,
              doc_q_len,
              doc_kv_len,
              query.dtype,
              query.device,
          )
          if block_mask is not None and not is_causal
          else None
      )

      if num_docs > 1:

        def fold(
            tensor: torch.Tensor,
        ) -> torch.Tensor:  # [1,H,T,D] -> [N,H,T/N,D]
          return tensor.reshape(
              tensor.shape[1], num_docs, doc_q_len, -1
          ).transpose(0, 1)

        query, key, value = fold(query), fold(key), fold(value)

      out = F.scaled_dot_product_attention(
          query,
          key,
          value,
          attn_mask=attn_bias,
          dropout_p=0.0,
          is_causal=is_causal,
          scale=scale,
          enable_gqa=enable_gqa,
      )

      if num_docs > 1:
        out = out.transpose(0, 1).reshape(1, -1, q_len, out.shape[-1])

      if return_aux is not None:
        return out, flex_attention.AuxOutput()
      return out

    return original_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
        return_aux=return_aux,
    )

  flex_attention.flex_attention = _flex_attention_tpu
