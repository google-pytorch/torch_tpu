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

"""Overrides for upstream PyTorch Meta and dispatch kernel registrations on TPU."""

import warnings

import torch
from torch import _meta_registrations

_meta_lib: torch.library.Library | None = None


def _patch_flex_attention() -> None:
  """Patches PyTorch flex_attention to support TPU tensors."""
  # pylint: disable=g-import-not-at-top,protected-access
  from torch._higher_order_ops.flex_attention import flex_attention_backward
  from torch.nn.attention import flex_attention
  import torch.nn.functional as F

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

  def _get_sdpa_attn_bias(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor:
    cache = getattr(block_mask, "_tpu_sdpa_mask_cache", None)
    if cache is None:
      cache = {}
      setattr(block_mask, "_tpu_sdpa_mask_cache", cache)
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
      attn_bias = (
          _get_sdpa_attn_bias(
              block_mask, q_len, kv_len, query.dtype, query.device
          )
          if block_mask is not None
          else None
      )
      out = F.scaled_dot_product_attention(
          query,
          key,
          value,
          attn_mask=attn_bias,
          dropout_p=0.0,
          is_causal=False,
          scale=scale,
          enable_gqa=enable_gqa,
      )
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
  # pylint: enable=g-import-not-at-top,protected-access


def _check_grouped_mm(cond: bool | torch.SymBool, msg: str) -> None:
  # `torch._check` defers unbacked (data-dependent) conditions to runtime.
  torch._check(cond, lambda: f"grouped_mm(): {msg}")  # pylint: disable=protected-access


def _dtype_name(dtype: torch.dtype) -> str:
  return str(dtype).removeprefix("torch.")


def _tpu_grouped_mm_meta(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
  """Validates inputs like the TPU eager kernel; returns a contiguous output.

  Upstream's meta requires 16-byte aligned strides and int32 `offs`, which TPU
  doesn't. Only TPU tensors are affected; other devices keep upstream's meta.
  Checks mirror `ValidateGroupedMmInputs` / `GetGroupedMmOutputSize` in
  csrc/ops/grouped_mm/grouped_mm_aten_kernels.cc so eager and compile agree.
  """
  dtype = self.dtype
  # Eager raises NotImplementedError for bias; match it.
  torch._check_not_implemented(  # pylint: disable=protected-access
      bias is None,
      lambda: "grouped_mm(): expected bias to be undefined, got defined tensor",
  )
  _check_grouped_mm(
      dtype == mat2.dtype,
      "expected self and mat2 to have the same dtype, got"
      f" {_dtype_name(dtype)} and {_dtype_name(mat2.dtype)}",
  )
  _check_grouped_mm(
      dtype in (torch.float32, torch.float16, torch.bfloat16),
      "expected self to be float32, float16, or bfloat16, got"
      f" {_dtype_name(dtype)}",
  )
  out_dtype = out_dtype or dtype
  _check_grouped_mm(
      out_dtype == dtype,
      f"expected out_dtype to match self dtype {_dtype_name(dtype)}, got"
      f" {_dtype_name(out_dtype)}",
  )
  if offs is not None:
    _check_grouped_mm(
        offs.dim() == 1, f"expected offs to be 1D, got {offs.dim()}D"
    )
    _check_grouped_mm(
        offs.dtype in (torch.int32, torch.int64),
        f"expected offs to be int32 or int64, got {_dtype_name(offs.dtype)}",
    )
    _check_grouped_mm(
        offs.stride(0) == 1,
        f"expected offs to have stride 1 along dim 0, got {offs.stride(0)}",
    )
  _check_grouped_mm(
      self.dim() in (2, 3), f"expected self to be 2D or 3D, got {self.dim()}D"
  )
  _check_grouped_mm(
      mat2.dim() in (2, 3), f"expected mat2 to be 2D or 3D, got {mat2.dim()}D"
  )

  a_is_2d, b_is_2d = self.dim() == 2, mat2.dim() == 2
  if a_is_2d or b_is_2d:
    _check_grouped_mm(
        offs is not None,
        "offsets must be provided when at least one of self or mat2 is a 2D"
        " tensor",
    )
  else:
    _check_grouped_mm(
        offs is None,
        "offsets must not be provided when both self and mat2 are 3D tensors",
    )
  num_offs = offs.size(0) if offs is not None else 0  # Presence checked above.

  if a_is_2d and b_is_2d:
    out_size = [num_offs, self.size(0), mat2.size(1)]
  elif a_is_2d:
    _check_grouped_mm(
        num_offs == mat2.size(0),
        "expected offs batch size to match mat2 batch size, got"
        f" {num_offs} and {mat2.size(0)}",
    )
    out_size = [self.size(0), mat2.size(-1)]
  elif b_is_2d:
    _check_grouped_mm(
        num_offs == self.size(0),
        "expected offs batch size to match self batch size, got"
        f" {num_offs} and {self.size(0)}",
    )
    out_size = [self.size(1), mat2.size(1)]
  else:
    _check_grouped_mm(
        self.size(0) == mat2.size(0),
        "expected self batch size to match mat2 batch size, got"
        f" {self.size(0)} and {mat2.size(0)}",
    )
    out_size = [self.size(0), self.size(1), mat2.size(-1)]
  _check_grouped_mm(
      self.size(-1) == mat2.size(-2),
      "expected the contraction dimension of self and mat2 to match, got"
      f" {self.size(-1)} and {mat2.size(-2)}",
  )
  return self.new_empty(out_size, dtype=out_dtype)


def _patch_grouped_mm_meta(meta_lib: torch.library.Library) -> None:
  """Replaces the `aten._grouped_mm.default` Meta kernel for TPU tensors."""
  op = torch.ops.aten._grouped_mm.default  # pylint: disable=protected-access
  meta_key = torch._C.DispatchKey.Meta  # pylint: disable=protected-access
  upstream_meta = op.py_kernels.pop(meta_key, None)
  if upstream_meta is None:
    # Don't break `import torch` on torch versions without this meta; the
    # grouped_mm meta tests catch the drift.
    warnings.warn(
        f"torch {torch.__version__} has no Python meta for {op}; TPU"
        " grouped_mm meta override not installed."
    )
    return

  def _grouped_mm_meta(
      self: torch.Tensor,
      mat2: torch.Tensor,
      offs: torch.Tensor | None = None,
      bias: torch.Tensor | None = None,
      out_dtype: torch.dtype | None = None,
  ) -> torch.Tensor:
    # FakeTensors report `meta` as their device here; `device_hint` returns
    # the fake device.
    if _meta_registrations.device_hint(self) != "tpu":
      return upstream_meta(self, mat2, offs, bias, out_dtype)
    return _tpu_grouped_mm_meta(self, mat2, offs, bias, out_dtype)

  # Upstream registers its meta both as a dispatcher Meta kernel and as a
  # Python-dispatcher kernel (see `activate_meta()`); replace both. Older torch
  # defaults `allow_override` to False.
  meta_lib.impl(op, _grouped_mm_meta, allow_override=True)
  op.py_impl(meta_key)(_grouped_mm_meta)


# TODO: Remove once PyTorch includes https://github.com/pytorch/pytorch/pull/198303.
def _patch_sdpa_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten._scaled_dot_product_fused_attention_overrideable` Meta kernel."""
  # pylint: disable=protected-access
  op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
  meta_key = torch._C.DispatchKey.Meta
  upstream_meta = op.py_kernels.pop(meta_key, None)
  if upstream_meta is None:
    return

  def _sdpa_overrideable_meta(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      attn_bias: torch.Tensor | None = None,
      dropout_p: float = 0.0,
      is_causal: bool = False,
      return_debug_mask: bool = False,
      scale: float | None = None,
  ):
    if _meta_registrations.device_hint(query) == "tpu" and query.dim() in (
        3,
        4,
    ):
      s_kv = key.size(-2)
      torch._check(
          s_kv == value.size(-2),
          lambda: (
              f"key sequence length ({s_kv}) must match "
              f"value sequence length ({value.size(-2)})"
          ),
      )
    return upstream_meta(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale,
    )

  meta_lib.impl(op, _sdpa_overrideable_meta, allow_override=True)
  op.py_impl(meta_key)(_sdpa_overrideable_meta)
  # pylint: enable=protected-access


def register() -> None:
  """Registers TPU Meta and dispatch kernel overrides."""
  global _meta_lib
  if _meta_lib is not None:
    return
  _meta_lib = torch.library.Library("aten", "IMPL", "Meta")
  _patch_flex_attention()
  _patch_grouped_mm_meta(_meta_lib)
  _patch_sdpa_meta(_meta_lib)
