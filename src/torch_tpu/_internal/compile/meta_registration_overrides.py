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

"""Overrides for upstream PyTorch Meta kernel registrations on TPU."""

from collections.abc import Callable
import inspect
from typing import Any

import torch
from torch import _decomp
from torch import _meta_registrations
from torch._decomp import decompositions
from torch._subclasses import fake_tensor

_meta_lib: torch.library.Library | None = None


@decompositions.out_wrapper(exact_dtype=True)
def _addmm_dtype_meta(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype,
    beta: Any = 1,
    alpha: Any = 1,
):
  if not self.is_floating_point() and not self.is_complex():
    beta = int(beta)
    alpha = int(alpha)
  out = alpha * torch.mm(mat1, mat2, out_dtype=out_dtype)
  if beta == 0:
    return out
  return out + beta * self.to(out_dtype).expand_as(out)


# Expose the replacement decomposition on `torch._decomp.decompositions` so
# `FakeTensorMode` (`torch_decomp_decompositions`) recognizes it on both
# static and symbolic shape paths.
_addmm_dtype_meta.__module__ = decompositions.__name__
setattr(decompositions, _addmm_dtype_meta.__name__, _addmm_dtype_meta)


# TODO: Remove once minimum supported PyTorch includes
# https://github.com/pytorch/pytorch/pull/179634 and
# https://github.com/pytorch/pytorch/pull/195594 (PyTorch 2.12+).
def _patch_addmm_dtype_meta(meta_lib: torch.library.Library) -> None:
  """Patches `aten.addmm.dtype` and `aten.addmm.dtype_out` Meta/decompositions."""
  # pylint: disable=protected-access
  meta_key = torch._C.DispatchKey.Meta
  op_to_fn: tuple[tuple[Any, Callable[..., Any]], ...] = (
      (torch.ops.aten.addmm.dtype, _addmm_dtype_meta),
      (torch.ops.aten.addmm.dtype_out, _addmm_dtype_meta),
  )
  patched = False
  for op, fn in op_to_fn:
    existing = _decomp.decomposition_table.get(op)
    if (
        existing is not None
        and "out_dtype" in inspect.signature(existing).parameters
    ):
      continue
    _decomp.decomposition_table[op] = fn
    if op in _decomp.pre_autograd_decomposition_table:
      _decomp.pre_autograd_decomposition_table[op] = fn
    op.py_kernels.pop(meta_key, None)
    meta_lib.impl(op, fn, allow_override=True)
    op.py_impl(meta_key)(fn)
    patched = True
  if patched:
    fake_tensor.torch_decomp_decompositions.cache_clear()
  # pylint: enable=protected-access


# TODO: Remove once PyTorch includes https://github.com/pytorch/pytorch/pull/194271
# and https://github.com/pytorch/pytorch/pull/198303 (PyTorch 2.15+).
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
      if attn_bias is not None:
        bias_s_kv = attn_bias.size(-1)
        if bias_s_kv != 1:
          torch._check(
              bias_s_kv == s_kv,
              lambda: (
                  f"attn_bias last dimension must match S_KV ({s_kv}) "
                  f"or be 1 for broadcasting, but got {bias_s_kv}"
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
  """Registers TPU Meta kernel overrides."""
  global _meta_lib
  if _meta_lib is not None:
    return
  _meta_lib = torch.library.Library("aten", "IMPL", "Meta")
  _patch_sdpa_meta(_meta_lib)
  _patch_addmm_dtype_meta(_meta_lib)
