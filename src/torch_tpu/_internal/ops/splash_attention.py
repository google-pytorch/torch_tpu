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

"""Splash Attention integration for torch_tpu via Pallas.

Wraps JAX's splash_attention_kernel for use with PyTorch tensors on TPU.
Uses torch_tpu._internal.pallas.custom_jax_kernel() to bridge JAX -> PyTorch.
"""

from collections.abc import Callable
import hashlib
import math
import sys
import typing
from typing import Optional

from absl import logging
# JAX imports
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import QKVLayout
import jax.numpy as jnp
import numpy as np
import torch
# torch_tpu Pallas bridge
from torch_tpu._internal import pallas as _pallas
from torch_tpu._internal.utils import annotations as _annotations

_custom_jax_kernel = _pallas.custom_jax_kernel

__all__ = [
    "splash_sdpa",
]

_DEFAULT_MASK_VALUE = splash_attention_kernel.DEFAULT_MASK_VALUE
_DEFAULT_MAX_BLOCK_SIZE = 512
_MIN_BLOCK_SIZE = 128


def _default_block_size(
    limit: int,
    max_block: int = _DEFAULT_MAX_BLOCK_SIZE,
    param_name: str = "seq_len",
) -> int:
  """Largest multiple of 128 (<= max_block) that divides limit."""
  if limit < _MIN_BLOCK_SIZE or limit % _MIN_BLOCK_SIZE != 0:
    raise ValueError(
        f"{param_name} ({limit}) must be a positive multiple of"
        f" {_MIN_BLOCK_SIZE}"
    )
  for size in range(
      min(max_block, limit), _MIN_BLOCK_SIZE - 1, -_MIN_BLOCK_SIZE
  ):
    if limit % size == 0:
      return size
  return _MIN_BLOCK_SIZE


def _to_numpy_mask_info(
    mask_info: splash_attention_mask_info.MaskInfo,
) -> splash_attention_mask_info.MaskInfo:
  """Normalizes MaskInfo arrays on host without placing buffers on TPU."""
  if mask_info.partial_mask_blocks is not None and isinstance(
      mask_info.partial_mask_blocks, np.ndarray
  ):
    mask_info = mask_info._replace(
        partial_mask_blocks=mask_info.partial_mask_blocks.astype(np.int8)
    )
  return mask_info


def _make_numpy_splash_mask_infos(
    mask: splash_attention_mask.MultiHeadMask,
    block_sizes: splash_attention_kernel.BlockSizes,
) -> tuple[
    splash_attention_mask_info.MaskInfo,
    splash_attention_mask_info.MaskInfo | None,
    splash_attention_mask_info.MaskInfo | None,
    Callable[..., typing.Any] | None,
]:
  """Builds forward and backward MaskInfo as host NumPy arrays.

  Avoids splash_attention_kernel.make_splash_{mha,mqa}, which eagerly call
  jnp.array(...) outside a JAX trace and stage mask buffers onto TPU via JAX's
  PJRT client while TorchTPU's C++ PJRT client may be executing on TPU.

  Args:
    mask: Multi-head attention mask specification.
    block_sizes: Tile sizes for forward and backward splash kernels.

  Returns:
    Tuple of (fwd_mask_info, dq_mask_info, dkv_mask_info, mask_function).

  Raises:
    ValueError: If required backward block sizes are missing or mask functions
      are inconsistent across forward and backward passes.
  """
  fwd_mask_info, mask_function_fwd = splash_attention_mask_info.process_mask(
      mask,
      (block_sizes.block_q, block_sizes.block_kv),
      downcast_smem_data=True,
      head_shards=1,
      q_seq_shards=1,
  )
  fwd_mask_info = _to_numpy_mask_info(fwd_mask_info)

  dq_mask_info = None
  dkv_mask_info = None
  if block_sizes.has_backward_blocks:
    if not block_sizes.use_fused_bwd_kernel:
      if block_sizes.block_q_dq is None or block_sizes.block_kv_dq is None:
        raise ValueError(
            "block_q_dq and block_kv_dq are required when "
            "use_fused_bwd_kernel is False."
        )
      dq_mask_info, mask_function_dq = splash_attention_mask_info.process_mask(
          mask,
          (block_sizes.block_q_dq, block_sizes.block_kv_dq),
          downcast_smem_data=True,
          head_shards=1,
          q_seq_shards=1,
      )
      if (mask_function_fwd is None) != (mask_function_dq is None):
        raise ValueError("Inconsistent forward and dQ mask functions.")
      dq_mask_info = _to_numpy_mask_info(dq_mask_info)

    if block_sizes.block_q_dkv is None or block_sizes.block_kv_dkv is None:
      raise ValueError(
          "block_q_dkv and block_kv_dkv are required when "
          "has_backward_blocks is True."
      )
    dkv_mask_info, mask_function_dkv = (
        splash_attention_mask_info.process_mask_dkv(
            mask,
            (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
            downcast_smem_data=True,
            head_shards=1,
            q_seq_shards=1,
            shrink_grid=not block_sizes.use_fused_bwd_kernel,
        )
    )
    if (mask_function_fwd is None) != (mask_function_dkv is None):
      raise ValueError("Inconsistent forward and dKV mask functions.")
    dkv_mask_info = _to_numpy_mask_info(dkv_mask_info)

  return fwd_mask_info, dq_mask_info, dkv_mask_info, mask_function_fwd


def _make_splash_attention_fn(
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    scale: float | None = None,
    is_causal: bool = True,
    local_window_size: int | None = None,
    block_q: int = 512,
    block_kv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
    use_vmap_bwd: bool = True,
):
  """Create JAX splash attention forward and backward functions.

  Args:
    seq_len: Sequence length.
    n_heads: Number of query heads.
    n_kv_heads: Number of key/value heads.
    scale: Scale factor for attention logits, or None.
    is_causal: Whether to apply causal mask.
    local_window_size: Size of sliding window attention mask, or None.
    block_q: Query block size.
    block_kv: Key/Value block size.
    block_kv_compute: Key/Value compute block size.
    block_q_dkv: Query block size for dK/dV computation.
    block_kv_dkv: Key/Value block size for dK/dV computation.
    block_kv_dkv_compute: Key/Value compute block size for dK/dV.
    block_q_dq: Query block size for dQ computation.
    block_kv_dq: Key/Value block size for dQ computation.
    use_fused_bwd_kernel: Whether to use fused backward kernel.
    q_layout: Query tensor layout.
    k_layout: Key tensor layout.
    v_layout: Value tensor layout.
    use_vmap_bwd: Whether to vmap backward pass.

  Returns:
    (splash_fn, splash_bwd_fn): forward function taking
      (q, k, v, segment_ids=None) returning (out, logsumexp), and backward
      function taking (q, k, v, out, logsumexp, grad_out, segment_ids=None)
      returning (grad_q, grad_k, grad_v) WITHOUT re-running the forward (uses
      saved residuals to call the backward kernel directly).
  """

  def _get_layout(layout_str):
    if layout_str == "HEAD_DIM_MINOR":
      return QKVLayout.HEAD_DIM_MINOR
    elif layout_str == "SEQ_MINOR":
      return QKVLayout.SEQ_MINOR
    else:
      raise ValueError(f"Unknown layout: {layout_str}")

  if use_fused_bwd_kernel:
    calc_block_q_dq = None
    calc_block_kv_dq = None
  else:
    calc_block_q_dq = block_q_dq if block_q_dq is not None else block_q
    calc_block_kv_dq = block_kv_dq if block_kv_dq is not None else block_kv

  block_sizes = splash_attention_kernel.BlockSizes(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=block_q_dkv,
      block_kv_dkv=block_kv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      block_q_dq=calc_block_q_dq,
      block_kv_dq=calc_block_kv_dq,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      q_layout=_get_layout(q_layout),
      k_layout=_get_layout(k_layout),
      v_layout=_get_layout(v_layout),
  )

  mask_shape = (seq_len, seq_len)
  if local_window_size is not None:
    # Sliding-window causal: each q attends to the previous local_window_size
    # keys plus itself. window_size=(left, right); right=0 ⇒ no future tokens
    # (still causal), offset=0 ⇒ q starts at kv start. The LocalMask is more
    # specific than CausalMask so the splash kernel skips the off-window
    # blocks entirely. With local_window_size >= seq_len-1, this is
    # equivalent to a CausalMask (every token attends to every prior token).
    single_mask = splash_attention_mask.LocalMask(
        shape=mask_shape,
        window_size=(local_window_size, 0),
        offset=0,
    )
  elif is_causal:
    single_mask = splash_attention_mask.CausalMask(shape=mask_shape)
  else:
    single_mask = splash_attention_mask.FullMask(mask_shape)

  multi_head_mask = splash_attention_mask.MultiHeadMask(
      masks=(single_mask,) * n_heads
  )

  is_mqa = n_kv_heads == 1

  # Keep static mask infos as host NumPy arrays so jax.export embeds them
  # directly as MLIR constants without eager JAX TPU buffer staging.
  fwd_mask_info, dq_mask_info, dkv_mask_info, _mask_function = (
      _make_numpy_splash_mask_infos(multi_head_mask, block_sizes)
  )
  _mask_value = _DEFAULT_MASK_VALUE
  _attn_logits_soft_cap = None
  _residual_checkpoint_name = None
  _interpret = False

  def _single_fwd_with_lse(q_b, k_b, v_b, seg_b):
    """Forward for one batch element; returns (out_b, logsumexp_b)."""
    # q_b is already scaled (q * 1/sqrt(head_dim)) from the caller.
    seg_ids = (
        splash_attention_kernel.SegmentIds(q=seg_b, kv=seg_b)
        if seg_b is not None
        else None
    )
    fwd_fn = getattr(splash_attention_kernel, "_splash_attention_forward")
    out_b, (logsumexp_b,) = fwd_fn(
        fwd_mask_info,
        q_b,
        k_b,
        v_b,
        segment_ids=seg_ids,
        sinks=None,
        mask_value=_mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        residual_checkpoint_name=_residual_checkpoint_name,
        save_residuals=True,
        mask_function=_mask_function,
        attn_logits_soft_cap=_attn_logits_soft_cap,
        interpret=_interpret,
    )
    return out_b, logsumexp_b

  @jax.jit
  def splash_fn(q, k, v, segment_ids=None):
    """Forward: returns (out, logsumexp) for the whole batch."""
    head_dim = q.shape[-1]
    attn_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    scale_val = jnp.array(attn_scale, dtype=q.dtype)
    q_scaled = q * scale_val
    out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_scaled, k, v, segment_ids)
    return out, logsumexp

  def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, seg_b):
    """Backward for one batch element using precomputed residuals.

    Calls _splash_attention_bwd directly — no extra forward pass.

    Args:
      q_scaled_b: Scaled query tensor for a single batch element.
      k_b: Key tensor for a single batch element.
      v_b: Value tensor for a single batch element.
      out_b: Output of the forward pass for a single batch element.
      logsumexp_b: Logsumexp from the forward pass for a single batch element.
      g_b: Gradient of the loss with respect to the output (`out_b`).
      seg_b: Optional 1D segment_ids tensor for a single batch element.

    Returns:
      Tuple of (dq_scaled, dk, dv) gradient tensors for the single batch item.
    """
    seg_ids = (
        splash_attention_kernel.SegmentIds(q=seg_b, kv=seg_b)
        if seg_b is not None
        else None
    )
    res = (
        q_scaled_b,
        k_b,
        v_b,
        seg_ids,
        None,  # sinks
        out_b,
        logsumexp_b,
        dq_mask_info,
        dkv_mask_info,
    )
    bwd_fn = getattr(splash_attention_kernel, "_splash_attention_bwd")
    _, _, _, dq_scaled, dk, dv, _, _ = bwd_fn(
        False,
        _mask_value,
        is_mqa,
        block_sizes,
        _residual_checkpoint_name,
        _mask_function,
        _attn_logits_soft_cap,
        _interpret,
        res,
        g_b,
    )
    return dq_scaled, dk, dv

  def _scan_bwd_body(carry, args):
    """Single-element backward body for jax.lax.scan (sequential over batch)."""
    q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, seg_b = args
    dq_scaled_b, dk_b, dv_b = _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b, seg_b
    )
    return carry, (dq_scaled_b, dk_b, dv_b)

  @jax.jit
  def splash_bwd_fn(q, k, v, out, logsumexp, g, segment_ids=None):
    """Backward: uses saved (out, logsumexp) — no extra forward pass.

    Supports both `jax.vmap` and `jax.lax.scan` over the batch dimension.
    By default, it uses `jax.lax.scan` to be memory safe, sizing the XLA
    backward program for one element at a time to avoid O(batch) growth in
    program size. `jax.vmap` can be enabled for better throughput but may
    hit memory limits for large batches.

    Args:
      q: Query tensor.
      k: Key tensor.
      v: Value tensor.
      out: Output of the forward pass.
      logsumexp: Logsumexp values from the forward pass.
      g: Gradient of the loss with respect to the output (`out`).
      segment_ids: Optional segment IDs tensor of shape (batch, seq_len).
    """
    head_dim = q.shape[-1]
    attn_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    scale_val = jnp.array(attn_scale, dtype=q.dtype)
    q_scaled = q * scale_val

    if use_vmap_bwd:
      dq_scaled, dk, dv = jax.vmap(_single_bwd)(
          q_scaled, k, v, out, logsumexp, g, segment_ids
      )
    else:
      _, (dq_scaled, dk, dv) = jax.lax.scan(
          _scan_bwd_body,
          None,
          (q_scaled, k, v, out, logsumexp, g, segment_ids),
      )
    # Chain rule: q_scaled = q * scale  =>  dq = dq_scaled * scale
    # TODO: b/543955134 - remove optimization barriers
    dq = jax.lax.optimization_barrier(dq_scaled * scale_val)
    dk = jax.lax.optimization_barrier(dk)
    dv = jax.lax.optimization_barrier(dv)
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


class _SplashAttentionFn(torch.autograd.Function):
  """torch.autograd.Function bridging splash forward and direct backward."""

  @staticmethod
  def forward(ctx, q, k, v, segment_ids, torch_fwd_fn, torch_bwd_fn):
    # torch_fwd_fn returns (out, logsumexp); both saved for the backward.
    out, logsumexp = torch_fwd_fn(q, k, v, segment_ids)
    if segment_ids is not None:
      ctx.save_for_backward(q, k, v, out, logsumexp, segment_ids)
      ctx.has_segment_ids = True
    else:
      ctx.save_for_backward(q, k, v, out, logsumexp)
      ctx.has_segment_ids = False
    ctx.torch_bwd_fn = torch_bwd_fn
    return out

  @staticmethod
  @torch.autograd.function.once_differentiable
  def backward(ctx, grad_output):  # pyrefly: ignore[bad-override]
    if ctx.has_segment_ids:
      q, k, v, out, logsumexp, segment_ids = ctx.saved_tensors
    else:
      q, k, v, out, logsumexp = ctx.saved_tensors
      segment_ids = None
    # Direct backward — no extra forward inside JAX.
    grad_q, grad_k, grad_v = ctx.torch_bwd_fn(
        q, k, v, out, logsumexp, grad_output, segment_ids
    )
    return (
        grad_q,
        grad_k,
        grad_v,
        None,
        None,
        None,
    )  # None for segment_ids, torch_fwd_fn, torch_bwd_fn


_splash_op_name_cache = {}


@torch.compiler.assume_constant_result
def _get_splash_op_names(**splash_kwargs):
  """Create and cache PyTorch forward/backward functions for splash attention."""
  cache_key = tuple(sorted(splash_kwargs.items()))
  if cache_key in _splash_op_name_cache:
    return _splash_op_name_cache[cache_key]

  splash_fn, splash_bwd_fn = _make_splash_attention_fn(**splash_kwargs)
  jax_fwd_callable = _custom_jax_kernel(splash_fn, name="splash_attention")
  jax_bwd_callable = _custom_jax_kernel(
      splash_bwd_fn, name="splash_attention_bwd"
  )

  identifier = hashlib.md5(str(cache_key).encode()).hexdigest()[:12]
  op_fwd_name = f"splash_attn_fwd_{identifier}"
  op_bwd_name = f"splash_attn_bwd_{identifier}"
  torch_fwd_fn = torch.library.custom_op(
      f"pallas::{op_fwd_name}",
      jax_fwd_callable,
      mutates_args=(),
      schema=(
          "(Tensor q, Tensor k, Tensor v, Tensor? segment_ids=None)"
          " -> (Tensor, Tensor)"
      ),
  )
  torch_fwd_fn.register_fake(
      # lse: (batch, num_q_heads, q_seq_len)
      lambda q, k, v, segment_ids=None: (
          # Kernel outputs are allocated via `torch.empty`,
          # which are contiguous whatever the input strides.
          torch.empty_like(q, memory_format=torch.contiguous_format),
          torch.empty(
              q.shape[0],
              q.shape[1],
              q.shape[2],
              dtype=torch.float32,
              device=q.device,
          ),
      )
  )

  torch_bwd_fn = torch.library.custom_op(
      f"pallas::{op_bwd_name}",
      jax_bwd_callable,
      mutates_args=(),
      schema=(
          "(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor"
          " grad_out, Tensor? segment_ids=None) -> (Tensor, Tensor, Tensor)"
      ),
  )

  # Register with TorchAX if active
  if "torchax" in sys.modules:
    try:
      torchax_mod = sys.modules["torchax"]

      jlibrary = getattr(torchax_mod, "ops").jlibrary

      op_fwd = getattr(torch.ops.pallas, op_fwd_name)
      op_bwd = getattr(torch.ops.pallas, op_bwd_name)
      logging.debug(
          "Registering fwd: pallas.%s for op: %s (type: %s)",
          op_fwd_name,
          op_fwd.default,
          type(op_fwd.default),
      )
      jlibrary.register_jax_composite(
          f"pallas.{op_fwd_name}", jax_fwd_callable.jit_fn, op_fwd.default
      )
      logging.debug(
          "Registering bwd: pallas.%s for op: %s (type: %s)",
          op_bwd_name,
          op_bwd.default,
          type(op_bwd.default),
      )
      jlibrary.register_jax_composite(
          f"pallas.{op_bwd_name}", jax_bwd_callable.jit_fn, op_bwd.default
      )
      logging.debug("Reloading ops in default env")
      getattr(torchax_mod, "default_env")().load_ops()
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        ImportError,
    ) as e:
      logging.warning("Failed to register JAX composite: %s", e)
  torch_bwd_fn.register_fake(
      lambda q, k, v, out, lse, grad_out, segment_ids=None: (
          # Kernel outputs are allocated via `torch.empty`,
          # which are contiguous whatever the input strides.
          torch.empty_like(q, memory_format=torch.contiguous_format),
          torch.empty_like(k, memory_format=torch.contiguous_format),
          torch.empty_like(v, memory_format=torch.contiguous_format),
      )
  )

  # Note: We return string names instead of the functions themselves.
  # Passing callables through activation checkpointing in Dynamo can trigger
  # safety guards that check for mutation. Since "function guards" are not yet
  # fully supported in PyTorch, this can cause failures. Using strings avoids
  # this issue as they are easily guarded by Dynamo.
  _splash_op_name_cache[cache_key] = (op_fwd_name, op_bwd_name)
  return op_fwd_name, op_bwd_name


@_annotations.experimental("splash_sdpa is experimental and subject to change.")
def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    segment_ids: Optional[torch.Tensor] = None,
    enable_gqa: bool = False,  # Kept for API compatibility
    block_q: Optional[int] = None,
    block_kv: Optional[int] = None,
    block_dkv: Optional[int] = None,
    block_kv_compute: Optional[int] = None,
    block_q_dkv: Optional[int] = None,
    block_kv_dkv: Optional[int] = None,
    block_kv_dkv_compute: Optional[int] = None,
    block_q_dq: Optional[int] = None,
    block_kv_dq: Optional[int] = None,
    use_fused_bwd_kernel: bool = True,
    use_vmap_bwd: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using splash attention.

  Supports backward pass: gradients flow through q, k, v via the JAX splash
  backward kernel. The forward saves (out, logsumexp) so the backward can call
  the pallas backward kernel directly — no extra forward pass at backward time.

  .. warning::
      This API is experimental and subject to change in future releases.

  Args:
    q: Query tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
    k: Key tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
    v: Value tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
    scale: Scale factor for attention logits.
    is_causal: Whether to apply causal mask.
    local_window_size: Size of sliding window attention mask, or None.
    segment_ids: Optional integer tensor of shape ``(batch, seq_len)`` with
      document/segment IDs for packed sequences. Tokens only attend to tokens
      with the same segment ID.
    enable_gqa: Kept for API compatibility.
    block_q: Block size for Q. Defaults to the largest multiple of 128 (<= 512)
      that divides ``seq_len``.
    block_kv: Block size for KV. Defaults to the largest multiple of 128 (<=
      512) that divides ``seq_len``.
    block_dkv: Shorthand that sets both ``block_q_dkv`` and ``block_kv_dkv``
      when they are not explicitly provided. Smaller values (e.g. 256) reduce
      the XLA program size (enabling larger batch sizes) at a small efficiency
      cost.
    block_kv_compute: Block size for KV compute. Defaults to the largest
      multiple of 128 (<= 512) that divides ``block_kv``.
    block_q_dkv: Block size for Q in DKV. Defaults to the largest multiple of
      128 (<= 512) that divides ``seq_len``.
    block_kv_dkv: Block size for KV in DKV. Defaults to the largest multiple of
      128 (<= 512) that divides ``seq_len``.
    block_kv_dkv_compute: Block size for KV compute in DKV. Defaults to the
      largest multiple of 128 (<= 512) that divides ``block_kv_dkv``.
    block_q_dq: Block size for Q in DQ (when ``use_fused_bwd_kernel=False``).
    block_kv_dq: Block size for KV in DQ (when ``use_fused_bwd_kernel=False``).
    use_fused_bwd_kernel: Whether to use fused backward kernel.
    use_vmap_bwd: Whether to use vmap in backward pass. Default True.
    q_layout: Layout for Q.
    k_layout: Layout for K.
    v_layout: Layout for V.

  Returns:
    Output tensor of scaled dot-product attention.
  """
  batch_size, n_heads, seq_len, _ = q.shape
  n_kv_heads = k.shape[1]

  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )
  if n_heads % n_kv_heads != 0:
    raise ValueError(
        f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    )

  if segment_ids is not None:
    if segment_ids.shape != (batch_size, seq_len):
      raise ValueError(
          "segment_ids must have shape (batch, seq_len) = "
          f"({batch_size}, {seq_len}), got {tuple(segment_ids.shape)}"
      )
    if segment_ids.dtype not in (torch.int32, torch.int64):
      raise ValueError(
          "segment_ids must have integer dtype (torch.int32 or torch.int64), "
          f"got {segment_ids.dtype}"
      )
    if segment_ids.dtype != torch.int32:
      segment_ids = segment_ids.to(torch.int32)

  if block_dkv is not None:
    if block_q_dkv is None:
      block_q_dkv = block_dkv
    if block_kv_dkv is None:
      block_kv_dkv = block_dkv

  block_q = (
      _default_block_size(seq_len) if block_q is None else min(block_q, seq_len)
  )
  block_kv = (
      _default_block_size(seq_len)
      if block_kv is None
      else min(block_kv, seq_len)
  )
  block_q_dkv = (
      _default_block_size(seq_len)
      if block_q_dkv is None
      else min(block_q_dkv, seq_len)
  )
  block_kv_dkv = (
      _default_block_size(seq_len)
      if block_kv_dkv is None
      else min(block_kv_dkv, seq_len)
  )
  block_kv_compute = (
      _default_block_size(block_kv, param_name="block_kv")
      if block_kv_compute is None
      else min(block_kv_compute, block_kv)
  )
  block_kv_dkv_compute = (
      _default_block_size(block_kv_dkv, param_name="block_kv_dkv")
      if block_kv_dkv_compute is None
      else min(block_kv_dkv_compute, block_kv_dkv)
  )

  if not use_fused_bwd_kernel:
    block_q_dq = block_q if block_q_dq is None else min(block_q_dq, seq_len)
    block_kv_dq = block_kv if block_kv_dq is None else min(block_kv_dq, seq_len)

  if n_kv_heads == 1:
    # For MQA (1 kv head) the splash attention kernel expects the kv
    # head dimension to be squeezed.
    k = torch.squeeze(k, dim=1)
    v = torch.squeeze(v, dim=1)

  splash_args = dict(
      seq_len=seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      scale=scale,
      is_causal=is_causal,
      local_window_size=local_window_size,
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=block_q_dkv,
      block_kv_dkv=block_kv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      block_q_dq=block_q_dq,
      block_kv_dq=block_kv_dq,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      use_vmap_bwd=use_vmap_bwd,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
  )
  torch_fwd_name, torch_bwd_name = _get_splash_op_names(**splash_args)
  torch_fwd_fn = getattr(torch.ops.pallas, torch_fwd_name)
  torch_bwd_fn = getattr(torch.ops.pallas, torch_bwd_name)

  result = _SplashAttentionFn.apply(
      q, k, v, segment_ids, torch_fwd_fn, torch_bwd_fn
  )
  return typing.cast(torch.Tensor, result)
