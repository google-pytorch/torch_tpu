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

import math
import os
import typing
from typing import Optional

# JAX imports
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import QKVLayout
import jax.numpy as jnp
import torch

# torch_tpu Pallas bridge
from torch_tpu._internal import pallas

custom_jax_kernel = pallas.custom_jax_kernel

_DEFAULT_MASK_VALUE = splash_attention_kernel.DEFAULT_MASK_VALUE


def _make_splash_attention_fn(
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
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
    use_vmap_bwd: bool = False,
):
  """Create JAX splash attention forward and backward functions.

  Returns:
    (splash_fn, splash_bwd_fn): forward function taking (q, k, v) returning
      (out, logsumexp), and backward function taking (q, k, v, out, logsumexp,
      grad_out) returning (grad_q, grad_k, grad_v) WITHOUT re-running the
      forward (uses saved residuals to call the backward kernel directly).
  """

  def _get_layout(layout_str):
    if layout_str == "HEAD_DIM_MINOR":
      return QKVLayout.HEAD_DIM_MINOR
    elif layout_str == "SEQ_MINOR":
      return QKVLayout.SEQ_MINOR
    else:
      raise ValueError(f"Unknown layout: {layout_str}")

  block_sizes = splash_attention_kernel.BlockSizes(
      block_q=min(block_q, seq_len),
      block_kv=min(block_kv, seq_len),
      block_kv_compute=min(block_kv_compute, seq_len),
      block_q_dkv=min(block_q_dkv, seq_len),
      block_kv_dkv=min(block_kv_dkv, seq_len),
      block_kv_dkv_compute=min(block_kv_dkv_compute, seq_len),
      block_q_dq=min(block_q_dq, seq_len)
      if not use_fused_bwd_kernel and block_q_dq is not None
      else None,
      block_kv_dq=min(block_kv_dq, seq_len)
      if not use_fused_bwd_kernel and block_kv_dq is not None
      else None,
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

  if is_mqa:
    splash_kernel = splash_attention_kernel.make_splash_mqa(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )
  else:
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )

  # Static mask infos and kernel kwargs from creation time.
  # These are the same for every forward/backward call with the same config.
  fwd_mask_info = splash_kernel.fwd_mask_info
  dq_mask_info = splash_kernel.dq_mask_info
  dkv_mask_info = splash_kernel.dkv_mask_info
  _mask_value = splash_kernel.kwargs.get("mask_value", _DEFAULT_MASK_VALUE)
  _mask_function = splash_kernel.kwargs.get("mask_function", None)
  _attn_logits_soft_cap = splash_kernel.kwargs.get("attn_logits_soft_cap", None)
  _residual_checkpoint_name = splash_kernel.kwargs.get(
      "residual_checkpoint_name", None
  )
  _interpret = splash_kernel.kwargs.get("interpret", False)

  def _single_fwd_with_lse(q_b, k_b, v_b):
    """Forward for one batch element; returns (out_b, logsumexp_b)."""
    # q_b is already scaled (q * 1/sqrt(head_dim)) from the caller.
    out_b, (logsumexp_b,) = splash_attention_kernel._splash_attention_forward(
        fwd_mask_info,
        q_b,
        k_b,
        v_b,
        segment_ids=None,
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
  def splash_fn(q, k, v):
    """Forward: returns (out, logsumexp) for the whole batch."""
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale
    out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_scaled, k, v)
    return out, logsumexp

  def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b):
    """Backward for one batch element using precomputed residuals.

    Calls _splash_attention_bwd directly — no extra forward pass.

    Args:
      q_scaled_b: Scaled query tensor for a single batch element.
      k_b: Key tensor for a single batch element.
      v_b: Value tensor for a single batch element.
      out_b: Output of the forward pass for a single batch element.
      logsumexp_b: Logsumexp from the forward pass for a single batch element.
      g_b: Gradient of the loss with respect to the output (`out_b`).
    """
    res = (
        q_scaled_b,
        k_b,
        v_b,
        None,  # segment_ids
        None,  # sinks
        out_b,
        logsumexp_b,
        dq_mask_info,
        dkv_mask_info,
    )
    all_grads = splash_attention_kernel._splash_attention_bwd(
        False,  # save_residuals (nondiff)
        _mask_value,  # mask_value (nondiff)
        is_mqa,  # is_mqa (nondiff)
        block_sizes,  # block_sizes (nondiff)
        _residual_checkpoint_name,  # residual_checkpoint_name (nondiff)
        _mask_function,  # mask_function (nondiff)
        _attn_logits_soft_cap,  # attn_logits_soft_cap (nondiff)
        _interpret,  # interpret (nondiff)
        res,  # residuals
        g_b,  # do (output gradient)
    )
    # all_grads = (dfwd_mask_info=None, ddq_mask_info=None, ddkv_mask_info=None,
    #              dq_scaled, dk, dv, dsegment_ids=None, dsinks)
    return all_grads[3], all_grads[4], all_grads[5]  # dq_scaled, dk, dv

  def _scan_bwd_body(carry, args):
    """Single-element backward body for jax.lax.scan (sequential over batch)."""
    q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b = args
    dq_scaled_b, dk_b, dv_b = _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b
    )
    return carry, (dq_scaled_b, dk_b, dv_b)

  @jax.jit
  def splash_bwd_fn(q, k, v, out, logsumexp, g):
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
    """
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale

    if use_vmap_bwd:
      dq_scaled, dk, dv = jax.vmap(_single_bwd)(
          q_scaled, k, v, out, logsumexp, g
      )
    else:
      _, (dq_scaled, dk, dv) = jax.lax.scan(
          _scan_bwd_body,
          None,
          (q_scaled, k, v, out, logsumexp, g),
      )
    # Chain rule: q_scaled = q * scale  =>  dq = dq_scaled * scale
    dq = dq_scaled * scale
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


class _SplashAttentionFn(torch.autograd.Function):
  """torch.autograd.Function bridging splash forward and direct backward."""

  @staticmethod
  def forward(ctx, q, k, v, torch_fwd_fn, torch_bwd_fn):
    # torch_fwd_fn returns (out, logsumexp); both saved for the backward.
    out, logsumexp = torch_fwd_fn(q, k, v)
    ctx.save_for_backward(q, k, v, out, logsumexp)
    ctx.torch_bwd_fn = torch_bwd_fn
    return out

  @staticmethod
  @torch.autograd.function.once_differentiable
  def backward(ctx, grad_output):
    q, k, v, out, logsumexp = ctx.saved_tensors
    # Direct backward — no extra forward inside JAX.
    grad_q, grad_k, grad_v = ctx.torch_bwd_fn(
        q, k, v, out, logsumexp, grad_output
    )
    return (
        grad_q,
        grad_k,
        grad_v,
        None,
        None,
    )  # None for torch_fwd_fn, torch_bwd_fn


_splash_op_name_cache = {}


@torch._dynamo.assume_constant_result
def _get_splash_op_names(**splash_kwargs):
  """Create and cache PyTorch forward/backward functions for splash attention."""
  cache_key = tuple(splash_kwargs.items())
  if cache_key in _splash_op_name_cache:
    return _splash_op_name_cache[cache_key]

  splash_fn, splash_bwd_fn = _make_splash_attention_fn(**splash_kwargs)
  jax_fwd_callable = custom_jax_kernel(splash_fn, name="splash_attention")
  jax_bwd_callable = custom_jax_kernel(
      splash_bwd_fn, name="splash_attention_bwd"
  )

  identifier = abs(hash(cache_key))  # TODO: use a better identifier
  op_fwd_name = f"splash_attn_fwd_{identifier}"
  op_bwd_name = f"splash_attn_bwd_{identifier}"
  torch_fwd_fn = torch.library.custom_op(
      f"pallas::{op_fwd_name}",
      jax_fwd_callable,
      mutates_args=(),
      schema="(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor)",
  )
  torch_fwd_fn.register_fake(
      # lse: (batch, num_q_heads, q_seq_len, ??)
      lambda q, k, v: (
          torch.empty_like(q),
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
          " grad_out) -> (Tensor, Tensor, Tensor)"
      ),
  )

  # Register with TorchAX if active
  import sys

  torch_keys = [k for k in sys.modules.keys() if "torch" in k]
  # print(f"[DEBUG_PALLAS_REG] Torch keys in sys.modules: {torch_keys}")
  if "torchax" in sys.modules:
    try:
      import torchax  # pytype: disable=import-error
      import torchax.ops.jlibrary as jlibrary  # buildcleaner: keep; pytype: disable=import-error

      op_fwd = getattr(torch.ops.pallas, op_fwd_name)
      op_bwd = getattr(torch.ops.pallas, op_bwd_name)
      print(
          f"[DEBUG_PALLAS_REG] Registering fwd: pallas.{op_fwd_name} for op:"
          f" {op_fwd.default} (type: {type(op_fwd.default)})"
      )
      jlibrary.register_jax_composite(
          f"pallas.{op_fwd_name}", jax_fwd_callable.jit_fn, op_fwd.default
      )
      print(
          f"[DEBUG_PALLAS_REG] Registering bwd: pallas.{op_bwd_name} for op:"
          f" {op_bwd.default} (type: {type(op_bwd.default)})"
      )
      jlibrary.register_jax_composite(
          f"pallas.{op_bwd_name}", jax_bwd_callable.jit_fn, op_bwd.default
      )
      print("[DEBUG_PALLAS_REG] Reloading ops in default env")
      torchax.default_env().load_ops()
    except Exception as e:
      print(f"[DEBUG_PALLAS_REG] Failed to register JAX composite: {e}")
  torch_bwd_fn.register_fake(
      lambda q, k, v, out, lse, grad_out: (
          torch.empty_like(q),
          torch.empty_like(k),
          torch.empty_like(v),
      )
  )

  # Note: We return string names instead of the functions themselves.
  # Passing callables through activation checkpointing in Dynamo can trigger
  # safety guards that check for mutation. Since "function guards" are not yet
  # fully supported in PyTorch, this can cause failures. Using strings avoids
  # this issue as they are easily guarded by Dynamo.
  _splash_op_name_cache[cache_key] = (op_fwd_name, op_bwd_name)
  return op_fwd_name, op_bwd_name


import os

DEBUG = os.environ.get("TORCH_TPU_DEBUG", "False").lower() == "true"


def get_stats_str(t: torch.Tensor) -> str:
  t_cpu = t.cpu().float()
  return (
      f"dtype={t.dtype}, mean={t_cpu.mean().item():.6f},"
      f" std={t_cpu.std().item():.6f}, max={t_cpu.abs().max().item():.6f}"
  )


def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    enable_gqa: bool = False,  # Kept for API compatibility
    block_q: int = 512,
    block_kv: int = 512,
    block_dkv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: Optional[int] = None,
    block_kv_dq: Optional[int] = None,
    use_fused_bwd_kernel: bool = True,
    use_vmap_bwd: bool = False,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using splash attention.

  Supports backward pass: gradients flow through q, k, v via the JAX splash
  backward kernel. The forward saves (out, logsumexp) so the backward can call
  the pallas backward kernel directly — no extra forward pass at backward time.

  Args:
    q: Query tensor.
    k: Key tensor.
    v: Value tensor.
    scale: Scale factor for attention logits.
    is_causal: Whether to apply causal mask.
    enable_gqa: Kept for API compatibility.
    block_q: Block size for Q.
    block_kv: Block size for KV.
    block_dkv: Block size for the dk/dv backward kernel tiles. Smaller values
      reduce the XLA program size (enabling larger batch sizes) at a small
      efficiency cost. Default 512 matches block_kv; use 256 for large batches.
    block_kv_compute: Block size for KV compute.
    block_q_dkv: Block size for Q in DKV.
    block_kv_dkv: Block size for KV in DKV.
    block_kv_dkv_compute: Block size for KV compute in DKV.
    block_q_dq: Block size for Q in DQ.
    block_kv_dq: Block size for KV in DQ.
    use_fused_bwd_kernel: Whether to use fused backward kernel.
    use_vmap_bwd: Whether to use vmap in backward pass. Default False (uses scan
      for memory safety).
    q_layout: Layout for Q.
    k_layout: Layout for K.
    v_layout: Layout for V.
  """

  block_q = int(os.environ.get("SPLASH_BLOCK_Q", block_q))
  block_kv = int(os.environ.get("SPLASH_BLOCK_KV", block_kv))
  block_dkv = int(os.environ.get("SPLASH_BLOCK_DKV", block_dkv))

  batch, n_heads, seq_len, head_dim = q.shape
  n_kv_heads = k.shape[1]

  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )

  splash_args = dict(
      seq_len=seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
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

  if scale is not None:
    default_scale = 1.0 / math.sqrt(head_dim)
    if abs(scale - default_scale) > 1e-6:
      scale_ratio = scale / default_scale
      q = q * torch.tensor(scale_ratio, dtype=q.dtype, device=q.device)

  input_dtype = q.dtype
  if n_kv_heads == 1:
    # For MQA (1 kv head) the splash attentionn kernel expects the kv
    # head dimension to be squeezed.
    k = torch.squeeze(k, dim=1)
    v = torch.squeeze(v, dim=1)
  result = _SplashAttentionFn.apply(q, k, v, torch_fwd_fn, torch_bwd_fn)
  result = typing.cast(torch.Tensor, result)

  if DEBUG:
    print(f"[DEBUG_SPLASH] device: {q.device} | Q: {get_stats_str(q)}")
    print(f"[DEBUG_SPLASH] device: {q.device} | K: {get_stats_str(k)}")
    print(f"[DEBUG_SPLASH] device: {q.device} | V: {get_stats_str(v)}")
    print(
        f"[DEBUG_SPLASH] device: {q.device} | local_window_size:"
        f" {local_window_size}, scale: {scale}"
    )
    print(f"[DEBUG_SPLASH] device: {q.device} | OUT: {get_stats_str(result)}")

  if result.dtype != input_dtype:
    result = result.to(input_dtype)
  return result
