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
Uses torch.tpu.pallas.jax_op() to bridge JAX -> PyTorch.
Based on torchtitan's implementation which can be found at
https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py

IMPORTANT — Why `jax` is imported here (Kernel Authoring vs. Runtime Execution):
  Pallas is distributed inside the `jax` package, so configuring and defining
  the kernel uses `jax` / `jax.numpy` syntax and `jax.Array` type annotations.
  However, **JAX is NOT executed at runtime**:
  1. On first invocation for a given shape/dtype signature,
     `torch.tpu.pallas.jax_op()` traces `splash_fn` and `splash_bwd_fn` once
     using abstract `jax.ShapeDtypeStruct` placeholders (not device arrays) and
     exports a serialized StableHLO / Mosaic MLIR module into the `torch_tpu`
     C++ custom-call registry.
  2. During training and inference (both eager and `torch.compile`), `torch_tpu`
     invokes the compiled XLA custom kernel directly on native PyTorch
     `torch.Tensor` TPU buffers — no `torch.Tensor` <-> `jax.Array` conversion
     or JAX runtime execution ever takes place.
"""

import hashlib
import math
import os
import typing

# JAX imports: used strictly at trace time as a kernel-authoring DSL to lower
# the Pallas SplashAttention kernel into a serialized StableHLO/Mosaic MLIR
# module via `torch.tpu.pallas.jax_op`. No JAX runtime executes during PyTorch
# forward or backward steps.
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import QKVLayout
import jax.numpy as jnp
import torch
import torch_tpu  # pylint: disable=unused-import

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

  Args:
    seq_len: Sequence length.
    n_heads: Number of query heads.
    n_kv_heads: Number of key/value heads.
    is_causal: Whether to apply causal attention masking.
    local_window_size: Sliding window size for local attention, or None.
    block_q: Query block size for forward pass.
    block_kv: Key/Value block size for forward pass.
    block_kv_compute: Key/Value compute block size for forward pass.
    block_q_dkv: Query block size for dK/dV backward pass.
    block_kv_dkv: Key/Value block size for dK/dV backward pass.
    block_kv_dkv_compute: Key/Value compute block size for dK/dV backward pass.
    block_q_dq: Query block size for dQ backward pass when not fused.
    block_kv_dq: Key/Value block size for dQ backward pass when not fused.
    use_fused_bwd_kernel: Whether to use fused dQ/dK/dV backward kernel.
    q_layout: Memory layout for Q ("HEAD_DIM_MINOR" or "SEQ_MINOR").
    k_layout: Memory layout for K ("HEAD_DIM_MINOR" or "SEQ_MINOR").
    v_layout: Memory layout for V ("HEAD_DIM_MINOR" or "SEQ_MINOR").
    use_vmap_bwd: Whether to parallelize backward over batch with jax.vmap.

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

  # Static mask metadata, computed on the host as NumPy arrays that get embedded
  # as constants when the kernels are traced for export. This is what
  # `splash_attention_kernel.make_splash_mha` (and `make_splash_mqa`) computes,
  # minus its final `jnp.array` conversion, which runs JAX programs to stage
  # the metadata on JAX's default device: on a TPU host, the very TPU that
  # TorchTPU concurrently executes on.
  fwd_mask_info, _mask_function = splash_attention_mask_info.process_mask(
      multi_head_mask,
      (block_sizes.block_q, block_sizes.block_kv),
      head_shards=1,
      q_seq_shards=1,
  )
  dq_mask_info = None
  dkv_mask_info = None
  if block_sizes.has_backward_blocks:
    if not use_fused_bwd_kernel:
      dq_mask_info, _ = splash_attention_mask_info.process_mask(
          multi_head_mask,
          (block_sizes.block_q_dq, block_sizes.block_kv_dq),
          head_shards=1,
          q_seq_shards=1,
      )
    dkv_mask_info, _ = splash_attention_mask_info.process_mask_dkv(
        multi_head_mask,
        (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
        head_shards=1,
        q_seq_shards=1,
        shrink_grid=not use_fused_bwd_kernel,
    )
  # The remaining kernel options keep the defaults of `make_splash_mha`.
  _mask_value = _DEFAULT_MASK_VALUE
  _attn_logits_soft_cap = None
  _residual_checkpoint_name = None
  _interpret = False

  def _single_fwd_with_lse(q_b, k_b, v_b):
    """Forward for one batch element; returns (out_b, logsumexp_b)."""
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

  # Note: `jax.Array` type annotations below allow `torch.tpu.pallas.jax_op` to
  # infer the PyTorch custom_op schema automatically. Both `splash_fn` and
  # `splash_bwd_fn` are traced only once with abstract ShapeDtypeStruct
  # placeholders to export a serialized MLIR module; they do not run JAX at
  # model execution time.
  def splash_fn(
      q: jax.Array, k: jax.Array, v: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Forward: returns (out, logsumexp) for the whole batch."""
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale
    out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_scaled, k, v)
    return out, logsumexp

  def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b):
    """Backward for one batch element using precomputed residuals."""
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
    return all_grads[3], all_grads[4], all_grads[5]  # dq_scaled, dk, dv

  def _scan_bwd_body(carry, args):
    """Single-element backward body for jax.lax.scan (sequential over batch)."""
    q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b = args
    dq_scaled_b, dk_b, dv_b = _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b
    )
    return carry, (dq_scaled_b, dk_b, dv_b)

  def splash_bwd_fn(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      out: jax.Array,
      logsumexp: jax.Array,
      g: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Backward: uses saved (out, logsumexp) — no extra forward pass."""
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
    dq = dq_scaled * scale
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


_splash_op_cache = {}


@torch._dynamo.assume_constant_result
def _get_splash_op(**splash_kwargs):
  """Create and cache PyTorch forward/backward custom ops for splash attention."""
  cache_key = tuple(sorted(splash_kwargs.items()))
  if cache_key in _splash_op_cache:
    return _splash_op_cache[cache_key]

  splash_fn, splash_bwd_fn = _make_splash_attention_fn(**splash_kwargs)

  identifier = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
  op_fwd_name = f"splash_attn_fwd_{identifier}"
  op_bwd_name = f"splash_attn_bwd_{identifier}"
  # `jax_op` lowers `splash_fn` and `splash_bwd_fn` to serialized MLIR modules
  # via abstract shape/dtype tracing and registers them as native PyTorch
  # `torch.library.custom_op` operators backed by the `torch_tpu` C++ runtime.
  torch_fwd_fn: typing.Any = torch.tpu.pallas.jax_op(
      f"pallas::{op_fwd_name}", splash_fn
  )
  torch_bwd_fn: typing.Any = torch.tpu.pallas.jax_op(
      f"pallas::{op_bwd_name}", splash_bwd_fn
  )

  def setup_context(ctx, inputs, output):
    q, k, v = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)

  def backward(ctx, grad_out, grad_lse):
    del grad_lse
    q, k, v, out, lse = ctx.saved_tensors
    grad_q, grad_k, grad_v = torch_bwd_fn(q, k, v, out, lse, grad_out)
    return grad_q, grad_k, grad_v

  torch_fwd_fn.register_autograd(backward, setup_context=setup_context)

  # Return the registered op name rather than the `CustomOpDef` object.
  # TorchDynamo cannot install guards on `CustomOpDef` objects returned from an
  # `assume_constant_result` function, which breaks `torch.compile`. Strings
  # are trivially guardable; callers look the op up via `torch.ops.pallas`.
  _splash_op_cache[cache_key] = op_fwd_name
  return op_fwd_name


def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = True,
    local_window_size: int | None = None,
    enable_gqa: bool = False,
    block_q: int = 512,
    block_kv: int = 512,
    block_dkv: int | None = None,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool = True,
    use_vmap_bwd: bool = False,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using splash attention.

  Accepts q, k, v in (B, H, S, D) format matching PyTorch SDPA.

  Args:
    q: Query tensor of shape (batch, n_heads, seq_len, head_dim) on TPU.
    k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim) on TPU.
    v: Value tensor of shape (batch, n_kv_heads, seq_len, head_dim) on TPU.
    scale: Optional custom scale factor for attention logits (currently unused;
      defaults to 1 / sqrt(head_dim)).
    is_causal: Whether to apply a causal attention mask.
    local_window_size: Sliding window size for local attention, or None for
      full/causal attention.
    enable_gqa: Whether grouped-query attention (n_heads != n_kv_heads) is
      enabled.
    block_q: Query tile size for the forward pass (adjusted to divide seq_len).
    block_kv: Key/Value tile size for the forward pass.
    block_dkv: If set, overrides both `block_q_dkv` and `block_kv_dkv` with this
      tile size. The `SPLASH_BLOCK_DKV` environment variable takes precedence.
    block_kv_compute: Key/Value compute sub-tile size for the forward pass.
    block_q_dkv: Query tile size for the dK/dV backward pass.
    block_kv_dkv: Key/Value tile size for the dK/dV backward pass.
    block_kv_dkv_compute: Key/Value compute sub-tile size for the dK/dV backward
      pass.
    block_q_dq: Query tile size for the dQ backward pass when unfused.
    block_kv_dq: Key/Value tile size for the dQ backward pass when unfused.
    use_fused_bwd_kernel: Whether to use the fused backward kernel.
    use_vmap_bwd: Whether to use jax.vmap (True) or jax.lax.scan (False) over
      the batch dimension in the backward pass.
    q_layout: Memory layout for Q ("HEAD_DIM_MINOR" or "SEQ_MINOR").
    k_layout: Memory layout for K ("HEAD_DIM_MINOR" or "SEQ_MINOR").
    v_layout: Memory layout for V ("HEAD_DIM_MINOR" or "SEQ_MINOR").

  Returns:
    Attention output tensor of shape (batch, n_heads, seq_len, head_dim) with
    the same dtype as q.
  """
  block_q = int(os.environ.get("SPLASH_BLOCK_Q", block_q))
  block_kv = int(os.environ.get("SPLASH_BLOCK_KV", block_kv))
  # `block_dkv` is a shorthand that sets both dK/dV backward tile sizes at once.
  # The kernel only reads `block_q_dkv` and `block_kv_dkv`, so the shorthand
  # has to be copied into them. It is applied only when set (by argument or by
  # `SPLASH_BLOCK_DKV`), so explicit `block_q_dkv`/`block_kv_dkv` values are
  # otherwise left untouched.
  dkv_env = os.environ.get("SPLASH_BLOCK_DKV")
  if dkv_env is not None:
    block_dkv = int(dkv_env)
  if block_dkv is not None:
    block_q_dkv = block_dkv
    block_kv_dkv = block_dkv

  batch, n_heads, seq_len, head_dim = q.shape
  n_kv_heads = k.shape[1]

  def _adjust_block(b_size: int, s_len: int) -> int:
    if s_len % b_size == 0:
      return b_size
    for candidate in [512, 256, 128, 64, 32]:
      if s_len % candidate == 0 and candidate <= b_size:
        return candidate
    return math.gcd(s_len, b_size)

  block_q = _adjust_block(block_q, seq_len)
  block_kv = _adjust_block(block_kv, seq_len)
  block_kv_compute = _adjust_block(block_kv_compute, seq_len)
  block_q_dkv = _adjust_block(block_q_dkv, seq_len)
  block_kv_dkv = _adjust_block(block_kv_dkv, seq_len)
  block_kv_dkv_compute = _adjust_block(block_kv_dkv_compute, seq_len)
  if block_q_dq is not None:
    block_q_dq = _adjust_block(block_q_dq, seq_len)
  if block_kv_dq is not None:
    block_kv_dq = _adjust_block(block_kv_dq, seq_len)

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
  # `_get_splash_op` returns the registered op name rather than the op object
  # so that it stays guardable under `torch.compile` (see `_get_splash_op`).
  # Look the op up by name through the `torch.ops.pallas` namespace.
  torch_fwd_fn = getattr(torch.ops.pallas, _get_splash_op(**splash_args))

  if scale is not None:
    default_scale = 1.0 / math.sqrt(head_dim)
    if abs(scale - default_scale) > 1e-6:
      scale_ratio = scale / default_scale
      q = q * scale_ratio

  input_dtype = q.dtype
  if n_kv_heads == 1:
    k = torch.squeeze(k, dim=1)
    v = torch.squeeze(v, dim=1)
  result, _ = torch_fwd_fn(q, k, v)
  result = typing.cast(torch.Tensor, result)

  if result.dtype != input_dtype:
    result = result.to(input_dtype)
  return result
