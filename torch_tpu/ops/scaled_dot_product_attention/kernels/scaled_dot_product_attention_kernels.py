# Copyright 2025 Google LLC
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

"""JAX/Pallas kernels scaled dot product attention."""

import dataclasses
import functools
from typing import cast

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu.ops.scaled_dot_product_attention.kernels import flash_attention
from torch_tpu.ops.scaled_dot_product_attention.kernels.splash import splash_attention

try:
  # Works whenever we have an updated jaxlib, i.e. g3 and when oss goes out.
  import jax.experimental.pallas  # pylint: disable=g-import-not-at-top

  pallas_costEstimate = jax.experimental.pallas.CostEstimate
  pallas_export_experimental = (
      jax.experimental.pallas.pallas_export_experimental
  )
except AttributeError:
  # Works in oss because we don't dependency check
  import jax._src.pallas.core as pallas_core  # pylint: disable=g-import-not-at-top

  pallas_export_experimental = pallas_core.pallas_export_experimental
  pallas_costEstimate = pallas_core.CostEstimate

DEFAULT_MASKED_VALUE = -1e30
_BLOCK_SIZE = 512


@dataclasses.dataclass
class InputSizes:
  batch_size: int | None
  q_num_heads: int | None
  kv_num_heads: int | None
  q_seq_len: int | None
  kv_seq_len: int | None
  qk_head_dim: int | None
  v_head_dim: int | None


################################################################################
## JAX/Torch translation functions ##
################################################################################


# Convert JAX arrays to Torch tensors.
def to_torch(x):
  if hasattr(x, "dtype") and x.dtype == jnp.bfloat16:
    return torch.from_numpy(np.array(x, dtype=np.float32)).to(torch.bfloat16)
  return torch.from_numpy(np.array(x))


def to_jax(x):
  if hasattr(x, "dtype") and x.dtype == torch.bfloat16:
    return jnp.array(x.to(torch.float32).numpy()).astype(jnp.bfloat16)
  return jnp.array(x.numpy())


################################################################################
# def sdpa_kernel_reference(q, k, v):
#   seq_len, head_dim = q.shape
#   scale = 1.0 / jnp.sqrt(head_dim)
#   attn_scores = jnp.einsum("qd,kd->qk", q, k) * scale
#   i = jnp.arange(seq_len)[:, None]
#   j = jnp.arange(seq_len)[None, :]
#   mask = j <= i
#   attn_scores = jnp.where(mask, attn_scores, DEFAULT_MASKED_VALUE)
#   attn_weights = jnn.softmax(attn_scores, axis=1)
#   return jnp.einsum("qk,kd->qd", attn_weights, v)


################################################################################
class SDPAKernelReferenceJax:
  """JAX reference implementation of SDPA."""

  @classmethod
  def forward(
      cls,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
  ):
    """Reference implementation using jax.nn.dot_product_attention."""

    del dropout_p, enable_gqa

    assert attn_mask is None

    return jnp.swapaxes(
        jnn.dot_product_attention(
            jnp.swapaxes(query, -3, -2),
            jnp.swapaxes(key, -3, -2),
            jnp.swapaxes(value, -3, -2),
            is_causal=is_causal,
            scale=1.0 / jnp.sqrt(query.shape[-1])
            if scale is None
            else scale,  # scale,
        ),
        -3,
        -2,
    )

  @classmethod
  def backward(
      cls,
      grad_out,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
  ):
    """Backward implementation for JAX reference."""
    del attn_mask, dropout_p
    return jax.vjp(
        functools.partial(
            cls.forward,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        ),
        query,
        key,
        value,
    )[1](grad_out)

  @staticmethod
  def get_shapes(dtype):
    """Returns symbolic shapes for SDPA inputs."""
    (
        # pylint: disable=invalid-name
        B_sym,
        Hq_sym,
        Hkv_sym,
        Lq_sym,
        Lkv_sym,
        Eqk_sym,
        Ev_sym,
        # pylint: enable=invalid-name
    ) = jax.export.symbolic_shape(
        "B,Hq,Hkv,Lq,Lkv,Eqk,Ev",
        constraints=(
            # "H*floordiv(Hq, H)==Hq",
            # The following are true given the above constraint, but the
            # symbolic shape resolution logic can't handle this currently.
            # "B*E*H*S*floordiv(Hq, Hkv)==B*Eqk*Hq*Lkv",
            # "B*Ey*H*S*floordiv(Hq, Hkv)==B*Ey*Hq*S",
            "mod(Hq, Hkv) == 0",
            "Eqk == Ev",
        ),
    )

    q_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, Lq_sym, Eqk_sym), dtype)
    k_shape = jax.ShapeDtypeStruct((B_sym, Hkv_sym, Lkv_sym, Eqk_sym), dtype)
    v_shape = jax.ShapeDtypeStruct((B_sym, Hkv_sym, Lkv_sym, Ev_sym), dtype)
    out_shape = jax.ShapeDtypeStruct((B_sym, Hq_sym, Lq_sym, Ev_sym), dtype)

    return q_shape, k_shape, v_shape, out_shape

  @classmethod
  def export_forward(cls, *, dtype, is_causal, **kwargs):
    """Exports JAX reference forward to MLIR."""
    del kwargs
    kernel_fn = functools.partial(
        cls.forward,
        is_causal=is_causal,
        enable_gqa=True,
    )
    q_shape, k_shape, v_shape, _ = cls.get_shapes(dtype)
    with pallas_export_experimental(dynamic_shapes=True):
      return jax.export.export(jax.jit(kernel_fn), platforms=["tpu"])(
          q_shape, k_shape, v_shape
      )

  @classmethod
  def export_backward(cls, *, dtype, is_causal, **kwargs):
    """Exports JAX reference backward to MLIR."""
    del kwargs
    kernel_fn = functools.partial(
        cls.backward,
        is_causal=is_causal,
        enable_gqa=True,
    )
    q_shape, k_shape, v_shape, out_shape = cls.get_shapes(dtype)
    with pallas_export_experimental(dynamic_shapes=True):
      return jax.export.export(jax.jit(kernel_fn), platforms=["tpu"])(
          out_shape, q_shape, k_shape, v_shape
      )


################################################################################
class SDPAKernelReferenceTorch:
  """Torch reference implementation of SDPA."""

  @classmethod
  def forward(
      cls,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
  ):
    """Forward implementation for Torch reference."""
    del enable_gqa
    # # pylint: disable=g-doc-args,g-doc-return-or-yield
    # """Port of reference implementation from Torch documentation page.

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # """
    # # pylint: enable=g-doc-args,g-doc-return-or-yield
    # assert attn_mask is None
    # assert dropout_p == 0.0

    # # Hq = query.shape[-2]
    # # H = key.shape[-2]
    # L, S = query.shape[-2], key.shape[-2]  # pylint: disable=invalid-name
    # scale_factor = 1 / jnp.sqrt(query.shape[-1]) if scale is None else scale
    # attn_bias = jnp.zeros(shape=(L, S), dtype=query.dtype)
    # if is_causal:
    #   assert attn_mask is None
    #   temp_mask = jnp.triu(jnp.ones(shape=(L, S), dtype=jnp.bool), k=1)
    #   attn_bias = temp_mask * DEFAULT_MASKED_VALUE

    # # TODO(elliotenglish): Add support for attn_mask.
    # # if attn_mask is not None:
    # #   if attn_mask.dtype == torch.bool:
    # #     attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    # #   else:
    # #     attn_bias = attn_mask + attn_bias

    # if enable_gqa:
    #   key = jnp.repeat(key, query.shape[-3] // key.shape[-3], -3)
    #   value = jnp.repeat(value, query.shape[-3] // value.shape[-3], -3)

    # attn_weight = query @ jnp.swapaxes(key, -2, -1) * scale_factor
    # attn_weight += attn_bias
    # attn_weight = jnn.softmax(attn_weight, axis=-1)
    # # TODO(elliotenglish): Add support for dropout.
    # # attn_weight = ...dropout(attn_weight, dropout_p, train=True)
    # return attn_weight @ value

    q_t = to_torch(query)
    k_t = to_torch(key)
    v_t = to_torch(value)
    mask_t = to_torch(attn_mask) if attn_mask is not None else None

    out_t = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=mask_t,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,  # enable_gqa,
    )

    return to_jax(out_t)

  @classmethod
  def backward(
      cls,
      grad_out,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
  ):
    """Reference backward implementation for Torch."""
    del enable_gqa

    q_t = to_torch(query)
    k_t = to_torch(key)
    v_t = to_torch(value)
    mask_t = to_torch(attn_mask) if attn_mask is not None else None
    grad_out_t = to_torch(grad_out)

    assert q_t is not None
    assert k_t is not None
    assert v_t is not None

    q_t.requires_grad_(True)
    k_t.requires_grad_(True)
    v_t.requires_grad_(True)

    # Run Torch SDPA forward.
    out_t = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=mask_t,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,  # enable_gqa,
    )

    # Run Torch SDPA backward.
    out_t.backward(grad_out_t)

    return to_jax(q_t.grad), to_jax(k_t.grad), to_jax(v_t.grad)

  @classmethod
  def export_forward(
      cls,
      *,
      dtype,
      is_causal,
  ):
    raise NotImplementedError

  @classmethod
  def export_backward(
      cls,
      *,
      dtype,
      is_causal,
  ):
    raise NotImplementedError


########################################################################
## Flash Attention Kernels ##
########################################################################
def flash_attention_gqa(q, k, v, causal, block_sizes=None):
  """Flash Attention wrapper that adds GQA using vmap."""
  q_heads = q.shape[-3]
  kv_heads = k.shape[-3]
  qk_head_dim = q.shape[-1]
  v_head_dim = v.shape[-1]

  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]

  groups = kv_heads
  batch_size = q.shape[0]

  q_grouped = q.reshape(
      batch_size, groups, q_heads // groups, q_seq_len, qk_head_dim
  )
  k_grouped = k.reshape(
      batch_size, groups, kv_heads // groups, kv_seq_len, qk_head_dim
  )
  v_grouped = v.reshape(
      batch_size, kv_heads, kv_heads // groups, kv_seq_len, v_head_dim
  )

  def process_group(q, k, v):

    def process_head(q, k, v):
      return flash_attention.flash_attention(
          q, k, v, causal=causal, block_sizes=block_sizes
      )

    return jax.vmap(process_head, in_axes=(1, None, None), out_axes=1)(
        q.reshape(batch_size, q_heads // groups, 1, q_seq_len, qk_head_dim),
        k,
        v,
    ).reshape(batch_size, q_heads // groups, q_seq_len, v_head_dim)

  return jax.vmap(
      process_group,
      in_axes=1,
      out_axes=1,
  )(
      q_grouped,
      k_grouped,
      v_grouped,
  ).reshape(batch_size, q_heads, q_seq_len, v_head_dim)


class SDPAKernelFlashAttention:
  """Flash Attention implementation of SDPA."""

  @classmethod
  def forward(
      cls,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
      block_size=_BLOCK_SIZE,
  ):
    """Forward implementation for Flash Attention."""
    del attn_mask, dropout_p, scale, enable_gqa
    q, k, v = query, key, value
    _, _, _, head_dim_qk = q.shape
    q = q / jnp.sqrt(head_dim_qk)
    out = flash_attention_gqa(
        q=q,
        k=k,
        v=v,
        block_sizes=flash_attention.BlockSizes(
            block_q=block_size,
            block_k=block_size,
            block_k_major=block_size,
            block_b=1,
            block_q_major_dkv=block_size,
            block_k_major_dkv=block_size,
            block_k_dkv=block_size,
            block_q_dkv=block_size,
            block_k_major_dq=block_size,
            block_k_dq=block_size,
            block_q_dq=block_size,
        ),
        causal=is_causal,
        # sm_scale=scale if scale is not None else 1.0,
    )
    return out

  @classmethod
  def backward(
      cls,
      grad_out,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
      block_size=_BLOCK_SIZE,
  ):
    """Wrapper for Flash Attention backward kernel."""
    grad_q, grad_k, grad_v = jax.vjp(
        functools.partial(
            cls.forward,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            block_size=block_size,
        ),
        query,
        key,
        value,
    )[1](grad_out)
    return grad_q, grad_k, grad_v

  @staticmethod
  def get_shapes(
      dtype,
      # parameters static shapes
      input_sizes: InputSizes,
      # parameters for shape constraints
      block_size=_BLOCK_SIZE,
  ):
    """Returns shapes for Flash Attention inputs."""
    if input_sizes is None:
      # Use symbolic shapes
      # This set of symbolic shapes set the constraint explicitly and
      # implicitly:
      # - batch_size >= 1
      # - q_seq_len and kv_seq_len are multiples of 512.
      # - head_dim is a multiple of 128.
      # - qk_head_dim == v_head_dim
      # - mod(num_q_heads, num_kv_heads) == 0

      (
          # pylint: disable=invalid-name
          batch_size_sym,
          num_q_heads_sym,
          num_kv_heads_sym,
          q_seq_len_sym,
          kv_seq_len_sym,
          qk_head_dim_sym,
          v_head_dim_sym,
          # pylint: enable=invalid-name
      ) = jax.export.symbolic_shape(
          """batch_size,
              num_q_heads,
              num_kv_heads,
              q_seq_len,
              kv_seq_len,
              qk_head_dim,
              v_head_dim""",
          constraints=(
              "batch_size >= 1",
              f"q_seq_len >= {block_size}",
              f"kv_seq_len >= {block_size}",
              f"mod(q_seq_len, {block_size}) == 0",
              f"mod(kv_seq_len, {block_size}) == 0",
              # TODO(elliotenglish): Add support for GQA.
              "mod(num_q_heads, num_kv_heads) == 0",
              "qk_head_dim == v_head_dim",
          ),
      )

      q_shape = (
          batch_size_sym,
          num_q_heads_sym,
          q_seq_len_sym,
          qk_head_dim_sym,
      )
      k_shape = (
          batch_size_sym,
          num_kv_heads_sym,
          kv_seq_len_sym,
          qk_head_dim_sym,
      )
      v_shape = (
          batch_size_sym,
          num_kv_heads_sym,
          kv_seq_len_sym,
          v_head_dim_sym,
      )
      out_shape = (
          batch_size_sym,
          num_q_heads_sym,
          q_seq_len_sym,
          v_head_dim_sym,
      )
    else:
      # Defaults if not symbolic
      q_shape = (
          input_sizes.batch_size,
          input_sizes.q_num_heads,
          input_sizes.q_seq_len,
          input_sizes.qk_head_dim,
      )
      k_shape = (
          input_sizes.batch_size,
          input_sizes.kv_num_heads,
          input_sizes.kv_seq_len,
          input_sizes.qk_head_dim,
      )
      v_shape = (
          input_sizes.batch_size,
          input_sizes.kv_num_heads,
          input_sizes.kv_seq_len,
          input_sizes.v_head_dim,
      )
      out_shape = (
          input_sizes.batch_size,
          input_sizes.q_num_heads,
          input_sizes.q_seq_len,
          input_sizes.v_head_dim,
      )

    q_dtype = jax.ShapeDtypeStruct(q_shape, dtype)
    k_dtype = jax.ShapeDtypeStruct(k_shape, dtype)
    v_dtype = jax.ShapeDtypeStruct(v_shape, dtype)
    grad_out_dtype = jax.ShapeDtypeStruct(out_shape, dtype)

    return q_dtype, k_dtype, v_dtype, grad_out_dtype

  @classmethod
  def export_forward(
      cls,
      *,
      dtype,
      is_causal,
      input_sizes: InputSizes,
      block_size=_BLOCK_SIZE,
      **kwargs,
  ):
    """Exports Flash Attention forward to MLIR."""
    q_dtype, k_dtype, v_dtype, _ = cls.get_shapes(
        dtype,
        input_sizes,
        block_size,
    )

    f_p = functools.partial(
        cls.forward, is_causal=is_causal, block_size=block_size, **kwargs
    )
    with pallas_export_experimental(dynamic_shapes=True):
      return jax.export.export(jax.jit(f_p), platforms=["tpu"])(
          q_dtype, k_dtype, v_dtype
      )

  @classmethod
  def export_backward(
      cls,
      *,
      dtype,
      is_causal,
      input_sizes: InputSizes,
      block_size=_BLOCK_SIZE,
      **kwargs,
  ):
    """Exports Flash Attention backward to MLIR."""

    q_dtype, k_dtype, v_dtype, out_dtype = cls.get_shapes(
        dtype,
        input_sizes,
        block_size,
    )

    f_p = functools.partial(
        cls.backward, is_causal=is_causal, block_size=block_size, **kwargs
    )
    with pallas_export_experimental(dynamic_shapes=True):
      return jax.export.export(jax.jit(f_p), platforms=["tpu"])(
          out_dtype, q_dtype, k_dtype, v_dtype
      )

########################################################################
## Splash Attention Kernels ##
########################################################################


class SDPAKernelSplashAttention:
  """Splash Attention implementation of SDPA."""

  @classmethod
  def forward(
      cls,
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
      enable_gqa=False,
      block_size=_BLOCK_SIZE,
  ):
    """Forward implementation for Splash Attention."""
    del attn_mask, dropout_p, scale

    _, _, q_seq_len, head_dim_qk = query.shape
    _, _, kv_seq_len, _ = key.shape
    _, _, _, v_head_dim = value.shape
    q = query / jnp.sqrt(head_dim_qk)

    # Splash Attention for a single batch expects 3D inputs:
    #   [num_heads, seq_len, head_dim]
    # We vmap over the batch dimension (axis 0).

    mask = (
        splash_attention.mask_lib.CausalMask((q_seq_len, kv_seq_len))
        if is_causal
        else splash_attention.mask_lib.FullMask((q_seq_len, kv_seq_len))
    )

    config = splash_attention.SplashConfig(
        block_q=block_size,
        block_kv=block_size,
        block_kv_compute=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
        # The cost estimate is added to avoid the computations of the default
        # cost estimate in Splash Attention kernel, since those only
        # work for static shapes.
        fwd_cost_estimate=pallas_costEstimate(
            flops=0,
            transcendentals=0,
            bytes_accessed=0,
        ),
    )

    mha_kernel = splash_attention.make_splash_mha_single_device(
        mask, config=config, mask_value=DEFAULT_MASKED_VALUE
    )

    def splash_attention_forward_single_batch(q, k, v):
      num_q_heads = q.shape[0]
      num_kv_heads = k.shape[0]

      if enable_gqa:
        groups = num_kv_heads
        query_heads_per_group = num_q_heads // groups
        kv_heads_per_group = num_kv_heads // groups

        q_grouped = q.reshape(
            groups, query_heads_per_group, q_seq_len, head_dim_qk
        )
        k_grouped = k.reshape(
            groups, kv_heads_per_group, kv_seq_len, head_dim_qk
        )
        v_grouped = v.reshape(
            groups, kv_heads_per_group, kv_seq_len, v_head_dim
        )

        def process_group(q_g, k_g, v_g):
          # q_g: [query_heads_per_group, q_seq_len, head_dim_qk]
          # k_g: [kv_heads_per_group, kv_seq_len, head_dim_qk]
          # v_g: [kv_heads_per_group, kv_seq_len, v_head_dim]

          def process_head(q_h_sub):
            # q_h_sub: [1, q_seq_len, head_dim_qk]
            return mha_kernel(q_h_sub, k_g, v_g)

          # 3rd inner vmap to force the inner kernel to see a static head
          # dimension of exactly 1. This is required to bypass symbolic shape
          # lookup crashes (KeyError/ValueError on _DimExpr) during Pallas
          # lowering with dynamic shapes. Due to vmap vectorization, this
          # shouldn't have any perf overhead.
          q_g_expanded = q_g.reshape(
              query_heads_per_group, 1, q_seq_len, head_dim_qk
          )
          # Cast the result to Array to reassure pytype.
          out_heads = cast(jax.Array, jax.vmap(process_head)(q_g_expanded))
          return out_heads.reshape(query_heads_per_group, q_seq_len, v_head_dim)

        out_grouped = jax.vmap(process_group)(q_grouped, k_grouped, v_grouped)
        return out_grouped.reshape(num_q_heads, q_seq_len, v_head_dim)
      else:
        return mha_kernel(q, k, v)

    out = jax.vmap(splash_attention_forward_single_batch)(q, key, value)
    return out

  @staticmethod
  def get_shapes(
      dtype,
      input_sizes: InputSizes | None,
      block_size=_BLOCK_SIZE,
      enable_gqa=False,
  ):
    """Determines input and output shapes."""
    if input_sizes is None:
      constraints = (
          "batch_size >= 1",
          f"q_seq_len >= {block_size}",
          f"kv_seq_len >= {block_size}",
          f"mod(q_seq_len, {block_size}) == 0",
          f"mod(kv_seq_len, {block_size}) == 0",
          "qk_head_dim == v_head_dim",
          "v_head_dim >= 128",
      )
      constraints += (
          "mod(num_q_heads, num_kv_heads) == 0"
          if enable_gqa
          else "num_q_heads == num_kv_heads",
      )

      (
          batch_size_sym,
          num_q_heads_sym,
          num_kv_heads_sym,
          q_seq_len_sym,
          kv_seq_len_sym,
          qk_head_dim_sym,
          v_head_dim_sym,
      ) = jax.export.symbolic_shape(
          """batch_size,
              num_q_heads,
              num_kv_heads,
              q_seq_len,
              kv_seq_len,
              qk_head_dim,
              v_head_dim""",
          constraints=constraints,
      )
      q_shape = (
          batch_size_sym,
          num_q_heads_sym,
          q_seq_len_sym,
          qk_head_dim_sym,
      )
      k_shape = (
          batch_size_sym,
          num_kv_heads_sym,
          kv_seq_len_sym,
          qk_head_dim_sym,
      )
      v_shape = (
          batch_size_sym,
          num_kv_heads_sym,
          kv_seq_len_sym,
          v_head_dim_sym,
      )
      out_shape = (
          batch_size_sym,
          num_q_heads_sym,
          q_seq_len_sym,
          v_head_dim_sym,
      )
    else:
      q_shape = (
          input_sizes.batch_size,
          input_sizes.q_num_heads,
          input_sizes.q_seq_len,
          input_sizes.qk_head_dim,
      )
      k_shape = (
          input_sizes.batch_size,
          input_sizes.kv_num_heads,
          input_sizes.kv_seq_len,
          input_sizes.qk_head_dim,
      )
      v_shape = (
          input_sizes.batch_size,
          input_sizes.kv_num_heads,
          input_sizes.kv_seq_len,
          input_sizes.v_head_dim,
      )
      out_shape = (
          input_sizes.batch_size,
          input_sizes.q_num_heads,
          input_sizes.q_seq_len,
          input_sizes.v_head_dim,
      )

    q_dtype = jax.ShapeDtypeStruct(q_shape, dtype)
    k_dtype = jax.ShapeDtypeStruct(k_shape, dtype)
    v_dtype = jax.ShapeDtypeStruct(v_shape, dtype)
    grad_out_dtype = jax.ShapeDtypeStruct(out_shape, dtype)

    return q_dtype, k_dtype, v_dtype, grad_out_dtype

  @classmethod
  def export_forward(
      cls,
      *,
      dtype,
      is_causal,
      input_sizes: InputSizes,
      block_size=_BLOCK_SIZE,
      **kwargs,
  ):
    """Exports Splash Attention forward to MLIR."""
    q_dtype, k_dtype, v_dtype, _ = cls.get_shapes(
        dtype,
        input_sizes,
        block_size,
        enable_gqa=kwargs.get("enable_gqa", False),
    )

    f_p = functools.partial(
        cls.forward, is_causal=is_causal, block_size=block_size, **kwargs
    )
    with pallas_export_experimental(dynamic_shapes=True):
      return jax.export.export(jax.jit(f_p), platforms=["tpu"])(
          q_dtype, k_dtype, v_dtype
      )

  @classmethod
  def export_backward(cls, *, dtype, is_causal, **kwargs):
    """Exports Splash Attention backward to MLIR."""
    raise NotImplementedError("Splash Attention backward is not implemented.")
