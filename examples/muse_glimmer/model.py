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

"""Standalone Muse Glimmer 30B model definition.

The architecture follows the reference implementation:
  * Grouped-query attention with a sigmoid output gate.
  * QK RMSNorm without a learned scale, and a constant Q scale factor.
  * RoPE applied in sliding-window layers only.
  * Sliding-window attention in every layer except every 4th one, which uses
    full causal attention.
  * SwiGLU feed-forward blocks with pre- and post-norms around both the
    attention and the feed-forward sub-blocks.

This module deliberately depends on nothing but PyTorch (no `torch_tpu` or
other helper libraries) so that it can run unmodified on TPU, GPU and CPU.
"""

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass
class Config:
  """Muse Glimmer 30B model and workload configuration.

  Attributes:
    vocab_size: Size of the token embedding table.
    num_layers: Number of decoder blocks.
    dmodel: Residual stream (model) dimension.
    dff: Hidden dimension of the feed-forward block.
    dhead: Per-head dimension of attention.
    num_q_heads: Number of query heads.
    num_kv_heads: Number of key/value heads (grouped-query attention).
    swa_size: Sliding attention window size, in tokens.
    rope_freq_base: RoPE theta, used by sliding-window layers.
    rmsnorm_eps: Epsilon of the pre-norms and of the QK norm.
    post_norm_eps: Epsilon of the post-norms.
    qk_scale_factor: Constant multiplier applied to the normalized queries.
    final_logit_softcapping: Tanh softcap applied to the output logits.
    output_multiplier: Logit multiplier applied before the softcap.
    batch_size: Batch size used by the benchmark workload.
    seq_len: Sequence length used by the benchmark workload.
  """

  # General model config.
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details.
  swa_size: int = 2048
  rope_freq_base: float = 500000.0
  rmsnorm_eps: float = 1e-5
  post_norm_eps: float = 1e-8
  qk_scale_factor: float = 3.87
  final_logit_softcapping: float = 20.0
  output_multiplier: float = 0.19611613513818404

  # Perf evaluation config.
  batch_size: int = 8
  seq_len: int = 5120


def is_sliding_window_layer(layer_idx: int) -> bool:
  """Returns whether layer `layer_idx` uses sliding-window attention."""
  # Every 4th layer (3, 7, 11, ...) uses full causal attention.
  return (layer_idx % 4) != 3


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates half of the trailing (head) dimension of `x`."""
  dh = x.shape[-1]
  x1 = x[..., : dh // 2]
  x2 = x[..., dh // 2 :]
  return torch.cat([-x2, x1], dim=-1)


def precompute_rope_sin_cos(
    seq_len: int, dim_head: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """Precomputes the RoPE sin/cos tables for all positions of `seq_len`.

  Args:
    seq_len: Number of positions to precompute.
    dim_head: Per-head dimension.
    rope_theta: RoPE frequency base.

  Returns:
    A (sin, cos) tuple, each shaped [1, 1, seq_len, dim_head].
  """
  inv_freq = 1.0 / (
      rope_theta
      ** (torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head)
  )
  inv_freq = inv_freq.repeat(2).view(1, 1, 1, dim_head)

  pos_idx = torch.arange(0, seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)

  sin = torch.sin(pos_idx * inv_freq)
  cos = torch.cos(pos_idx * inv_freq)
  return sin, cos


def apply_rope(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
  """Applies RoPE to `x` ([batch, heads, seq_len, dhead]) in float32."""
  dtype = x.dtype
  x = x.to(dtype=torch.float32)
  return ((x * cos) + (rotate_half(x) * sin)).to(dtype=dtype)


def build_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
  """Builds an additive causal sliding-window mask of [1, 1, S, S]."""
  mask = torch.ones(1, 1, seq_len, seq_len)
  mask = mask.tril()  # Remove future tokens.
  mask = mask.triu(-window_size + 1)  # Remove tokens outside of the window.
  return torch.where(mask == 1, 0, float("-inf"))


class Attention(nn.Module):
  """Gated grouped-query attention, sliding-window or full causal."""

  def __init__(self, config: Config, is_swa: bool):
    super().__init__()
    self.config = config
    self.is_swa = is_swa

    dm, dh = config.dmodel, config.dhead
    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads

    self.q_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
    self.k_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)
    self.v_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)

    self.gate_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
    self.o_proj = nn.Linear(num_q_heads * dh, dm, bias=False)

    # QK RMSNorm without learned scale parameters.
    self.qk_norm = nn.RMSNorm(
        normalized_shape=(dh,),
        eps=config.rmsnorm_eps,
        dtype=torch.float32,
        elementwise_affine=False,
    )

  def forward(
      self,
      h: torch.Tensor,
      rope_sin_cos: tuple[torch.Tensor, torch.Tensor] | None = None,
      swa_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Runs attention over `h` ([batch, seq_len, dmodel]).

    Args:
      h: Input activations.
      rope_sin_cos: Precomputed (sin, cos) tables; required for SWA layers.
      swa_mask: Additive sliding-window mask; required for SWA layers. Shared
        across layers by the caller so that it is materialized only once.

    Returns:
      The attention output, shaped like `h`.
    """
    local_bsz, seq_len, _ = h.shape

    dh = self.config.dhead
    num_q_heads = self.config.num_q_heads
    num_kv_heads = self.config.num_kv_heads

    q = self.q_proj(h).view(local_bsz, seq_len, num_q_heads, dh)
    k = self.k_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)
    v = self.v_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)
    gate = F.sigmoid(self.gate_proj(h))

    qk_scale = self.config.qk_scale_factor
    q = self.qk_norm(q.transpose(1, 2).contiguous()) * qk_scale
    k = self.qk_norm(k.transpose(1, 2).contiguous())
    v = v.transpose(1, 2).contiguous()

    if self.is_swa:
      if rope_sin_cos is None or swa_mask is None:
        raise ValueError(
            "sliding-window attention requires both rope_sin_cos and swa_mask"
        )
      # In Muse Glimmer 30B, RoPE is only applied in SWA layers.
      rs, rc = rope_sin_cos
      q = apply_rope(q, rs, rc)
      k = apply_rope(k, rs, rc)
      attn_out = F.scaled_dot_product_attention(
          q, k, v, attn_mask=swa_mask, enable_gqa=True
      )
    else:
      attn_out = F.scaled_dot_product_attention(
          q, k, v, is_causal=True, enable_gqa=True
      )
    attn_out = attn_out.transpose(1, 2)
    attn_out = attn_out.reshape(local_bsz, seq_len, num_q_heads * dh)

    return self.o_proj(attn_out * gate)


class FeedForwardSwiGLU(nn.Module):
  """Simple FFN with SwiGLU activation."""

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    dm, df = config.dmodel, config.dff
    self.up_proj = nn.Linear(dm, df, bias=False)
    self.gate_proj = nn.Linear(dm, df, bias=False)
    self.down_proj = nn.Linear(df, dm, bias=False)

  def forward(self, h: torch.Tensor) -> torch.Tensor:
    h = self.up_proj(h) * F.silu(self.gate_proj(h))
    return self.down_proj(h)


class DecoderBlock(nn.Module):
  """Decoder block: gated attention and SwiGLU FFN, each with pre/post norms."""

  def __init__(self, config: Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx

    self.self_attn = Attention(
        config, is_swa=is_sliding_window_layer(layer_idx)
    )
    self.mlp = FeedForwardSwiGLU(config)

    dm = config.dmodel
    f32 = torch.float32
    eps_pre, eps_post = config.rmsnorm_eps, config.post_norm_eps
    self.input_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_attention_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)
    self.pre_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)

  def forward(
      self,
      h: torch.Tensor,
      rope_sin_cos: tuple[torch.Tensor, torch.Tensor] | None = None,
      swa_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos, swa_mask)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h


class MuseGlimmer30B(nn.Module):
  """Muse Glimmer 30B decoder stack.

  The RoPE tables and the sliding-window mask are owned by this module and
  passed to every layer, so that each is materialized exactly once instead of
  once per layer.
  """

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )
    self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)

    # The LM head is part of the model, but during training it is often better
    # placed elsewhere (fused with the loss calculation), so this benchmark
    # model stops at the final norm.

    rope_sin, rope_cos = precompute_rope_sin_cos(
        seq_len=config.seq_len,
        dim_head=config.dhead,
        rope_theta=config.rope_freq_base,
    )
    self.rope_sin = nn.Buffer(rope_sin, persistent=False)
    self.rope_cos = nn.Buffer(rope_cos, persistent=False)
    self.swa_mask = nn.Buffer(
        build_sliding_window_mask(config.seq_len, config.swa_size),
        persistent=False,
    )

  def forward(self, tok: torch.Tensor) -> torch.Tensor:
    h = self.embed_tokens(tok)
    rope_sin_cos = (self.rope_sin, self.rope_cos)
    for layer in self.layers:
      h = layer(h, rope_sin_cos, self.swa_mask)
    return self.norm(h)

  def apply_torch_compile(self, backend: str) -> None:
    """Compiles the model layer by layer with the given dynamo backend."""
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)  # pyrefly: ignore[bad-argument-type]
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)  # pyrefly: ignore[bad-assignment]


@torch.no_grad()
def init_nn_module(m: nn.Module) -> None:
  """Initializes all parameters of `m` with small random values."""
  for p in m.parameters():
    p.data.copy_(torch.randn_like(p.data) * 0.05)


def count_parameters(m: nn.Module) -> int:
  """Returns the total number of parameters of `m`."""
  return sum(p.numel() for p in m.parameters())


def print_module(m: nn.Module) -> None:
  """Prints the buffer and parameter shapes/dtypes of `m`."""
  print("Model buffers:", flush=True)
  for n, p in m.named_buffers(recurse=True):
    print(f" {n}: {p.shape}, {p.dtype}", flush=True)

  print("Model parameters:", flush=True)
  for n, p in m.named_parameters(recurse=True):
    print(
        f" {n}: {p.shape}, {p.dtype}, trainable={p.requires_grad}", flush=True
    )
