# Copyright 2026 The TorchTPU Authors
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

"""OpenXLA and TorchTPU Compatible DeepSeek-V4-Flash Model Architecture.

This module replaces custom CUDA / TileLang kernels and in-place tensor mutations
with purely functional, XLA-compilable PyTorch operations.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

try:
  from .fallbacks import (
      act_quant_fallback as act_quant,
      apply_rotary_emb_functional,
      fp4_act_quant_fallback as fp4_act_quant,
      fp4_gemm_fallback as fp4_gemm,
      fp8_gemm_fallback as fp8_gemm,
      hadamard_transform_fallback,
      hc_split_sinkhorn_fallback as hc_split_sinkhorn,
      mla_sparse_attn_fallback as sparse_attn,
      update_kv_cache_functional,
  )
except ImportError:
  from fallbacks import (
      act_quant_fallback as act_quant,
      apply_rotary_emb_functional,
      fp4_act_quant_fallback as fp4_act_quant,
      fp4_gemm_fallback as fp4_gemm,
      fp8_gemm_fallback as fp8_gemm,
      hadamard_transform_fallback,
      hc_split_sinkhorn_fallback as hc_split_sinkhorn,
      mla_sparse_attn_fallback as sparse_attn,
      update_kv_cache_functional,
  )

world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32


@contextmanager
def set_dtype(dtype):
  """Temporarily override torch default dtype, restoring it on exit."""
  prev = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
  """Model hyperparameters for DeepSeek-V4-Flash inference on Cloud TPU."""

  max_batch_size: int = 4
  max_seq_len: int = 4096
  dtype: Literal["bf16", "fp8"] = "bf16"
  scale_fmt: Literal[None, "ue8m0"] = None
  expert_dtype: Literal[None, "fp4"] = None
  scale_dtype: Literal["fp32", "fp8"] = "fp32"
  vocab_size: int = 129280
  dim: int = 4096
  moe_inter_dim: int = 2048
  n_layers: int = 43
  n_hash_layers: int = 3
  n_mtp_layers: int = 0
  n_heads: int = 64
  # moe
  n_routed_experts: int = 256
  n_shared_experts: int = 1
  n_activated_experts: int = 6
  score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
  route_scale: float = 1.5
  swiglu_limit: float = 10.0
  # mla
  q_lora_rank: int = 1024
  head_dim: int = 512
  rope_head_dim: int = 64
  norm_eps: float = 1e-6
  o_groups: int = 8
  o_lora_rank: int = 1024
  window_size: int = 128
  compress_ratios: Tuple[int, ...] = (
      0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
      128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
      128, 4, 128, 4, 128, 4, 0,
  )
  # yarn
  compress_rope_theta: float = 160000.0
  original_seq_len: int = 65536
  rope_theta: float = 10000.0
  rope_factor: float = 16.0
  beta_fast: int = 32
  beta_slow: int = 1
  # indexer
  index_n_heads: int = 64
  index_head_dim: int = 128
  index_topk: int = 512
  # hyper-connections (hc)
  hc_mult: int = 4
  hc_sinkhorn_iters: int = 20
  hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
  """Embedding layer supporting single-device and tensor-parallel sharding."""

  def __init__(self, vocab_size: int, dim: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.part_vocab_size = vocab_size // world_size
    self.vocab_start_idx = rank * self.part_vocab_size
    self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
    self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if world_size > 1:
      mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
      x_local = torch.clamp(
          x - self.vocab_start_idx, 0, self.part_vocab_size - 1
      )
      y = F.embedding(x_local, self.weight)
      y = torch.where(mask.unsqueeze(-1), torch.zeros_like(y), y)
      dist.all_reduce(y)
      return y
    return F.embedding(x, self.weight)


def linear_tpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Dispatches matrix multiplication to TPU-compliant fallbacks or native linear."""
  if hasattr(weight, "dtype") and weight.dtype == torch.float8_e4m3fn:
    x_q, x_scale = act_quant(x, block_size, scale_fmt, scale_dtype)
    w_scale = getattr(weight, "scale", None)
    out = fp8_gemm(x_q, x_scale, weight, w_scale, scale_dtype)
  else:
    if x.dtype != weight.dtype:
      x = x.to(weight.dtype)
    out = F.linear(x, weight)
  if bias is not None:
    out = out + bias
  return out


class Linear(nn.Module):
  """Linear layer supporting BF16 and quantized weights for Cloud TPU execution."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      bias: bool = False,
      dtype=None,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    dtype = dtype or default_dtype
    self.weight = nn.Parameter(
        torch.empty(out_features, in_features, dtype=dtype)
    )
    if bias:
      self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
    else:
      self.register_parameter("bias", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return linear_tpu(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):

  def __init__(
      self,
      in_features: int,
      out_features: int,
      bias: bool = False,
      dtype=None,
  ):
    assert out_features % world_size == 0
    part_out = out_features // world_size
    super().__init__(in_features, part_out, bias, dtype)


class RowParallelLinear(Linear):

  def __init__(
      self,
      in_features: int,
      out_features: int,
      bias: bool = False,
      dtype=None,
  ):
    assert in_features % world_size == 0
    part_in = in_features // world_size
    super().__init__(part_in, out_features, bias, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = linear_tpu(x, self.weight, None)
    if world_size > 1:
      dist.all_reduce(y)
    if self.bias is not None:
      y = y + self.bias
    return y


class RMSNorm(nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.square().mean(-1, keepdim=True)
    normed = x_f * torch.rsqrt(var + self.eps)
    return (normed * self.weight).to(orig_dtype)


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
  """Precomputes complex exponentials for rotary embeddings with YaRN scaling."""

  def find_correction_dim(num_rotations, dim, base, max_seq_len):
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )

  def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min_val, max_val, dim):
    if min_val == max_val:
      max_val += 0.001
    linear_func = (
        torch.arange(dim, dtype=torch.float32) - min_val
    ) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)

  freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  if original_seq_len > 0:
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_seq_len
    )
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth

  t = torch.arange(seqlen)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis


def apply_rotary_emb_to_tail(
    x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False
) -> torch.Tensor:
  """Applies functional rotary embedding only to the last rd dimensions."""
  if rd == 0:
    return x
  tail = x[..., -rd:]
  head = x[..., :-rd]
  rotated_tail = apply_rotary_emb_functional(tail, freqs_cis, inverse=inverse)
  return torch.cat([head, rotated_tail], dim=-1)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
  """Applies pure PyTorch Fast Walsh-Hadamard Transform fallback."""
  return hadamard_transform_fallback(x, scale=x.size(-1) ** -0.5)


@lru_cache(1)
def get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
):
  if start_pos >= window_size - 1:
    sp = start_pos % window_size
    matrix = torch.cat(
        [torch.arange(sp + 1, window_size), torch.arange(0, sp + 1)], dim=0
    )
  elif start_pos > 0:
    matrix = F.pad(
        torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1
    )
  else:
    base = torch.arange(seqlen).unsqueeze(1)
    matrix = (base - window_size + 1).clamp(0) + torch.arange(
        min(seqlen, window_size)
    )
    matrix = torch.where(matrix > base, -1, matrix)
  return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
  """Compresses KV cache via learned gated pooling over consecutive tokens."""

  def __init__(
      self,
      args: ModelArgs,
      compress_ratio: int = 4,
      head_dim: int = 512,
      rotate: bool = False,
  ):
    super().__init__()
    self.dim = args.dim
    self.head_dim = head_dim
    self.rope_head_dim = args.rope_head_dim
    self.nope_head_dim = head_dim - args.rope_head_dim
    self.compress_ratio = compress_ratio
    self.overlap = compress_ratio == 4
    self.rotate = rotate
    coff = 1 + self.overlap

    self.ape = nn.Parameter(
        torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
    )
    self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
    self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
    self.norm = RMSNorm(self.head_dim, args.norm_eps)
    self.kv_cache: Optional[torch.Tensor] = None
    self.freqs_cis: Optional[torch.Tensor] = None

  def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
    assert self.kv_cache is not None
    bsz, seqlen, _ = x.size()
    ratio, d, rd = self.compress_ratio, self.head_dim, self.rope_head_dim
    dtype = x.dtype
    x_f = x.float()
    kv = self.wkv(x_f)
    score = self.wgate(x_f)

    if start_pos == 0:
      should_compress = seqlen >= ratio
      remainder = seqlen % ratio
      cutoff = seqlen - remainder
      if remainder > 0:
        kv = kv[:, :cutoff]
        score = score[:, :cutoff]
      if not should_compress:
        return None
      kv = kv.unflatten(1, (-1, ratio))
      score = score.unflatten(1, (-1, ratio)) + self.ape
      kv = (kv * score.softmax(dim=2)).sum(dim=2)
    else:
      should_compress = (start_pos + 1) % self.compress_ratio == 0
      if not should_compress:
        return None
      kv = kv.squeeze(1).unsqueeze(1)

    kv = self.norm(kv.to(dtype))
    if start_pos == 0:
      freqs_cis = self.freqs_cis[:cutoff:ratio]
    else:
      freqs_cis = self.freqs_cis[
          start_pos + 1 - self.compress_ratio
      ].unsqueeze(0)

    kv = apply_rotary_emb_to_tail(kv, freqs_cis, rd)
    if self.rotate:
      kv = rotate_activation(kv)
    return kv


class Indexer(nn.Module):
  """Learned scoring module to select top-k compressed KV positions."""

  def __init__(self, args: ModelArgs, compress_ratio: int = 4):
    super().__init__()
    self.dim = args.dim
    self.n_heads = args.index_n_heads
    self.n_local_heads = args.index_n_heads // world_size
    self.head_dim = args.index_head_dim
    self.rope_head_dim = args.rope_head_dim
    self.index_topk = args.index_topk
    self.q_lora_rank = args.q_lora_rank
    self.wq_b = ColumnParallelLinear(
        self.q_lora_rank, self.n_heads * self.head_dim
    )
    self.weights_proj = ColumnParallelLinear(
        self.dim, self.n_heads, dtype=torch.bfloat16
    )
    self.softmax_scale = self.head_dim ** -0.5
    self.compress_ratio = compress_ratio

    self.compressor = Compressor(
        args, compress_ratio, self.head_dim, rotate=True
    )
    self.register_buffer(
        "kv_cache",
        torch.zeros(
            args.max_batch_size,
            args.max_seq_len // compress_ratio,
            self.head_dim,
        ),
        persistent=False,
    )
    self.freqs_cis = None

  def forward(
      self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
  ) -> torch.Tensor:
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    ratio = self.compress_ratio
    rd = self.rope_head_dim
    end_pos = start_pos + seqlen

    if self.compressor.kv_cache is None:
      self.compressor.kv_cache = self.kv_cache
      self.compressor.freqs_cis = self.freqs_cis

    q = self.wq_b(qr)
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    q = apply_rotary_emb_to_tail(q, freqs_cis, rd)
    q = rotate_activation(q)

    self.compressor(x, start_pos)
    weights = self.weights_proj(x) * (
        self.softmax_scale * (self.n_heads ** -0.5)
    )
    kv_slice = self.kv_cache[:bsz, : max(1, end_pos // ratio)]
    index_score = torch.einsum("bshd,btd->bsht", q, kv_slice)
    index_score = (F.relu(index_score) * weights.unsqueeze(-1)).sum(dim=2)

    if world_size > 1:
      dist.all_reduce(index_score)

    k_val = min(self.index_topk, index_score.size(-1))
    topk_idxs = index_score.topk(k_val, dim=-1)[1] + offset
    return topk_idxs


class Attention(nn.Module):
  """Multi-head Latent Attention (MLA) with sliding window and functional KV caching."""

  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.layer_id = layer_id
    self.dim = args.dim
    self.n_heads = args.n_heads
    self.n_local_heads = args.n_heads // world_size
    self.q_lora_rank = args.q_lora_rank
    self.o_lora_rank = args.o_lora_rank
    self.head_dim = args.head_dim
    self.rope_head_dim = args.rope_head_dim
    self.nope_head_dim = args.head_dim - args.rope_head_dim
    self.n_groups = args.o_groups
    self.n_local_groups = self.n_groups // world_size
    self.window_size = args.window_size
    self.compress_ratio = args.compress_ratios[layer_id]
    self.eps = args.norm_eps

    self.attn_sink = nn.Parameter(
        torch.empty(self.n_local_heads, dtype=torch.float32)
    )
    self.wq_a = Linear(self.dim, self.q_lora_rank)
    self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
    self.wq_b = ColumnParallelLinear(
        self.q_lora_rank, self.n_heads * self.head_dim
    )
    self.wkv = Linear(self.dim, self.head_dim)
    self.kv_norm = RMSNorm(self.head_dim, self.eps)
    self.wo_a = ColumnParallelLinear(
        self.n_heads * self.head_dim // self.n_groups,
        self.n_groups * args.o_lora_rank,
        dtype=torch.bfloat16,
    )
    self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
    self.softmax_scale = self.head_dim ** -0.5

    if self.compress_ratio:
      self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
      if self.compress_ratio == 4:
        self.indexer = Indexer(args, self.compress_ratio)
      else:
        self.indexer = None
    else:
      self.compressor = None
      self.indexer = None

    kv_cache_size = args.window_size + (
        args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
    )
    self.register_buffer(
        "kv_cache",
        torch.zeros(
            args.max_batch_size,
            kv_cache_size,
            self.head_dim,
            dtype=torch.bfloat16,
        ),
        persistent=False,
    )

    rope_theta = (
        args.compress_rope_theta if self.compress_ratio else args.rope_theta
    )
    orig_seq = args.original_seq_len if self.compress_ratio else 0
    freqs_cis = precompute_freqs_cis(
        self.rope_head_dim,
        args.max_seq_len,
        orig_seq,
        rope_theta,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )
    self.register_buffer("freqs_cis", freqs_cis, persistent=False)

  def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    win = self.window_size
    rd = self.rope_head_dim

    qr = self.q_norm(self.wq_a(x))
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    q = apply_rotary_emb_to_tail(q, freqs_cis, rd)

    kv = self.kv_norm(self.wkv(x))
    kv = apply_rotary_emb_to_tail(kv, freqs_cis, rd)

    topk_idxs = (
        get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device).int()
    )

    if start_pos == 0:
      if seqlen <= win:
        self.kv_cache = update_kv_cache_functional(self.kv_cache, kv, 0)
      else:
        self.kv_cache = update_kv_cache_functional(
            self.kv_cache, kv[:, -win:], 0
        )
      o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    else:
      self.kv_cache = update_kv_cache_functional(
          self.kv_cache, kv, start_pos % win
      )
      o = sparse_attn(
          q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
      )

    o = apply_rotary_emb_to_tail(o, freqs_cis, rd, inverse=True)

    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    out = self.wo_b(o.flatten(2))
    return out


class Gate(nn.Module):
  """MoE Gating without device-to-host CPU synchronization."""

  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.dim = args.dim
    self.topk = args.n_activated_experts
    self.score_func = args.score_func
    self.route_scale = args.route_scale
    self.hash = layer_id < args.n_hash_layers
    self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
    if self.hash:
      self.tid2eid = nn.Parameter(
          torch.empty(
              args.vocab_size, args.n_activated_experts, dtype=torch.int32
          ),
          requires_grad=False,
      )
      self.bias = None
    else:
      self.bias = nn.Parameter(
          torch.empty(args.n_routed_experts, dtype=torch.float32)
      )

  def forward(
      self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = linear_tpu(x.float(), self.weight.float())
    if self.score_func == "softmax":
      scores = scores.softmax(dim=-1)
    elif self.score_func == "sigmoid":
      scores = scores.sigmoid()
    else:
      scores = F.softplus(scores).sqrt()
    original_scores = scores

    if self.bias is not None:
      scores = scores + self.bias
    if self.hash and input_ids is not None:
      indices = self.tid2eid[input_ids]
    else:
      indices = scores.topk(self.topk, dim=-1)[1]

    weights = original_scores.gather(1, indices)
    if self.score_func != "softmax":
      weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    weights = weights * self.route_scale
    return weights, indices


class Expert(nn.Module):
  """SwiGLU FFN Expert evaluated in float32 for numerical stability on TPUs."""

  def __init__(
      self,
      dim: int,
      inter_dim: int,
      dtype=None,
      swiglu_limit: float = 0.0,
  ):
    super().__init__()
    self.w1 = Linear(dim, inter_dim, dtype=dtype)
    self.w2 = Linear(inter_dim, dim, dtype=dtype)
    self.w3 = Linear(dim, inter_dim, dtype=dtype)
    self.swiglu_limit = swiglu_limit

  def forward(
      self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    orig_dtype = x.dtype
    gate = self.w1(x).float()
    up = self.w3(x).float()
    if self.swiglu_limit > 0:
      up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
      gate = torch.clamp(gate, max=self.swiglu_limit)
    h = F.silu(gate) * up
    if weights is not None:
      h = h * weights
    return self.w2(h.to(orig_dtype))


class MoE(nn.Module):
  """Mixture-of-Experts with TPU-native index_add reduction and zero CPU graph breaks."""

  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.layer_id = layer_id
    self.dim = args.dim
    self.n_routed_experts = args.n_routed_experts
    self.n_local_experts = args.n_routed_experts // world_size
    self.n_activated_experts = args.n_activated_experts
    self.experts_start_idx = rank * self.n_local_experts
    self.experts_end_idx = self.experts_start_idx + self.n_local_experts
    self.gate = Gate(layer_id, args)

    self.experts = nn.ModuleList([
        (
            Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
            if self.experts_start_idx <= i < self.experts_end_idx
            else None
        )
        for i in range(self.n_routed_experts)
    ])
    self.shared_experts = Expert(
        args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
    )

  def forward(
      self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    shape = x.size()
    x_flat = x.view(-1, self.dim)
    input_ids_flat = input_ids.flatten() if input_ids is not None else None
    weights, indices = self.gate(x_flat, input_ids_flat)

    y = torch.zeros_like(x_flat, dtype=torch.float32)

    # Device-side routing using functional index_add
    for i in range(self.experts_start_idx, self.experts_end_idx):
      mask = indices == i
      if not mask.any():
        continue
      idx, top = torch.where(mask)
      expert = self.experts[i]
      expert_out = expert(x_flat[idx], weights[idx, top, None])
      y = y.index_add(0, idx, expert_out.float())

    if world_size > 1:
      dist.all_reduce(y)

    y = y + self.shared_experts(x_flat).float()
    return y.to(x.dtype).view(shape)


class Block(nn.Module):
  """Transformer block with Hyper-Connections (HC) and iterative Sinkhorn mixing."""

  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.layer_id = layer_id
    self.norm_eps = args.norm_eps
    self.attn = Attention(layer_id, args)
    self.ffn = MoE(layer_id, args)
    self.attn_norm = RMSNorm(args.dim, self.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
    self.hc_mult = hc_mult = args.hc_mult
    self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
    self.hc_eps = args.hc_eps

    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * args.dim
    with set_dtype(torch.float32):
      self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
      self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
      self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
      self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
      self.hc_attn_scale = nn.Parameter(torch.empty(3))
      self.hc_ffn_scale = nn.Parameter(torch.empty(3))

  def hc_pre(
      self,
      x: torch.Tensor,
      hc_fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
  ):
    shape, orig_dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        self.hc_mult,
        self.hc_sinkhorn_iters,
        self.hc_eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(orig_dtype), post, comb

  def hc_post(
      self,
      x: torch.Tensor,
      residual: torch.Tensor,
      post: torch.Tensor,
      comb: torch.Tensor,
  ):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.to(x.dtype)

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      input_ids: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    residual = x
    x_pre, post, comb = self.hc_pre(
        x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
    )
    x_attn = self.attn(self.attn_norm(x_pre), start_pos)
    x = self.hc_post(x_attn, residual, post, comb)

    residual = x
    x_pre, post, comb = self.hc_pre(
        x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
    )
    x_ffn = self.ffn(self.ffn_norm(x_pre), input_ids)
    x = self.hc_post(x_ffn, residual, post, comb)
    return x


class ParallelHead(nn.Module):
  """LM Head projecting final hidden states to vocabulary logits."""

  def __init__(
      self,
      vocab_size: int,
      dim: int,
      norm_eps: float = 1e-6,
      hc_eps: float = 1e-6,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.norm_eps = norm_eps
    self.hc_eps = hc_eps
    self.part_vocab_size = vocab_size // world_size
    self.weight = nn.Parameter(
        torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32)
    )

  def get_logits(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x[:, -1].float(), self.weight)

  def hc_head(
      self,
      x: torch.Tensor,
      hc_fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
  ):
    shape, orig_dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(orig_dtype)

  def forward(
      self,
      x: torch.Tensor,
      hc_fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
      norm: RMSNorm,
  ) -> torch.Tensor:
    x = self.hc_head(x, hc_fn, hc_scale, hc_base)
    logits = self.get_logits(norm(x))
    if world_size > 1:
      gathered = [torch.empty_like(logits) for _ in range(world_size)]
      dist.all_gather(gathered, logits)
      logits = torch.cat(gathered, dim=-1)
    return logits


class Transformer(nn.Module):
  """DeepSeek-V4-Flash Transformer Architecture optimized for Cloud TPU execution."""

  def __init__(self, args: ModelArgs):
    global world_size, rank, default_dtype
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    default_dtype = torch.bfloat16
    super().__init__()

    self.max_seq_len = args.max_seq_len
    self.norm_eps = args.norm_eps
    self.hc_eps = args.hc_eps
    self.embed = ParallelEmbedding(args.vocab_size, args.dim)
    self.layers = nn.ModuleList(
        [Block(layer_id, args) for layer_id in range(args.n_layers)]
    )
    self.norm = RMSNorm(args.dim, self.norm_eps)
    self.head = ParallelHead(
        args.vocab_size, args.dim, self.norm_eps, self.hc_eps
    )

    self.hc_mult = hc_mult = args.hc_mult
    hc_dim = hc_mult * args.dim
    with set_dtype(torch.float32):
      self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
      self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
      self.hc_head_scale = nn.Parameter(torch.empty(1))

  @torch.inference_mode()
  def forward(
      self, input_ids: torch.Tensor, start_pos: int = 0
  ) -> torch.Tensor:
    h = self.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    for layer in self.layers:
      h = layer(h, start_pos, input_ids)
    logits = self.head(
        h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
    )
    return logits
