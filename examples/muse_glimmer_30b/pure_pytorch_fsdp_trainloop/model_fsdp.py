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

from collections.abc import Sequence
import dataclasses

import torch
import torch.distributed._functional_collectives as func_col
import torch.nn as nn
import torch.nn.functional as F

from . import rope

CHUNKWISE_FUSED_LOSS_CHUNK_SIZE = 4096  # Chunk size for fused loss


@dataclasses.dataclass
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  rope_freq_base: float = 500000.0
  rmsnorm_eps: float = 1e-5
  post_norm_eps: float = 1e-8
  qk_scale_factor: float = 3.87
  final_logit_softcapping: float = 20.0
  output_multiplier: float = 0.19611613513818404

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120  # Full sequence length


class MuseGlimmer30B(nn.Module):
  """Muse Glimmer 30B with built-in 'manual' FSDP."""

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
    self.embed_norm = nn.RMSNorm(
        (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
    )
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )
    self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)

    # NOTE: There is no LM head here -- instead it (together with tanh softcap)
    # is fused with the cross-entropy loss function.

    rope_sin, rope_cos = rope.precompute_rope_sin_cos(
        seq_len=config.seq_len,
        dim_head=config.dhead,
        rope_theta=config.rope_freq_base,
    )
    self.rope_sin = nn.Buffer(rope_sin, persistent=False)
    self.rope_cos = nn.Buffer(rope_cos, persistent=False)

  def forward(self, tok: torch.Tensor) -> torch.Tensor:
    h = self.embed_norm(self.embed_tokens(tok))
    for layer in self.layers:
      h = layer(h, (self.rope_sin, self.rope_cos))
    h = self.norm(h)
    return h

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)  # pyrefly: ignore[bad-assignment]
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)  # pyrefly: ignore[bad-argument-type]
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)  # pyrefly: ignore[bad-assignment]


class DecoderBlock(nn.Module):

  def __init__(self, config: Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx

    self.self_attn = Attention(config, is_swa=((layer_idx % 4) != 3))
    self.mlp = FeedForwardSwiGLU(config)

    dm = config.dmodel
    f32 = torch.float32
    eps_pre, eps_post = config.rmsnorm_eps, config.post_norm_eps
    # NOTE: These RMSNorms should use centered rmsnorm. This would be relevant
    # for weight decay and when loading weights from HF checkpoint.
    self.input_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_attention_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)
    self.pre_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    # We use activation checkpointing (rematerialization) per layer, matching
    # the scope of torch.compile.
    #
    # NOTE: Compile can also be applied on larger scope (e.g. whole model),
    # but what's important is that activation checkpointing is *inside* the
    # the scope of torch.compile, not the other way.
    # Here specifically, this means that torch.compile can be applied for the
    # whole DecodeBlock module (or higher), but cannot be applied only for the
    # individual Attention and FeedForwardSwiGLU layers.
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h


class Attention(torch.nn.Module):
  """Grouped-query Attention."""

  def __init__(self, config: Config, is_swa: bool):
    super().__init__()
    self.config = config
    self.is_swa = is_swa

    dm, dh = config.dmodel, config.dhead
    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads

    world_size = torch.distributed.get_world_size()
    self.pg: torch.distributed.ProcessGroup = torch.distributed.group.WORLD  # pyrefly: ignore[bad-assignment]
    assert self.pg is not None

    assert dm % world_size == 0
    assert (num_q_heads * dh) % world_size == 0

    # QKV projections are fully sharded:
    self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
    self.k_proj = nn.Linear(dm, num_kv_heads * dh // world_size, bias=False)
    self.v_proj = nn.Linear(dm, num_kv_heads * dh // world_size, bias=False)

    # Gate and output projections also fully sharded:
    self.gate_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
    self.o_proj = nn.Linear(num_q_heads * dh, dm // world_size, bias=False)

    # Glimmer has QK RMSNorm without learned scale parameters
    self.qk_norm = torch.nn.RMSNorm(
        normalized_shape=(dh,),
        eps=config.rmsnorm_eps,
        dtype=torch.float32,
        elementwise_affine=False,
    )

  def forward(self, h, rope_sin_cos):
    local_bsz, seq_len, dm = h.shape

    dh = self.config.dhead
    num_q_heads = self.config.num_q_heads
    num_kv_heads = self.config.num_kv_heads

    # Temporarily all-gather all FSDP-sharded projections. PyTorch tracks
    # autograd here, so reduce-scatter will run during the backward pass.
    q_proj_w = func_col.all_gather_single(
        self.q_proj.weight, gather_dim=0, group=self.pg
    )
    k_proj_w = func_col.all_gather_single(
        self.k_proj.weight, gather_dim=0, group=self.pg
    )
    v_proj_w = func_col.all_gather_single(
        self.v_proj.weight, gather_dim=0, group=self.pg
    )

    gate_proj_w = func_col.all_gather_single(
        self.gate_proj.weight, gather_dim=0, group=self.pg
    )
    o_proj_w = func_col.all_gather_single(
        self.o_proj.weight, gather_dim=0, group=self.pg
    )

    q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
    k = F.linear(h, k_proj_w).view(local_bsz, seq_len, num_kv_heads, dh)
    v = F.linear(h, v_proj_w).view(local_bsz, seq_len, num_kv_heads, dh)

    qk_scale = self.config.qk_scale_factor
    q = self.qk_norm(q.transpose(1, 2).contiguous()) * qk_scale
    k = self.qk_norm(k.transpose(1, 2).contiguous())
    v = v.transpose(1, 2).contiguous()

    if self.is_swa:
      # In Muse Glimmer 30B, RoPE is only applied in SWA layers
      rs, rc = rope_sin_cos
      q = rope.apply_rope(q, rs, rc)
      k = rope.apply_rope(k, rs, rc)
      attn_mask = compute_swa_mask(
          seq_len, self.config.swa_size, device=q.device
      )
      attn_out = F.scaled_dot_product_attention(
          q, k, v, attn_mask=attn_mask, enable_gqa=True
      )
    else:
      # Full attention layers use NoPE.
      attn_out = F.scaled_dot_product_attention(
          q, k, v, is_causal=True, enable_gqa=True
      )
    attn_out = attn_out.transpose(1, 2)
    attn_out = attn_out.reshape(local_bsz, seq_len, num_q_heads * dh)

    gate = F.sigmoid(F.linear(h, gate_proj_w))
    out = F.linear(attn_out * gate, o_proj_w)

    return out


class FeedForwardSwiGLU(nn.Module):
  """Simple FFN with SwiGLU activation."""

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    dm, df = config.dmodel, config.dff

    world_size = torch.distributed.get_world_size()
    self.pg: torch.distributed.ProcessGroup = torch.distributed.group.WORLD  # pyrefly: ignore[bad-assignment]
    assert self.pg is not None

    assert dm % world_size == 0
    assert df % world_size == 0

    # Up/gate/down projections are fully sharded (FSDP).
    self.up_proj = nn.Linear(dm, df // world_size, bias=False)
    self.gate_proj = nn.Linear(dm, df // world_size, bias=False)
    self.down_proj = nn.Linear(df, dm // world_size, bias=False)

  def forward(self, h):
    # Temporarily all-gather all FSDP-sharded projections. PyTorch tracks
    # autograd here, so reduce-scatter will run during the backward pass.
    up = func_col.all_gather_single(
        self.up_proj.weight, gather_dim=0, group=self.pg
    )
    gate = func_col.all_gather_single(
        self.gate_proj.weight, gather_dim=0, group=self.pg
    )
    down = func_col.all_gather_single(
        self.down_proj.weight, gather_dim=0, group=self.pg
    )

    h = F.linear(h, up) * F.silu(F.linear(h, gate))
    h = F.linear(h, down)

    return h


class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  """Fused LMHead + CrossEntropy loss computation.

  A naive implementation would require materializing [batch*seq, vocab_size]
  intermediate tensors, which would consume way too much HBM.

  One solution, demonstrated here, is to perform the LM head (matmul by vocab
  weight matrix) and apply cross-entropy loss on "chunks" of tokens, and then
  combine the results.

  This allows us to limit intermediate tensor shapes to [chunk_size, vocab].
  """

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self.lm_head = nn.Linear(config.dmodel, config.vocab_size, bias=False)

    # Chunkwise LM head (with tanh softcap and scaling) fused with CE loss
    # forward and backward. This is necessary to avoid materializing giant
    # intermediate O(batch * seq_len * vocab_size) tensors.
    self.fused_loss_fn_fwd_bwd = (
        MuseGlimmerChunkwiseFusedLMHeadLoss._chunk_fused_lmhead_loss_fwd_bwd
    )

  def forward(self, h: torch.Tensor, tgt: torch.LongTensor) -> torch.Tensor:
    bsz, seq_len, dm = h.shape

    # We directly return h.grad, instead of having autograd propagate it.
    assert h.requires_grad == False

    h = h.view(-1, dm)
    tgt = tgt.view(-1)

    chunk_size = CHUNKWISE_FUSED_LOSS_CHUNK_SIZE
    assert h.shape[0] % chunk_size == 0

    alpha = self.config.output_multiplier
    cap = self.config.final_logit_softcapping

    h_chunk = torch.zeros(chunk_size, dm, dtype=h.dtype, device=h.device)
    h_chunk.requires_grad_(True)
    tgt_chunk = torch.zeros(chunk_size, dtype=tgt.dtype, device=tgt.device)

    # We want grads based on global avg of loss per token.
    world_size = torch.distributed.get_world_size()
    loss_scaler = 1.0 / (world_size * bsz * seq_len)

    h_chunk_grads = []
    for i in range(0, h.shape[0], chunk_size):
      h_chunk.data.copy_(h[i : (i + chunk_size)])
      tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

      self.fused_loss_fn_fwd_bwd(
          h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
      )

      h_chunk_grads.append(h_chunk.grad)
      h_chunk.grad = None

    return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)

  def apply_torch_compile(self, backend):
    self.fused_loss_fn_fwd_bwd = torch.compile(  # pyrefly: ignore[bad-assignment]
        self.fused_loss_fn_fwd_bwd, dynamic=False, backend=backend
    )

  @staticmethod
  def _chunk_fused_lmhead_loss_fwd_bwd(
      h, tgt, lm_head_w, alpha, cap, loss_scaler
  ):
    logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
    loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
    loss *= loss_scaler
    # Note: in recent versions of PyTorch, .backward is no longer a graph break!
    loss.backward()
    return


def compute_swa_mask(seq_len: int, swa_size: int, device) -> torch.Tensor:
  mask = torch.ones(1, 1, seq_len, seq_len, device=device)
  mask = mask.tril()  # Remove future tokens
  mask = mask.triu(-swa_size + 1)  # Remove tokens outside of window
  mask = torch.where(mask == 1, 0, float("-inf"))
  return mask


def get_non_fsdp_params(m: nn.Module) -> Sequence[nn.Parameter]:
  fsdp_param_suffixes = [
      ".self_attn.q_proj.weight",
      ".self_attn.k_proj.weight",
      ".self_attn.v_proj.weight",
      ".self_attn.gate_proj.weight",
      ".self_attn.o_proj.weight",
      ".mlp.up_proj.weight",
      ".mlp.gate_proj.weight",
      ".mlp.down_proj.weight",
  ]
  non_fsdp_params = []
  for n, p in m.named_parameters():
    if not p.requires_grad:
      continue
    is_fsdp = False
    for suffix in fsdp_param_suffixes:
      if n.endswith(suffix):
        is_fsdp = True
        break
    if not is_fsdp:
      non_fsdp_params.append(p)

  return non_fsdp_params


@torch.no_grad()
def init_nn_module(m: torch.nn.Module):
  for p in m.parameters():
    p.data.copy_(torch.randn_like(p.data) * 0.05)


def all_reduce_grads(params):
  for p in params:
    torch.distributed.all_reduce(p.grad)
