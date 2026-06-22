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

"""Gemma4 model implementation for TorchTPU."""

import math

import torch
from torch import nn
import torch.nn.functional as F
from examples.gemma4 import splash_attention


class Gemma4Config:
  """Config class for Gemma4."""

  def __init__(
      self,
      vocab_size=262144,
      hidden_size=5376,
      intermediate_size=21504,
      num_hidden_layers=60,
      num_attention_heads=32,
      num_key_value_heads=16,
      num_global_kv_heads=None,
      head_dim=256,
      max_position_embeddings=262144,
      initializer_range=0.02,
      rms_norm_eps=1e-6,
      pad_token_id=0,
      bos_token_id=2,
      eos_token_id=1,
      tie_word_embeddings=True,
      rope_theta=1000000.0,
      rope_local_base_freq=10000.0,
      global_rope_proportion=0.25,
      attention_bias=False,
      sliding_window=1024,
      layer_types=None,
      enable_moe=False,
      num_experts=128,
      expert_dim=704,
      top_k_experts=8,
      moe_dense_hidden_dim=2112,
      use_bidirectional_attention='vision',
      use_post_ffw_norm=True,
      **kwargs,
  ):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.num_global_kv_heads = num_global_kv_heads
    self.head_dim = head_dim
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.pad_token_id = pad_token_id
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.tie_word_embeddings = tie_word_embeddings
    self.rope_theta = rope_theta
    self.rope_local_base_freq = rope_local_base_freq
    self.global_rope_proportion = global_rope_proportion
    self.attention_bias = attention_bias
    self.sliding_window = sliding_window
    self.enable_moe = enable_moe
    self.num_experts = num_experts
    self.expert_dim = expert_dim
    self.top_k_experts = top_k_experts
    self.moe_dense_hidden_dim = moe_dense_hidden_dim
    self.use_bidirectional_attention = use_bidirectional_attention
    self.use_post_ffw_norm = use_post_ffw_norm

    if layer_types is None:
      # Default pattern: 5 sliding, 1 global
      self.layer_types = []
      for i in range(num_hidden_layers):
        if (i + 1) % 6 == 0:
          self.layer_types.append('full_attention')
        else:
          self.layer_types.append('sliding_attention')
    else:
      self.layer_types = layer_types

    self.kwargs = kwargs


class GELUTanh(nn.Module):
  """Gemma activation function (GELU with tanh approximation)."""

  def __init__(self):
    super().__init__()
    self._const1 = math.sqrt(2.0 / math.pi)
    self._const2 = 0.044715

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5) * (
        1.0 + torch.tanh(self._const1 * (x + self._const2 * torch.pow(x, 3.0)))
    )


class Gemma4RMSNorm(nn.Module):
  """Gemma specific RMSNorm."""

  def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
    super().__init__()
    self.eps = eps
    self.with_scale = with_scale
    if with_scale:
      self.weight = nn.Parameter(torch.ones(dim))
    else:
      self.register_parameter('weight', None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.float()
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    if self.with_scale:
      output = output * self.weight.float()
    return output.to(input_dtype)


class Gemma4RotaryEmbedding(nn.Module):
  """RoPE implementation for Gemma4."""

  def __init__(self, config: Gemma4Config, is_sliding: bool):
    super().__init__()
    self.config = config
    self.is_sliding = is_sliding

    if is_sliding:
      self.base = config.rope_local_base_freq
      self.rope_proportion = 1.0
    else:
      self.base = config.rope_theta
      self.rope_proportion = config.global_rope_proportion

    self.head_dim = config.head_dim
    self.rope_angles = int(self.rope_proportion * self.head_dim // 2)

    # Compute inv_freq for rotated part
    inv_freq = 1.0 / (
        self.base
        ** (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.int64).float()
            / self.head_dim
        )
    )

    # Pad with zeros for non-rotated part
    nope_angles = self.head_dim // 2 - self.rope_angles
    if nope_angles > 0:
      inv_freq = torch.cat(
          (inv_freq, torch.zeros(nope_angles, dtype=torch.float32)), dim=0
      )

    self.register_buffer('inv_freq', inv_freq, persistent=False)

  def forward(self, x, position_ids):
    # position_ids: [B, S]
    inv_freq_expanded = (
        self.inv_freq[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    with torch.autocast(device_type=x.device.type, enabled=False):
      freqs = (
          inv_freq_expanded.float() @ position_ids_expanded.float()
      ).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


class Gemma4MLP(nn.Module):
  """Gemma4 MLP."""

  def __init__(self, config: Gemma4Config, intermediate_size=None):
    super().__init__()
    self.config = config
    if intermediate_size is None:
      intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(
        config.hidden_size, intermediate_size, bias=False
    )
    self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(
        intermediate_size, config.hidden_size, bias=False
    )
    self.act_fn = GELUTanh()

  def forward(self, x):
    gate = self.gate_proj(x)
    activated = self.act_fn(gate)
    up = self.up_proj(x)
    gated_up = activated * up
    down = self.down_proj(gated_up)
    return down


class Gemma4Attention(nn.Module):
  """Gemma4 Attention."""

  def __init__(self, config: Gemma4Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.is_sliding = config.layer_types[layer_idx] == 'sliding_attention'

    self.head_dim = config.head_dim
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.num_global_kv_heads = config.num_global_kv_heads

    # In Gemma4, we might have different number of KV heads for global layers.
    # Let's check if that's the case.
    # From JAX config: num_kv_heads=16, num_global_kv_heads=4 for 31B.
    # So yes, they can be different.

    if not self.is_sliding and self.num_global_kv_heads is not None:
      self.current_kv_heads = self.num_global_kv_heads
    else:
      self.current_kv_heads = self.num_kv_heads

    self.q_proj = nn.Linear(
        config.hidden_size,
        self.num_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        self.current_kv_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        self.current_kv_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        self.num_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    self.q_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
    self.v_norm = Gemma4RMSNorm(
        dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False
    )

    self.rotary_emb = Gemma4RotaryEmbedding(config, is_sliding=self.is_sliding)

  def forward(
      self,
      hidden_states,
      position_ids,
      attention_mask=None,
      skip_sliding_mask=False,
  ):
    bsz, seq_len, _ = hidden_states.shape

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, seq_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, seq_len, self.current_kv_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, seq_len, self.current_kv_heads, self.head_dim)
        .transpose(1, 2)
    )
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)
    value_states = self.v_norm(value_states)

    cos, sin = self.rotary_emb(query_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    use_splash = (
        query_states.device.type in ('xla', 'tpu')
        and self.is_sliding
        and not skip_sliding_mask
        and attention_mask is None
    )

    if use_splash:
      enable_gqa = self.num_heads != self.current_kv_heads
      attn_output = splash_attention.splash_sdpa(
          query_states,
          key_states,
          value_states,
          scale=1.0,
          is_causal=True,
          local_window_size=self.config.sliding_window - 1,
          enable_gqa=enable_gqa,
      )
      attn_output = (
          attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
      )
      attn_output = self.o_proj(attn_output)
      return attn_output, None
    else:
      num_key_value_groups = self.num_heads // self.current_kv_heads

      if num_key_value_groups > 1:
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(
            num_key_value_groups, dim=1
        )

      attn_mask = attention_mask
      is_causal = attention_mask is None

      if self.is_sliding and not skip_sliding_mask:
        indices = torch.arange(seq_len, device=query_states.device)
        q_idx = indices.unsqueeze(1)
        kv_idx = indices.unsqueeze(0)
        allowed = (kv_idx <= q_idx) & (
            kv_idx > q_idx - self.config.sliding_window
        )
        swa_mask = torch.where(allowed, 0.0, float('-inf'))
        swa_mask = swa_mask.view(1, 1, seq_len, seq_len)
        if attn_mask is not None:
          attn_mask = attn_mask + swa_mask
        else:
          attn_mask = swa_mask
        is_causal = False

      attn_output = F.scaled_dot_product_attention(
          query_states,
          key_states,
          value_states,
          attn_mask=attn_mask,
          is_causal=is_causal,
          scale=1.0,
      )

      attn_output = (
          attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
      )

      attn_output = self.o_proj(attn_output)

      return attn_output, None


class Gemma4MoERagged(nn.Module):
  """Mixture of Experts (MoE) module with top-k routing using ragged_dot."""

  def __init__(self, config: Gemma4Config):
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = config.top_k_experts
    self.hidden_size = config.hidden_size
    self.expert_dim = config.expert_dim

    self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    self.up = nn.Parameter(
        torch.randn(config.num_experts, config.hidden_size, config.expert_dim)
        / math.sqrt(config.hidden_size)
    )
    self.gate = nn.Parameter(
        torch.randn(config.num_experts, config.hidden_size, config.expert_dim)
        / math.sqrt(config.hidden_size)
    )
    self.down = nn.Parameter(
        torch.randn(config.num_experts, config.expert_dim, config.hidden_size)
        / math.sqrt(config.expert_dim)
    )

    self.act_fn = GELUTanh()
    if hasattr(torch.ops, 'torch_tpu') and hasattr(
        torch.ops.torch_tpu, 'ragged_dot'
    ):
      self.ragged_dot_impl = torch.ops.torch_tpu.ragged_dot
    else:
      self.ragged_dot_impl = None
    self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
    self.router_norm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, with_scale=False
    )
    self.router_scale = nn.Parameter(torch.ones(config.hidden_size))
    self.register_buffer(
        'root_size',
        torch.rsqrt(torch.tensor(config.hidden_size, dtype=torch.float32)),
    )

  def forward(self, x):
    batch_size, sequence_length, hidden_size = x.shape
    batch_fused = batch_size * sequence_length
    h = x.view(batch_fused, hidden_size)

    # Router: norm, scale, then logits
    if self.ragged_dot_impl is None:
      raise RuntimeError(
          'ragged_dot is not available. Ensure you are running on TPU or have'
          ' patched it in tests.'
      )
    router_input = self.router_norm(h)
    router_input = (
        router_input * self.root_size * self.router_scale.to(router_input.dtype)
    )

    router_logits = self.router(router_input)
    router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    selected_weights, selected_indices = torch.topk(
        router_weights, self.top_k, dim=-1
    )

    selected_weights = selected_weights / selected_weights.sum(
        dim=-1, keepdim=True
    )
    selected_weights = selected_weights.to(dtype=h.dtype)

    selected_indices = selected_indices.flatten()
    sortidx = torch.argsort(selected_indices)
    reverse_sortidx = torch.argsort(sortidx)

    group_sizes = torch.zeros(
        self.num_experts, dtype=torch.int32, device=h.device
    )
    group_sizes.scatter_add_(
        dim=0,
        index=selected_indices,
        src=torch.ones(
            batch_fused * self.top_k, dtype=torch.int32, device=h.device
        ),
    )

    h = h.view(batch_fused, 1, hidden_size).broadcast_to(
        batch_fused, self.top_k, hidden_size
    )
    h = h.reshape(-1, hidden_size)
    h = h[sortidx, :]

    h_up = self.ragged_dot_impl(h, self.up, group_sizes)
    h_gate = self.ragged_dot_impl(h, self.gate, group_sizes)
    h = h_up * self.act_fn(h_gate)
    h = self.ragged_dot_impl(h, self.down, group_sizes)

    # Avoid CPU fallback by using sortidx to get expert indices
    expert_indices = selected_indices[sortidx]
    h = h * self.per_expert_scale[expert_indices, None]

    h = h[reverse_sortidx, :].view(batch_fused, self.top_k, hidden_size)
    h = (h * selected_weights.view(batch_fused, self.top_k, 1)).sum(dim=1)

    return h.view(batch_size, sequence_length, hidden_size)


class Gemma4DecoderLayer(nn.Module):
  """Gemma4 decoder layer."""

  def __init__(self, config: Gemma4Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.self_attn = Gemma4Attention(config, layer_idx)

    if config.enable_moe:
      self.moe = Gemma4MoERagged(config)
      self.mlp2 = Gemma4MLP(
          config, intermediate_size=config.moe_dense_hidden_dim
      )

      self.pre_ffw2_norm = Gemma4RMSNorm(
          config.hidden_size, eps=config.rms_norm_eps
      )
      self.post_ffw2_norm = Gemma4RMSNorm(
          config.hidden_size, eps=config.rms_norm_eps
      )
      self.post_ffw1_norm = Gemma4RMSNorm(
          config.hidden_size, eps=config.rms_norm_eps
      )
    else:
      self.mlp = Gemma4MLP(config)

    self.input_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.pre_feedforward_layernorm = Gemma4RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )

    if getattr(config, 'use_post_ffw_norm', True):
      self.post_feedforward_layernorm = Gemma4RMSNorm(
          config.hidden_size, eps=config.rms_norm_eps
      )
    else:
      self.post_feedforward_layernorm = nn.Identity()
    self.register_buffer('layer_scalar', torch.ones(1))

  def forward(
      self,
      hidden_states,
      position_ids,
      attention_mask=None,
      skip_sliding_mask=False,
  ):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, _ = self.self_attn(
        hidden_states,
        position_ids,
        attention_mask,
        skip_sliding_mask=skip_sliding_mask,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states

    if self.config.enable_moe:
      # Dense shared branch
      dense_in = self.pre_ffw2_norm(hidden_states)
      dense_out = self.mlp2(dense_in)
      dense_out = self.post_ffw2_norm(dense_out)

      # MoE branch
      moe_in = self.pre_feedforward_layernorm(hidden_states)
      moe_out = self.moe(moe_in)
      moe_out = self.post_ffw1_norm(moe_out)

      # Combine
      hidden_states = dense_out + moe_out
      hidden_states = self.post_feedforward_layernorm(hidden_states)
    else:
      hidden_states = self.pre_feedforward_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = self.post_feedforward_layernorm(hidden_states)

    hidden_states = residual + hidden_states
    hidden_states = hidden_states * self.layer_scalar

    return hidden_states


class Gemma4MultiModalProjector(nn.Module):
  """Projects multimodal features to the text embedding space."""

  def __init__(self, input_dim: int, output_dim: int):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim, bias=False)
    self.eps = 1e-6

  def forward(self, x):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    x = x.to(self.linear.weight.dtype)
    return self.linear(x)


class Gemma4Model(nn.Module):
  """Gemma4 Model."""

  def __init__(self, config: Gemma4Config):
    super().__init__()
    self.config = config
    self.embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
    )
    self.layers = nn.ModuleList(
        [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
    )
    self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Add projectors for multimodal inputs if configured
    self.vision_proj = None
    if hasattr(config, 'vision_proj_dim') and config.vision_proj_dim:
      self.vision_proj = Gemma4MultiModalProjector(
          config.vision_proj_dim, config.hidden_size
      )

    self.audio_proj = None
    if hasattr(config, 'audio_proj_dim') and config.audio_proj_dim:
      self.audio_proj = Gemma4MultiModalProjector(
          config.audio_proj_dim, config.hidden_size
      )

  def forward(
      self,
      input_ids,
      position_ids=None,
      attention_mask=None,
      pixel_values=None,
      audio_features=None,
      mm_token_type_ids=None,
      skip_sliding_mask=None,
  ):
    _, seq_len = input_ids.shape

    if position_ids is None:
      position_ids = torch.arange(
          seq_len, dtype=torch.long, device=input_ids.device
      ).unsqueeze(0)

    hidden_states = self.embed_tokens(input_ids)

    # Gemma scales embeddings by sqrt(hidden_size)
    hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

    # TODO(peterpc): Implement multimodal embedding merging here if needed.
    # This typically involves projecting pixel_values/audio_features and
    # interleaving them with hidden_states based on mm_token_type_ids.
    # For now, we assume input_ids already contains the correct sequence
    # and we are just processing the embeddings.

    for layer in self.layers:
      layer_skip_sliding_mask = skip_sliding_mask
      if layer_skip_sliding_mask is None:
        layer_skip_sliding_mask = (
            self.config.use_bidirectional_attention == 'vision'
            and layer.self_attn.is_sliding
        )
      hidden_states = layer(
          hidden_states,
          position_ids,
          attention_mask,
          skip_sliding_mask=layer_skip_sliding_mask,
      )

    hidden_states = self.norm(hidden_states)

    return hidden_states


class Gemma4ForCausalLM(nn.Module):
  """Gemma4 Causal LM."""

  def __init__(self, config: Gemma4Config):
    super().__init__()
    self.model = Gemma4Model(config)

  def forward(
      self,
      input_ids,
      position_ids=None,
      attention_mask=None,
      skip_sliding_mask=None,
  ):
    hidden_states = self.model(
        input_ids,
        position_ids,
        attention_mask,
        skip_sliding_mask=skip_sliding_mask,
    )
    logits = torch.matmul(hidden_states, self.model.embed_tokens.weight.t())
    return logits
