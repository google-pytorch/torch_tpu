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

"""Nemotron 3 model implementation"""

import torch
from torch import nn
import torch.nn.functional as F
from examples.nemotron3.mamba2 import NemotronHMamba2Mixer


class NemotronHConfig:
  """Configuration for Nemotron-3-Nano-30B-A3B"""

  def __init__(
      self,
      vocab_size: int = 131072,
      hidden_size: int = 2688,
      layers_block_type: list[str] | None = None,
      intermediate_size: int = 21504,
      # Mamba 2 params
      ssm_state_size: int = 128,
      mamba_num_heads: int = 64,
      mamba_head_dim: int = 64,
      mamba_hidden_act: str = "silu",
      n_groups: int = 8,
      conv_kernel: int = 4,
      expand: int = 2,
      time_step_min: float = 0.001,
      time_step_max: float = 0.1,
      time_step_limit: list[float] | None = None,
      use_conv_bias: bool = True,
      chunk_size: int = 128,
      mamba_proj_bias: bool = False,
      # MoE params
      n_routed_experts: int = 8,  # scaled down for lightweight baseline
      num_experts_per_tok: int = 2,
      routed_scaling_factor: float = 1.0,
      n_group: int = 1,
      topk_group: int = 1,
      norm_topk_prob: bool = True,
      moe_intermediate_size: int = 1856,
      moe_shared_expert_intermediate_size: int = 1856,
      moe_latent_size: int | None = None,
      # Attention params
      num_attention_heads: int = 32,
      num_key_value_heads: int = 8,
      head_dim: int = 128,
      attention_dropout: float = 0.0,
      # Common params
      mlp_bias: bool = False,
      mlp_hidden_act: str = "silu",
      use_bias: bool = False,
      layer_norm_epsilon: float = 1e-5,
  ):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.use_bias = use_bias
    self.layer_norm_epsilon = layer_norm_epsilon

    if layers_block_type is None:
      self.layers_block_type = ["mamba", "moe", "mamba", "attention", "moe"]
    else:
      self.layers_block_type = layers_block_type

    # Mamba 2
    self.ssm_state_size = ssm_state_size
    self.mamba_num_heads = mamba_num_heads
    self.mamba_head_dim = mamba_head_dim
    self.mamba_hidden_act = mamba_hidden_act
    self.n_groups = n_groups
    self.conv_kernel = conv_kernel
    self.expand = expand
    self.time_step_min = time_step_min
    self.time_step_max = time_step_max
    self.time_step_limit = time_step_limit or [0.0, float("inf")]
    self.use_conv_bias = use_conv_bias
    self.chunk_size = chunk_size
    self.mamba_proj_bias = mamba_proj_bias

    # MoE
    self.n_routed_experts = n_routed_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.routed_scaling_factor = routed_scaling_factor
    self.n_group = n_group
    self.topk_group = topk_group
    self.norm_topk_prob = norm_topk_prob
    self.moe_intermediate_size = moe_intermediate_size
    self.moe_shared_expert_intermediate_size = (
        moe_shared_expert_intermediate_size
    )
    self.moe_latent_size = moe_latent_size

    # Attention
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.head_dim = head_dim
    self.attention_dropout = attention_dropout

    # MLP
    self.mlp_bias = mlp_bias
    self.mlp_hidden_act = mlp_hidden_act


configs = {
    "Toy": {
        "vocab_size": 1000,
        "hidden_size": 512,
        "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
        "mamba_num_heads": 8,
        "mamba_head_dim": 64,
        "n_groups": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "moe_intermediate_size": 512,
        "moe_shared_expert_intermediate_size": 512,
    },
    "30B": {
        "vocab_size": 131072,
        "hidden_size": 2688,
        "intermediate_size": 21504,
        "ssm_state_size": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "conv_kernel": 4,
        "expand": 2,
        "chunk_size": 128,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 1856,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}


class NemotronHRMSNorm(nn.Module):
  """RMSNorm with float32 upcasting for variance calculation."""

  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(
        variance + self.variance_epsilon
    )
    return self.weight * hidden_states.to(input_dtype)


class NemotronHMLP(nn.Module):
  """Gated SwiGLU MLP block used in Nemotron-3 (shared experts and standalone MLP)."""

  def __init__(
      self, config: NemotronHConfig, intermediate_size: int | None = None
  ):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.intermediate_size = intermediate_size or config.intermediate_size
    self.up_proj = nn.Linear(
        self.hidden_size, self.intermediate_size, bias=config.mlp_bias
    )
    self.down_proj = nn.Linear(
        self.intermediate_size, self.hidden_size, bias=config.mlp_bias
    )
    # Mapping simple silu string
    self.act_fn = F.silu

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
  """Router for Top-k experts routing in MoE."""

  def __init__(self, config: NemotronHConfig):
    super().__init__()
    self.n_routed_experts = config.n_routed_experts
    self.weight = nn.Parameter(
        torch.empty((self.n_routed_experts, config.hidden_size))
    )
    nn.init.normal_(self.weight, std=0.02)
    # Initialize score correction bias
    self.register_buffer(
        "e_score_correction_bias", torch.zeros(self.n_routed_experts)
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    router_logits = F.linear(
        hidden_states.type(torch.float32), self.weight.type(torch.float32)
    )
    return router_logits


class NemotronHExperts(nn.Module):
  """Non-gated MLP Experts block for Mixture of Experts (MoE)."""

  def __init__(self, config: NemotronHConfig):
    super().__init__()
    self.num_experts = config.n_routed_experts
    self.hidden_dim = config.hidden_size
    self.intermediate_dim = config.moe_intermediate_size

    input_dim = (
        config.moe_latent_size
        if config.moe_latent_size is not None
        else config.hidden_size
    )

    # expert weights stored as 3D tensors: (num_experts, out_dim, in_dim)
    self.up_proj = nn.Parameter(
        torch.empty(self.num_experts, self.intermediate_dim, input_dim)
    )
    self.down_proj = nn.Parameter(
        torch.empty(self.num_experts, input_dim, self.intermediate_dim)
    )
    nn.init.normal_(self.up_proj, std=0.02)
    nn.init.normal_(self.down_proj, std=0.02)
    self.act_fn = F.silu

  def forward(
      self,
      hidden_states: torch.Tensor,
      top_k_index: torch.Tensor,
      top_k_weights: torch.Tensor,
  ) -> torch.Tensor:
    # Parallel Up Projection
    # hidden_states: [T, I] -> [1, T, I]
    # self.up_proj: [E, J, I] -> transpose(1, 2) -> [E, I, J]
    # matmul: [1, T, I] x [E, I, J] -> [E, T, J]
    up_out = torch.matmul(
        hidden_states.unsqueeze(0), self.up_proj.transpose(1, 2)
    )
    hidden = self.act_fn(up_out)

    # Parallel Down Projection
    # hidden: [E, T, J]
    # self.down_proj: [E, I, J] -> transpose(1, 2) -> [E, J, I]
    # matmul: [E, T, J] x [E, J, I] -> [E, T, I]
    down_out = torch.matmul(hidden, self.down_proj.transpose(1, 2))

    # Masking
    # top_k_index: [T, K]
    # top_k_weights: [T, K]
    one_hot = F.one_hot(top_k_index, num_classes=self.num_experts)  # [T, K, E]
    weighted_one_hot = one_hot * top_k_weights.unsqueeze(-1)
    gating_weights = weighted_one_hot.sum(dim=1).t()  # [E, T]

    # Weighted Sum
    # gating_weights: [E, T] -> unsqueeze(-1) -> [E, T, 1]
    # down_out: [E, T, I]
    output = (down_out * gating_weights.unsqueeze(-1)).sum(dim=0)
    return output.to(hidden_states.dtype)


class NemotronHMoE(nn.Module):
  """Mixture of Experts block with latent projection layers."""

  def __init__(self, config: NemotronHConfig):
    super().__init__()
    self.config = config
    self.experts = NemotronHExperts(config)
    self.gate = NemotronHTopkRouter(config)

    self.shared_experts = NemotronHMLP(
        config=config,
        intermediate_size=config.moe_shared_expert_intermediate_size,
    )
    self.n_routed_experts = config.n_routed_experts
    self.n_group = config.n_group
    self.topk_group = config.topk_group
    self.norm_topk_prob = config.norm_topk_prob
    self.routed_scaling_factor = config.routed_scaling_factor
    self.top_k = config.num_experts_per_tok

    if config.moe_latent_size is not None:
      self.fc1_latent_proj = nn.Linear(
          config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
      )
      self.fc2_latent_proj = nn.Linear(
          config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
      )
    else:
      self.fc1_latent_proj = nn.Identity()
      self.fc2_latent_proj = nn.Identity()

  def route_tokens_to_experts(self, router_logits: torch.Tensor):
    router_logits = router_logits.sigmoid()
    router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

    # Group scores calculation
    group_scores = (
        router_logits_for_choice.view(
            -1, self.n_group, self.n_routed_experts // self.n_group
        )
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )

    group_idx = torch.topk(
        group_scores, k=self.topk_group, dim=-1, sorted=False
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
        .reshape(-1, self.n_routed_experts)
    )

    scores_for_choice = router_logits_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )
    topk_indices = torch.topk(
        scores_for_choice, k=self.top_k, dim=-1, sorted=False
    )[1]
    topk_weights = router_logits.gather(1, topk_indices)

    if self.norm_topk_prob:
      denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
      topk_weights /= denominator

    topk_weights = topk_weights * self.routed_scaling_factor
    return topk_indices, topk_weights

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residuals = hidden_states
    orig_shape = hidden_states.shape
    router_logits = self.gate(hidden_states)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # Apply Latent Projection
    hidden_states = self.fc1_latent_proj(hidden_states)
    hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
    hidden_states = self.fc2_latent_proj(hidden_states)

    hidden_states = hidden_states.view(*orig_shape)
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states


# 3. NemotronH Attention Layer
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """Repeat key-value heads for Grouped-Query Attention (GQA)."""
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(
      batch, num_key_value_heads, n_rep, slen, head_dim
  )
  return hidden_states.reshape(
      batch, num_key_value_heads * n_rep, slen, head_dim
  )


class NemotronHAttention(nn.Module):
  """Self-Attention block with GQA (Grouped Query Attention)."""

  def __init__(self, config: NemotronHConfig):
    super().__init__()
    self.config = config
    self.head_dim = config.head_dim
    self.num_key_value_groups = (
        config.num_attention_heads // config.num_key_value_heads
    )
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=False,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=False,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=False,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=False,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch, seqlen, _ = hidden_states.shape
    hidden_shape = (batch, seqlen, -1, self.head_dim)

    # Project to attention dimension (may project from 2688 to 4096)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Repeat Key-Value states for GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Eager scaled dot-product attention
    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    )
    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    if self.training and self.attention_dropout > 0:
      attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(batch, seqlen, -1)
    return self.o_proj(attn_output)


MIXER_TYPES = {
    "mamba": NemotronHMamba2Mixer,
    "attention": NemotronHAttention,
    "moe": NemotronHMoE,
    "mlp": NemotronHMLP,
}


class NemotronHBlock(nn.Module):
  """A single block wrapping a mixer type and applying pre-RMSNorm."""

  def __init__(self, config: NemotronHConfig, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.norm = NemotronHRMSNorm(
        config.hidden_size, eps=config.layer_norm_epsilon
    )

    self.block_type = config.layers_block_type[layer_idx]
    if self.block_type == "mamba":
      self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
    elif self.block_type == "attention":
      self.mixer = NemotronHAttention(config)
    elif self.block_type == "moe":
      self.mixer = NemotronHMoE(config)
    elif self.block_type == "mlp":
      self.mixer = NemotronHMLP(config)
    else:
      raise ValueError(f"Unknown block type: {self.block_type}")

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

    if self.block_type == "mamba":
      hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
    elif self.block_type == "attention":
      hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
    else:
      hidden_states = self.mixer(hidden_states)

    hidden_states = residual + hidden_states
    return hidden_states


class NemotronHModel(nn.Module):
  """Backbone Nemotron-H model with alternating layer stack."""

  def __init__(self, config: NemotronHConfig):
    super().__init__()
    self.config = config
    self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

    self.layers = nn.ModuleList([
        NemotronHBlock(config, layer_idx=idx)
        for idx in range(len(config.layers_block_type))
    ])

    self.norm_f = NemotronHRMSNorm(
        config.hidden_size, eps=config.layer_norm_epsilon
    )

  def forward(
      self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Create 4D causal mask for attention layers
    causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = (
        causal_mask.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, 1, seq_len, seq_len)
    )

    if attention_mask is not None and attention_mask.dim() == 2:
      # Combine 2D padding mask with 4D causal mask
      # 2D mask has 1 for valid tokens, 0 for pad tokens
      padding_mask = torch.where(
          attention_mask == 1,
          torch.tensor(0.0, device=device),
          torch.tensor(float("-inf"), device=device),
      )
      # shape: (batch, 1, 1, seq_len)
      padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
      causal_mask = causal_mask + padding_mask

    x = self.embeddings(input_ids)
    for layer in self.layers:
      if layer.block_type == "attention":
        x = layer(x, attention_mask=causal_mask)
      elif layer.block_type == "mamba":
        # Pass 2D mask to Mamba (only if it is 2D, if we received a 4D mask for some reason, don't pass it or it will crash)
        # But we assume the input attention_mask to the model is always 2D (padding mask).
        mamba_mask = (
            attention_mask
            if (attention_mask is not None and attention_mask.dim() == 2)
            else None
        )
        x = layer(x, attention_mask=mamba_mask)
      else:
        x = layer(x)
    x = self.norm_f(x)
    return x


class NemotronHForCausalLM(nn.Module):
  """Nemotron-H Model with Causal LM head."""

  def __init__(self, config: NemotronHConfig):
    super().__init__()
    self.model = NemotronHModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  def forward(
      self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    hidden_states = self.model(input_ids, attention_mask=attention_mask)
    return self.lm_head(hidden_states)
