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

"""Gemma3 model implementation.

Note: Attribute names are identical to the HuggingFace implementation
and cannot be changed without breaking weight loading compatibility.

Note on rope embeddings: There are actually three types of seq len we care
about. Inside a local attention layer, the sliding window is always 1024.
Inside a global attention layer, the window can go up to the
max_positional_embeddings of 128K (131_072). However, that is after a
long-context window training. The model is natively trained at 32K during
pretraining. To keep compatibility with HF pretrained weights, we will only
support 128K max_positional_embeddings for now.

Architecture Note on `num_hidden_layers` (e.g., 62 for 27B):
  The `num_hidden_layers` parameter defines the number of Transformer Decoder
  layers.
  It DOES NOT include the Embedding layer or the final H->V projection.

  For the 27B model with 62 layers and a sliding window pattern of 6 (5 local +
  1 global):
  - The model consists of 62 stacked `Gemma3DecoderLayer` blocks.
  - Pattern: 10 full repetitions of [5 Local Attention + 1 Global Attention]
  layers (60 layers).
  - Plus 2 remaining layers (Local Attention).
  - Total: 60 + 2 = 62 layers.

  The Embedding and H->V projection are separate modules wrapping this stack.
"""

import copy
import math
import types
from typing import Any
from typing import cast
from typing import Optional
from absl import logging
import torch
from torch import nn
from torch_tpu._internal.utils import log_utils

log_utils.log_to_stderr()


# This was established to be equiv to text part of Gemma3 27B in tool.py.
# Directly modify the num_hidden_layers to 6 to create a mini version.
GEMMA3_27B_TEXT = """
{
  "architectures": [
    "Gemma3ForCausalLM"
  ],
  "head_dim": 128,
  "hidden_size": 5376,
  "intermediate_size": 21504,
  "model_type": "gemma3_text",
  "num_attention_heads": 32,
  "num_hidden_layers": 62,
  "num_key_value_heads": 16,
  "query_pre_attn_scalar": 168,
  "rope_scaling": {
    "factor": 8.0,
    "rope_type": "linear"
  },
  "sliding_window": 1024,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.50.0.dev0"
}
"""


class Gemma3TextConfig:
  """Config class, mostly ducktype compatible with transformers.Gemma3TextConfig."""

  def __init__(
      self,
      architectures=None,  # Unused
      model_type=None,  # Unused
      torch_dtype=None,  # Unused
      transformers_version=None,  # Unused
      vocab_size=262_208,
      hidden_size=2304,
      intermediate_size=9216,
      num_hidden_layers=26,
      num_attention_heads=8,
      num_key_value_heads=4,
      head_dim=None,
      # hidden_activation="gelu_pytorch_tanh",
      max_position_embeddings=131_072,
      initializer_range=0.02,
      rms_norm_eps=1e-6,
      # use_cache=True,
      pad_token_id=0,
      eos_token_id=1,
      bos_token_id=2,
      tie_word_embeddings=True,
      rope_theta=1_000_000.0,
      attention_bias=False,
      # attention_dropout=0.0,
      query_pre_attn_scalar=None,
      sliding_window=None,
      # layer_types=None,
      # final_logit_softcapping=None,
      # attn_logit_softcapping=None,
      rope_scaling=None,
      rope_local_base_freq=10_000.0,
      # use_bidirectional_attention=False,
  ):
    if torch_dtype != "bfloat16":
      raise ValueError(f"torch_dtype not set to bfloat16: {torch_dtype}")
    if head_dim is None:
      raise ValueError("head_dim must be specified")
    if rope_scaling is None:
      raise ValueError("rope_scaling must be specified")
    if query_pre_attn_scalar is None:
      raise ValueError("query_pre_attn_scalar must be specified")
    if sliding_window is None:
      raise ValueError("sliding_window must be specified")

    del architectures  # Unused
    del model_type  # Unused
    del torch_dtype  # Unused
    del transformers_version

    self.pad_token_id = pad_token_id
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.tie_word_embeddings = tie_word_embeddings

    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.head_dim = head_dim
    self.num_key_value_heads = num_key_value_heads
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    # self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.attention_bias = attention_bias
    # self.attention_dropout = attention_dropout
    # self.hidden_activation = hidden_activation
    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.sliding_window = sliding_window
    # self.final_logit_softcapping = final_logit_softcapping
    # self.attn_logit_softcapping = attn_logit_softcapping
    # self.layer_types = layer_types
    # self.use_bidirectional_attention = use_bidirectional_attention
    # if use_bidirectional_attention:
    # self.sliding_window = (self.sliding_window // 2) + 1

    self.rope_local_base_freq = rope_local_base_freq

    if isinstance(rope_scaling, dict):
      self.rope_scaling = types.SimpleNamespace(
          **cast(dict[Any, str], rope_scaling)
      )
    else:
      self.rope_scaling = rope_scaling

    self._sliding_window_pattern = 6  # kwargs.get("sliding_window_pattern", 6)

    self.layer_types = [
        "sliding_attention"
        if bool((i + 1) % self._sliding_window_pattern)
        else "full_attention"
        for i in range(self.num_hidden_layers)
    ]


class GELUTanh(nn.Module):
  """Gemma3 activation function.

  Based on HuggingFace implementation.

  This version uses the python implementation,
  not the C implementation, from HF.
  """

  def __init__(self):
    super().__init__()
    self._const1 = math.sqrt(2.0 / math.pi)
    self._const2 = 0.044715

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
      x: Input value

    Returns:
      activated value.
    """

    return (x * 0.5) * (
        1.0 + torch.tanh(self._const1 * (x + self._const2 * torch.pow(x, 3.0)))
    )


class Gemma3TextScaledWordEmbedding(nn.Embedding):
  """Gemma3 embedding layer.

  This module extends nn.Embedding to include a scale factor.
  """

  def __init__(
      self,
      num_embeddings: int,
      embedding_dim: int,
      padding_idx: int,
      embed_scale: float = 1.0,
  ):
    """Constructor.

    Args:
      num_embeddings: same as nn.Embedding
      embedding_dim: same as nn.Embedding
      padding_idx: same as nn.Embedding
      embed_scale: scaling factor
    """
    super().__init__(num_embeddings, embedding_dim, padding_idx)
    self.register_buffer(
        "embed_scale", torch.tensor(embed_scale), persistent=False
    )

  def forward(self, input_ids: torch.Tensor):
    return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class GateProjLinear(nn.Linear):
  pass


class UpProjLinear(nn.Linear):
  pass


class DownProjLinear(nn.Linear):
  pass


class Gemma3MLP(nn.Module):
  """Gemma3 MLP layer."""

  def __init__(self, config: Gemma3TextConfig):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = GateProjLinear(
        self.hidden_size, self.intermediate_size, bias=False
    )
    self.up_proj = UpProjLinear(
        self.hidden_size, self.intermediate_size, bias=False
    )
    self.down_proj = DownProjLinear(
        self.intermediate_size, self.hidden_size, bias=False
    )
    self.act_fn = GELUTanh()

  def forward(self, x):
    gate = self.gate_proj(x)
    activated = self.act_fn(gate)
    up = self.up_proj(x)
    gated_up = activated * up
    down = self.down_proj(gated_up)
    return down


class Gemma3RMSNorm(nn.Module):
  """Gemma3 RMSNorm.

  This is a basic RMSNorm but Gemma3 is specific about initialization, epsilon,
  and order of mixed precision operations.
  """

  def __init__(self, dim: int, eps: float = 1e-6):
    """Constructor.

    Args:
      dim: the hidden dim
      eps: for numerical stability
    """

    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.zeros(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
      x: input tensor

    Returns:
      normalized tensor per RMSNorm algorithm
    """
    x = x.float()
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


def _compute_default_rope_parameters(
    config: Gemma3TextConfig,
) -> tuple["torch.Tensor", float]:
  """Computes rope frequencies.

  Args:
    config: config with key architecture shapes

  Returns:
    Tuple of rope frequencies and scaling factor.
  """
  base = config.rope_theta

  # Compute the inverse frequencies
  inv_freq = 1.0 / (
      base
      ** (
          torch.arange(0, config.head_dim, 2, dtype=torch.int64).float()
          / config.head_dim
      )
  )
  return inv_freq, 1.0


def _compute_linear_scaling_rope_parameters(
    config: Gemma3TextConfig,
) -> tuple["torch.Tensor", float]:
  """Computes the inverse frequencies with linear scaling."""
  factor = config.rope_scaling.factor
  inv_freq, attention_factor = _compute_default_rope_parameters(config)
  inv_freq /= factor
  return inv_freq, attention_factor


class Gemma3RotaryEmbedding(nn.Module):
  """RoPE implementation for Gemma3."""

  inv_freq: torch.Tensor

  def __init__(self, config: Gemma3TextConfig):
    super().__init__()
    # self.rope_type = config.rope_scaling

    self.max_seq_len_cached = config.max_position_embeddings
    self.original_max_seq_len = config.max_position_embeddings

    self.config = config

    # See note in module docstring on multiple
    # rope configs in Gemma3. Default is
    # used for local attn layers.
    if config.rope_scaling.rope_type == "default":
      inv_freq, self.attention_scaling = _compute_default_rope_parameters(
          self.config
      )
    elif (
        config.rope_scaling.rope_type == "linear"
        and config.rope_scaling.factor == 8
    ):
      inv_freq, self.attention_scaling = (
          _compute_linear_scaling_rope_parameters(self.config)
      )
    else:
      raise ValueError(f"Unsupported rope scaling: {config.rope_scaling}")

    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.original_inv_freq = self.inv_freq

  @torch.no_grad()
  def forward(self, x, position_ids):
    inv_freq_expanded = (
        self.inv_freq[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    # Removed some HF code to deal with MPS.
    # https://github.com/huggingface/transformers/blame/f1f34de0a68497ebc95ae690893fb75b89814c0d/src/transformers/models/gemma3/modeling_gemma3.py#L223
    with torch.autocast(device_type=x.device.type, enabled=False):
      freqs = (
          inv_freq_expanded.float() @ position_ids_expanded.float()
      ).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies Rotary Position Embedding to the query and key tensors.

  Args:
    q (`torch.Tensor`): The query tensor.
    k (`torch.Tensor`): The key tensor.
    cos (`torch.Tensor`): The cosine part of the rotary embedding.
    sin (`torch.Tensor`): The sine part of the rotary embedding.
    unsqueeze_dim (`int`, *optional*, defaults to 1): The 'unsqueeze_dim'
      argument specifies the dimension along which to unsqueeze
      cos[position_ids] and sin[position_ids] so that they can be properly
      broadcasted to the dimensions of q and k. For example, note that
      cos[position_ids] and sin[position_ids] have the shape [batch_size,
      seq_len, head_dim]. Then, if q and k have the shape [batch_size, heads,
      seq_len, head_dim], then setting unsqueeze_dim=1 makes cos[position_ids]
      and sin[position_ids] broadcastable to the shapes of q and k. Similarly,
      if q and k have the shape [batch_size, seq_len, heads, head_dim], then set
      unsqueeze_dim=2.

  Returns:
      Tuple of rotated query and key tensors.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

  The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
  (batch, num_attention_heads, seqlen, head_dim).

  Args:
    hidden_states: KV values
    n_rep: number of repeats on the KV heads.

  Returns:
    hidden_states with repeated KV heads to match Q.
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(
      batch, num_key_value_heads, n_rep, slen, head_dim
  )
  return hidden_states.reshape(
      batch, num_key_value_heads * n_rep, slen, head_dim
  )


def eager_attention_forward(
    # module: nn.Module,
    num_key_value_groups: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    # dropout: float = 0.0,
    scaling: float,
    # softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Pure python implementation of self attention.

  Arg sizes are for Gemma27B.

  Args:
    num_key_value_groups: Number of key-value heads.
    query: (B, num_attention_heads, seq_len, head_dim)
    key: (B, num_key_value_heads, seq_len, head_dim)
    value: (B, num_key_value_heads, seq_len, head_dim)
    attention_mask: (B, 1, seq_len, seq_len)
    scaling: Scaling factor.

  Returns:
    torch.Tensor:
  """
  bsz, num_attn_heads, seq_len, head_dim = query.shape
  if head_dim != 128:
    raise ValueError(f"head_dim is {head_dim}, expected 128")
  if num_attn_heads != 32:
    raise ValueError(f"num_attn_heads is {num_attn_heads}, expected 32")

  assert attention_mask.shape == (1, 1, seq_len, seq_len), (
      f"attention_mask.shape: {attention_mask.shape}, expected"
      f" {(1, 1, seq_len, seq_len)}"
  )

  # if scaling is None:
  #   scaling = module.head_dim**-0.5

  key_states = repeat_kv(key, num_key_value_groups)  # (B, 32, S, 128)
  value_states = repeat_kv(value, num_key_value_groups)  # (B, 32, S, 128)

  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attn_weights.shape != (bsz, 32, seq_len, seq_len):
    raise ValueError(
        f"attn_weights.shape: {attn_weights.shape}, expected"
        f" {(bsz, 32, seq_len, seq_len)}"
    )

  # if softcap is not None:
  #   attn_weights = attn_weights / softcap
  #   attn_weights = torch.tanh(attn_weights)
  #   attn_weights = attn_weights * softcap

  # Note: I find this slice of the 4th dim, but not the 3rd, to be sus.
  logging.info("attention_mask.shape: %s", attention_mask.shape)
  logging.info("key_states.shape: %s", key_states.shape)
  causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
  attn_weights = attn_weights + causal_mask

  # upcast attention to fp32
  attn_weights = nn.functional.softmax(
      attn_weights, dim=-1, dtype=torch.float32
  ).to(query.dtype)
  # attn_weights = nn.functional.dropout(
  #     attn_weights, p=dropout, training=module.training
  # )
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()
  return attn_output, attn_weights


class QProjLinear(nn.Linear):
  pass


class KProjLinear(nn.Linear):
  pass


class VProjLinear(nn.Linear):
  pass


class OProjLinear(nn.Linear):
  pass


class Gemma3Attention(nn.Module):
  """Gemma3 attention layer."""

  def __init__(self, config: Gemma3TextConfig, layer_idx: int):
    super().__init__()
    self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = config.head_dim  # = getattr(
    #     config, "head_dim", config.hidden_size // config.num_attention_heads
    # )
    self.num_key_value_groups = (
        config.num_attention_heads // config.num_key_value_heads
    )
    self.scaling = config.query_pre_attn_scalar**-0.5

    # self.attention_dropout = self.config.attention_dropout
    # self.is_causal = True not self.config.use_bidirectional_attention

    # Q is 32 heads, KV is 16 heads (num_attention_heads, num_key_value_heads),
    # so this is a 1:2 GQA (Grouped Query Attention) ratio. Gemma
    # does NOT enforce that head dim * heads = hidden_size. The
    # hidden size is 5376, while head dim * heads = 32 * 128 = 4096.
    self.q_proj = QProjLinear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = KProjLinear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = VProjLinear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = OProjLinear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )
    # self.attn_logit_softcapping = self.config.attn_logit_softcapping

    self.sliding_window = config.sliding_window if self.is_sliding else None

    self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: torch.Tensor,
      attention_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, None]:
    """Forward pass for Gemma3 decoder block.

    Args:
      hidden_states: Shape (batch_size, seq_len, hidden_size) aka BSH
      position_embeddings: Position emeddings
      attention_mask: Attention mask.

    Returns:
      Tuple[torch.Tensor, None]: The output of the model, followed by None.
        This is solely for signature compatibility with the HF reference model.
    """
    input_shape = hidden_states.shape[:-1]  # (B,S)
    hidden_shape = (*input_shape, -1, self.head_dim)  # (B,S,-1,128)

    query_states = (
        self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    )  # (B,32,S,128)
    key_states = (
        self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    )  # (B,16,S,128)
    value_states = (
        self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    )  # (B,16,S,128)

    query_states = self.q_norm(query_states)  # (B,32,S,128)
    key_states = self.k_norm(key_states)  # (B,16,S,128)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # This will eventually need to be replaced with FA2/FA3 with SWA.
    # Naive attn does not understand the concept of a sliding window;
    # it just sees a truncated causal mask.
    attn_output, _ = eager_attention_forward(
        self.num_key_value_groups,  # original: self
        query_states,  # (B,32,S,128)
        key_states,  # (B,16,S,128)
        value_states,  # (B,16,S,128)
        attention_mask,  # (B,1,S,S)
        # dropout=0.0,
        scaling=self.scaling,
        # sliding_window=self.sliding_window,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


class Gemma3DecoderLayer(nn.Module):
  """Gemma3 decoder layer."""

  def __init__(self, config: Gemma3TextConfig, layer_idx: int):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.layer_idx = layer_idx
    self.attention_type = config.layer_types[layer_idx]
    self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
    self.mlp = Gemma3MLP(config)
    self.input_layernorm = Gemma3RMSNorm(
        self.hidden_size, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = Gemma3RMSNorm(
        self.hidden_size, eps=config.rms_norm_eps
    )
    self.pre_feedforward_layernorm = Gemma3RMSNorm(
        self.hidden_size, eps=config.rms_norm_eps
    )
    self.post_feedforward_layernorm = Gemma3RMSNorm(
        self.hidden_size, eps=config.rms_norm_eps
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
  ) -> tuple[torch.Tensor]:
    # pylint: disable=g-one-element-tuple
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def _create_causal_mask(
    seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
  """Creates a causal mask (lower triangular)."""
  # (B, 1, S, S)
  mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
  mask.masked_fill_(
      torch.triu(
          torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
          diagonal=1,
      ),
      torch.finfo(dtype).min,
  )
  return mask[None, None, :, :]


def _create_sliding_window_causal_mask(
    seq_len: int, sliding_window: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
  """Creates a sliding window causal mask."""
  # (B, 1, S, S)
  mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)

  # Mask future
  future_mask = torch.triu(
      torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
  )

  # Mask past (too far away)
  past_mask = torch.tril(
      torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
      diagonal=-sliding_window - 1,
  )

  mask.masked_fill_(future_mask | past_mask, torch.finfo(dtype).min)
  # Broadcasting batch dim will not work when we add image support.
  return mask[None, None, :, :]


class Gemma3TextModel(nn.Module):
  """Gemma3 Text Model."""

  def __init__(self, config: Gemma3TextConfig):
    super().__init__()
    self.config = config
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = Gemma3TextScaledWordEmbedding(
        config.vocab_size,
        config.hidden_size,
        self.padding_idx,
        embed_scale=self.config.hidden_size**0.5,
    )
    self.layers = nn.ModuleList([
        Gemma3DecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ])
    self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = Gemma3RotaryEmbedding(config=config)
    # self.gradient_checkpointing = False

    # Create a second rope_scaling for local attn layers.
    # Copies the entire config (for architecture sizes),
    # then replaces:
    # * rope_theta: 1_000_000 -> 10_000
    # * rope_scaling: {factor:8, rope_type:linear} -> "default"

    config = copy.deepcopy(config)
    config.rope_theta = config.rope_local_base_freq
    config.rope_scaling = types.SimpleNamespace(rope_type="default")
    self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

  def forward(
      self,
      input_ids: torch.Tensor,
      # attention_mask: Optional[torch.Tensor] = None,
      # position_ids: torch.Tensor,
  ) -> torch.Tensor:
    # Attention masks can look forward in multimodal world.
    # But for text only, it's just a lower triangle or
    # triangular band for sliding window.
    # TODO: There is high risk of error here. Verify if these masks
    # match HF.

    bsz, seq_len = input_ids.shape
    del bsz
    device = input_ids.device

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    # Create the masks
    mask_kwargs = {
        "seq_len": seq_len,
        "dtype": self.embed_tokens.weight.dtype,
        "device": device,
    }

    # TODO: This logic needs to change for decode and image support.
    causal_mask_mapping = {
        "full_attention": _create_causal_mask(**mask_kwargs),
        "sliding_attention": _create_sliding_window_causal_mask(
            sliding_window=self.config.sliding_window, **mask_kwargs
        ),
    }

    # embed positions
    hidden_states = self.embed_tokens(input_ids)

    # create position embeddings to be shared across the decoder layers
    rope_mapping = {
        "full_attention": self.rotary_emb,
        "sliding_attention": self.rotary_emb_local,
    }
    position_embeddings = {}
    for layer_type in set(self.config.layer_types):
      position_embeddings[layer_type] = rope_mapping[layer_type](
          hidden_states, position_ids
      )

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
      hidden_states = decoder_layer(
          hidden_states,
          position_embeddings=position_embeddings[decoder_layer.attention_type],
          attention_mask=causal_mask_mapping[decoder_layer.attention_type],
          position_ids=position_ids,
      )

    hidden_states = self.norm(hidden_states)

    return hidden_states


class Gemma3ForCausalLM(nn.Module):
  """Gemma3 Causal LM."""

  def __init__(self, config: Gemma3TextConfig):
    super().__init__()
    self.model = Gemma3TextModel(config)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward pass for Causal LM."""

    hidden_states = self.model(input_ids=input_ids)

    # Gemma3 re-uses the embedding weights for the final proj.
    logits = torch.matmul(hidden_states, self.model.embed_tokens.weight.t())

    return logits
