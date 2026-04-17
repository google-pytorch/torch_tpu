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

"""An initial implementation of Qwen3.

This implementation was inspired by the book:
  Build a Large Language Model by Sebastian Raschka

Currently, this file only contains the model instantiation.
Weights and other configurations will come later.

To run the MoE model:
  blaze test --test_arg=--tpu --test_arg=--model=MoEmini qwen3

///////////////////////////////////////////////////////////////
NOTE: This file/model is meant to DEBUG and TRIAGE remaining
      issues in the torch_tpu implementation. It will contain
      some experimental things and lots of TODOs to bypass
      existing problems.
//////////////////////////////////////////////////////////////
"""


from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.utils import utils

# Global model configurations. For now we only support the smallest
# one but we will add more once things start working.
configs = {
    # A tiny version for Integration Tests
    "Integration_Test": {
        "vocab_size": 1234,  # Vocabulary size
        "context_length": (
            1024
        ),  # Context length that was used to train the model
        "emb_dim": 256,  # Embedding dimension
        "n_heads": 8,  # Number of attention heads
        "n_layers": 2,  # Number of layers
        "hidden_dim": 128,  # Size of the intermediate dimension in FeedForward
        "head_dim": 32,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        "dtype": torch.float32,  # Lower-precision dtype to reduce memory usage
        "use_cache": False,
    },
    "MoEmini": {
        "vocab_size": 151_936,
        "context_length": 262_144,
        "emb_dim": 128,  # 2048,
        "n_heads": 4,  # 32,
        "n_layers": 4,  # 48,
        "head_dim": 4,  # , 128,
        "qk_norm": True,
        "n_kv_groups": 4,
        "rope_base": 10_000_000.0,
        # TODO: There are still issues with type casts.
        #       See TODO's below. Setting the whole model to float32
        #       works but is only a short term workaround.
        # "dtype": torch.bfloat16,  # Lower-precision dtype
        "dtype": torch.float32,
        "num_experts": 4,  # 128,
        "num_experts_per_tok": 4,  # 8,
        "moe_intermediate_size": 768,
        "use_cache": False,
    },
    "MoE": {
        "vocab_size": 151_936,
        "context_length": 262_144,
        "emb_dim": 2048,
        "n_heads": 32,
        "n_layers": 48,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 4,
        "rope_base": 10_000_000.0,
        # TODO: There are still issues with type casts.
        #       See TODO's below. Setting the whole model to float32
        #       works but is only a short term workaround.
        # "dtype": torch.bfloat16,  # Lower-precision dtype
        "dtype": torch.float32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "use_cache": False,
    },
    "0.6B": {
        "vocab_size": 151_936,  # Vocabulary size
        "context_length": (
            40_960
        ),  # Context length that was used to train the model
        "emb_dim": 1024,  # Embedding dimension
        "n_heads": 16,  # Number of attention heads
        "n_layers": 28,  # Number of layers
        "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
        "head_dim": 128,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        # TODO: There are still issues with type casts.
        #       See TODO's below. Setting the whole model to float32
        #       works but is only a short term workaround.
        # "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        "dtype": torch.float32,  # Lower-precision dtype to reduce memory usage
        "use_cache": False,
    },
    "1.7B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2048,  # 2x larger than above
        "n_heads": 16,
        "n_layers": 28,
        "hidden_dim": 6144,  # 2x larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,  # TODO: bfloat16,
        "use_cache": False,
    },
    "4B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2560,  # 25% larger than above
        "n_heads": 32,  # 2x larger than above
        "n_layers": 36,  # 29% larger than above
        "hidden_dim": 9728,  # ~3x larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,  # TODO: bfloat16,
        "use_cache": False,
    },
    "8B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 4096,  # 60% larger than above
        "n_heads": 32,
        "n_layers": 36,  # 26% larger than above
        "hidden_dim": 12288,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,  # TODO: bfloat16,
        "use_cache": False,
    },
    "14B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,  # 25% larger than above
        "n_heads": 40,  # 25% larger than above
        "n_layers": 40,  # 11% larger than above
        "hidden_dim": 17408,  # 42% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,  # TODO: bfloat16,
        "use_cache": False,
    },
    "32B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,  # 60% larger than above
        "n_layers": 64,  # 60% larger than above
        "hidden_dim": 25600,  # 47% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,  # TODO: bfloat16,
        "use_cache": False,
    },
}


class MoEFeedForward(nn.Module):

  def __init__(self, cfg):
    super().__init__()
    self.num_experts_per_tok = cfg["num_experts_per_tok"]
    self.num_experts = cfg["num_experts"]
    self.gate = nn.Linear(
        cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"]
    )

    # meta device to reduce memory pressure when initializing the
    # model before loading weights
    # meta_device = torch.device("meta")
    self.fc1 = nn.ModuleList([
        nn.Linear(
            cfg["emb_dim"],
            cfg["moe_intermediate_size"],
            bias=False,
            dtype=cfg["dtype"],
            # device=meta_device,
        )
        for _ in range(cfg["num_experts"])
    ])
    self.fc2 = nn.ModuleList([
        nn.Linear(
            cfg["emb_dim"],
            cfg["moe_intermediate_size"],
            bias=False,
            dtype=cfg["dtype"],
            # device=meta_device,
        )
        for _ in range(cfg["num_experts"])
    ])
    self.fc3 = nn.ModuleList([
        nn.Linear(
            cfg["moe_intermediate_size"],
            cfg["emb_dim"],
            bias=False,
            dtype=cfg["dtype"],
            # device=meta_device,
        )
        for _ in range(cfg["num_experts"])
    ])

  def forward(self, x):
    scores = self.gate(x)
    # TODO:
    # Once everything works, in the original implementation:
    #
    # https://github.com/huggingface/transformers/blob/
    # a5923d4de7df2fbd1f373dfcfe983216b79b6937/src/transformers/
    # models/qwen3_moe/modeling_qwen3_moe.py#L235
    #
    # The softmax is before the topk. Investigate performance
    # once everything works.
    topk_scores, topk_indices = torch.topk(
        scores, self.num_experts_per_tok, dim=-1
    )
    topk_probs = torch.softmax(topk_scores, dim=-1)

    # This loop computes all expert networks for all tokens,
    # which can be very inefficient, especially with a large
    # number of experts. A more typical implementation would
    # only compute the experts that were selected by the gating
    # network for each token.
    #
    # This implementation will be very slow.
    #
    # A common pattern for MoE is to reshape the input x to
    # (batch_size * seq_len, embed_dim), then use the
    # topk_indices to route tokens to their respective experts,
    # and only then perform the expert computation.
    #
    expert_outputs = []
    for e in range(self.num_experts):
      hidden = torch.nn.functional.silu(self.fc1[e](x)) * self.fc2[e](x)
      out = self.fc3[e](hidden)
      expert_outputs.append(out.unsqueeze(-2))
    expert_outputs = torch.cat(expert_outputs, dim=-2)

    gating_probs = torch.zeros_like(scores)

    # TODO: There may be a potential to shorten this loop (wan@)
    for i in range(self.num_experts_per_tok):
      indices = topk_indices[..., i : i + 1]
      prob = topk_probs[..., i : i + 1]
      gating_probs.scatter_(dim=-1, index=indices, src=prob)

    # gating_probs is shape BSE where E is total number of experts. Along that
    # dimension, exactly num_experts_per_tok values should be non-zero.
    # TODO: b/439709255 - Remove .cpu() once count_nonzero is supported.
    assert (
        self.num_experts_per_tok == gating_probs.cpu().count_nonzero(-1)
    ).all()

    # Weighted sum over experts.
    gating_probs = gating_probs.unsqueeze(-1)
    y = (gating_probs * expert_outputs).sum(dim=-2)
    return y


class FeedForward(nn.Module):
  """Traditional feed forward layer with 2 FC's, a silu, and a final FC."""

  def __init__(self, cfg):
    super().__init__()
    self.fc1 = nn.Linear(
        cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
    )
    self.fc2 = nn.Linear(
        cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
    )
    self.fc3 = nn.Linear(
        cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
    )

  def forward(self, x):
    x_fc1 = self.fc1(x)
    x_fc2 = self.fc2(x)
    x = nn.functional.silu(x_fc1) * x_fc2
    return self.fc3(x)


class RMSNorm(nn.Module):
  """Standard RMS Norm implementation (with float32 casts)."""

  def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
    super().__init__()
    self.eps = eps
    self.qwen3_compatible = qwen3_compatible
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

  def forward(self, x):
    input_dtype = x.dtype

    if self.qwen3_compatible:
      x = x.to(torch.float32)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    norm_x = x * torch.rsqrt(variance + self.eps)
    norm_x = norm_x * self.scale

    if self.shift is not None:
      norm_x = norm_x + self.shift

    return norm_x.to(input_dtype)


def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):
  """TODO: Docstring."""

  assert head_dim % 2 == 0, "Embedding dimension must be even"

  # Compute the inverse frequencies
  inv_freq = 1.0 / (
      theta_base
      ** (
          torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
          / head_dim
      )
  )

  # Generate position indices
  positions = torch.arange(context_length, dtype=dtype)

  # Compute the angles
  angles = (
      positions[:, None] * inv_freq[None, :]
  )  # Shape: (context_length, head_dim // 2)

  # Expand angles to match the head_dim
  angles = torch.cat(
      [angles, angles], dim=1
  )  # Shape: (context_length, head_dim)

  # Precompute sine and cosine
  cos = torch.cos(angles)
  sin = torch.sin(angles)

  return cos, sin


def apply_rope(x, cos, sin):
  """TODO: Docstring."""

  # x: (batch_size, num_heads, seq_len, head_dim)
  _, _, _, head_dim = x.shape
  assert head_dim % 2 == 0, "Head dimension must be even"

  # Split x into first half and second half
  x1 = x[..., : head_dim // 2]  # First half
  x2 = x[..., head_dim // 2 :]  # Second half

  # Adjust sin and cos shapes
  cos = cos.unsqueeze(0).unsqueeze(0)
  sin = sin.unsqueeze(0).unsqueeze(0)

  # Apply the rotary transformation
  rotated = torch.cat((-x2, x1), dim=-1)
  x_rotated = (x * cos) + (rotated * sin)

  # It's ok to use lower-precision after applying cos and sin rotation

  # TODO: There is a problem here with a mismatch of buffer sizes in
  #       Torch and PjRt (I think). Not casting works but is a workaround.
  # return x_rotated.to(dtype=x.dtype)
  return x_rotated


class GroupedQueryAttention(nn.Module):
  """TODO: Docstring."""

  def __init__(
      self,
      d_in,
      num_heads,
      num_kv_groups,
      head_dim=None,
      qk_norm=False,
      dtype=None,
      context_length=4096,
      use_cache=False,
  ):
    super().__init__()
    assert (
        num_heads % num_kv_groups == 0
    ), "num_heads must be divisible by num_kv_groups"

    self.num_heads = num_heads
    self.num_kv_groups = num_kv_groups
    self.group_size = num_heads // num_kv_groups

    if head_dim is None:
      assert (
          d_in % num_heads == 0
      ), "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
      head_dim = d_in // num_heads

    self.head_dim = head_dim
    self.d_out = num_heads * head_dim

    self.w_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
    self.w_key = nn.Linear(
        d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
    )
    self.w_value = nn.Linear(
        d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
    )

    self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    if qk_norm:
      self.q_norm = RMSNorm(head_dim, eps=1e-6)
      self.k_norm = RMSNorm(head_dim, eps=1e-6)
    else:
      self.q_norm = self.k_norm = None

    self.register_buffer("cache_k", None, persistent=False)
    self.register_buffer("cache_v", None, persistent=False)

    self.ptr_current_pos = 0
    self.use_cache = use_cache

  def forward(self, x, cos, sin, mask):
    """TODO: Docstring."""

    b, num_tokens, _ = x.shape

    # Apply projections
    queries = self.w_query(x)  # (b, num_tokens, num_heads * head_dim)
    keys = self.w_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
    values = self.w_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

    # Reshape
    queries = queries.view(
        b, num_tokens, self.num_heads, self.head_dim
    ).transpose(1, 2)
    keys = keys.view(
        b, num_tokens, self.num_kv_groups, self.head_dim
    ).transpose(1, 2)
    values = values.view(
        b, num_tokens, self.num_kv_groups, self.head_dim
    ).transpose(1, 2)

    # Optional normalization
    if self.q_norm:
      queries = self.q_norm(queries)
    if self.k_norm:
      keys = self.k_norm(keys)

    # Apply RoPE
    current_cos = cos[self.ptr_current_pos : self.ptr_current_pos + num_tokens]
    current_sin = sin[self.ptr_current_pos : self.ptr_current_pos + num_tokens]
    queries = apply_rope(queries, current_cos, current_sin)
    keys = apply_rope(keys, current_cos, current_sin)

    # Expand K and V to match number of heads
    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    if self.use_cache:
      if self.cache_k is None:
        self.cache_k, self.cache_v = keys, values
      else:
        self.cache_k = torch.cat([self.cache_k, keys], dim=2)
        self.cache_v = torch.cat([self.cache_v, values], dim=2)
      keys, values = self.cache_k, self.cache_v

    # Attention
    attn_scores = queries @ keys.transpose(2, 3)

    num_tokens_q = queries.shape[-2]
    num_tokens_k = keys.shape[-2]
    if self.use_cache:
      mask_bool = mask[
          self.ptr_current_pos : self.ptr_current_pos + num_tokens_q,
          :num_tokens_k,
      ]
      self.ptr_current_pos += num_tokens_q
    else:
      mask_bool = mask[:num_tokens_q, :num_tokens_k]
    attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
    attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

    # TODO: Problems with float @ bf16.
    context = (
        (attn_weights @ values)
        .transpose(1, 2)
        .reshape(b, num_tokens, self.d_out)
    )
    return self.out_proj(context)

  def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0


class TransformerBlock(nn.Module):
  """TODO: Docstring."""

  def __init__(self, cfg):
    super().__init__()
    self.att = GroupedQueryAttention(
        d_in=cfg["emb_dim"],
        num_heads=cfg["n_heads"],
        head_dim=cfg["head_dim"],
        num_kv_groups=cfg["n_kv_groups"],
        qk_norm=cfg["qk_norm"],
        dtype=cfg["dtype"],
        context_length=cfg["context_length"],
        use_cache=cfg["use_cache"],
    )
    if "num_experts" in cfg:
      self.ff = MoEFeedForward(cfg)
    else:
      self.ff = FeedForward(cfg)
    self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
    self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

  def forward(self, x, cos, sin, mask):
    # Shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x, cos, sin, mask)  # Shape [batch_size, num_tokens, emb_size]
    x = x + shortcut  # Add the original input back

    # Shortcut connection for feed-forward block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = x + shortcut  # Add the original input back

    return x

  def reset_cache(self):
    self.att.reset_cache()


class Qwen3Model(nn.Module):
  """TODO: Docstring."""

  def __init__(self, cfg):
    super().__init__()

    # Main model parameters
    self.tok_emb = nn.Embedding(
        cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
    )

    # ModuleList since Sequential can only accept one input,
    # and we need `x, mask, cos, sin`
    self.trf_blocks = nn.ModuleList(
        [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )

    self.final_norm = RMSNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
    )

    # Reusuable utilities
    if cfg["head_dim"] is None:
      head_dim = cfg["emb_dim"] // cfg["n_heads"]
    else:
      head_dim = cfg["head_dim"]
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=cfg["rope_base"],
        context_length=cfg["context_length"],
    )
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
    self.register_buffer(
        "mask",
        torch.triu(
            torch.ones(
                cfg["context_length"], cfg["context_length"], dtype=torch.bool
            ),
            diagonal=1,
        ),
        persistent=False,
    )
    self.cfg = cfg

  def forward(self, in_idx):
    # Forward pass
    tok_embeds = self.tok_emb(in_idx)
    x = tok_embeds

    for block in self.trf_blocks:
      x = block(x, self.cos, self.sin, self.mask)

    x = self.final_norm(x)
    logits = self.out_head(x.to(self.cfg["dtype"]))
    return logits

  def reset_cache(self):
    for block in self.trf_blocks:
      block.reset_cache()
