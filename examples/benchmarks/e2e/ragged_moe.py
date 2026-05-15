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

"""MoE implementation for Qwen3 model using XLA ragged dot op."""

import math

import torch
import torch.nn.functional as F


class RaggedMoeQwen3(torch.nn.Module):
  """Drop-in replacement for Qwen3MoeSparseMoeBlock using XLA ragged dot."""

  def __init__(self, config, is_tensor_parallel=False):
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = getattr(
        config,
        "num_experts_per_tok",
        getattr(config, "num_experts_per_token", 8),
    )
    self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

    self.is_tensor_parallel = is_tensor_parallel
    world_size = torch.distributed.get_world_size() if is_tensor_parallel else 1
    assert config.moe_intermediate_size % world_size == 0, (
        "For tensor parallel MoE, model moe_intermediate_size"
        f" ({config.moe_intermediate_size}) must be divisible by world_size"
        f" ({world_size})"
    )

    # Router
    self.router = torch.nn.Linear(
        config.hidden_size, config.num_experts, bias=False
    )

    self.up = torch.nn.Parameter(
        torch.randn(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size // world_size,
        )
        / math.sqrt(config.hidden_size)
    )

    self.gate = torch.nn.Parameter(
        torch.randn(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size // world_size,
        )
        / math.sqrt(config.hidden_size)
    )

    self.down = torch.nn.Parameter(
        torch.randn(
            config.num_experts,
            config.moe_intermediate_size // world_size,
            config.hidden_size,
        )
        / math.sqrt(config.moe_intermediate_size)
    )

    self.ragged_dot_impl = torch.ops.torch_tpu.ragged_dot

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = hidden_states.shape

    # Fuse the sequence dimension into batch dimension
    batch_fused = batch_size * sequence_length
    h = hidden_states.view(batch_fused, hidden_size)  # [B, dm]

    router_logits = self.router(h)  # [B, E]
    router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

    selected_weights, selected_indices = torch.topk(
        router_weights, self.top_k, dim=-1
    )  # [B, K]

    if self.norm_topk_prob:
      selected_weights /= selected_weights.sum(dim=-1, keepdim=True)  # [B, K]
    selected_weights = selected_weights.to(dtype=h.dtype)

    # Prepare shuffling to/from sorted (by expert).
    selected_indices = selected_indices.flatten()  # [B*K]
    sortidx = torch.argsort(selected_indices)  # [B*K]
    reverse_sortidx = torch.argsort(sortidx)  # [B*K]

    # Prepare group_sizes (without bincount)
    group_sizes = torch.zeros(
        self.num_experts, dtype=torch.int32, device=h.device
    )  # [E]
    group_sizes.scatter_add_(
        dim=0,
        index=selected_indices,
        src=torch.ones(
            batch_fused * self.top_k, dtype=torch.int32, device=h.device
        ),
    )

    # Alternative 1: using bincount (slightly slower):
    # group_sizes = torch.bincount(selected_indices, minlength=self.num_experts)
    # group_sizes = group_sizes.to(dtype=torch.int32)

    # Alternative 2: using F.one_hot + sum (slightly slower):
    # group_sizes = F.one_hot(selected_indices, num_classes=self.num_experts)
    # group_sizes = group_sizes.sum(dim=0).to(dtype=torch.int32)

    # Prepare input (h) into broadcasted and sorted for ragged matmul
    h = h.view(batch_fused, 1, hidden_size)  # [B, 1, dm]
    h = h.broadcast_to(batch_fused, self.top_k, hidden_size)  # [B, K, dm]
    h = h.reshape(-1, hidden_size)  #  [B*K, dm]
    h = h[sortidx, :]  # [B*K, dm] - sorted by expert id

    # Apply SwiGLU MoE
    h_up = self.ragged_dot_impl(h, self.up, group_sizes)  # [B*K, df]
    h_gate = self.ragged_dot_impl(h, self.gate, group_sizes)
    h = h_up * F.silu(h_gate)  # [B*K, df]
    h = self.ragged_dot_impl(h, self.down, group_sizes)  # [B*K, dm]

    # Restore original order, and apply sum over selected experts
    h = h[reverse_sortidx, :].view(batch_fused, self.top_k, hidden_size)
    h = (h * selected_weights.view(batch_fused, self.top_k, 1)).sum(dim=1)

    # AllReduce after downward projection and expert summation
    if self.is_tensor_parallel:
      torch.distributed.all_reduce(h)

    # Split back the sequence dimension
    return h.view(batch_size, sequence_length, hidden_size), router_logits
