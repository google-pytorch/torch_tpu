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

"""MoE implementation for Qwen3.5 model using XLA ragged dot op."""

from collections.abc import Callable
import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe


class RaggedMoeQwen35(torch.nn.Module):
  """Drop-in replacement for Qwen3_5MoeSparseMoeBlock using XLA ragged dot."""

  def __init__(
      self,
      config: Any,
      is_tensor_parallel: bool = False,
      ragged_dot_impl: (
          Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
          | None
      ) = None,
  ):
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = True

    self.is_tensor_parallel = is_tensor_parallel
    world_size = (
        torch.distributed.get_world_size()
        if is_tensor_parallel and torch.distributed.is_initialized()
        else 1
    )
    if config.moe_intermediate_size % world_size != 0:
      raise ValueError(
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

    # Shared Expert for Qwen3.5
    self.shared_expert = modeling_qwen3_5_moe.Qwen3_5MoeMLP(
        config, intermediate_size=config.shared_expert_intermediate_size
    )
    self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    if ragged_dot_impl is not None:
      self.ragged_dot_impl = ragged_dot_impl
    elif hasattr(torch.ops, "tpu") and hasattr(torch.ops.tpu, "ragged_dot"):
      self.ragged_dot_impl = torch.ops.tpu.ragged_dot
    else:
      self.ragged_dot_impl = None

  def forward(
      self, hidden_states: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        hidden_states.device.type == "cuda"
        and hasattr(torch, "_grouped_mm")
        and (
            self.ragged_dot_impl is None
            or self.ragged_dot_impl
            == getattr(getattr(torch.ops, "tpu", None), "ragged_dot", None)
        )
    ):

      def _cuda_ragged_dot(x, w, groups):
        offs = torch.cumsum(groups, dim=0).to(torch.int32)
        orig_dtype = x.dtype
        return torch._grouped_mm(
            x.to(torch.bfloat16), w.to(torch.bfloat16), offs=offs
        ).to(orig_dtype)

      ragged_dot_fn = _cuda_ragged_dot
    else:
      if self.ragged_dot_impl is None:
        raise RuntimeError(
            "torch.ops.tpu.ragged_dot is not available and no alternative"
            " ragged_dot_impl was provided."
        )
      ragged_dot_fn = self.ragged_dot_impl

    batch_size, sequence_length, hidden_size = hidden_states.shape

    # Fuse the sequence dimension into batch dimension
    batch_fused = batch_size * sequence_length
    h = hidden_states.view(batch_fused, hidden_size)  # [B*S, dm]

    # Shared Expert for Qwen3.5
    shared_out = self.shared_expert(h)
    shared_gate_score = F.sigmoid(self.shared_expert_gate(h))
    shared_out = shared_out * shared_gate_score

    router_logits = self.router(h)  # [B*S, E]
    router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

    selected_weights, selected_indices = torch.topk(
        router_weights, self.top_k, dim=-1
    )

    if self.norm_topk_prob:
      selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
    selected_weights = selected_weights.to(dtype=h.dtype)

    # Prepare shuffling to/from sorted (by expert).
    selected_indices = selected_indices.flatten()  # [(B*S)*K]
    sortidx = torch.argsort(selected_indices)
    reverse_sortidx = torch.argsort(sortidx)

    # Prepare group_sizes (without bincount)
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

    # Prepare input (h) into broadcasted and sorted for ragged matmul
    h_sparse = h.view(batch_fused, 1, hidden_size)  # [B, 1, dm]
    h_sparse = h_sparse.broadcast_to(
        batch_fused, self.top_k, hidden_size
    )  # [B, K, dm]
    h_sparse = h_sparse.reshape(-1, hidden_size)  #  [B*K, dm]
    h_sparse = h_sparse[sortidx, :]  # [B*K, dm] sorted by expert id

    # Apply SwiGLU MoE
    h_up = ragged_dot_fn(h_sparse, self.up, group_sizes)  # [B*K, df]
    h_gate = ragged_dot_fn(h_sparse, self.gate, group_sizes)
    h_sparse = h_up * F.silu(h_gate)  # [B*K, df]
    h_sparse = ragged_dot_fn(h_sparse, self.down, group_sizes)  # [B*K, dm]

    # Restore original order, and apply sum over selected experts
    h_sparse = h_sparse[reverse_sortidx, :].view(
        batch_fused, self.top_k, hidden_size
    )
    h_sparse = (
        h_sparse * selected_weights.view(batch_fused, self.top_k, 1)
    ).sum(dim=1)

    # AllReduce sparse expert output across TP ranks before adding shared expert.
    # Shared expert down_proj is a RowParallelLinear which already performs
    # an all-reduce internally.
    if self.is_tensor_parallel:
      torch.distributed.all_reduce(h_sparse)

    # Add shared expert output
    final_h = h_sparse + shared_out

    # Split back the sequence dimension
    return final_h.view(batch_size, sequence_length, hidden_size), router_logits
