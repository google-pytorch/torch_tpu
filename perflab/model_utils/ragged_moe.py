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

"""MoE implementations for Qwen3.5 and Gemma 4 models using XLA ragged dot op."""

from collections.abc import Callable
import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers import activations
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe


def _cuda_ragged_dot(
    x: torch.Tensor, w: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
  """CUDA fallback implementation of ragged dot using `torch._grouped_mm`."""
  offs = torch.cumsum(groups, dim=0).to(torch.int32)
  return torch._grouped_mm(
      x.to(torch.bfloat16), w.to(torch.bfloat16), offs=offs
  ).to(x.dtype)


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
    elif hasattr(torch, "tpu") and hasattr(torch.tpu, "ops"):
      self.ragged_dot_impl = torch.tpu.ops.ragged_dot
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
            == getattr(
                getattr(getattr(torch, "tpu", None), "ops", None),
                "ragged_dot",
                None,
            )
        )
    ):
      ragged_dot_fn = _cuda_ragged_dot
    else:
      if self.ragged_dot_impl is None:
        raise RuntimeError(
            "torch.tpu.ops.ragged_dot is not available and no alternative"
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


class RaggedExpertsGemma4(torch.nn.Module):
  """Drop-in replacement for Gemma4TextExperts using XLA ragged dot.

  `Gemma4TextDecoderLayer` routes the tokens itself and calls the experts with
  the flattened hidden states and the top-k expert indices and weights.
  """

  def __init__(
      self,
      config: Any,
      ragged_dot_impl: (
          Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
          | None
      ) = None,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = config.top_k_experts
    self.hidden_size = config.hidden_size
    self.moe_intermediate_size = config.moe_intermediate_size

    self.up = torch.nn.Parameter(
        torch.randn(
            self.num_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            dtype=dtype,
        )
        / math.sqrt(self.hidden_size)
    )
    self.gate = torch.nn.Parameter(
        torch.randn(
            self.num_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            dtype=dtype,
        )
        / math.sqrt(self.hidden_size)
    )
    self.down = torch.nn.Parameter(
        torch.randn(
            self.num_experts,
            self.moe_intermediate_size,
            self.hidden_size,
            dtype=dtype,
        )
        / math.sqrt(self.moe_intermediate_size)
    )
    self.act_fn = activations.ACT2FN[config.hidden_activation]

    if ragged_dot_impl is not None:
      self.ragged_dot_impl = ragged_dot_impl
    elif hasattr(torch, "tpu") and hasattr(torch.tpu, "ops"):
      self.ragged_dot_impl = torch.tpu.ops.ragged_dot
    else:
      self.ragged_dot_impl = None

  def _ragged_dot_fn(
      self, device_type: str
  ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns the ragged dot implementation for `device_type`."""
    tpu_ragged_dot = getattr(
        getattr(getattr(torch, "tpu", None), "ops", None), "ragged_dot", None
    )
    if (
        device_type == "cuda"
        and hasattr(torch, "_grouped_mm")
        and (
            self.ragged_dot_impl is None
            or self.ragged_dot_impl == tpu_ragged_dot
        )
    ):
      return _cuda_ragged_dot
    if self.ragged_dot_impl is None:
      raise RuntimeError(
          "torch.tpu.ops.ragged_dot is not available and no alternative"
          " ragged_dot_impl was provided."
      )
    return self.ragged_dot_impl

  def forward(
      self,
      hidden_states: torch.Tensor,
      top_k_index: torch.Tensor,
      top_k_weights: torch.Tensor,
  ) -> torch.Tensor:
    ragged_dot_fn = self._ragged_dot_fn(hidden_states.device.type)
    num_tokens, hidden_size = hidden_states.shape

    top_k_weights = top_k_weights.to(dtype=hidden_states.dtype)
    selected_indices = top_k_index.flatten()  # [T*K]
    sortidx = torch.argsort(selected_indices)  # [T*K]
    reverse_sortidx = torch.argsort(sortidx)  # [T*K]

    # Prepare group_sizes (without bincount)
    group_sizes = torch.zeros(
        self.num_experts, dtype=torch.int32, device=hidden_states.device
    )  # [E]
    group_sizes.scatter_add_(
        dim=0,
        index=selected_indices,
        src=torch.ones(
            num_tokens * self.top_k,
            dtype=torch.int32,
            device=hidden_states.device,
        ),
    )

    # Broadcast each token to its top-k experts and sort by expert id.
    h = hidden_states.view(num_tokens, 1, hidden_size)  # [T, 1, dm]
    h = h.broadcast_to(num_tokens, self.top_k, hidden_size)  # [T, K, dm]
    h = h.reshape(-1, hidden_size)  # [T*K, dm]
    h = h[sortidx, :]  # [T*K, dm] sorted by expert id

    h_up = ragged_dot_fn(h, self.up, group_sizes)  # [T*K, df]
    h_gate = ragged_dot_fn(h, self.gate, group_sizes)  # [T*K, df]
    h = h_up * self.act_fn(h_gate)  # [T*K, df]
    h = ragged_dot_fn(h, self.down, group_sizes)  # [T*K, dm]

    # Restore the original order and sum over the selected experts.
    h = h[reverse_sortidx, :].view(num_tokens, self.top_k, hidden_size)
    return (h * top_k_weights.view(num_tokens, self.top_k, 1)).sum(dim=1)
