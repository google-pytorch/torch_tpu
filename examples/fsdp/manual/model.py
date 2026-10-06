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

"""Manual Fully Sharded Data Parallel (FSDP) model and collective operations.

This module implements FSDP manually using explicit collective operations:
  1. Sharding: Each rank stores only a shard of each layer's parameters.
  2. Forward Pass: Parameters are all-gathered across ranks using
     `torch.distributed._functional_collectives.all_gather_tensor` to
     reconstruct the full layer parameters. Computation is performed using
     standard PyTorch operations (F.linear).
  3. Backward Pass: Autograd natively computes parameter and activation
     gradients. Functional collectives have adjoint formulas registered with
     PyTorch autograd, so all-gather in the forward pass automatically invokes
     the adjoint reduce-scatter in the backward pass, delivering sharded
     gradients directly to each rank without custom autograd functions.
"""

import math
from typing import Optional

import torch
from torch import distributed as dist
from torch import nn
import torch.distributed._functional_collectives as _func_col
import torch.nn.functional as F
from torch.utils import checkpoint as checkpoint_utils


class ManualFSDPLinear(nn.Module):
  """Linear layer implementing manual Fully Sharded Data Parallelism.

  The weight matrix (out_features, in_features) is sharded along dimension 0
  across all ranks, while the bias vector (out_features,) is kept unsharded.
  Collectives have adjoint formulas registered with PyTorch autograd,
  allowing all-gather in forward to automatically reduce-scatter in backward.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      group: Optional[dist.ProcessGroup] = None,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.group: dist.ProcessGroup = (
        group
        if group is not None
        else dist.distributed_c10d._get_default_group()
    )

    world_size = dist.get_world_size(self.group)
    if out_features % world_size != 0:
      raise ValueError(
          f"out_features ({out_features}) must be divisible by world_size"
          f" ({world_size})"
      )
    self.shard_out_features = out_features // world_size

    # Sharded weight parameter: shape (out_features // world_size, in_features)
    self.weight = nn.Parameter(
        torch.empty(self.shard_out_features, in_features)
    )
    # Unsharded bias parameter: shape (out_features,)
    self.bias = nn.Parameter(torch.empty(out_features))

    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Initializes weights with Kaiming uniform and biases to zero."""
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    nn.init.zeros_(self.bias)

  def get_full_weight(self) -> torch.Tensor:
    """Reconstructs the full weight tensor via all-gather across all ranks."""
    return _func_col.all_gather_tensor(
        self.weight, gather_dim=0, group=self.group
    )

  def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.get_full_weight(), self.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Wrap the layer in activation checkpointing so unsharded weights
    # are freed immediately after forward and re-all-gathered during backward.
    return checkpoint_utils.checkpoint(
        self._forward_impl,
        x,
        use_reentrant=False,
    )  # pyrefly: ignore[bad-return]


class ManualSimpleNN(nn.Module):
  """Two-layer model using manual FSDP linear layers."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      group: Optional[dist.ProcessGroup] = None,
  ):
    super().__init__()
    self.linear1 = ManualFSDPLinear(
        in_features=in_features,
        out_features=in_features,
        group=group,
    )
    self.silu = nn.SiLU()
    self.linear2 = ManualFSDPLinear(
        in_features=in_features,
        out_features=out_features,
        group=group,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear1(x)
    x = self.silu(x)
    x = self.linear2(x)
    return x


# Aliases for backward compatibility
Simple2LayerModel = ManualSimpleNN
ManualFSDP2LayerModel = ManualSimpleNN
