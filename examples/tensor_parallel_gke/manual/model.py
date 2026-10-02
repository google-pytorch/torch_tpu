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

"""Example of "manual" tensor parallelism.

This is a minimal example of tensor parallelism in `torch_tpu`, done in a way
that looks like simple/normal PyTorch.

'Manual' means that the user's model code explicitly calls the necessary
collective ops, as opposed to using something like DTensor, or another wrapper.
"""

import math

from torch import nn
import torch.distributed


class MyModel(nn.Module):
  """Simple example model with optional tensor parallelism."""

  def __init__(self, dmodel, use_tp=False):
    super().__init__()
    if use_tp:
      self.ff1 = FeedForwardTensorParallel(dmodel)
      self.ff2 = FeedForwardTensorParallel(dmodel)
      self.ff3 = FeedForwardTensorParallel(dmodel)
    else:
      self.ff1 = FeedForward(dmodel)
      self.ff2 = FeedForward(dmodel)
      self.ff3 = FeedForward(dmodel)

  def forward(self, x):
    x = x + self.ff1(x)
    x = x + self.ff2(x)
    x = x + self.ff3(x)
    return x


class FeedForward(nn.Module):
  """Non-distributed feed-forward layer."""

  def __init__(self, dmodel):
    super().__init__()
    self.w1 = MyLinear(dmodel, 4 * dmodel)
    self.act = nn.SiLU()
    self.w2 = MyLinear(4 * dmodel, dmodel)

  def forward(self, x):
    y = self.w1(x)
    y = self.act(y)
    y = self.w2(y)
    return y


class FeedForwardTensorParallel(nn.Module):
  """Tensor-parallel feed-forward layer."""

  def __init__(self, dmodel):
    super().__init__()
    self.w1 = ColumnParallelLinear(dmodel, 4 * dmodel)
    self.act = nn.SiLU()
    self.w2 = RowParallelLinear(4 * dmodel, dmodel)

  def forward(self, x):
    y = self.w1(x)
    y = self.act(y)  # Note: sharded computation in case of TP!
    y = self.w2(y)
    return y


class ColumnParallelLinear(nn.Module):
  """First part of a tensor-parallel feed-forward layer.

  This implements the FFN part of Megatron-LM paper,
  see https://arxiv.org/abs/1909.08053 for details.
  """

  def __init__(self, in_features, out_features):
    super().__init__()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert out_features % world_size == 0

    # Initialize distributed weights in a way equivalent to the non-TP version.
    w = torch.randn(in_features, out_features) / math.sqrt(in_features)
    stride = out_features // world_size
    w = w[..., rank * stride : (rank + 1) * stride].clone()

    self.w_sharded = MyLinear(
        in_features, out_features // world_size, init_weights=w
    )

  def forward(self, x):
    # NOTE: no collective ops on forward pass (but all-reduce on backward).
    return self.w_sharded(x)


class RowParallelLinear(nn.Module):
  """Second part of a tensor-parallel feed-forward layer.

  This implements the FFN part of Megatron-LM paper,
  see https://arxiv.org/abs/1909.08053 for details.
  """

  def __init__(self, in_features, out_features):
    super().__init__()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert out_features % world_size == 0

    # Initialize distributed weights in a way equivalent to the non-TP version.
    w = torch.randn(in_features, out_features) / math.sqrt(in_features)
    stride = in_features // world_size
    w = w[rank * stride : (rank + 1) * stride, ...].clone()

    self.w_sharded = MyLinear(
        in_features // world_size, out_features, init_weights=w
    )

  def forward(self, x):
    # NOTE: all-reduce on forward pass (but nothing on backward).
    x = self.w_sharded(x)
    torch.distributed.all_reduce(x, torch.distributed.ReduceOp.SUM)
    return x


class MyLinear(nn.Module):
  """A simple linear layer used by both regular and tensor-parallel versions."""

  def __init__(self, in_features, out_features, init_weights=None):
    super().__init__()

    if init_weights is None:
      w = torch.randn(in_features, out_features) / math.sqrt(in_features)
    else:
      w = init_weights

    self.w = nn.Parameter(w.requires_grad_(True))

  def forward(self, x):
    return x @ self.w
