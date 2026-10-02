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

"""Example model for tensor parallelism using PyTorch wrappers."""

import math

import torch
from torch import nn


class FeedForward(nn.Module):
  """Feed-forward layer using nn.Linear for compatibility with PyTorch TP wrappers."""

  def __init__(self, dmodel):
    super().__init__()
    self.w1 = nn.Linear(dmodel, 4 * dmodel, bias=False)
    self.act = nn.SiLU()
    self.w2 = nn.Linear(4 * dmodel, dmodel, bias=False)

    # Random initialization.
    with torch.no_grad():
      w1_init = torch.randn(dmodel, 4 * dmodel) / math.sqrt(dmodel)
      self.w1.weight.copy_(w1_init.t())
      w2_init = torch.randn(4 * dmodel, dmodel) / math.sqrt(4 * dmodel)
      self.w2.weight.copy_(w2_init.t())

  def forward(self, x):
    y = self.w1(x)
    y = self.act(y)
    y = self.w2(y)
    return y


class MyModel(nn.Module):
  """Model built with nn.Linear for PyTorch wrapper tensor parallelism."""

  def __init__(self, dmodel):
    super().__init__()
    self.ff1 = FeedForward(dmodel)
    self.ff2 = FeedForward(dmodel)
    self.ff3 = FeedForward(dmodel)

  def forward(self, x):
    x = x + self.ff1(x)
    x = x + self.ff2(x)
    x = x + self.ff3(x)
    return x
