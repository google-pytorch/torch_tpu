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

"""Multi-layer model to demonstrate FSDPv2 vs SimpleFSDP migration."""

import torch
from torch import nn


class FeedForwardBlock(nn.Module):
  """A 4-layer feed-forward network used to compare FSDPv2 and SimpleFSDP."""

  def __init__(
      self,
      in_features: int,
      hidden_features: int,
      out_features: int,
      bias: bool = True,
  ):
    super().__init__()
    self.layer1 = nn.Linear(in_features, hidden_features, bias=bias)
    self.layer2 = nn.Linear(hidden_features, hidden_features, bias=bias)
    self.layer3 = nn.Linear(hidden_features, hidden_features, bias=bias)
    self.layer4 = nn.Linear(hidden_features, out_features, bias=bias)
    self.act = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.act(self.layer1(x))
    x = self.act(self.layer2(x))
    x = self.act(self.layer3(x))
    x = self.layer4(x)
    return x
