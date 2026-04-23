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

"""A few ways to implement LayerNorm to validate mean() and var().

This is WIP. At time of first checkin, mean() and var() were not
implemented yet.
"""

import random
import sys

from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.utils import utils


class LayerNormManual(nn.Module):
  """Manual implementation of LayerNorm with explicit mean() and var() calls."""

  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift


class LayerNormTorch(nn.Module):
  """Torch-based implementation of LayerNorm."""

  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
    self.ln = nn.LayerNorm(emb_dim)

  def forward(self, x):
    return self.ln(x)


_DIM = flags.DEFINE_integer("dim", 1024, "Embeddings Dimension.")


def main(argv):
  del argv  # Unused.
  emb_dim = _DIM.value
  print("Layer Norm on vector of size {emb_dim}.", file=sys.stderr)

  input_ = torch.rand(emb_dim)

  model1 = LayerNormManual(emb_dim=emb_dim).eval()
  output1 = model1(input_)
  model2 = LayerNormTorch(emb_dim=emb_dim).eval()
  output2 = model2(input_)
  utils.assert_close(output1, output2, rtol=1e-3, atol=1e-5)

  try:
    device = torch.device("tpu")
    model1.to(device)
    output_tpu = model1(input_.to(device)).to("cpu")
    utils.assert_close(output_tpu, output1, rtol=1e-3, atol=1e-5)
  except Exception as e:
    print("TPU model not fully supported yet.", e)


if __name__ == "__main__":
  # Set a fixed random seed to avoid flakes.
  random.seed(42)
  torch.manual_seed(42)
  app.run(main)
