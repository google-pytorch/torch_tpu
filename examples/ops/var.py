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

"""Implement var() in several ways and compare.

We can implement var() in several ways:
  - simply use torch.var()
  - compute it explicitly utilizing other functions, such as the mean()
  - we can run these on CPU and TPU and compare numerics

Ideally, performance and resource constraints of the manual version will
match the highly optimized torch.var() function. This is something we
can evaluate with this code later.

We also use this code to drive the development of the missing TPU ops:
  mean()           (done)
  div.out()
  var.correction()
"""

import sys

from absl import app
from absl import flags
import torch
from torch_tpu._internal.utils import utils


class VarManual(torch.nn.Module):
  """Manual implementation of var()."""

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.sum((x - x.mean()) ** 2) / (x.numel() - 1)


class VarTorch(torch.nn.Module):
  """Torch-based implementation of var()."""

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.var(x)


_DIM = flags.DEFINE_integer("dim", 128, "Size of cube dimension.")


def main(argv):
  size_1d = _DIM.value
  print(f"var() on cubic tensor of dimension {size_1d}**3.", file=sys.stderr)

  torch.manual_seed(0)
  tensor_in = torch.rand(size_1d, size_1d, size_1d)

  # CPU Comparisons.
  out_torch = VarTorch().eval()(tensor_in)
  out_manual = VarManual().eval()(tensor_in)
  utils.assert_close(out_torch, out_manual, rtol=1e-3, atol=1e-5)

  # TPU Comparisons.
  tpu = torch.accelerator.current_accelerator()

  def run_model(model: torch.nn.Module, tensor_in: torch.Tensor, desc: str):
    print(desc, file=sys.stderr)
    try:
      tpu_model = model.to(tpu)
      out_tpu = tpu_model(tensor_in.to(tpu)).to("cpu")
      utils.assert_close(out_tpu, out_torch, rtol=1e-3, atol=1e-5)
    except NotImplementedError as ne:
      print("*** TPU model not working yet.***")
      print(ne)

  run_model(VarTorch(), tensor_in, "TPU: Torch model")
  run_model(VarManual(), tensor_in, "TPU: Manual model")


if __name__ == "__main__":
  app.run(main)
