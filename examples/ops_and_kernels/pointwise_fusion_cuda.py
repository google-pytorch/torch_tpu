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

"""Example of a simple pointwise addition and ReLU fusion on CUDA."""

import os
import sys

# Set the flags to print generated code
os.environ["TORCH_COMPILE_LOGS"] = "output_code"
os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "0"

import torch


@torch.compile(backend="inductor")
def fwd(x, y):
  return torch.nn.functional.relu(torch.add(x, y))


def main():
  if not torch.cuda.is_available():
    print("CUDA not available. This test requires a GPU.")
    return 0

  x = torch.randn(1024, device="cuda")
  y = torch.randn(1024, device="cuda")

  fwd(x, y)

  print("Success.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
