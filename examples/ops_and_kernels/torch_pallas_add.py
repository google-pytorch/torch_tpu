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

"""TorchTPU adapter for the Pallas addition kernel."""

import torch
from examples.ops_and_kernels import pallas_add

# Register the JAX function as a PyTorch custom operation.
pallas_add_op = torch.tpu.pallas.jax_op(
    "custom_pallas::add_vectors",
    pallas_add.add_vectors_jax,
)


def pallas_add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """User-facing PyTorch function that invokes the Pallas addition kernel."""
  return pallas_add_op(x, y)
