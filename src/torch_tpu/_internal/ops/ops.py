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

"""TorchTPU custom operators exposed publicly under `torch.tpu.ops`."""

from collections.abc import Sequence

import torch
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.ops.optimization_barrier is experimental and subject to change."
)
def optimization_barrier(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
  """Inserts an XLA optimization barrier around `inputs`.

  The barrier lowers to `stablehlo.optimization_barrier`
  (https://openxla.org/stablehlo/spec#optimization_barrier). It is an identity
  operation on its operands, but the XLA compiler treats it as opaque: no
  computation is moved across it and computations on either side of it are not
  fused together or deduplicated (e.g. by common subexpression elimination).

  This is useful to control scheduling and memory usage, for example to force
  a recomputation in activation checkpointing to happen where it is placed
  rather than being merged with the original computation, or to prevent the
  compiler from fusing a numerically sensitive region with its neighbors.

  Example:
    a, b = torch.tpu.ops.optimization_barrier([a, b])

  Args:
    inputs: The TPU tensors to pass through the barrier. Every computation
      producing these tensors happens before the barrier, and every computation
      consuming the returned tensors happens after it.

  Returns:
    A list of tensors with the same values, shapes, dtypes and strides as
    `inputs`, in the same order. Use these tensors, rather than `inputs`, in
    subsequent computations so that they are ordered after the barrier.
  """
  return torch.ops.tpu._optimization_barrier(inputs)  # pylint: disable=protected-access


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "optimization_barrier",
    # go/keep-sorted end
]
