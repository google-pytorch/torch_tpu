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

"""Defines the collective-op registry used by TorchTPU FX logic."""

import torch

# pylint: disable=protected-access
COLLECTIVE_OPS = (
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.all_reduce_,
    torch.ops._c10d_functional.all_reduce_coalesced,
    torch.ops._c10d_functional.all_reduce_coalesced_,
    torch.ops._c10d_functional.all_gather_into_tensor,
    torch.ops._c10d_functional.all_gather_into_tensor_out,
    torch.ops._c10d_functional.all_gather_into_tensor_coalesced,
    torch.ops._c10d_functional.reduce_scatter_tensor,
    torch.ops._c10d_functional.reduce_scatter_tensor_out,
    torch.ops._c10d_functional.reduce_scatter_tensor_coalesced,
    torch.ops._c10d_functional.all_to_all_single,
    torch.ops._c10d_functional.broadcast,
    torch.ops._c10d_functional.broadcast_,
)
# pylint: enable=protected-access

# This module can be imported before TorchTPU's C++ schemas are registered, so
# match backend-local collectives by schema name instead of resolving
# torch.ops.tpu attributes at import time.
_TPU_COLLECTIVE_NAMES = frozenset(
    {
        "tpu::experimental_all_gather_dim",
        "tpu::experimental_reduce_scatter_dim",
    }
)


def is_collective_op(op: object) -> bool:
  """Returns whether an operator is a collective recognized by TorchTPU."""
  op = getattr(op, "overloadpacket", op)
  if op in COLLECTIVE_OPS:
    return True
  default = getattr(op, "default", None)
  schema = getattr(default, "_schema", None)
  return schema is not None and schema.name in _TPU_COLLECTIVE_NAMES
