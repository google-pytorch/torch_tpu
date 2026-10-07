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

"""Experimental collectives that gather or scatter along any tensor dimension.

These functions accept a `ProcessGroup` and call the registered
`torch.ops.tpu._all_gather_dim` and `torch.ops.tpu._reduce_scatter_dim` ops,
which identify the group by name.
Under `torch.compile` the group size and name are traced as constants, so the
compiled graph contains only the registered ops. Autograd is not supported.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch_tpu._internal.utils import annotations as _annotations

__all__ = [
    "all_gather_dim",
    "reduce_scatter_dim",
]


def _resolve_group(
    group: Optional[dist.ProcessGroup],
) -> tuple[int, str]:
  group = dist.group.WORLD if group is None else group
  if group is None:
    raise ValueError(
        "Default process group has not been initialized, please make sure to"
        " call init_process_group."
    )
  return group.size(), group.group_name


@_annotations.experimental(
    "torch.tpu.ops.all_gather_dim is experimental and subject to change."
)
def all_gather_dim(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
  """All-gathers `input` from every rank and concatenates along `dim`.

  Autograd is not supported: no backward is registered for this op, so
  gradients do not flow back to `input`.

  Args:
    input: The local TPU tensor. Must have at least one dimension.
    dim: The dimension to concatenate along. May be negative.
    group: The process group. Defaults to the world group.

  Returns:
    A tensor whose size along `dim` is `input.size(dim) * group.size()`.
  """
  group_size, group_name = _resolve_group(group)
  return torch.ops.tpu._all_gather_dim(  # pylint: disable=protected-access
      input, dim, group_size, group_name
  )


@_annotations.experimental(
    "torch.tpu.ops.reduce_scatter_dim is experimental and subject to change."
)
def reduce_scatter_dim(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
    reduce_op: str = "sum",
) -> torch.Tensor:
  """Reduces `input` across ranks and returns this rank's chunk along `dim`.

  Autograd is not supported: no backward is registered for this op, so
  gradients do not flow back to `input`.

  Args:
    input: The local TPU tensor. `input.size(dim)` must be divisible by the
      group size.
    dim: The dimension to scatter along. May be negative.
    group: The process group. Defaults to the world group.
    reduce_op: One of "sum", "avg", "product", "min", "max", "band", "bor", or
      "bxor". The bitwise ops require an integer dtype.

  Returns:
    A tensor whose size along `dim` is `input.size(dim) // group.size()`.
  """
  group_size, group_name = _resolve_group(group)
  return torch.ops.tpu._reduce_scatter_dim(  # pylint: disable=protected-access
      input, reduce_op, dim, group_size, group_name
  )
