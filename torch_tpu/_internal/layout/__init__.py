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

"""Layout-pinned TPU tensor allocation.

This module exposes a single helper, :func:`allocate_with_layout`, which copies
a CPU torch.Tensor onto the TPU device backed by a PJRT buffer with an
explicit XLA layout.

Use this internal helper when a persistent `nn.Parameter` must match the
kernel's expected `entry_computation_layout` so that XLA can DCE the relayout
that would otherwise show up at runtime as `tt_jit_as_strided` /
`copy_bitcast_fusion` modules. This is not a general `.to("tpu")`
replacement; the resulting buffer should only be fed to consumers that expect
the requested layout.

Example:

    >>> import torch
    >>> from torch_tpu._internal.layout import allocate_with_layout
    >>> # GMM kernel expects f32[E,K,1,N] in layout {1,3,2,0}.
    >>> cpu = torch.zeros(20, 6144, 1, 1, dtype=torch.float32)
    >>> tpu = allocate_with_layout(cpu, minor_to_major=(1, 3, 2, 0))
    >>> tpu.device.type
    'tpu'
"""

from typing import Sequence

import torch

from torch_tpu._internal import layout_impl


def allocate_with_layout(
    cpu_tensor: torch.Tensor,
    *,
    minor_to_major: Sequence[int],
) -> torch.Tensor:
    """Allocate a TPU tensor backed by a PJRT buffer with the given XLA layout.

    Args:
      cpu_tensor: a CPU `torch.Tensor`. Its data is copied to TPU. Must be
          contiguous from the caller's perspective; non-contiguous inputs are
          forced through `.contiguous()` on the CPU side before transfer.
      minor_to_major: the XLA layout's minor-to-major axis order. The length
          must equal `cpu_tensor.dim()` and the entries must be a permutation
          of `range(cpu_tensor.dim())`. For an `f32[E,K,1,N]` tensor consumed
          by a kernel whose `entry_computation_layout` shows
          `{1,3,2,0:T(...)}`, pass `(1, 3, 2, 0)`.

    Returns:
      A TPU `torch.Tensor` whose underlying PJRT buffer has the requested XLA
      layout. The returned tensor's logical shape and dtype match
      `cpu_tensor`.
    """
    if not isinstance(cpu_tensor, torch.Tensor):
        raise TypeError(
            "allocate_with_layout: cpu_tensor must be a torch.Tensor, "
            f"got {type(cpu_tensor).__name__}.")
    if cpu_tensor.device.type != "cpu":
        raise ValueError(
            "allocate_with_layout: cpu_tensor must be on CPU; "
            f"got device {cpu_tensor.device}.")
    minor_to_major_list = [int(a) for a in minor_to_major]
    return layout_impl.allocate_with_layout(cpu_tensor, minor_to_major_list)


__all__ = ["allocate_with_layout"]
