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

"""Implementation of the `torch.tpu.distributed` properties and functions.

`torch_tpu._internal.public_api` installs the properties of
`DistributedModuleProperties` on the type of the `torch.tpu.distributed` module,
so that e.g. `torch.tpu.distributed.window_num_stripes = 8` invokes the setter.
"""

from collections.abc import Callable
from typing import Any

from lazy_importer import lazy_importer
import torch
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.utils import annotations

__lazy_modules__ = frozenset({"torch_tpu._internal.distributed.device_mesh"})

with lazy_importer():
  import torch_tpu._internal.distributed.device_mesh as device_mesh


def _get_backend_fn(name: str) -> Callable[..., Any]:
  """Returns the `_device_ops_backend` function `name`."""
  fn = getattr(_device_ops_backend, name, None)
  if fn is None:
    raise RuntimeError("TorchTPU device ops backend is not initialized.")
  return fn


class DistributedModuleProperties:
  """Properties of the `torch.tpu.distributed` module.

  The getters and setters receive the `torch.tpu.distributed` module as `self`.
  """

  @property
  @annotations.experimental(
      "torch.tpu.distributed.window_num_stripes is experimental and subject to"
      " change."
  )
  def window_num_stripes(self) -> int:
    """Maximum concurrent DMA stripes for WindowTpu peer transfers.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    del self  # Unused.
    return _get_backend_fn("_get_window_num_stripes")()

  @window_num_stripes.setter
  @annotations.experimental(
      "torch.tpu.distributed.window_num_stripes is experimental and subject to"
      " change."
  )
  def window_num_stripes(self, value: int) -> None:
    """Sets maximum concurrent DMA stripes for WindowTpu peer transfers.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      value: The maximum number of concurrent DMA stripes. Must be positive.
    """
    del self  # Unused.
    _get_backend_fn("_set_window_num_stripes")(value)

  @property
  @annotations.experimental(
      "torch.tpu.distributed.window_stripe_chunk_mb is experimental and"
      " subject to change."
  )
  def window_stripe_chunk_mb(self) -> int:
    """Minimum payload size in MB per parallel DMA stripe for WindowTpu.

    .. warning::
        This API is experimental and subject to change in future releases.
    """
    del self  # Unused.
    return _get_backend_fn("_get_window_stripe_chunk_mb")()

  @window_stripe_chunk_mb.setter
  @annotations.experimental(
      "torch.tpu.distributed.window_stripe_chunk_mb is experimental and"
      " subject to change."
  )
  def window_stripe_chunk_mb(self, value: int) -> None:
    """Sets minimum payload size in MB per parallel DMA stripe for WindowTpu.

    .. warning::
        This API is experimental and subject to change in future releases.

    Args:
      value: The minimum payload size in MB per DMA stripe. Must be positive.
    """
    del self  # Unused.
    _get_backend_fn("_set_window_stripe_chunk_mb")(value)


@annotations.experimental(
    "torch.tpu.distributed.topology_aware_mesh is experimental and subject to"
    " change."
)
def topology_aware_mesh(
    mesh_shape: tuple[int, ...],
    *,
    dcn_mesh_shape: tuple[int, ...] | None = None,
    allow_split_physical_axes: bool = False,
    contiguous_submeshes: bool = False,
) -> torch.Tensor:
  """Computes a topology-aware rank layout tensor.

  Returns an int64 tensor of global torch ranks with shape `mesh_shape`,
  mapping logically adjacent coordinate positions to physically adjacent TPU
  devices on the ICI fabric. Whether the job spans one slice or several is
  read off the devices, so the same call covers both.

  .. warning::
      This API is experimental and subject to change in future releases.

  Args:
    mesh_shape: Full logical mesh shape, ordered by increasing network
      intensity. Its product must equal the world size, on multi-slice jobs too.
    dcn_mesh_shape: Which logical axes are split across slices. Same rank as
      `mesh_shape`, product equal to the number of slices, and must divide
      `mesh_shape` elementwise. Inferred when None.
    allow_split_physical_axes: Permits splitting a physical axis across logical
      axes when required.
    contiguous_submeshes: Forwards to create_device_mesh.

  Returns:
    An int64 tensor of global torch ranks mapping the requested layout.
  """
  return device_mesh.topology_aware_mesh(
      "tpu",
      mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      allow_split_physical_axes=allow_split_physical_axes,
      contiguous_submeshes=contiguous_submeshes,
  )
