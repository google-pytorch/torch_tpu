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

"""Utilities for hardware discovery and configuration.

This is for reading from the pcie devs, and then from that, determining the
generation of TPU that we have (and that we have a TPU at all). If the user
decides to call the wrapper to generate our two additional environment
variables, then we use this to decode their topology given the generation of the
device and the world size.

`TpuSpec` holds the specs of each TPU generation that TorchTPU supports, and
`get_tpu_spec()` returns the specs of the attached TPU.
"""

from __future__ import annotations

from collections.abc import Set
import os
from typing import Final

from torch_tpu._internal.utils import _hardware_backend
from torch_tpu._internal.utils import annotations

# GPU device names
_NVIDIA_GPU_DEVICES: Final[Set[str]] = frozenset([
    "/dev/nvidia0",
    "/dev/nvidiactl",  # Docker/Kubernetes
    "/dev/dxg",  # WSL2
])

SparseCoreSpec = _hardware_backend.SparseCoreSpec
TpuSpec = _hardware_backend.TpuSpec


def get_tpu_spec() -> TpuSpec | None:
  """Returns the specs of the attached TPU device.

  Scans PCI devices for a Google TPU to identify its generation. Does not
  initialize any TPU runtime — safe to call at import time and from concurrent
  processes.

  Returns:
    The `TpuSpec` of the attached TPU device, or None if no TPU generation that
    TorchTPU supports is attached.
  """
  return _hardware_backend.get_tpu_spec()


@annotations.deprecated(
    version="0.2.1",
    reason="Use torch.tpu.get_device_properties().name instead.",
)
def get_tpu_device_name() -> str:
  """Returns a human-readable name for the attached TPU device.

  Deprecated: code outside TorchTPU should use
  `torch.tpu.get_device_properties().name` instead, which raises a RuntimeError
  rather than returning "TPU" if no TPU generation that TorchTPU supports is
  attached.

  Like `get_tpu_spec()`, does not initialize any TPU runtime.

  Returns:
    The `TpuSpec.name` of the attached TPU device (e.g., "TPU v7"), or "TPU" if
    no TPU generation that TorchTPU supports is attached.
  """
  tpu_spec = get_tpu_spec()
  return "TPU" if tpu_spec is None else tpu_spec.name


@annotations.deprecated(
    version="0.2.1",
    reason=(
        "Use torch.tpu.get_device_properties().hbm_bytes_per_device instead."
    ),
)
def get_hbm_bytes_per_device() -> int | None:
  """Returns the physical HBM capacity, in bytes, of one attached TPU device.

  Deprecated: code outside TorchTPU should use
  `torch.tpu.get_device_properties().hbm_bytes_per_device` instead, which raises
  a RuntimeError rather than returning None if no TPU generation that TorchTPU
  supports is attached.

  Returns:
    The `TpuSpec.hbm_bytes_per_device` of the attached TPU device, or None if no
    TPU generation that TorchTPU supports is attached.
  """
  tpu_spec = get_tpu_spec()
  return None if tpu_spec is None else tpu_spec.hbm_bytes_per_device


@annotations.deprecated(
    version="0.2.1", reason="Use torch.accelerator.device_count() instead."
)
def get_tpu_device_count() -> int:
  """Returns how many TPU PCI devices on this host this process can access.

  Does not initialize the PJRT runtime. Counts the PCI devices in
  `/sys/bus/pci/devices` with Google's vendor ID and a TPU device ID in
  `TpuSpec.specs_by_pci_device_id()` whose IOMMU group is exposed under
  `/dev/vfio` as a character device that this process can read and write.
  Devices whose sysfs entries can't be read are skipped, and if
  `/sys/bus/pci/devices` can't be read at all, this returns 0. The count does
  NOT include:

  - TPUs whose PCI device ID isn't in `TpuSpec.specs_by_pci_device_id()`, e.g.
    future generations;
  - TPUs whose IOMMU group isn't exposed under `/dev/vfio`, which is normal in
    containers that expose only the VFIO groups allocated to them;
  - TPUs whose `/dev/vfio` path holds a placeholder rather than a device node
    this process can read and write, e.g. the empty, unreadable regular files
    that sandboxes put at the paths of TPUs not allocated to them;
  - TPUs bound to a non-VFIO driver, e.g. reached through `/dev/accel*`.

  So 0 doesn't mean the host has no TPUs. It can also differ from the number of
  devices the PJRT runtime will use, e.g. if other runtime settings or process
  bounds restrict them.

  Deprecated: code outside TorchTPU should call
  `torch.accelerator.device_count()` instead, which returns this count until the
  TPU runtime is initialized, and the runtime's device count afterwards.

  Returns:
    The integer number of TPU devices detected.
  """
  return _hardware_backend.get_tpu_device_count()


def get_visible_tpu_device_count() -> int:
  """Returns how many TPU devices this process may use, per the PCI devices.

  Like `get_tpu_device_count()`, but if `TPU_VISIBLE_DEVICES` is a
  comma-separated list of chip indices, the count is capped at the number of
  devices on that many chips, like `CUDA_VISIBLE_DEVICES` caps the GPU count.
  `TPU_VISIBLE_DEVICES` is read on every call. TorchTPU may rewrite it when it
  initializes the TPU runtime, so this is meant for use before then, e.g. by
  `torch.tpu.device_count()`.

  Returns:
    The integer number of TPU devices this process may use.
  """
  return _hardware_backend.get_visible_tpu_device_count()


@annotations.deprecated(
    version="0.2.1",
    reason="Use torch.tpu.get_device_properties().topology() instead.",
)
def get_tpu_topology(world_size: int | None = None) -> str | None:
  """Returns the topology string of the attached TPU devices.

  Deprecated: code outside TorchTPU should call
  `torch.tpu.get_device_properties().topology()` instead, which raises a
  RuntimeError rather than returning None if no TPU generation that TorchTPU
  supports is attached.

  Args:
    world_size: If provided, use this as the target device count for topology
      lookup instead of the auto-detected count. Allows sub-slicing only on
      single-host, in supported topologies.

  Returns:
    A topology string like "2,2,1", or None if VFIO exposes no TPU device.

  Raises:
    RuntimeError: If a TPU device is detected but no topology is found for the
      `target_count` (either `world_size` or the auto-detected count).
  """
  return _hardware_backend.get_tpu_topology(world_size)


def has_nvidia_gpu() -> bool:
  """True if there's a visible nvidia gpu available on device, False otherwise."""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)
