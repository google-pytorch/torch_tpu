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

from collections.abc import Mapping, Set
import dataclasses
import os
import pathlib
from typing import ClassVar, Final

import immutabledict
from torch_tpu._internal.utils import annotations

# Google's PCI vendor ID (found in /sys/bus/pci/devices/*/vendor)
_GOOGLE_PCI_VENDOR_ID: Final[str] = "0x1ae0"

# Directory with an entry per PCI device of the host.
_PCI_DEVICES_DIR: Final[pathlib.Path] = pathlib.Path("/sys/bus/pci/devices")

# Directory with character device nodes for accelerators.
_DEV_DIR: Final[pathlib.Path] = pathlib.Path("/dev")

# Directory with a node per IOMMU group that VFIO exposes to user space.
_VFIO_DIR: Final[pathlib.Path] = pathlib.Path("/dev/vfio")

# TPU topology maps represent the logical mesh of TPU chips as (X, Y, Z)
# dimensions. For example, (2, 2, 1) indicates a 2x2x1 mesh of 4 chips.
# The keys are the number of TPU devices, and the values are the topology
# dimensions. Generations that expose each TensorCore of a chip as a separate
# device (TPU v7 and TPU v8i) have a trailing cores-per-chip dimension, e.g.
# (1, 2, 1, 2) for the 4 devices of a 1x2x1 mesh of 2 chips.
_V4_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        1: (1, 1, 1),
        2: (1, 2, 1),
        4: (2, 2, 1),
        8: (2, 2, 2),
    })
)
_V5P_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        1: (1, 1, 1),
        2: (1, 2, 1),
        4: (2, 2, 1),
        8: (2, 2, 2),
    })
)

_V5E_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        1: (1, 1, 1),
        4: (2, 2, 1),
        # TPU v5e (8-chip) topology differs between environments:
        # - Cloud TPU VMs: The physical hardware interconnect is wired as a 2D mesh
        #   (2x4):
        #   https://docs.cloud.google.com/tpu/docs/v5e#tpu-v5e-config
        # - Internal test environments: A virtual 3D cube mapping (2x2x2) is used to
        #   prevent coordinate collisions during local discovery in the absence of
        #   physical wrap cabling.
        # copybara:uncomment 8: (2, 2, 2),
        8: (2, 4, 1),  # copybara:comment(oss-only)
    })
)

_V6E_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        1: (1, 1, 1),
        4: (2, 2, 1),
        8: (2, 4, 1),
    })
)

_V7_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        2: (1, 1, 1, 2),
        4: (1, 2, 1, 2),
        8: (2, 2, 1, 2),
    })
)

_V8I_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        2: (1, 1, 1, 2),
        4: (2, 1, 1, 2),
        8: (4, 1, 1, 2),
    })
)

# TPU v8t has one TensorCore per chip, unlike TPU v7 and TPU v8i, so its
# topologies have no trailing cores-per-chip dimension.
_V8T_TOPOLOGY: Final[Mapping[int, tuple[int, ...]]] = (
    immutabledict.immutabledict({
        1: (1, 1, 1),
        2: (1, 2, 1),
        4: (2, 2, 1),
    })
)

# GPU device names
_NVIDIA_GPU_DEVICES: Final[Set[str]] = frozenset([
    "/dev/nvidia0",
    "/dev/nvidiactl",  # Docker/Kubernetes
    "/dev/dxg",  # WSL2
])


@dataclasses.dataclass(frozen=True)
class SparseCoreSpec:
  """SparseCore specs of a TPU generation.

  SparseCores are TPU processors specialized for sparse operations such as
  embedding lookups.

  Attributes:
    sparse_cores_per_device: Number of SparseCores per TPU device. On
      generations that expose each TensorCore of a chip as a separate device
      (e.g. TPU v7), this counts the SparseCores of one device, not of the chip.
    simd_width: SIMD width (number of vector lanes) of each SparseCore.
  """

  sparse_cores_per_device: int
  simd_width: int


@dataclasses.dataclass(frozen=True)
class TpuSpec:
  """Specs of a TPU generation that TorchTPU supports.

  TorchTPU doesn't run on TPU v2/v3 and older generations, so they have no
  constants on `TpuSpec`.

  Attributes:
    name: Human-readable name of the generation, as returned by
      `torch.tpu.get_device_properties().name`.
    devices_per_chip: Number of separately-addressable devices the chip exposes
      to the runtime, used as the divisor that turns a per-chip quantity into
      what one device sees. This is megacore-aware (not the physical TensorCore
      count): 1 for single-core chips (v5e, v6e, v8t) and megacore chips (v5p)
      that fuse their two physical cores into one addressable device, and 2 for
      dual-core chips without megacore (v7, v8i).
    hbm_bytes_per_chip: Physical HBM capacity per chip, in bytes. On chips that
      expose several devices (v7, v8i), this HBM is shared across the devices;
      it is reported per chip (the published physical capacity), not divided per
      exposed device.
    bf16_flops_per_chip: Peak dense bf16 compute per chip, in FLOP/s. Stored as
      the published per-chip figure (easy to check against the docs); the
      per-device value is derived via `devices_per_chip`.
    topologies: Topology dimensions of each number of devices that a host can
      use (see the TPU topology maps above).
    sparse_core_spec: SparseCore specs of the generation, or None if it has no
      SparseCores.
  """

  name: str
  devices_per_chip: int
  hbm_bytes_per_chip: int
  bf16_flops_per_chip: int
  topologies: Mapping[int, tuple[int, ...]]
  sparse_core_spec: SparseCoreSpec | None = None

  V4: ClassVar[TpuSpec]
  V5P: ClassVar[TpuSpec]
  V5E: ClassVar[TpuSpec]
  V6E: ClassVar[TpuSpec]
  V7: ClassVar[TpuSpec]
  V8I: ClassVar[TpuSpec]
  V8T: ClassVar[TpuSpec]

  @property
  def hbm_bytes_per_device(self) -> int:
    """Physical HBM capacity, in bytes, visible to a single device.

    The per-chip capacity divided by the chip's exposed device count — what one
    device can actually address (e.g. ~96 GB per device on a 192 GB v7 chip).
    The divisor is `devices_per_chip`, so this equals `hbm_bytes_per_chip`
    except where a generation exposes more than one device (v7, v8i).
    """
    return self.hbm_bytes_per_chip // self.devices_per_chip

  @property
  def bf16_flops_per_device(self) -> int:
    """Peak dense bf16 compute, in FLOP/s, for a single device.

    The published per-chip peak divided by the chip's exposed device count. This
    is the correct denominator for per-device MFU (each rank drives one exposed
    device). The divisor is `devices_per_chip`, so this equals
    `bf16_flops_per_chip` except where a generation exposes more than one device
    (v7, v8i).
    """
    return self.bf16_flops_per_chip // self.devices_per_chip

  def topology(self, world_size: int | None = None) -> tuple[int, ...]:
    """Returns the topology of TPU devices of this generation.

    Args:
      world_size: The number of devices to return the topology of. Defaults to
        the number of TPU devices that VFIO exposes on this host. Allows
        sub-slicing only on a single host, in supported topologies.

    Returns:
      The topology dimensions, like (2, 2, 1) or (1, 2, 1, 2).

    Raises:
      RuntimeError: If the generation has no topology for `world_size` devices.
    """
    if world_size is None:
      world_size, _ = _scan_pci_tpus()
    if world_size not in self.topologies:
      # torchtpu_agents' engine_compatibility_check.py matches this message.
      raise RuntimeError(f"No TPU topology found for count: {world_size}")
    return self.topologies[world_size]


# HBM and bf16 FLOP/s sources:
# - v5e, v5p, v6e, v7: https://cloud.google.com/tpu/docs/{v5e,v5p,v6e,tpu7x}
# - v8i, v8t: `hbm_capacity_bytes` and `bf16_ops_per_second` (per TensorCore)
#   times TensorCores per chip in
#   https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py
# SparseCore specs come from `SparseCoreInfo` in
# https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py.
# The C++ counterpart, `torch_tpu::TpuDeviceSupportsSparseCore()`, checks the
# device kind of the initialized PJRT device instead. It lists the generations
# without SparseCores, so unlike `TpuSpec`, it assumes that generations it
# doesn't know have SparseCores. Keep both in sync when adding TPU generations.
TpuSpec.V4 = TpuSpec(
    name="TPU v4",
    devices_per_chip=1,
    hbm_bytes_per_chip=32 * 1024**3,
    bf16_flops_per_chip=275 * 10**12,
    topologies=_V4_TOPOLOGY,
    sparse_core_spec=None,
)
TpuSpec.V5P = TpuSpec(
    name="TPU v5p",
    devices_per_chip=1,
    hbm_bytes_per_chip=95 * 1024**3,
    bf16_flops_per_chip=459 * 10**12,
    topologies=_V5P_TOPOLOGY,
    sparse_core_spec=SparseCoreSpec(sparse_cores_per_device=4, simd_width=8),
)
TpuSpec.V5E = TpuSpec(
    name="TPU v5e",
    devices_per_chip=1,
    hbm_bytes_per_chip=16 * 1024**3,
    bf16_flops_per_chip=197 * 10**12,
    topologies=_V5E_TOPOLOGY,
    sparse_core_spec=None,
)
TpuSpec.V6E = TpuSpec(
    name="TPU v6e",
    devices_per_chip=1,
    hbm_bytes_per_chip=32 * 1024**3,
    bf16_flops_per_chip=918 * 10**12,
    topologies=_V6E_TOPOLOGY,
    sparse_core_spec=SparseCoreSpec(sparse_cores_per_device=2, simd_width=8),
)
TpuSpec.V7 = TpuSpec(
    name="TPU v7",
    devices_per_chip=2,
    hbm_bytes_per_chip=192 * 1024**3,
    bf16_flops_per_chip=2307 * 10**12,
    topologies=_V7_TOPOLOGY,
    sparse_core_spec=SparseCoreSpec(sparse_cores_per_device=2, simd_width=16),
)
TpuSpec.V8I = TpuSpec(
    name="TPU v8i",
    devices_per_chip=2,
    hbm_bytes_per_chip=309 * 10**9,
    bf16_flops_per_chip=1101 * 10**12,
    topologies=_V8I_TOPOLOGY,
    sparse_core_spec=SparseCoreSpec(sparse_cores_per_device=1, simd_width=16),
)
TpuSpec.V8T = TpuSpec(
    name="TPU v8t",
    devices_per_chip=1,
    hbm_bytes_per_chip=231 * 10**9,
    bf16_flops_per_chip=9961 * 10**11,
    topologies=_V8T_TOPOLOGY,
    sparse_core_spec=SparseCoreSpec(sparse_cores_per_device=2, simd_width=16),
)

# The `TpuSpec` of each TPU PCI device ID. Verified IDs from
# util/platforminfo/pci_ids.h.
_TPU_SPEC_BY_PCI_DEVICE_ID: Final[Mapping[str, TpuSpec]] = (
    immutabledict.immutabledict({
        "0x005e": TpuSpec.V4,
        "0x0062": TpuSpec.V5P,
        "0x0063": TpuSpec.V5E,
        "0x006f": TpuSpec.V6E,
        "0x0076": TpuSpec.V7,
        "0x007c": TpuSpec.V8T,
        "0x0083": TpuSpec.V8I,
    })
)


def get_tpu_spec() -> TpuSpec | None:
  """Returns the specs of the attached TPU device.

  Scans PCI devices for a Google TPU to identify its generation. Does not
  initialize any TPU runtime — safe to call at import time and from concurrent
  processes.

  Returns:
    The `TpuSpec` of the attached TPU device, or None if no TPU generation that
    TorchTPU supports is attached.
  """
  try:
    for device_path in _PCI_DEVICES_DIR.iterdir():
      try:
        vendor_id = (device_path / "vendor").read_text().strip()
        if vendor_id != _GOOGLE_PCI_VENDOR_ID:
          continue
        device_id = (device_path / "device").read_text().strip()
        if device_id in _TPU_SPEC_BY_PCI_DEVICE_ID:
          return _TPU_SPEC_BY_PCI_DEVICE_ID[device_id]
      except (OSError, IOError):
        continue
  except (OSError, IOError):
    pass
  return None


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


def _scan_pci_tpus() -> tuple[int, TpuSpec | None]:
  """Scans PCI bus for TPU devices.

  Returns:
    A tuple of (count, tpu_spec). `count` is the number of TPU devices exposed
    via VFIO. `tpu_spec` is the `TpuSpec` of the first of them, or None if VFIO
    exposes no TPU device.
  """
  count = 0
  first_tpu_spec = None

  try:
    for device_path in _PCI_DEVICES_DIR.iterdir():
      try:
        vendor_id = (device_path / "vendor").read_text().strip()
        if vendor_id != _GOOGLE_PCI_VENDOR_ID:
          continue

        device_id = (device_path / "device").read_text().strip()

        tpu_spec = _TPU_SPEC_BY_PCI_DEVICE_ID.get(device_id)
        if tpu_spec is not None:
          # Some devices have multiple tensorcores sharing the same bus.
          # We count only those that are exposed via VFIO.
          try:
            group_id = (device_path / "iommu_group").readlink().name
            (_VFIO_DIR / group_id).stat()
          except (OSError, IOError):
            # Cloud TPU v4 VMs expose chips via /dev/accel* rather than /dev/vfio.
            if not any(_DEV_DIR.glob("accel*")):
              continue

          count += 1
          if first_tpu_spec is None:
            first_tpu_spec = tpu_spec
      except (OSError, IOError):
        continue
  except (OSError, IOError):
    pass

  return count, first_tpu_spec


@annotations.deprecated(
    version="0.2.1", reason="Use torch.accelerator.device_count() instead."
)
def get_tpu_device_count() -> int:
  """Returns how many physical TPU PCI devices are attached to this host.

  Does not initialize the PJRT runtime. Counts the PCI devices in
  `/sys/bus/pci/devices` with Google's vendor ID and a TPU device ID in
  `_TPU_SPEC_BY_PCI_DEVICE_ID` whose IOMMU group is exposed under `/dev/vfio`.
  Devices whose sysfs entries can't be read are skipped, and if
  `/sys/bus/pci/devices` can't be read at all, this returns 0. The count does
  NOT include:

  - TPUs whose PCI device ID isn't in `_TPU_SPEC_BY_PCI_DEVICE_ID`, e.g. future
    generations;
  - TPUs whose IOMMU group isn't exposed under `/dev/vfio` (or `/dev/accel*`
    on TPU v4), which is normal in containers that expose only the devices
    allocated to them.

  So 0 doesn't mean the host has no TPUs. It's also not the number of devices
  the PJRT runtime will use, which can be smaller.

  Deprecated: code outside TorchTPU should call
  `torch.accelerator.device_count()` instead, which returns this count until the
  TPU runtime is initialized, and the runtime's device count afterwards.

  Returns:
    The integer number of TPU devices detected.
  """
  count, _ = _scan_pci_tpus()
  return count


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
  count, tpu_spec = _scan_pci_tpus()
  if tpu_spec is None:
    return None
  topology = tpu_spec.topology(count if world_size is None else world_size)
  return ",".join(map(str, topology))


def has_nvidia_gpu() -> bool:
  """True if there's a visible nvidia gpu available on device, False otherwise."""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)
