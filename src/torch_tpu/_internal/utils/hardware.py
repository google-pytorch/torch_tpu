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
"""

from collections.abc import Mapping, Set
import enum
import os
import pathlib
from typing import Final

import immutabledict

# Google's PCI vendor ID (found in /sys/bus/pci/devices/*/vendor)
_GOOGLE_PCI_VENDOR_ID: Final[str] = "0x1ae0"

# TPU topology maps represent the logical mesh of TPU chips as (X, Y, Z)
# coordinates. For example, "2,2,1" indicates a 2x2x1 mesh of 4 chips.
# The keys are the number of chips, and the values are the topology strings.
_V4_TOPOLOGY: Final[Mapping[int, str]] = immutabledict.immutabledict({
    1: "1,1,1",
    2: "1,2,1",
    4: "2,2,1",
    8: "2,2,2",
})

_V5P_TOPOLOGY: Final[Mapping[int, str]] = immutabledict.immutabledict({
    1: "1,1,1",
    2: "1,2,1",
    4: "2,2,1",
    8: "2,2,2",
})

_V5E_TOPOLOGY: Final[Mapping[int, str]] = immutabledict.immutabledict({
    1: "1,1,1",
    4: "2,2,1",
    # TPU v5e (8-chip) topology differs between environments:
    # - Cloud TPU VMs: The physical hardware interconnect is wired as a 2D mesh
    #   (2x4):
    #   https://docs.cloud.google.com/tpu/docs/v5e#tpu-v5e-config
    # - Internal test environments: A virtual 3D cube mapping (2x2x2) is used to
    #   prevent coordinate collisions during local discovery in the absence of
    #   physical wrap cabling.
    # copybara:uncomment 8: "2,2,2",
    8: "2,4,1",  # copybara:comment(oss-only)
})

_V6E_TOPOLOGY: Final[Mapping[int, str]] = immutabledict.immutabledict({
    1: "1,1,1",
    4: "2,2,1",
    8: "2,4,1",
})

_V7_TOPOLOGY: Final[Mapping[int, str]] = immutabledict.immutabledict({
    2: "1,1,1,2",
    4: "1,2,1,2",
    8: "2,2,1,2",
})

# TPU_DEVICE_0083_TOPOLOGY

# Verified IDs from util/platforminfo/pci_ids.h
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID: Final[Mapping[str, Mapping[int, str]]] = (
    immutabledict.immutabledict({
        "0x005e": _V4_TOPOLOGY,  # TPU v4
        "0x0062": _V5P_TOPOLOGY,  # TPU v5p
        "0x0063": _V5E_TOPOLOGY,  # TPU v5e
        "0x006f": _V6E_TOPOLOGY,  # TPU v6e
        "0x0076": _V7_TOPOLOGY,  # TPU v7
        # TPU_DEVICE_0083_TOPOLOGY_BY_TPU_PCI_DEVICE_ID
    })
)

_PCI_DEVICE_ID_TO_NAME: Final[Mapping[str, str]] = immutabledict.immutabledict({
    "0x005e": "TPU v4",
    "0x0062": "TPU v5p",
    "0x0063": "TPU v5e",
    "0x006f": "TPU v6e",
    "0x0076": "TPU v7",
    # TPU_DEVICE_0083_PCI_DEVICE_ID_TO_NAME,
})

# GPU device names
_NVIDIA_GPU_DEVICES: Final[Set[str]] = frozenset([
    "/dev/nvidia0",
    "/dev/nvidiactl",  # Docker/Kubernetes
    "/dev/dxg",  # WSL2
])


class TpuVersion(enum.Enum):
  V4 = "TPU v4"
  V5P = "TPU v5p"
  V5E = "TPU v5e"
  V6E = "TPU v6e"
  V7 = "TPU v7"
  # TPU_DEVICE_0083_TPU_VERSION_ENUM
  UNKNOWN = "TPU"


def get_tpu_device_name() -> str:
  """Returns a human-readable name for the attached TPU device.

  Scans PCI devices for a Google TPU and returns a name like "TPU v7".
  Does not initialize any TPU runtime — safe to call at import time and from
  concurrent processes.

  Returns:
    A string containing the TPU name (e.g., "TPU v7") or "TPU" if no recognized
    device is found.
  """
  pci_devices = pathlib.Path("/sys/bus/pci/devices")
  try:
    for device_path in pci_devices.iterdir():
      try:
        vendor_id = (device_path / "vendor").read_text().strip()
        if vendor_id != _GOOGLE_PCI_VENDOR_ID:
          continue
        device_id = (device_path / "device").read_text().strip()
        if device_id in _PCI_DEVICE_ID_TO_NAME:
          return _PCI_DEVICE_ID_TO_NAME[device_id]
      except (OSError, IOError):
        continue
  except (OSError, IOError):
    pass
  return "TPU"


def get_tpu_version() -> TpuVersion:
  """Returns the TpuVersion enum of the attached TPU device.

  Scans PCI devices to identify the TPU generation and returns the
  corresponding TpuVersion enum.
  """
  name = get_tpu_device_name()
  try:
    return TpuVersion(name)
  except ValueError:
    return TpuVersion.UNKNOWN


# Physical HBM capacity per chip, in bytes, keyed by TPU generation. On
# multi-core chips (v7) this HBM is shared across the chip's TensorCores; it
# is reported per chip (the published physical capacity), not divided per
# exposed device.
# Sources: https://cloud.google.com/tpu/docs/{v4,v5e,v5p,v6e,tpu7x}
_HBM_BYTES_PER_CHIP: Final[Mapping[TpuVersion, int]] = (
    immutabledict.immutabledict({
        TpuVersion.V4: 32 * 1024**3,
        TpuVersion.V5E: 16 * 1024**3,
        TpuVersion.V5P: 95 * 1024**3,
        TpuVersion.V6E: 32 * 1024**3,
        TpuVersion.V7: 192 * 1024**3,
    })
)


def get_hbm_bytes_per_chip() -> int | None:
  """Physical HBM capacity per chip, in bytes, for the attached TPU.

  Returns:
    The physical HBM capacity per chip in bytes, or None if the generation is
    unknown or unrecognized.
  """
  return _HBM_BYTES_PER_CHIP.get(get_tpu_version())


# Exposed devices per chip: how many separately-addressable devices the chip
# presents to the runtime, used as the divisor that turns a per-chip quantity
# into what one device sees. This is deliberately NOT the physical TensorCore
# count. The default is 1, which covers both single-core chips (v5e, v6e) and
# "megacore" chips (v4, v5p) that fuse their two physical cores into a single
# addressable device. Only generations that expose MORE than one device are
# listed here: v7 splits its two cores into two devices (verified — one device
# reports ~96 GB of the 192 GB chip). The default of 1 for the other
# generations, and v7 = 2, are to be confirmed against Google before landing.
_DEVICES_PER_CHIP: Final[Mapping[TpuVersion, int]] = (
    immutabledict.immutabledict({
        TpuVersion.V7: 2,
    })
)

# Peak dense bf16 compute per chip, in FLOP/s, keyed by TPU generation. Stored
# as the published *per-chip* figure (easy to check against the docs); the
# per-device value is derived below via _DEVICES_PER_CHIP.
# Sources: https://cloud.google.com/tpu/docs/{v4,v5e,v5p,v6e,tpu7x}
_BF16_FLOPS_PER_CHIP: Final[Mapping[TpuVersion, int]] = (
    immutabledict.immutabledict({
        TpuVersion.V4: 275 * 10**12,
        TpuVersion.V5E: 197 * 10**12,
        TpuVersion.V5P: 459 * 10**12,
        TpuVersion.V6E: 918 * 10**12,
        TpuVersion.V7: 2307 * 10**12,
    })
)


def get_devices_per_chip() -> int | None:
  """Number of separately-addressable devices the attached chip exposes.

  This is megacore-aware (not the physical TensorCore count): 1 for single-core
  and megacore-fused generations, 2 for v7.

  Returns:
    The number of exposed devices per chip, or None if no recognized TPU is
    attached.
  """
  version = get_tpu_version()
  if version is TpuVersion.UNKNOWN:
    return None
  return _DEVICES_PER_CHIP.get(version, 1)


def get_bf16_flops_per_chip() -> int | None:
  """Peak dense bf16 compute per chip, in FLOP/s, for the attached TPU.

  Returns:
    The peak bf16 FLOP/s per chip, or None if the generation is unknown or
    unrecognized.
  """
  return _BF16_FLOPS_PER_CHIP.get(get_tpu_version())


def get_hbm_bytes_per_device() -> int | None:
  """Physical HBM capacity, in bytes, visible to a single device.

  The per-chip capacity divided by the chip's exposed device count — what one
  device can actually address (e.g. ~96 GB per device on a 192 GB v7 chip).
  The divisor defaults to 1, so this equals the per-chip value except where a
  generation is known to expose more (v7).

  Returns:
    The physical HBM capacity visible to a single device in bytes, or None if no
    recognized TPU is attached.
  """
  version = get_tpu_version()
  per_chip = _HBM_BYTES_PER_CHIP.get(version)
  if per_chip is None:
    return None
  return per_chip // _DEVICES_PER_CHIP.get(version, 1)


def get_bf16_flops_per_device() -> int | None:
  """Peak dense bf16 compute, in FLOP/s, for a single device.

  The published per-chip peak divided by the chip's exposed device count. This
  is the correct denominator for per-device MFU (each rank drives one exposed
  device). The divisor defaults to 1, so this equals the per-chip value except
  where a generation is known to expose more (v7).

  Returns:
    The peak dense bf16 compute in FLOP/s for a single device, or None if no
    recognized TPU is attached.
  """
  version = get_tpu_version()
  per_chip = _BF16_FLOPS_PER_CHIP.get(version)
  if per_chip is None:
    return None
  return per_chip // _DEVICES_PER_CHIP.get(version, 1)


def _scan_pci_tpus() -> tuple[int, Mapping[int, str] | None]:
  """Scans PCI bus for TPU devices.

  Returns:
    A tuple of (count, topology_map). `count` is the number of TPU devices
    exposed via VFIO. `topology_map` is the Mapping of chip counts to
    topology strings for the detected TPU generation.
  """
  count = 0
  topology_map = None

  pci_devices = pathlib.Path("/sys/bus/pci/devices")
  try:
    for device_path in pci_devices.iterdir():
      try:
        vendor_id = (device_path / "vendor").read_text().strip()
        if vendor_id != _GOOGLE_PCI_VENDOR_ID:
          continue

        device_id = (device_path / "device").read_text().strip()

        if device_id in _TOPOLOGY_BY_TPU_PCI_DEVICE_ID:
          # Some devices have multiple tensorcores sharing the same bus.
          # We count only those that are exposed via VFIO.
          try:
            group_id = (device_path / "iommu_group").readlink().name
            (pathlib.Path("/dev/vfio") / group_id).stat()
          except (OSError, IOError):
            continue

          count += 1
          # Use the first matching device's topology map.
          if topology_map is None:
            topology_map = _TOPOLOGY_BY_TPU_PCI_DEVICE_ID[device_id]
      except (OSError, IOError):
        continue
  except (OSError, IOError):
    pass

  return count, topology_map


def get_tpu_device_count() -> int:
  """Returns the number of TPU devices found on the system.

  Returns:
    The integer number of TPU devices detected.
  """
  count, _ = _scan_pci_tpus()
  return count


def get_tpu_topology(world_size: int | None = None) -> str | None:
  """Returns the topology string and device count for attached TPU devices.

  Args:
    world_size: If provided, use this as the target device count for topology
      lookup instead of the auto-detected count. Allows sub-slicing only on
      single-host, in supported topologies.

  Returns:
    A topology string like "2,2,1" or None if no TPU is found.

  Raises:
    RuntimeError: If a TPU device is detected but no topology is found for the
      `target_count` (either `world_size` or the auto-detected count).
  """
  count, topology_map = _scan_pci_tpus()

  if topology_map:
    target_size = world_size if world_size is not None else count
    if target_size in topology_map:
      return topology_map[target_size]
    else:
      raise RuntimeError(f"No TPU topology found for count: {target_size}")

  return None


def has_nvidia_gpu() -> bool:
  """True if there's a visible nvidia gpu available on device, False otherwise."""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)
