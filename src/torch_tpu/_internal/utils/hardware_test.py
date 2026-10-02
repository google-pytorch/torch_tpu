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

import dataclasses
import pathlib
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu._internal.utils import hardware
from tests import seed_test_utils

_GOOGLE_VENDOR_ID = "0x1ae0"


@dataclasses.dataclass(frozen=True)
class _FakePciDevice:
  """A PCI device entry to materialize in the fake sysfs tree."""

  addr: str
  device_id: str
  vendor_id: str = _GOOGLE_VENDOR_ID
  # IOMMU group id, or None if the device has no iommu_group link.
  iommu_group: str | None = None
  # Whether /dev/vfio/<iommu_group> exists.
  has_vfio_node: bool = True


class _FakeSysfsTestCase(seed_test_utils.RepeatableTest):
  """Points hardware.py at a fake sysfs/devfs tree in a temp directory."""

  def setUp(self):
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self._pci_dir = root / "sys" / "bus" / "pci" / "devices"
    self._iommu_groups_dir = root / "sys" / "kernel" / "iommu_groups"
    self._vfio_dir = root / "dev" / "vfio"
    for d in (self._pci_dir, self._iommu_groups_dir, self._vfio_dir):
      d.mkdir(parents=True)
    self.enter_context(
        mock.patch.object(hardware, "_PCI_DEVICES_DIR", self._pci_dir)
    )
    self.enter_context(mock.patch.object(hardware, "_VFIO_DIR", self._vfio_dir))

  def _add_devices(self, devices: list[_FakePciDevice]) -> None:
    for d in devices:
      dev_dir = self._pci_dir / d.addr
      dev_dir.mkdir()
      (dev_dir / "vendor").write_text(f"{d.vendor_id}\n")
      (dev_dir / "device").write_text(f"{d.device_id}\n")
      if d.iommu_group is not None:
        group_dir = self._iommu_groups_dir / d.iommu_group
        group_dir.mkdir(exist_ok=True)
        (dev_dir / "iommu_group").symlink_to(group_dir)
        if d.has_vfio_node:
          (self._vfio_dir / d.iommu_group).touch()


class TpuTopologyTest(_FakeSysfsTestCase):

  def test_v7_8_chips(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:{bus}:01.{fn}",
            device_id="0x0076",
            iommu_group=str(i * 2 + int(fn)),
        )
        for i, bus in enumerate(["16", "38", "98", "b8"])
        for fn in ["0", "1"]
    ])
    self.assertEqual(hardware.get_tpu_topology(), hardware._V7_TOPOLOGY[8])

  def test_v5e_8_chips(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:03:01.{i}", device_id="0x0063", iommu_group=str(146 + i)
        )
        for i in range(8)
    ])
    self.assertEqual(hardware.get_tpu_topology(), hardware._V5E_TOPOLOGY[8])

  def test_v5p_4_chips(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:05:01.{i}", device_id="0x0062", iommu_group=str(439 + i)
        )
        for i in range(4)
    ])
    self.assertEqual(hardware.get_tpu_topology(), hardware._V5P_TOPOLOGY[4])

  def test_v6e_8_chips(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:41:01.{i}", device_id="0x006f", iommu_group=str(131 + i)
        )
        for i in range(8)
    ])
    self.assertEqual(hardware.get_tpu_topology(), hardware._V6E_TOPOLOGY[8])

  def test_devices_without_vfio_node_are_not_counted(self):
    # Eight v6e functions on the bus, but only four are exposed via VFIO.
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:41:01.{i}",
            device_id="0x006f",
            iommu_group=str(131 + i),
            has_vfio_node=i % 2 == 0,
        )
        for i in range(8)
    ])
    self.assertEqual(hardware.get_tpu_device_count(), 4)
    self.assertEqual(hardware.get_tpu_topology(), hardware._V6E_TOPOLOGY[4])

  def test_devices_without_iommu_group_are_not_counted(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:05:01.{i}",
            device_id="0x0062",
            iommu_group=str(439 + i) if i < 2 else None,
        )
        for i in range(4)
    ])
    self.assertEqual(hardware.get_tpu_device_count(), 2)
    self.assertEqual(hardware.get_tpu_topology(), hardware._V5P_TOPOLOGY[2])

  def test_non_google_and_unknown_devices_are_ignored(self):
    self._add_devices([
        # Non-Google vendor reusing a TPU device id.
        _FakePciDevice(
            addr="0000:00:01.0",
            device_id="0x006f",
            vendor_id="0x8086",
            iommu_group="1",
        ),
        # Google device that is not a known TPU.
        _FakePciDevice(
            addr="0000:00:02.0", device_id="0xffff", iommu_group="2"
        ),
        _FakePciDevice(
            addr="0000:41:01.0", device_id="0x006f", iommu_group="3"
        ),
    ])
    self.assertEqual(hardware.get_tpu_device_count(), 1)
    self.assertEqual(hardware.get_tpu_topology(), hardware._V6E_TOPOLOGY[1])

  def test_world_size_overrides_detected_count(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:41:01.{i}", device_id="0x006f", iommu_group=str(131 + i)
        )
        for i in range(8)
    ])
    self.assertEqual(
        hardware.get_tpu_topology(world_size=4), hardware._V6E_TOPOLOGY[4]
    )

  def test_unsupported_count_raises(self):
    self._add_devices([
        _FakePciDevice(
            addr=f"0000:41:01.{i}", device_id="0x006f", iommu_group=str(131 + i)
        )
        for i in range(3)
    ])
    with self.assertRaisesRegex(RuntimeError, "No TPU topology found"):
      hardware.get_tpu_topology()

  def test_no_tpu_returns_none(self):
    self.assertIsNone(hardware.get_tpu_topology())
    self.assertEqual(hardware.get_tpu_device_count(), 0)


class NvidiaGpuTest(seed_test_utils.RepeatableTest):
  """Tests has_nvidia_gpu against fake device nodes in a temp directory."""

  def setUp(self):
    super().setUp()
    self._dev_dir = pathlib.Path(self.create_tempdir().full_path)
    self.enter_context(
        mock.patch.object(
            hardware,
            "_NVIDIA_GPU_DEVICES",
            frozenset(
                str(self._dev_dir / name)
                for name in ("nvidia0", "nvidiactl", "dxg")
            ),
        )
    )

  @parameterized.parameters("nvidia0", "nvidiactl", "dxg")
  def test_has_nvidia_gpu_true(self, device_name):
    (self._dev_dir / device_name).touch()
    self.assertTrue(hardware.has_nvidia_gpu())

  def test_has_nvidia_gpu_false(self):
    (self._dev_dir / "unrelated_device").touch()
    self.assertFalse(hardware.has_nvidia_gpu())


class DeviceCapabilityTest(_FakeSysfsTestCase):

  def test_v7_name_flops_memory(self):
    self._add_devices([_FakePciDevice(addr="0000:16:01.0", device_id="0x0076")])
    # PCI device id -> name -> version mapping
    self.assertEqual(hardware.get_tpu_device_name(), "TPU v7")
    self.assertEqual(hardware.get_tpu_version(), hardware.TpuVersion.V7)
    # v7 splits its two cores into two separately-addressable devices
    self.assertEqual(hardware.get_devices_per_chip(), 2)
    # flops: 2307 TFLOP/s per chip, half of that per device
    self.assertEqual(hardware.get_bf16_flops_per_chip(), 2307 * 10**12)
    self.assertEqual(hardware.get_bf16_flops_per_device(), 2307 * 10**12 // 2)
    # memory: 192 GiB per chip, 96 GiB per device
    self.assertEqual(hardware.get_hbm_bytes_per_chip(), 192 * 1024**3)
    self.assertEqual(hardware.get_hbm_bytes_per_device(), 192 * 1024**3 // 2)

  def test_no_tpu_is_unknown(self):
    self.assertEqual(hardware.get_tpu_device_name(), "TPU")
    self.assertEqual(hardware.get_tpu_version(), hardware.TpuVersion.UNKNOWN)
    self.assertIsNone(hardware.get_devices_per_chip())
    self.assertIsNone(hardware.get_bf16_flops_per_device())
    self.assertIsNone(hardware.get_hbm_bytes_per_device())


if __name__ == "__main__":
  absltest.main()
