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

from collections.abc import Callable, Sequence
import dataclasses
import math
import os
import pathlib
import types
from typing import Any
import unittest
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import hardware
from tests import op_testing
from tests import seed_test_utils


@dataclasses.dataclass(frozen=True)
class _FakePciDevice:
  """A PCI device in the fake sysfs that `_set_up_pci_fs()` creates.

  Attributes:
    address: PCI address, the name of the device's sysfs directory.
    device_id: Contents of the device's `device` file, or None to omit it.
    vendor_id: Contents of the device's `vendor` file, or None to omit it.
    iommu_group: IOMMU group that the device's `iommu_group` symlink points to,
      or None to omit the symlink.
    vfio_exposed: Whether `/dev/vfio` has a node for the IOMMU group.
  """

  address: str
  device_id: str | None
  vendor_id: str | None = "0x1ae0"  # Google
  iommu_group: str | None = None
  vfio_exposed: bool = True
  accel_exposed: bool = False


def _fake_tpus(
    device_id: str, num_chips: int, devices_per_chip: int = 1
) -> list[_FakePciDevice]:
  """Returns the VFIO-exposed PCI devices of `num_chips` TPU chips.

  Each chip has its own bus, with a PCI function and IOMMU group per device.

  Args:
    device_id: PCI device ID of the TPU generation.
    num_chips: Number of TPU chips.
    devices_per_chip: Number of devices that each chip exposes.
  """
  return [
      _FakePciDevice(
          address=f"0000:{0x10 + chip:02x}:01.{function}",
          device_id=device_id,
          iommu_group=str(devices_per_chip * chip + function),
      )
      for chip in range(num_chips)
      for function in range(devices_per_chip)
  ]


class TpuTopologyMockTest(seed_test_utils.RepeatableTest):
  """Tests TPU detection from the PCI devices in a fake sysfs."""

  @parameterized.parameters(
      (1, "1,1,1"),
      (2, "1,2,1"),
      (4, "2,2,1"),
      (8, "2,2,2"),
  )
  def test_v5p_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x0062", num_chips),
        expected_topology,
        expected_device_count=num_chips,
        expected_device_name="TPU v5p",
        expected_spec=hardware.TpuSpec.V5P,
    )

  @parameterized.parameters(
      (1, "1,1,1"),
      (4, "2,2,1"),
      # The 8-chip topology differs between internal and OSS builds.
      (8, ",".join(map(str, hardware._V5E_TOPOLOGY[8]))),
  )
  def test_v5e_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x0063", num_chips),
        expected_topology,
        expected_device_count=num_chips,
        expected_device_name="TPU v5e",
        expected_spec=hardware.TpuSpec.V5E,
    )

  @parameterized.parameters(
      (1, "1,1,1"),
      (4, "2,2,1"),
      (8, "2,4,1"),
  )
  def test_v6e_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x006f", num_chips),
        expected_topology,
        expected_device_count=num_chips,
        expected_device_name="TPU v6e",
        expected_spec=hardware.TpuSpec.V6E,
    )

  @parameterized.parameters(
      (1, "1,1,1,2"),
      (2, "1,2,1,2"),
      (4, "2,2,1,2"),
  )
  def test_v7_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x0076", num_chips, devices_per_chip=2),
        expected_topology,
        expected_device_count=2 * num_chips,
        expected_device_name="TPU v7",
        expected_spec=hardware.TpuSpec.V7,
    )

  @parameterized.parameters(
      (1, "1,1,1,2"),
      (2, "2,1,1,2"),
      (4, "4,1,1,2"),
  )
  def test_v8i_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x0083", num_chips, devices_per_chip=2),
        expected_topology,
        expected_device_count=2 * num_chips,
        expected_device_name="TPU v8i",
        expected_spec=hardware.TpuSpec.V8I,
    )

  @parameterized.parameters(
      (1, "1,1,1"),
      (2, "1,2,1"),
      (4, "2,2,1"),
  )
  def test_v8t_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x007c", num_chips),
        expected_topology,
        expected_device_count=num_chips,
        expected_device_name="TPU v8t",
        expected_spec=hardware.TpuSpec.V8T,
    )

  @parameterized.parameters(
      (1, "1,1,1"),
      (2, "1,2,1"),
      (4, "2,2,1"),
      (8, "2,2,2"),
  )
  def test_v4_chips(self, num_chips, expected_topology):
    self._run_mock_test(
        _fake_tpus("0x005e", num_chips),
        expected_topology,
        expected_device_count=num_chips,
        expected_device_name="TPU v4",
        expected_spec=hardware.TpuSpec.V4,
    )
  def test_v4_chips_with_accel_fallback(self):
    # On Cloud TPU v4 VMs, chips are exposed via /dev/accel* rather than VFIO.
    self._run_mock_test(
        [
            _FakePciDevice(
                address="0000:00:04.0",
                device_id="0x005e",
                iommu_group=None,
                accel_exposed=True,
            )
        ],
        "1,1,1",
        expected_device_count=1,
        expected_device_name="TPU v4",
        expected_spec=hardware.TpuSpec.V4,
    )
  def test_no_tpu_devices(self):
    self._run_mock_test(
        [],
        None,
        expected_device_count=0,
        expected_device_name="TPU",
        expected_spec=None,
    )
    self.assertIsNone(hardware.get_tpu_topology(world_size=4))

  def test_no_pci_devices_dir(self):
    missing_dir = pathlib.Path(self.create_tempdir().full_path) / "devices"
    self.enter_context(
        mock.patch.object(hardware, "_PCI_DEVICES_DIR", missing_dir)
    )

    self.assertIsNone(hardware.get_tpu_topology())
    self.assertEqual(hardware.get_tpu_device_count(), 0)
    self.assertEqual(hardware.get_tpu_device_name(), "TPU")
    self.assertIsNone(hardware.get_tpu_spec())
    self.assertIsNone(hardware.get_hbm_bytes_per_device())

  def test_counts_only_vfio_exposed_devices(self):
    self._run_mock_test(
        [
            *_fake_tpus("0x0076", 1, devices_per_chip=2),
            # A device whose IOMMU group VFIO doesn't expose.
            _FakePciDevice(
                address="0000:20:01.0",
                device_id="0x0076",
                iommu_group="20",
                vfio_exposed=False,
            ),
            # A device without an IOMMU group.
            _FakePciDevice(address="0000:20:01.1", device_id="0x0076"),
        ],
        "1,1,1,2",
        expected_device_count=2,
        expected_device_name="TPU v7",
        expected_spec=hardware.TpuSpec.V7,
    )

  def test_identifies_tpu_without_vfio_exposed_devices(self):
    # The TPU generation comes from the PCI device ID alone, while only devices
    # that VFIO exposes are counted.
    self._run_mock_test(
        [
            _FakePciDevice(
                address="0000:10:01.0",
                device_id="0x006f",
                iommu_group="0",
                vfio_exposed=False,
            )
        ],
        None,
        expected_device_count=0,
        expected_device_name="TPU v6e",
        expected_spec=hardware.TpuSpec.V6E,
    )

  def test_ignores_non_tpu_pci_devices(self):
    self._run_mock_test(
        [
            # A non-Google device whose device ID is the one of TPU v5p.
            _FakePciDevice(
                address="0000:00:01.0",
                device_id="0x0062",
                vendor_id="0x8086",
                iommu_group="101",
            ),
            # A Google gVNIC network device.
            _FakePciDevice(
                address="0000:00:02.0", device_id="0x0042", iommu_group="102"
            ),
            # A Google device without a `device` file.
            _FakePciDevice(
                address="0000:00:03.0", device_id=None, iommu_group="103"
            ),
            # A device without `vendor` and `device` files.
            _FakePciDevice(
                address="0000:00:04.0", device_id=None, vendor_id=None
            ),
            *_fake_tpus("0x0062", 1),
        ],
        "1,1,1",
        expected_device_count=1,
        expected_device_name="TPU v5p",
        expected_spec=hardware.TpuSpec.V5P,
    )

  @parameterized.parameters(
      (2, "1,1,1,2"),
      (4, "1,2,1,2"),
      (8, "2,2,1,2"),
  )
  def test_get_tpu_topology_with_world_size(self, world_size, expected):
    self._set_up_pci_fs(_fake_tpus("0x0076", 4, devices_per_chip=2))
    self.assertEqual(hardware.get_tpu_topology(world_size=world_size), expected)

  def test_get_tpu_topology_raises_for_unsupported_world_size(self):
    self._set_up_pci_fs(_fake_tpus("0x0076", 4, devices_per_chip=2))
    with self.assertRaisesRegex(
        RuntimeError, "No TPU topology found for count: 3"
    ):
      hardware.get_tpu_topology(world_size=3)

  def test_get_tpu_topology_raises_for_unsupported_device_count(self):
    self._set_up_pci_fs(_fake_tpus("0x0062", 3))
    with self.assertRaisesRegex(
        RuntimeError, "No TPU topology found for count: 3"
    ):
      hardware.get_tpu_topology()

  @parameterized.parameters(
      (2, (1, 1, 1, 2)),
      (4, (1, 2, 1, 2)),
      (8, (2, 2, 1, 2)),
  )
  def test_tpu_spec_topology_with_world_size(self, world_size, expected):
    self.assertEqual(hardware.TpuSpec.V7.topology(world_size), expected)

  def test_tpu_spec_topology_defaults_to_device_count(self):
    self._set_up_pci_fs(_fake_tpus("0x0076", 4, devices_per_chip=2))
    self.assertEqual(hardware.TpuSpec.V7.topology(), (2, 2, 1, 2))

  def test_tpu_spec_topology_raises_for_unsupported_world_size(self):
    with self.assertRaisesRegex(
        RuntimeError, "No TPU topology found for count: 3"
    ):
      hardware.TpuSpec.V7.topology(3)

  def _set_up_pci_fs(self, devices: Sequence[_FakePciDevice]) -> None:
    """Makes `hardware` scan a fake sysfs and devfs with the given devices."""
    root = pathlib.Path(self.create_tempdir().full_path)
    pci_devices_dir = root / "sys" / "bus" / "pci" / "devices"
    dev_dir = root / "dev"
    vfio_dir = dev_dir / "vfio"
    pci_devices_dir.mkdir(parents=True)
    vfio_dir.mkdir(parents=True)
    for device in devices:
      device_dir = pci_devices_dir / device.address
      device_dir.mkdir()
      if device.vendor_id is not None:
        (device_dir / "vendor").write_text(f"{device.vendor_id}\n")
      if device.device_id is not None:
        (device_dir / "device").write_text(f"{device.device_id}\n")
      if device.iommu_group is not None:
        (device_dir / "iommu_group").symlink_to(
            root / "sys" / "kernel" / "iommu_groups" / device.iommu_group
        )
        if device.vfio_exposed:
          (vfio_dir / device.iommu_group).touch()
      if device.accel_exposed:
        (dev_dir / "accel0").touch()

    self.enter_context(
        mock.patch.object(hardware, "_PCI_DEVICES_DIR", pci_devices_dir)
    )
    self.enter_context(mock.patch.object(hardware, "_DEV_DIR", dev_dir))
    self.enter_context(mock.patch.object(hardware, "_VFIO_DIR", vfio_dir))

  def _run_mock_test(
      self,
      devices: Sequence[_FakePciDevice],
      expected_topology: str | None,
      *,
      expected_device_count: int,
      expected_device_name: str,
      expected_spec: hardware.TpuSpec | None,
  ) -> None:
    self._set_up_pci_fs(devices)
    self.assertEqual(hardware.get_tpu_topology(), expected_topology)
    self.assertEqual(hardware.get_tpu_device_count(), expected_device_count)
    self.assertEqual(hardware.get_tpu_device_name(), expected_device_name)
    self.assertEqual(hardware.get_tpu_spec(), expected_spec)
    self.assertEqual(
        hardware.get_hbm_bytes_per_device(),
        None if expected_spec is None else expected_spec.hbm_bytes_per_device,
    )


def _reset_warn_once(api: Callable[..., Any]) -> None:
  """Resets the per-process one-time warning flag of an annotated function."""
  idx = api.__code__.co_freevars.index("_warned")
  api.__closure__[idx].cell_contents = False


class DeprecatedFunctionsTest(seed_test_utils.RepeatableTest):
  """Tests the functions that code outside TorchTPU should no longer call."""

  @parameterized.named_parameters(
      (
          "get_hbm_bytes_per_device",
          hardware.get_hbm_bytes_per_device,
          "torch.tpu.get_device_properties().hbm_bytes_per_device",
      ),
      (
          "get_tpu_device_count",
          hardware.get_tpu_device_count,
          "torch.accelerator.device_count()",
      ),
      (
          "get_tpu_device_name",
          hardware.get_tpu_device_name,
          "torch.tpu.get_device_properties().name",
      ),
      (
          "get_tpu_topology",
          hardware.get_tpu_topology,
          "torch.tpu.get_device_properties().topology()",
      ),
  )
  def test_deprecated_in_favor_of_public_api(self, function, replacement):
    self.assertEqual(
        getattr(function, annotations.TT_API_STAGE),
        annotations.Stage.DEPRECATED.value,
    )
    self.assertEqual(
        getattr(function, annotations.TT_API_DEPRECATED_VERSION), "0.2.1"
    )
    self.assertEqual(
        getattr(function, annotations.TT_API_STAGE_REASON),
        f"Use {replacement} instead.",
    )
    _reset_warn_once(function)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      function()
      function()
    self.assertLen(w, 1)
    self.assertTrue(issubclass(w[0].category, DeprecationWarning))
    self.assertEqual(
        str(w[0].message),
        f"<TorchTpu/ApiStage/DeprecatedApi/> {function.__name__} is deprecated"
        " as of TorchTPU 0.2.1 and will be removed in a future release: Use"
        f" {replacement} instead.",
    )


class NvidiaGpuMockTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      ("/dev/nvidia0",),
      ("/dev/nvidiactl",),
      ("/dev/dxg",),
  )
  @mock.patch.object(os.path, "exists")
  def test_has_nvidia_gpu_true(self, path_to_exist, mock_exists):
    mock_exists.side_effect = lambda path: path == path_to_exist
    self.assertTrue(hardware.has_nvidia_gpu())

  @mock.patch.object(os.path, "exists")
  def test_has_nvidia_gpu_false(self, mock_exists):
    mock_exists.return_value = False
    self.assertFalse(hardware.has_nvidia_gpu())


class DeviceCapabilityTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      (
          hardware.TpuSpec.V4,
          "TPU v4",
          1,
          32 * 1024**3,
          32 * 1024**3,
          275 * 10**12,
          275 * 10**12,
          None,
      ),
      (
          hardware.TpuSpec.V5P,
          "TPU v5p",
          1,
          95 * 1024**3,
          95 * 1024**3,
          459 * 10**12,
          459 * 10**12,
          hardware.SparseCoreSpec(sparse_cores_per_device=4, simd_width=8),
      ),
      (
          hardware.TpuSpec.V5E,
          "TPU v5e",
          1,
          16 * 1024**3,
          16 * 1024**3,
          197 * 10**12,
          197 * 10**12,
          None,
      ),
      (
          hardware.TpuSpec.V6E,
          "TPU v6e",
          1,
          32 * 1024**3,
          32 * 1024**3,
          918 * 10**12,
          918 * 10**12,
          hardware.SparseCoreSpec(sparse_cores_per_device=2, simd_width=8),
      ),
      (
          hardware.TpuSpec.V7,
          "TPU v7",
          2,
          192 * 1024**3,
          192 * 1024**3 // 2,
          2307 * 10**12,
          2307 * 10**12 // 2,
          hardware.SparseCoreSpec(sparse_cores_per_device=2, simd_width=16),
      ),
      (
          hardware.TpuSpec.V8I,
          "TPU v8i",
          2,
          309 * 10**9,
          309 * 10**9 // 2,
          1101 * 10**12,
          1101 * 10**12 // 2,
          hardware.SparseCoreSpec(sparse_cores_per_device=1, simd_width=16),
      ),
      (
          hardware.TpuSpec.V8T,
          "TPU v8t",
          1,
          231 * 10**9,
          231 * 10**9,
          9961 * 10**11,
          9961 * 10**11,
          hardware.SparseCoreSpec(sparse_cores_per_device=2, simd_width=16),
      ),
  )
  def test_tpu_spec(
      self,
      tpu_spec,
      expected_name,
      expected_devices_per_chip,
      expected_hbm_bytes_per_chip,
      expected_hbm_bytes_per_device,
      expected_bf16_flops_per_chip,
      expected_bf16_flops_per_device,
      expected_sparse_core_spec,
  ):
    self.assertEqual(tpu_spec.name, expected_name)
    self.assertEqual(tpu_spec.devices_per_chip, expected_devices_per_chip)
    self.assertEqual(tpu_spec.hbm_bytes_per_chip, expected_hbm_bytes_per_chip)
    self.assertEqual(
        tpu_spec.hbm_bytes_per_device, expected_hbm_bytes_per_device
    )
    self.assertEqual(tpu_spec.bf16_flops_per_chip, expected_bf16_flops_per_chip)
    self.assertEqual(
        tpu_spec.bf16_flops_per_device, expected_bf16_flops_per_device
    )
    self.assertEqual(tpu_spec.sparse_core_spec, expected_sparse_core_spec)

  def test_each_tpu_spec_has_pci_device_id_and_topologies(self):
    tpu_specs = [
        value
        for value in vars(hardware.TpuSpec).values()
        if isinstance(value, hardware.TpuSpec)
    ]
    self.assertCountEqual(
        hardware._TPU_SPEC_BY_PCI_DEVICE_ID.values(), tpu_specs
    )
    for tpu_spec in tpu_specs:
      with self.subTest(tpu=tpu_spec.name):
        self.assertNotEmpty(tpu_spec.topologies)

  def test_topologies_match_device_counts_and_devices_per_chip(self):
    for tpu_spec in hardware._TPU_SPEC_BY_PCI_DEVICE_ID.values():
      for num_devices, topology in tpu_spec.topologies.items():
        with self.subTest(tpu=tpu_spec.name, num_devices=num_devices):
          self.assertEqual(math.prod(topology), num_devices)
          # Only generations with several devices per chip have a trailing
          # cores-per-chip dimension.
          self.assertEqual(math.prod(topology[3:]), tpu_spec.devices_per_chip)

  @parameterized.parameters(
      (hardware.TpuSpec.V4, True),
      (hardware.TpuSpec.V5P, False),
      (hardware.TpuSpec.V5E, True),
      (hardware.TpuSpec.V6E, False),
      (hardware.TpuSpec.V7, False),
      (hardware.TpuSpec.V8I, False),
      (hardware.TpuSpec.V8T, False),
      (None, True),
  )
  def test_skip_if_sparsecore_unsupported(self, tpu_spec, expect_skip):
    # TorchTPU only has TPU devices of generations that it supports.
    has_tpu_devices = tpu_spec is not None
    with (
        mock.patch.object(
            torch.accelerator,
            "current_accelerator",
            return_value=(
                types.SimpleNamespace(type="tpu") if has_tpu_devices else None
            ),
        ),
        mock.patch.object(
            torch.accelerator, "is_available", return_value=has_tpu_devices
        ),
        mock.patch.object(hardware, "get_tpu_spec", return_value=tpu_spec),
    ):
      if expect_skip:
        with self.assertRaisesRegex(
            unittest.case.SkipTest,
            "SparseCore is not supported on this TPU device.",
        ):
          op_testing.TorchTpuTestBase._skip_if_sparsecore_unsupported(self)
      else:
        op_testing.TorchTpuTestBase._skip_if_sparsecore_unsupported(self)


if __name__ == "__main__":
  absltest.main()
