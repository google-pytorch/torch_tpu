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

import os
import pathlib
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu._internal.utils import hardware
from tests import seed_test_utils


class TpuTopologyMockTest(seed_test_utils.RepeatableTest):

  def test_v7_8_chips(self):
    tpus = []
    for bus in ["16", "38", "98", "b8"]:
      for fn in ["0", "1"]:
        addr = f"0000:{bus}:01.{fn}"
        tpus.append(
            {"addr": addr, "id": "0x0076", "vfio_group": f"grp_{bus}_{fn}"}
        )

    self._run_mock_test(tpus, "2,2,1,2")

  def test_v5e_8_chips(self):
    tpus = []
    for i in range(8):
      addr = f"0000:03:01.{i}"
      tpus.append({"addr": addr, "id": "0x0063", "vfio_group": str(146 + i)})

    self._run_mock_test(tpus, "2,2,2")

  def test_v5p_4_chips(self):
    tpus = []
    for i in range(4):
      addr = f"0000:05:01.{i}"
      tpus.append({"addr": addr, "id": "0x0062", "vfio_group": str(439 + i)})

    self._run_mock_test(tpus, "2,2,1")

  def test_v6e_8_chips(self):
    tpus = []
    for i in range(8):
      addr = f"0000:41:01.{i}"
      tpus.append({"addr": addr, "id": "0x006f", "vfio_group": str(131 + i)})

    self._run_mock_test(tpus, "2,4,1")

  def _run_mock_test(self, tpus, expected_topology):
    # Map of path strings to mock Path objects
    path_mocks = {}
    real_path_cls = pathlib.Path

    def get_path_mock(path_str):
      path_str = str(path_str)
      if path_str not in path_mocks:
        m = mock.MagicMock(spec=real_path_cls)
        m.__str__.return_value = path_str
        m.__truediv__.side_effect = lambda other: get_path_mock(
            f"{path_str}/{other}"
        )
        m.name = path_str.split("/")[-1]

        # Default behaviors
        m.exists.return_value = False

        # Handle /sys/bus/pci/devices
        if path_str == "/sys/bus/pci/devices":
          m.exists.return_value = True
          m.iterdir.return_value = [
              get_path_mock(f"/sys/bus/pci/devices/{d['addr']}") for d in tpus
          ]

        # Handle device paths
        for d in tpus:
          device_root = f"/sys/bus/pci/devices/{d['addr']}"
          if path_str == device_root:
            m.exists.return_value = True
          elif path_str == f"{device_root}/vendor":
            m.read_text.return_value = "0x1ae0"
          elif path_str == f"{device_root}/device":
            m.read_text.return_value = d["id"]
          elif path_str == f"{device_root}/iommu_group":
            m.exists.return_value = True
            m.readlink.return_value = get_path_mock(d["vfio_group"])

        # Handle /dev/vfio
        if path_str == "/dev/vfio":
          m.exists.return_value = True
        elif path_str.startswith("/dev/vfio/"):
          group_id = path_str.split("/")[-1]
          m.exists.return_value = any(d["vfio_group"] == group_id for d in tpus)

        path_mocks[path_str] = m
      return path_mocks[path_str]

    with mock.patch.object(pathlib, "Path", side_effect=get_path_mock):
      topology = hardware.get_tpu_topology()
      self.assertEqual(topology, expected_topology)


class NvidiaGpuMockTest(parameterized.TestCase):

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

  @mock.patch.object(hardware, "get_tpu_device_name", return_value="TPU v7")
  def test_v7_name_flops_memory(self, mock_name):
    del mock_name  # Unused
    # name -> version mapping
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


if __name__ == "__main__":
  absltest.main()
