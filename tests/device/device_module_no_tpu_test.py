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

"""Tests for the device module on hosts without a TPU."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import hardware


# Not a `seed_test_utils` test class: those seed the RNGs in `setUp()`, which
# would obscure what the test body checks.
class NoTpuDeviceModuleTest(
    parameterized.TestCase  # ABSLTEST_OK=tests seeding
):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Fail rather than skip, so that running this test on a TPU host is noticed
    # and fixed instead of silently losing coverage.
    if torch.accelerator.is_available():
      raise AssertionError("expected a host without TPU devices")

  def test_import_does_not_set_premapped_buffer_size_env_var(self):
    self.assertNotIn("TPU_PREMAPPED_BUFFER_SIZE", os.environ)

  def test_manual_seed_is_noop(self):
    # Like `torch.cuda`, seeding is a no-op when no device is present, so that
    # `torch.manual_seed()` keeps working on CPU-only hosts.
    torch.manual_seed(0)
    torch.tpu.manual_seed(0)
    torch.tpu.manual_seed_all(0)

  def test_get_device_name_returns_none_when_tpu_unrecognized(self):
    self.assertIsNone(torch.tpu.get_device_name())

  @parameterized.parameters(*hardware.TpuSpec.specs_by_pci_device_id().values())
  def test_get_device_name_uses_tpu_spec(self, tpu_spec):
    with mock.patch.object(hardware, "get_tpu_spec", return_value=tpu_spec):
      self.assertEqual(torch.tpu.get_device_name(), tpu_spec.name)

  def test_get_device_name_is_deprecated(self):
    # Reset the one-time warning flag, which other tests may have set.
    wrapper = torch.tpu.get_device_name.__func__
    wrapper.__closure__[
        wrapper.__code__.co_freevars.index("_warned")
    ].cell_contents = False
    with self.assertWarnsRegex(
        DeprecationWarning,
        r"^//TorchTpu/ApiStage/DeprecatedApi// get_device_name is deprecated as"
        r" of TorchTPU 0\.2\.1 and will be removed in a future release: Use"
        r" torch\.tpu\.get_device_properties\(\)\.name instead, which mirrors"
        r" torch\.cuda\.get_device_properties\(\)\.name\.$",
    ):
      torch.tpu.get_device_name()

  def test_get_device_properties_raises_when_tpu_unrecognized(self):
    with self.assertRaisesWithLiteralMatch(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
        RuntimeError,
        "cannot determine TPU device properties: no TPU generation that"
        " TorchTPU supports is attached",
    ):
      torch.tpu.get_device_properties()

  @parameterized.named_parameters(
      ("v5p", hardware.TpuSpec.V5P, 4, 8),
      ("v5e", hardware.TpuSpec.V5E, 0, None),
      ("v6e", hardware.TpuSpec.V6E, 2, 8),
      ("v7", hardware.TpuSpec.V7, 2, 16),
      ("v8i", hardware.TpuSpec.V8I, 1, 16),
      ("v8t", hardware.TpuSpec.V8T, 2, 16),
  )
  def test_get_device_properties_uses_tpu_spec(
      self,
      tpu_spec,
      expected_sparse_cores_per_device,
      expected_sparse_core_simd_width,
  ):
    with mock.patch.object(hardware, "get_tpu_spec", return_value=tpu_spec):
      props = torch.tpu.get_device_properties()
    self.assertEqual(props.name, tpu_spec.name)
    self.assertEqual(props.devices_per_chip, tpu_spec.devices_per_chip)
    self.assertEqual(props.hbm_bytes_per_chip, tpu_spec.hbm_bytes_per_chip)
    self.assertEqual(props.hbm_bytes_per_device, tpu_spec.hbm_bytes_per_device)
    self.assertEqual(props.bf16_flops_per_chip, tpu_spec.bf16_flops_per_chip)
    self.assertEqual(
        props.bf16_flops_per_device, tpu_spec.bf16_flops_per_device
    )
    self.assertEqual(
        props.sparse_cores_per_device, expected_sparse_cores_per_device
    )
    self.assertEqual(
        props.sparse_core_simd_width, expected_sparse_core_simd_width
    )
    for num_devices, topology in tpu_spec.topologies.items():
      self.assertEqual(props.topology(num_devices), topology)

  def test_device_properties_multi_processor_count_is_deprecated(self):
    tpu_spec = hardware.TpuSpec.V7
    with mock.patch.object(hardware, "get_tpu_spec", return_value=tpu_spec):
      props = torch.tpu.get_device_properties()
    with self.assertWarnsRegex(
        DeprecationWarning,
        r"^//TorchTpu/ApiStage/DeprecatedApi// multi_processor_count is"
        r" deprecated as of TorchTPU 0\.2\.1 and will be removed in a future"
        r" release: multi_processor_count is a CUDA concept with no TPU"
        r" equivalent\.$",
    ):
      self.assertEqual(props.multi_processor_count, 1)

  def test_device_properties_total_memory_is_deprecated(self):
    tpu_spec = hardware.TpuSpec.V7
    with mock.patch.object(hardware, "get_tpu_spec", return_value=tpu_spec):
      props = torch.tpu.get_device_properties()
    with self.assertWarnsRegex(
        DeprecationWarning,
        r"^//TorchTpu/ApiStage/DeprecatedApi// total_memory is deprecated as of"
        r" TorchTPU 0\.2\.1 and will be removed in a future release: Use"
        r" torch\.tpu\.get_device_properties\(\)\.hbm_bytes_per_device"
        r" instead\.$",
    ):
      self.assertEqual(props.total_memory, tpu_spec.hbm_bytes_per_device)


if __name__ == "__main__":
  absltest.main()
