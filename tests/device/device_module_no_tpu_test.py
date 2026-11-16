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
      (
          "v5p",
          hardware.TpuSpec.for_version(hardware.TpuVersion.V5P),
          hardware.TpuVersion.V5P,
          4,
          8,
      ),
      (
          "v5e",
          hardware.TpuSpec.for_version(hardware.TpuVersion.V5E),
          hardware.TpuVersion.V5E,
          0,
          None,
      ),
      (
          "v6e",
          hardware.TpuSpec.for_version(hardware.TpuVersion.V6E),
          hardware.TpuVersion.V6E,
          2,
          8,
      ),
      (
          "v7",
          hardware.TpuSpec.for_version(hardware.TpuVersion.V7),
          hardware.TpuVersion.V7,
          2,
          16,
      ),
      (
          "v8i",
          hardware.TpuSpec.for_version(hardware.TpuVersion.V8I),
          hardware.TpuVersion.V8I,
          1,
          16,
      ),
      (
          "v8t",
          hardware.TpuSpec.for_version(hardware.TpuVersion.V8T),
          hardware.TpuVersion.V8T,
          2,
          16,
      ),
  )
  def test_get_device_properties_uses_tpu_spec(
      self,
      tpu_spec,
      expected_version,
      expected_sparse_cores_per_device,
      expected_sparse_core_simd_width,
  ):
    with mock.patch.object(hardware, "get_tpu_spec", return_value=tpu_spec):
      props = torch.tpu.get_device_properties()
    self.assertEqual(props.name, tpu_spec.name)
    self.assertIs(props.version, expected_version)
    self.assertIs(
        props.version, getattr(torch.tpu.TpuVersion, expected_version.name)
    )
    self.assertFalse(hasattr(props, "kind"))
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
    tpu_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V7)
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
    tpu_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V7)
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

  # Note: `torch.device("tpu")` is not registered on hosts without TPUs
  # (`_loader.load` skips `rename_privateuse1_backend` when device count is 0).
  @parameterized.parameters(
      (None,),
      (0,),
      (1,),
      ("tpu",),
      ("tpu:0",),
      ("tpu:1",),
  )
  def test_device_arg_validation_accepts_valid_specifiers(self, valid_dev):
    tpu_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V7)
    with (
        mock.patch.object(
            hardware, "get_tpu_spec", autospec=True, return_value=tpu_spec
        ),
        mock.patch.object(
            hardware,
            "get_visible_tpu_device_count",
            autospec=True,
            return_value=2,
        ),
    ):
      self.assertEqual(
          torch.tpu.get_device_properties(valid_dev).version,
          hardware.TpuVersion.V7,
      )
      self.assertEqual(torch.tpu.get_device_name(valid_dev), "TPU v7")

  @parameterized.parameters((True,), (1.5,), ((1, 2),))
  def test_device_arg_validation_rejects_invalid_types(self, bad_type):
    tpu_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V7)
    with (
        mock.patch.object(
            hardware, "get_tpu_spec", autospec=True, return_value=tpu_spec
        ),
        mock.patch.object(
            hardware,
            "get_visible_tpu_device_count",
            autospec=True,
            return_value=2,
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          TypeError,
          "expected device to be a torch.device, str, int, or None",
      ):
        torch.tpu.get_device_properties(bad_type)  # type: ignore[arg-type]
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          TypeError,
          "expected device to be a torch.device, str, int, or None",
      ):
        torch.tpu.get_device_name(bad_type)  # type: ignore[arg-type]

  @parameterized.parameters(
      ("cpu",),
      ("cuda:0",),
      ("tpu:",),
      ("tpu:+0",),
      ("tpu:-0",),
      ("tpu:01",),
      ("tpu:0_0",),
      ("tpu:abc",),
      (torch.device("cpu"),),
      (torch.device("cpu", 0),),
      (torch.device("meta", 0),),
  )
  def test_device_arg_validation_rejects_non_tpu_specifiers(self, bad_val):
    tpu_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V7)
    with (
        mock.patch.object(
            hardware, "get_tpu_spec", autospec=True, return_value=tpu_spec
        ),
        mock.patch.object(
            hardware,
            "get_visible_tpu_device_count",
            autospec=True,
            return_value=2,
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError, r"^expected a 'tpu' device, got "
      ):
        torch.tpu.get_device_properties(bad_val)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError, r"^expected a 'tpu' device, got "
      ):
        torch.tpu.get_device_name(bad_val)

  @parameterized.parameters(
      (-1, r"^device index must be non-negative, got -1$"),
      ("tpu:-1", r"^device index must be non-negative, got -1$"),
      (
          2,
          r"^device index 2 is out of range for 2 visible TPU device\(s\)$",
      ),
      (
          "tpu:2",
          r"^device index 2 is out of range for 2 visible TPU device\(s\)$",
      ),
  )
  def test_device_arg_validation_rejects_out_of_range_indices(
      self, bad_idx, expected_regex
  ):
    tpu_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V7)
    with (
        mock.patch.object(
            hardware, "get_tpu_spec", autospec=True, return_value=tpu_spec
        ),
        mock.patch.object(
            hardware,
            "get_visible_tpu_device_count",
            autospec=True,
            return_value=2,
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError, expected_regex
      ):
        torch.tpu.get_device_properties(bad_idx)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError, expected_regex
      ):
        torch.tpu.get_device_name(bad_idx)

  def test_device_arg_validation_handles_zero_visible_and_cached_count(self):
    v5e_spec = hardware.TpuSpec.for_version(hardware.TpuVersion.V5E)
    with (
        mock.patch.object(
            hardware, "get_tpu_spec", autospec=True, return_value=v5e_spec
        ),
        mock.patch.object(
            hardware,
            "get_visible_tpu_device_count",
            autospec=True,
            return_value=0,
        ),
    ):
      self.assertEqual(torch.tpu.get_device_name(0), "TPU v5e")
      self.assertEqual(
          torch.tpu.get_device_properties("tpu:0").version,
          hardware.TpuVersion.V5E,
      )
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError, r"^device index must be non-negative, got -1$"
      ):
        torch.tpu.get_device_name(-1)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError, r"^expected a 'tpu' device, got 'tpu:01'$"
      ):
        torch.tpu.get_device_properties("tpu:01")

    with (
        mock.patch.object(
            hardware, "get_tpu_spec", autospec=True, return_value=v5e_spec
        ),
        mock.patch.object(
            hardware,
            "get_visible_tpu_device_count",
            autospec=True,
            return_value=4,
        ),
        mock.patch.object(
            torch.tpu.get_device_properties.__self__, "_device_count", 1
        ),
    ):
      self.assertEqual(torch.tpu.get_device_properties(0).name, "TPU v5e")
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Needs a host without a recognized TPU.
          ValueError,
          r"^device index 1 is out of range for 1 visible TPU device\(s\)$",
      ):
        torch.tpu.get_device_properties(1)

  def test_stream_is_type(self):
    self.assertTrue(issubclass(torch.tpu.Stream, torch.Stream))
    self.assertIsInstance(None, torch.tpu.Stream | None)

  def test_event_is_type(self):
    self.assertTrue(issubclass(torch.tpu.Event, torch.Event))
    self.assertIsInstance(None, torch.tpu.Event | None)


if __name__ == "__main__":
  absltest.main()
