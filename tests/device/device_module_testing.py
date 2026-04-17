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

"""Base class defining tests for the device module."""

import abc
import contextlib
import threading
from typing import Final
from unittest import mock
from absl.testing import absltest
import torch
from torch_tpu._internal.device import _device_module

_DEVICE_LOCK: Final[threading.Lock] = threading.Lock()


# pylint: disable=protected-access
class DeviceModuleBase(absltest.TestCase, metaclass=abc.ABCMeta):
  """Abstract base class containing tests for the device module."""

  @property
  @abc.abstractmethod
  def device_module(self):
    """The device_module under test."""
    raise NotImplementedError("Must be implemented in child class")

  def setUp(self):
    super().setUp()

    # Acquire a lock to be released on cleanup, since we are modifying a global
    self.enter_context(_DEVICE_LOCK)

    # Reset class variables before each test
    self.device_module._autocast_enabled = False
    self.device_module._autocast_dtype = torch.float16

  @contextlib.contextmanager
  def patch_current_device(self, device_id: int = 0):
    patch = mock.patch.object(
        self.device_module,
        "current_device",
        return_value=device_id,
        autospec=True,
    )
    with patch:
      yield patch

  def test_get_amp_supported_dtype(self):
    supported_dtypes = self.device_module.get_amp_supported_dtype()
    self.assertIsInstance(supported_dtypes, list)
    self.assertNotEmpty(supported_dtypes)
    for dtype in supported_dtypes:
      with self.subTest(dtype):
        self.assertIsInstance(dtype, torch.dtype)

  def test_is_autocast_enabled(self):
    self.assertFalse(self.device_module._is_autocast_enabled())
    self.device_module._set_autocast_enabled(True)
    self.assertTrue(self.device_module._is_autocast_enabled())

  def test_get_autocast_dtype(self):
    self.assertEqual(self.device_module._get_autocast_dtype(), torch.float16)
    self.device_module._set_autocast_dtype(torch.bfloat16)
    self.assertEqual(self.device_module._get_autocast_dtype(), torch.bfloat16)

  def test_set_autocast_enabled(self):
    self.device_module._set_autocast_enabled(True)
    self.assertTrue(self.device_module._autocast_enabled)
    self.device_module._set_autocast_enabled(False)
    self.assertFalse(self.device_module._autocast_enabled)

  def test_set_autocast_dtype(self):
    supported_dtypes = self.device_module.get_amp_supported_dtype()
    for dtype in supported_dtypes:
      with self.subTest(dtype):
        self.device_module._set_autocast_dtype(dtype)
        self.assertEqual(self.device_module._autocast_dtype, dtype)

    with self.assertRaises(ValueError):
      self.device_module._set_autocast_dtype(None)

  def test_memory_stats(self):
    stats = torch.accelerator.memory_stats()
    self.assertIn("active_bytes.all.current", stats)
    self.assertIn("active_bytes.all.peak", stats)
    self.assertIn("reserved_bytes.all.current", stats)
    self.assertIn("reserved_bytes.all.peak", stats)

  @absltest.skip(reason="b/502954625: Fails if any other test allocates memory")
  def test_memory_allocation(self):
    """Tests that memory allocation works with the device."""
    # Get initial state
    allocated_before = torch.accelerator.memory_allocated()
    max_allocated_before = torch.accelerator.max_memory_allocated()
    self.assertGreaterEqual(allocated_before, 0)
    self.assertEqual(max_allocated_before, allocated_before)

    # Allocate a known amount of memory (e.g., 4MB)
    # float32 is 4 bytes, so 1024 * 1024 elements is 4MB
    size_in_bytes = 4 * 1024 * 1024
    num_elements = size_in_bytes // 4
    t = torch.randn(
        num_elements, device=torch.accelerator.current_accelerator()
    )
    t = t + 1
    # Synchronize by copying to CPU (forces execution)
    _ = t.cpu()

    # Verify allocation increased
    allocated_after = torch.accelerator.memory_allocated()
    max_allocated_after = torch.accelerator.max_memory_allocated()

    self.assertGreater(allocated_after, allocated_before)
    self.assertGreaterEqual(allocated_after, allocated_before + size_in_bytes)

    # Verify max memory allocated updated
    self.assertGreater(max_allocated_after, 0)
    self.assertGreater(max_allocated_after, max_allocated_before)
    self.assertGreaterEqual(max_allocated_after, allocated_after)

  def test_initialized(self):
    # Because of lazy initialization, the device is only initialized when we
    # actually use the device, before which the state depends on what other
    # tests have run. In order to make this test consistently give an
    # initialized device, we'll materialize a tensor on the device. Lazy
    # initialization itself is tested in the C++ tests.
    tensor = torch.zeros(1, device=torch.accelerator.current_accelerator())
    _ = tensor.item()  # Materialize to trigger initialization

    self.assertTrue(self.device_module.is_initialized())
    self.assertGreater(self.device_module.device_count(), 0)

  def test_dump_on_miss(self):
    # Default should be False
    self.assertFalse(self.device_module._get_dump_on_cache_miss())
    self.device_module._set_dump_on_cache_miss(True)
    self.assertTrue(self.device_module._get_dump_on_cache_miss())
    self.device_module._set_dump_on_cache_miss(False)
    self.assertFalse(self.device_module._get_dump_on_cache_miss())

  def test_rng_validate_device_index(self):
    """Test valid string and int."""
    _device_module._rng_validate_device_index("tpu", 0)
    _device_module._rng_validate_device_index(0, 0)

    # Test valid torch.device.
    _device_module._rng_validate_device_index(
        mock.MagicMock(spec=torch.device, type="tpu", index=0), 0
    )

    # Test invalid device type.
    with self.assertRaisesRegex(
        ValueError, "RNG state can only be accessed on TPU"
    ):
      _device_module._rng_validate_device_index("cpu", 0)

    # Test invalid device index logs warning.
    with mock.patch(
        "torch_tpu._internal.device._device_module.logging.warning"
    ) as mock_warning:
      _device_module._rng_validate_device_index(1, 0)
      mock_warning.assert_called_once()

  def test_set_device_valid(self):
    with self.patch_current_device():
      # Test valid int
      self.device_module.set_device(0)

      # Test valid string
      self.device_module.set_device("tpu")

  def test_set_device_invalid_index(self):
    with self.patch_current_device():
      with self.assertRaisesRegex(
          ValueError, "Cannot set TPU device to index 1"
      ):
        self.device_module.set_device(1)

  def test_set_device_invalid_string(self):
    with self.patch_current_device():
      with self.assertRaisesRegex(ValueError, "Invalid device string cpu"):
        self.device_module.set_device("cpu")

  def test_set_device_invalid_device_type(self):
    with self.patch_current_device():
      with self.assertRaisesRegex(ValueError, "Invalid device type cpu"):
        self.device_module.set_device(torch.device("cpu"))

  def test_set_device_invalid_type(self):
    with self.patch_current_device():
      with self.assertRaisesRegex(TypeError, "Got unrecognized device type"):
        self.device_module.set_device(1.0)


# pylint: enable=protected-access

if __name__ == "__main__":
  absltest.main()
