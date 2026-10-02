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
import gc
import threading
from typing import Final
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.device import _device_module
from torch_tpu._internal.device import _device_ops_backend
from tests import oss_utils
from tests import seed_test_utils

_DEVICE_LOCK: Final[threading.Lock] = threading.Lock()


class _DeviceModuleBaseMeta(parameterized.TestGeneratorMetaclass, abc.ABCMeta):
  pass


# pylint: disable=protected-access
class DeviceModuleBase(
    seed_test_utils.RepeatableTest, metaclass=_DeviceModuleBaseMeta
):
  """Abstract base class containing tests for the device module."""

  @property
  @abc.abstractmethod
  def device_module(self):
    """The device_module under test."""
    raise NotImplementedError("Must be implemented in child class")

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

    # Acquire a lock to be released on cleanup, since we are modifying a global
    self.enter_context(_DEVICE_LOCK)

    # Reset class variables before each test
    self.device_module._autocast_enabled = False
    self.device_module._autocast_dtype = torch.float16

  def _bound_device_index(self) -> int:
    """Returns the real device index of this process, checking it is valid."""
    idx = torch.accelerator.current_device_index()
    self.assertBetween(idx, 0, self.device_module.device_count() - 1)
    return idx

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

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 42),
      "TODO(b/565506249): Requires libtpu >= 0.0.42 for"
      " reset_peak_memory_stats.",
  )
  def test_reset_memory_stats(self):
    """Tests that resetting memory stats works correctly."""
    # 1. Reset stats to establish a clean baseline where peak == current
    torch.accelerator.reset_peak_memory_stats()

    allocated_before = torch.accelerator.memory_allocated()
    max_allocated_before = torch.accelerator.max_memory_allocated()
    self.assertEqual(max_allocated_before, allocated_before)

    # 2. Allocate some memory (e.g., 4MB)
    size_in_bytes = 4 * 1024 * 1024
    num_elements = size_in_bytes // 4
    t = torch.randn(num_elements, device="cpu").to(
        torch.accelerator.current_accelerator()
    )
    # 3. Verify peak memory increased
    allocated_after = torch.accelerator.memory_allocated()
    max_allocated_after = torch.accelerator.max_memory_allocated()
    self.assertGreaterEqual(allocated_after, allocated_before + size_in_bytes)
    self.assertGreaterEqual(max_allocated_after, allocated_after)
    # 4. Free the allocated tensor
    del t
    gc.collect()
    allocated_after_free = torch.accelerator.memory_allocated()
    self.assertLess(allocated_after_free, allocated_after)

    # 5. Reset peak memory stats again
    torch.accelerator.reset_peak_memory_stats()

    max_allocated_after_reset = torch.accelerator.max_memory_allocated()
    self.assertEqual(max_allocated_after_reset, allocated_after_free)

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 42),
      "TODO(b/565506249): Requires libtpu >= 0.0.42 for"
      " reset_peak_memory_stats.",
  )
  def test_memory_allocation(self):
    """Tests that memory allocation works with the device."""
    torch.accelerator.reset_peak_memory_stats()
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

  def test_get_cache_stats(self):
    stats = self.device_module._get_cache_stats()
    self.assertIsNotNone(stats)
    self.assertGreaterEqual(stats.num_cache_reqs, 0)
    self.assertGreaterEqual(stats.num_cache_hits, 0)
    self.assertFalse(stats.peak_compilation_host_memory_bytes)
    self.assertIsInstance(stats.per_entry_stats, list)

  def test_rng_validate_device_index(self):
    """Test valid string and int."""
    _device_module._rng_validate_device_index("tpu", 0)
    _device_module._rng_validate_device_index(0, 0)

    # Test valid torch.device.
    _device_module._rng_validate_device_index(torch.device("tpu", 0), 0)

    # Test invalid device type.
    with self.assertRaisesRegex(
        ValueError, "expected device type 'tpu' for RNG state access, got 'cpu'"
    ):
      _device_module._rng_validate_device_index("cpu", 0)

    # Test invalid device index raises ValueError.
    with self.assertRaisesRegex(
        ValueError,
        "expected local device index 0, got 1: accessing RNG state of a"
        " non-current TPU device is not supported",
    ):
      _device_module._rng_validate_device_index(1, 0)

  def test_bound_device_index_is_valid(self):
    idx = self._bound_device_index()
    if self.device_module.device_count() == 1:
      self.assertEqual(idx, 0)

  def test_set_device_valid(self):
    idx = self._bound_device_index()

    # TPU processes are bound to one device, so selecting the bound device must
    # not raise and must not change the current device.
    # Test valid integer index
    self.device_module.set_device(idx)
    self.assertEqual(self.device_module.current_device(), idx)

    # Setting through torch.accelerator is reflected by current_device()
    torch.accelerator.set_device_index(idx)
    self.assertEqual(self.device_module.current_device(), idx)

    # Test valid string and torch.device
    if self.device_module._device_type == "tpu":
      self.device_module.set_device("tpu")
      self.assertEqual(self.device_module.current_device(), idx)

      self.device_module.set_device(torch.device("tpu"))
      self.assertEqual(self.device_module.current_device(), idx)

      self.device_module.set_device(torch.device("tpu", idx))
      self.assertEqual(self.device_module.current_device(), idx)

  def test_set_device_invalid_index(self):
    idx = self._bound_device_index()
    invalid_idx = self.device_module.device_count()
    with self.assertRaisesRegex(
        ValueError, f"Cannot set TPU device to index {invalid_idx}"
    ):
      self.device_module.set_device(invalid_idx)
    # A failed set_device does not change the current device.
    self.assertEqual(self.device_module.current_device(), idx)

  def test_set_device_invalid_string(self):
    with self.assertRaisesRegex(ValueError, "Invalid device string cpu"):
      self.device_module.set_device("cpu")

  def test_set_device_invalid_device_type(self):
    with self.assertRaisesRegex(ValueError, "Invalid device type cpu"):
      self.device_module.set_device(torch.device("cpu"))

  def test_set_device_invalid_type(self):
    with self.assertRaisesRegex(TypeError, "Got unrecognized device type"):
      self.device_module.set_device(1.0)

  def test_device_context_manager(self):
    idx = self._bound_device_index()

    # Valid int: verifies device selection and context restoration
    ctx = self.device_module.device(idx)
    self.assertEqual(ctx.idx, idx)
    with ctx:
      self.assertEqual(ctx.prev_idx, idx)
      self.assertEqual(self.device_module.current_device(), idx)
    self.assertEqual(ctx.idx, idx)
    self.assertEqual(self.device_module.current_device(), idx)

    # Valid string and torch.device if registered
    if self.device_module._device_type == "tpu":
      with self.device_module.device("tpu"):
        self.assertEqual(self.device_module.current_device(), idx)

      with self.device_module.device(torch.device("tpu", idx)):
        self.assertEqual(self.device_module.current_device(), idx)

    # None resolves to the current device, keeping the active device context
    none_ctx = self.device_module.device(None)
    self.assertEqual(none_ctx.idx, idx)
    with none_ctx:
      self.assertEqual(none_ctx.prev_idx, idx)

    # Negative int is a no-op
    neg_ctx = self.device_module.device(-1)
    self.assertEqual(neg_ctx.idx, -1)
    with neg_ctx:
      self.assertEqual(neg_ctx.prev_idx, -1)
      self.assertEqual(self.device_module.current_device(), idx)

    # Invalid device index raises ValueError and keeps the current device
    invalid_idx = self.device_module.device_count()
    with self.assertRaisesRegex(
        ValueError, f"Cannot set TPU device to index {invalid_idx}"
    ):
      with self.device_module.device(invalid_idx):
        pass
    self.assertEqual(self.device_module.current_device(), idx)

  def test_device_of_context_manager(self):
    idx = self._bound_device_index()

    # If device backend is TPU, a TPU tensor selects its own device index
    if self.device_module._device_type == "tpu":
      tpu_tensor = torch.zeros(1, device="tpu")
      tpu_ctx = self.device_module.device_of(tpu_tensor)
      self.assertEqual(tpu_ctx.idx, idx)
      with tpu_ctx:
        self.assertEqual(tpu_ctx.prev_idx, idx)

    # CPU tensor is not on TPU, so it acts as a no-op (idx = -1)
    cpu_tensor = torch.zeros(1, device="cpu")
    cpu_ctx = self.device_module.device_of(cpu_tensor)
    self.assertEqual(cpu_ctx.idx, -1)
    with cpu_ctx:
      self.assertEqual(cpu_ctx.prev_idx, -1)

  def test_is_available(self):
    """Verifies that is_available returns a boolean."""
    is_avail = self.device_module.is_available()
    self.assertIsInstance(is_avail, bool)
    if self.device_module._device_type == "tpu":
      self.assertEqual(is_avail, self.device_module.device_count() > 0)

  def test_synchronize(self):
    """Verifies that device_module.synchronize executes cleanly."""
    self.device_module.synchronize()
    self.device_module.synchronize(0)
    self.device_module.synchronize(None)

  def test_exchange_device(self):
    idx = self._bound_device_index()
    self.assertEqual(self.device_module._exchange_device(idx), idx)
    self.assertEqual(self.device_module._exchange_device(-1), -1)
    self.assertEqual(self.device_module._maybe_exchange_device(idx), idx)
    self.assertEqual(self.device_module._maybe_exchange_device(-1), -1)
    self.assertEqual(self.device_module.current_device(), idx)

  def test_get_local_device_attributes(self):
    """Tests the local device attributes dictionary contains correct keys."""
    tensor = torch.zeros(1, device=torch.accelerator.current_accelerator())
    _ = tensor.item()  # Materialize to trigger initialization

    attrs = _device_ops_backend._get_local_device_attributes()
    self.assertIsInstance(attrs, dict)
    self.assertIn("id", attrs)
    self.assertIn("process_index", attrs)
    self.assertIn("device_kind", attrs)
    if "coords" in attrs:
      self.assertIsInstance(attrs["coords"], (list, tuple))
    if "core_on_chip" in attrs:
      self.assertIsInstance(attrs["core_on_chip"], int)
    if "slice_index" in attrs:
      self.assertIsInstance(attrs["slice_index"], int)


# pylint: enable=protected-access

if __name__ == "__main__":
  absltest.main()
