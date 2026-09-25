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

"""Tests for torchax_device_ops."""

from unittest import mock

from absl.testing import absltest
from perflab import target as target_lib
from perflab.torchax import torchax_device_ops
from tests import seed_test_utils


class TorchaxDeviceOpsTest(seed_test_utils.RepeatableTest):

  def test_init_success(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torchax_device_ops.TorchaxDeviceOps(target)
    self.assertEqual(ops.target, target)
    self.assertEqual(ops.compile_count(), 0)

  @mock.patch("jax.device_count")
  def test_init_device_count_mismatch(self, mock_device_count):
    mock_device_count.return_value = 0
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        torchax_device_ops.DeviceCountMismatch,
        r"expects 1 device\(s\) but the host has 0",
    ):
      torchax_device_ops.TorchaxDeviceOps(target)

  def test_sync_jax_device_with_leaf_methods(self):
    mock_leaf1 = mock.MagicMock()
    mock_leaf2 = mock.MagicMock()
    # mock_leaf1 has .jax().block_until_ready()
    mock_jax_obj = mock.MagicMock()
    mock_leaf1.jax.return_value = mock_jax_obj
    # mock_leaf2 has direct .block_until_ready()
    del mock_leaf2.jax

    structure = {"a": mock_leaf1, "b": [mock_leaf2, 42, None]}
    torchax_device_ops._sync_jax_device(structure)

    mock_leaf1.jax.assert_called_once()
    mock_jax_obj.block_until_ready.assert_called_once()
    mock_leaf2.block_until_ready.assert_called_once()

  def test_await_result(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torchax_device_ops.TorchaxDeviceOps(target)

    mock_leaf = mock.MagicMock()
    del mock_leaf.jax
    ops.await_result({"out": mock_leaf})
    mock_leaf.block_until_ready.assert_called_once()

  def test_reset_peak_memory(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torchax_device_ops.TorchaxDeviceOps(target)
    # Should complete without error on CPU
    ops.reset_peak_memory()

    mock_dev = mock.MagicMock()
    with mock.patch("jax.local_devices", return_value=[mock_dev]):
      ops.reset_peak_memory()
      mock_dev.reset_memory_stats.assert_called_once()

  def test_peak_memory_mb(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torchax_device_ops.TorchaxDeviceOps(target)
    self.assertIsNone(ops.peak_memory_mb())

    mock_dev = mock.MagicMock()
    mock_dev.memory_stats.return_value = {
        "peak_bytes_in_use": 128 * 1024 * 1024
    }
    with mock.patch("jax.local_devices", return_value=[mock_dev]):
      self.assertEqual(ops.peak_memory_mb(), 128.0)

    mock_dev.memory_stats.return_value = {
        "peak_bytes_in_use": -1,
        "bytes_in_use": 64 * 1024 * 1024,
    }
    with mock.patch("jax.local_devices", return_value=[mock_dev]):
      self.assertEqual(ops.peak_memory_mb(), 64.0)

    # Verify negative sentinel values (-1) return None rather than -1.
    mock_dev.memory_stats.return_value = {
        "peak_bytes_in_use": -1,
        "bytes_in_use": -1,
    }
    with mock.patch("jax.local_devices", return_value=[mock_dev]):
      self.assertIsNone(ops.peak_memory_mb())

    # Verify None stats or exceptions return None rather than -1.
    mock_dev.memory_stats.return_value = None
    with mock.patch("jax.local_devices", return_value=[mock_dev]):
      self.assertIsNone(ops.peak_memory_mb())

    mock_dev.memory_stats.side_effect = RuntimeError("memory stats unavailable")
    with mock.patch("jax.local_devices", return_value=[mock_dev]):
      self.assertIsNone(ops.peak_memory_mb())
    with mock.patch("jax.local_devices", return_value=[]):
      self.assertIsNone(ops.peak_memory_mb())

  def test_compile_count(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torchax_device_ops.TorchaxDeviceOps(target)
    self.assertEqual(ops.compile_count(), 0)


if __name__ == "__main__":
  absltest.main()
