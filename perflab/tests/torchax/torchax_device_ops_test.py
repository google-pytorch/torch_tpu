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

from typing import Any
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from perflab import target as target_lib
from perflab.torchax import torchax_device_ops
from tests import seed_test_utils

_MB = 2**20


@jax.jit
def _slow_matmul_chain(a: jax.Array) -> jax.Array:
  return jax.lax.fori_loop(0, 50, lambda _, b: jnp.tanh(b @ a), a)


def _pending_jax_array() -> jax.Array:
  """Returns a jax.Array whose computation was dispatched asynchronously."""
  a = jnp.full((512, 512), 1e-3, dtype=jnp.float32)
  _slow_matmul_chain(a).block_until_ready()  # Compile before dispatching.
  return _slow_matmul_chain(a + 1e-3)


class _JaxBackedLeaf:
  """Leaf exposing `.jax()`, the accessor torchax tensors provide."""

  # TODO(wan): revisit DECISION (b/568593189): this stands in for a real
  # torchax tensor so the test does not depend on torch_xla2, whose pinned
  # OSS flax dependency fails to import on Python 3.14+.

  def __init__(self, array: jax.Array):
    self._array = array

  def jax(self) -> jax.Array:
    return self._array


class _FakeDevice:
  """Stateful fake of a JAX device's memory-stats API."""

  def __init__(self, stats: Any):
    self.stats = stats

  def memory_stats(self) -> Any:
    if isinstance(self.stats, Exception):
      raise self.stats
    return self.stats

  def reset_memory_stats(self) -> None:
    self.stats = dict(self.stats, peak_bytes_in_use=self.stats["bytes_in_use"])


def _cpu_ops() -> torchax_device_ops.TorchaxDeviceOps:
  return torchax_device_ops.TorchaxDeviceOps(
      target_lib.make_target(platform=target_lib.Platform.CPU)
  )


class TorchaxDeviceOpsTest(seed_test_utils.RepeatableTest):

  def test_init_success(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ops = torchax_device_ops.TorchaxDeviceOps(target)
    self.assertEqual(ops.target, target)
    self.assertEqual(ops.compile_count(), 0)

  def test_init_device_count_mismatch(self):
    actual_devices = jax.device_count()
    expected_devices = actual_devices + 1
    target = target_lib.Target(
        platform=target_lib.Platform.V7_2X2X1,
        platform_spec=target_lib.PlatformSpec(
            kind=target_lib.DeviceKind.TPU,
            topology=target_lib.Topology(
                nnodes=1, nprocs_per_node=expected_devices, chip="v7"
            ),
        ),
        dtype=target_lib.DType.BF16,
    )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        torchax_device_ops.DeviceCountMismatch,
        rf"expects {expected_devices} device\(s\) but the host has"
        rf" {actual_devices}\b",
    ):
      torchax_device_ops.TorchaxDeviceOps(target)

  def test_sync_jax_device_waits_for_every_leaf(self):
    direct = _pending_jax_array()
    via_accessor = _pending_jax_array()
    structure = {"a": _JaxBackedLeaf(via_accessor), "b": [direct, 42, None]}

    torchax_device_ops._sync_jax_device(structure)

    self.assertTrue(direct.is_ready())
    self.assertTrue(via_accessor.is_ready())

  def test_await_result(self):
    ops = _cpu_ops()
    out = _pending_jax_array()

    ops.await_result({"out": out})

    self.assertTrue(out.is_ready())

  def test_memory_stats_on_cpu(self):
    ops = _cpu_ops()

    ops.reset_peak_memory()  # Should complete without error on CPU.

    self.assertIsNone(ops.peak_memory_mb())

  def test_reset_peak_memory_resets_device_peak(self):
    ops = _cpu_ops()
    device = _FakeDevice(
        {"peak_bytes_in_use": 128 * _MB, "bytes_in_use": 64 * _MB}
    )

    with mock.patch.object(jax, "local_devices", return_value=[device]):
      self.assertEqual(ops.peak_memory_mb(), 128.0)
      ops.reset_peak_memory()
      self.assertEqual(ops.peak_memory_mb(), 64.0)

  def test_peak_memory_mb(self):
    ops = _cpu_ops()
    device = _FakeDevice(None)

    with mock.patch.object(jax, "local_devices", return_value=[device]):
      device.stats = {"peak_bytes_in_use": -1, "bytes_in_use": 64 * _MB}
      self.assertEqual(ops.peak_memory_mb(), 64.0)

      # Verify negative sentinel values (-1) return None rather than -1.
      device.stats = {"peak_bytes_in_use": -1, "bytes_in_use": -1}
      self.assertIsNone(ops.peak_memory_mb())

      # Verify None stats or exceptions return None rather than -1.
      device.stats = None
      self.assertIsNone(ops.peak_memory_mb())

      device.stats = RuntimeError("memory stats unavailable")
      self.assertIsNone(ops.peak_memory_mb())

    with mock.patch.object(jax, "local_devices", return_value=[]):
      self.assertIsNone(ops.peak_memory_mb())

  def test_compile_count(self):
    ops = _cpu_ops()
    self.assertEqual(ops.compile_count(), 0)


if __name__ == "__main__":
  absltest.main()
