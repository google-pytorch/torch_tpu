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

"""Tests for torch_device_ops."""

import sys
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab import target as target_lib
from perflab import torch_device_ops
from tests import oss_utils
from tests import seed_test_utils

_MB = 2**20


def _tpu_ops() -> torch_device_ops.TorchDeviceOps:
  return torch_device_ops.TorchDeviceOps(
      target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
  )


class TorchDeviceOpsTest(seed_test_utils.RepeatableTest):

  def test_init_success(self):
    target = target_lib.make_target(
        platform=target_lib.Platform.V5E_1X1,
        dtype=target_lib.DType.BF16,
    )
    ops = torch_device_ops.TorchDeviceOps(target)
    self.assertEqual(ops.device.type, "tpu")
    self.assertEqual(ops.dtype, torch.bfloat16)

  def test_init_device_count_mismatch(self):
    actual_devices = torch.accelerator.device_count()
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
        torch_device_ops.DeviceCountMismatch,
        rf"expects {expected_devices} device\(s\) but the host has"
        rf" {actual_devices}\b",
    ):
      torch_device_ops.TorchDeviceOps(target)

  def test_await_result_completes_pending_device_work(self):
    ops = _tpu_ops()
    num_elements = 4 * _MB
    src = torch.full((num_elements,), 3.0, device=ops.device)
    dst = torch.zeros(num_elements, pin_memory=True)

    dst.copy_(src, non_blocking=True)
    done = torch.Event(device=ops.device)
    done.record()
    ops.await_result(dst)

    self.assertTrue(done.query())
    test_utils.assert_close(dst, torch.full((num_elements,), 3.0))

  def test_peak_memory_tracks_allocations(self):
    ops = _tpu_ops()
    ops.reset_peak_memory()
    peak_before = ops.peak_memory_mb()
    self.assertGreaterEqual(peak_before, 0.0)

    # Allocate 32 MB on top of any prior baseline so peak_memory_mb() must
    # advance even on libtpu < 0.0.42 where reset_peak_memory() is a no-op.
    alloc_mb = int(peak_before) + 32
    t = torch.randn(alloc_mb * _MB // 4, device=ops.device)
    ops.await_result(t)
    peak_after_alloc = ops.peak_memory_mb()
    self.assertGreaterEqual(peak_after_alloc, float(alloc_mb))

    del t
    ops.await_result(None)
    self.assertEqual(ops.peak_memory_mb(), peak_after_alloc)

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 42),
      "TODO(b/565506249): Requires libtpu >= 0.0.42 for"
      " reset_peak_memory_stats.",
  )
  def test_reset_peak_memory_clears_peak(self):
    ops = _tpu_ops()
    ops.reset_peak_memory()

    t = torch.randn(32 * _MB // 4, device=ops.device)  # 32 MB of float32.
    ops.await_result(t)
    peak_after_alloc = ops.peak_memory_mb()
    self.assertGreaterEqual(peak_after_alloc, 32.0)

    del t
    ops.await_result(None)
    ops.reset_peak_memory()
    self.assertLess(ops.peak_memory_mb(), peak_after_alloc)

  def test_cpu_memory_ops_are_noops(self):
    ops = torch_device_ops.TorchDeviceOps(
        target_lib.make_target(platform=target_lib.Platform.CPU)
    )

    ops.reset_peak_memory()
    ops.await_result(None)

    self.assertEqual(ops.peak_memory_mb(), 0.0)

  def test_compile_count_dynamo_real_compile(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)

    ops = torch_device_ops.TorchDeviceOps(target)
    count_before = ops.compile_count()
    model = torch.nn.Linear(2, 2)
    compiled_model = torch.compile(model, backend="eager")
    _ = compiled_model(torch.randn(2, 2))

    self.assertGreater(ops.compile_count(), count_before)

  def test_compile_count_dynamo_unavailable(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)

    ops = torch_device_ops.TorchDeviceOps(target)
    # A None entry in sys.modules makes `from torch._dynamo import ...` raise
    # ImportError, as on a build without Dynamo.
    with mock.patch.dict(sys.modules, {"torch._dynamo": None}):
      count = ops.compile_count()

    self.assertEqual(count, 0)

  def test_compile_count_tpu_counts_new_executables(self):
    ops = _tpu_ops()
    count_before = ops.compile_count()

    # Eager TPU ops on a shape not used elsewhere compile a new executable
    # without going through Dynamo.
    x = torch.randn(5, 11, device=ops.device)
    _ = (x * 3.0).sin().cpu()

    self.assertGreater(ops.compile_count(), count_before)


if __name__ == "__main__":
  absltest.main()
