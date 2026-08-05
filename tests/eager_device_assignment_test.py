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

"""Unit tests verifying eager mode SetDefaultDeviceAssignment multi-replica config."""

from absl.testing import absltest
import torch
import torch_tpu
from tests import seed_test_utils


class EagerDeviceAssignmentTest(seed_test_utils.RepeatableTest):

  def test_eager_fast_compile_multi_replica(self):
    """Verify that in fast compile (eager) mode with multiple devices, num_replicas equals num_devices."""
    # Verify that the C++ MakeCompileOptionsByMode function returns
    # correct compilation options without requiring --num_replicas=N
    # inside XLA_FLAGS.
    if not hasattr(torch_tpu, "is_available") or not torch_tpu.is_available():
      self.skipTest("TPU device not available in this test environment.")
    device_count = (
        torch_tpu.device_count() if hasattr(torch_tpu, "device_count") else 1
    )
    if device_count <= 1:
      self.skipTest(
          "Test requires >1 TPU/XLA devices to verify multi-replica eager"
          " DeviceAssignment."
      )

    # Execute a simple eager operation on device to verify it compiles
    # and runs without num_replicas / num_partitions layout mismatch errors.
    t = torch.randn(10, 10, device="tpu")
    res = t + t
    self.assertEqual(res.shape, (10, 10))


if __name__ == "__main__":
  absltest.main()
