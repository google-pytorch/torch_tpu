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

"""Tests CPU-only autograd on hosts without TPU device nodes."""

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing


class CpuBackwardNoTpuTest(
    absltest.TestCase  # ABSLTEST_OK=no randomness to seed
):

  def test_cpu_backward_does_not_initialize_tpu_runtime(self):
    # This test is flaky (~5% failure rate on CPU machines): some CPU hosts
    # happen to have non-TPU device files (like VFIO or accel) in /dev. When
    # those files are present, TorchTPU assumes a TPU might be attached and
    # tries to start PjRt, so we skip the test on those hosts.
    if tt_testing.has_tpu_device_nodes():
      self.skipTest(
          "Host has VFIO/accel device files, so PjRt initialization will be "
          "attempted."
      )
    self.assertFalse(tt_testing.pjrt_init_attempted())

    # The autograd engine asks every registered device type for its device
    # count on the first backward(), even for a CPU-only graph. On a host
    # without TPU device nodes, that must not start the TPU runtime, which can
    # take minutes before it fails.
    x = torch.ones(2, requires_grad=True)
    (x * 3).sum().backward()

    self.assertEqual(x.grad.tolist(), [3.0, 3.0])
    self.assertFalse(tt_testing.pjrt_init_attempted())


if __name__ == "__main__":
  absltest.main()
