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

from absl.testing import absltest
import torch
from tests import seed_test_utils


class NoTpuDeviceModuleTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Fail rather than skip, so that running this test on a TPU host is noticed
    # and fixed instead of silently losing coverage.
    if torch.accelerator.is_available():
      raise AssertionError("expected a host without TPU devices")

  def test_manual_seed_is_noop(self):
    # Like `torch.cuda`, seeding is a no-op when no device is present, so that
    # `torch.manual_seed()` keeps working on CPU-only hosts.
    torch.manual_seed(0)
    torch.tpu.manual_seed(0)
    torch.tpu.manual_seed_all(0)


if __name__ == "__main__":
  absltest.main()
