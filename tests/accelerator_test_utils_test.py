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

from unittest import mock

from absl.testing import absltest
from torch_tpu._internal.utils import hardware
from tests import accelerator_test_utils
from tests import oss_utils
from tests import seed_test_utils


class AcceleratorTestUtilsTest(seed_test_utils.RepeatableTest):

  def test_get_tpu_device_count_returns_default_when_no_tpus_detected(self):
    with mock.patch.object(hardware, "get_tpu_device_count", return_value=0):
      self.assertEqual(accelerator_test_utils.get_tpu_device_count(), 0)
      self.assertEqual(
          accelerator_test_utils.get_tpu_device_count(default=2), 2
      )

  def test_get_tpu_device_count_caps_at_four_in_oss(self):
    with (
        mock.patch.object(hardware, "get_tpu_device_count", return_value=8),
        mock.patch.object(oss_utils, "is_oss", return_value=True),
    ):
      self.assertEqual(accelerator_test_utils.get_tpu_device_count(), 4)

  def test_get_tpu_device_count_returns_full_count_outside_oss(self):
    with (
        mock.patch.object(hardware, "get_tpu_device_count", return_value=8),
        mock.patch.object(oss_utils, "is_oss", return_value=False),
    ):
      self.assertEqual(accelerator_test_utils.get_tpu_device_count(), 8)


if __name__ == "__main__":
  absltest.main()
