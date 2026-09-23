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

import sys
from unittest import mock

from absl.testing import absltest
from tests import oss_utils
from tests import seed_test_utils


class OssUtilsTest(seed_test_utils.RepeatableTest):

  def test_running_in_cloud_when_libtpu_available(self):
    mock_libtpu = mock.MagicMock()
    with mock.patch.dict(sys.modules, {'libtpu': mock_libtpu}):
      self.assertTrue(oss_utils.running_in_cloud())

  def test_running_in_cloud_when_libtpu_unavailable(self):
    with mock.patch.dict(sys.modules, {'libtpu': None}):
      self.assertFalse(oss_utils.running_in_cloud())


if __name__ == '__main__':
  absltest.main()
