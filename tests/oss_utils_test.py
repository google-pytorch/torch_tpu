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

  def test_libtpu_version_when_libtpu_available(self):
    mock_libtpu = mock.MagicMock()
    mock_libtpu.__version__ = '0.4.0'
    with mock.patch.dict(sys.modules, {'libtpu': mock_libtpu}):
      self.assertEqual(oss_utils.libtpu_version(), '0.4.0')

  def test_libtpu_version_when_libtpu_unavailable(self):
    with mock.patch.dict(sys.modules, {'libtpu': None}):
      with self.assertRaisesRegex(ValueError, 'libtpu is not available'):
        oss_utils.libtpu_version()

  def test_skip_libtpu_older_than_min_version(self):
    mock_libtpu = mock.MagicMock()
    mock_libtpu.__version__ = '0.0.40'
    with mock.patch.dict(sys.modules, {'libtpu': mock_libtpu}):

      @oss_utils.skip_if_cloud_and_libtpu_older_than('0.0.42')
      def dummy_test(test_class):  # pylint: disable=unused-argument
        pass

      mock_self = mock.MagicMock(
          spec=absltest.TestCase  # ABSLTEST_OK=Mock spec for absltest
      )
      dummy_test(mock_self)
      mock_self.skipTest.assert_called_once_with(
          'dummy_test requires libtpu >= 0.0.42 in cloud, but got 0.0.40'
      )

  def test_no_skip_libtpu_newer_than_min_version(self):
    mock_libtpu = mock.MagicMock()
    mock_libtpu.__version__ = '0.0.43'
    with mock.patch.dict(sys.modules, {'libtpu': mock_libtpu}):

      @oss_utils.skip_if_cloud_and_libtpu_older_than('0.0.42')
      def dummy_test(test_class):  # pylint: disable=unused-argument
        pass

      mock_self = mock.MagicMock(
          spec=absltest.TestCase  # ABSLTEST_OK=Mock spec for absltest
      )
      dummy_test(mock_self)
      mock_self.skipTest.assert_not_called()

  def test_no_skip_outside_cloud(self):
    with mock.patch.dict(sys.modules, {'libtpu': None}):

      @oss_utils.skip_if_cloud_and_libtpu_older_than('0.0.42')
      def dummy_test(test_class):  # pylint: disable=unused-argument
        pass

      mock_self = mock.MagicMock(
          spec=absltest.TestCase  # ABSLTEST_OK=Mock spec for absltest
      )
      dummy_test(mock_self)
      mock_self.skipTest.assert_not_called()


if __name__ == '__main__':
  absltest.main()
