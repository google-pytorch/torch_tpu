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
import unittest
from unittest import mock

from absl.testing import absltest
from torch import torch_version
from torch_tpu._internal.utils import utils
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
      ver = utils.libtpu_version()
      self.assertIsInstance(ver, torch_version.TorchVersion)
      self.assertEqual(ver, torch_version.TorchVersion('0.4.0'))

  def test_libtpu_version_when_libtpu_unavailable(self):
    with mock.patch.dict(sys.modules, {'libtpu': None}):
      self.assertIsNone(utils.libtpu_version())

  def test_skip_if_libtpu_version_older_than_min_version(self):
    mock_libtpu = mock.MagicMock()
    mock_libtpu.__version__ = '0.0.40'
    with mock.patch.dict(sys.modules, {'libtpu': mock_libtpu}):

      @oss_utils.skip_if_libtpu_version(
          lambda v: v < (0, 0, 42),
          'Requires libtpu >= 0.0.42',
      )
      def dummy_test():
        pass

      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing unittest.SkipTest decorator behavior
          unittest.SkipTest, 'Requires libtpu >= 0.0.42'
      ):
        dummy_test()

  def test_skip_if_libtpu_version_no_skip_newer_than_min_version(self):
    mock_libtpu = mock.MagicMock()
    mock_libtpu.__version__ = '0.0.43'
    with mock.patch.dict(sys.modules, {'libtpu': mock_libtpu}):

      @oss_utils.skip_if_libtpu_version(
          lambda v: v < (0, 0, 42),
          'Requires libtpu >= 0.0.42',
      )
      def dummy_test():
        return 'ok'

      self.assertEqual(dummy_test(), 'ok')

  def test_skip_if_libtpu_version_no_skip_outside_oss(self):
    with mock.patch.dict(sys.modules, {'libtpu': None}):
      with mock.patch.object(oss_utils, 'is_oss', return_value=False):

        @oss_utils.skip_if_libtpu_version(
            lambda v: v < (0, 0, 42),
            'Requires libtpu >= 0.0.42',
        )
        def dummy_test():
          return 'ok'

        self.assertEqual(dummy_test(), 'ok')


if __name__ == '__main__':
  absltest.main()
