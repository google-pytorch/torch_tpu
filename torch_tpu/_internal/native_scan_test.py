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

"""Tests for the libtpu native-scan support gate."""

import sys
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal import native_scan
from tests import seed_test_utils


class NativeScanTest(seed_test_utils.RepeatableTest):

  def test_parse_version_plain(self):
    self.assertEqual(native_scan._parse_version("0.0.41"), (0, 0, 41))

  def test_parse_version_strips_prerelease_suffix(self):
    self.assertEqual(
        native_scan._parse_version("0.0.41.dev20260101"), (0, 0, 41)
    )

  def test_parse_version_multi_digit_components_ordered_numerically(self):
    # Guards against lexical string comparison: 0.0.40 < 0.0.41 < 0.0.100.
    self.assertLess(
        native_scan._parse_version("0.0.40"),
        native_scan._parse_version("0.0.100"),
    )

  def _patch_libtpu(self, version):
    fake = mock.MagicMock()
    fake.__version__ = version
    return mock.patch.dict(sys.modules, {"libtpu": fake})

  def test_supported_when_version_above_minimum(self):
    with self._patch_libtpu("0.0.44"):
      self.assertTrue(native_scan.libtpu_supports_native_scan())

  def test_supported_when_version_well_above_minimum(self):
    with self._patch_libtpu("0.4.0"):
      self.assertTrue(native_scan.libtpu_supports_native_scan())

  def test_unsupported_when_version_at_pinned_oss_wheel(self):
    # torchtpu-vllm pins libtpu 0.0.40, which predates the native scan emitter.
    with self._patch_libtpu("0.0.40"):
      self.assertFalse(native_scan.libtpu_supports_native_scan())

  def test_unsupported_when_libtpu_missing(self):
    with mock.patch.dict(sys.modules, {"libtpu": None}):
      self.assertFalse(native_scan.libtpu_supports_native_scan())

  def test_unsupported_when_version_missing(self):
    fake = mock.MagicMock()
    fake.__version__ = ""
    with mock.patch.dict(sys.modules, {"libtpu": fake}):
      self.assertFalse(native_scan.libtpu_supports_native_scan())


if __name__ == "__main__":
  absltest.main()
