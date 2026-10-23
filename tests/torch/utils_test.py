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

"""Unit tests for TorchTPU upstream test utilities."""

import importlib
import os
import unittest
from unittest import mock

from absl.testing import absltest
import pytest
from tests.torch import utils


class UtilsTest(absltest.TestCase):  # ABSLTEST_OK=test runner utility

  def test_idempotent_patch_pytest_xfail_for_unittest(self):
    utils.patch_pytest_xfail_for_unittest()
    utils.patch_pytest_xfail_for_unittest()

  def test_patch_pytest_xfail_raises_skip_test(self):
    utils.patch_pytest_xfail_for_unittest()

    with self.assertRaises(  # ASSERT_RAISES_OK=Testing unittest.SkipTest
        unittest.SkipTest
    ) as cm:
      pytest.xfail("test imperative xfail")
    self.assertIn("pytest.xfail: test imperative xfail", str(cm.exception))

    outcomes = importlib.import_module("_pytest.outcomes")
    with self.assertRaises(  # ASSERT_RAISES_OK=Testing unittest.SkipTest
        unittest.SkipTest
    ) as cm:
      outcomes.xfail("test outcome xfail")
    self.assertIn("pytest.xfail: test outcome xfail", str(cm.exception))

  def test_patch_pytest_xfail_under_pytest(self):
    utils.patch_pytest_xfail_for_unittest()
    outcomes = importlib.import_module("_pytest.outcomes")

    with mock.patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_dummy"}):
      with self.assertRaises(  # ASSERT_RAISES_OK=Testing pytest XFailed
          outcomes.XFailed
      ) as cm:
        pytest.xfail("test native xfail")
      self.assertIn("test native xfail", str(cm.exception))


if __name__ == "__main__":
  absltest.main()
