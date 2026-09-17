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

"""Tests for ci/tools/check_python_version_sync.py."""

import pathlib
import sys
import unittest

_CI_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CI_TOOLS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools import check_python_version_sync
from ci.tools import python_versions


class RepositoryIsInSyncTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):
  """The real tree must be in sync; this is the check that has teeth."""

  def test_no_drift(self):
    errors = check_python_version_sync.run_all_checks()
    self.assertEqual(errors, [], "\n".join(errors))


class DriftIsDetectedTest(
    unittest.TestCase  # UNITTEST_OK=CI utility script test without torch_tpu dependencies
):
  """A checker that can only pass is worthless, so assert it can fail."""

  def setUp(self):
    super().setUp()
    self.constants = python_versions.load_constants()

  def _drifted(self, **overrides):
    return {**self.constants, **overrides}

  def test_detects_dropped_version_in_module_bazel(self):
    versions = self.constants["SUPPORTED_PYTHON_VERSIONS"][:-1]
    errors = check_python_version_sync.check_module_bazel(
        self._drifted(SUPPORTED_PYTHON_VERSIONS=versions)
    )
    self.assertTrue(errors)
    self.assertIn("MODULE.bazel", errors[0])

  def test_detects_changed_default_in_bazelrc(self):
    errors = check_python_version_sync.check_bazelrc(
        self._drifted(DEFAULT_PYTHON_VERSION="9.99")
    )
    self.assertTrue(errors)
    self.assertIn("HERMETIC_PYTHON_VERSION", errors[0])

  def test_detects_dropped_version_in_benchmark_choices(self):
    versions = self.constants["SUPPORTED_PYTHON_VERSIONS"][:-1]
    errors = check_python_version_sync.check_run_benchmarks_choices(
        self._drifted(SUPPORTED_PYTHON_VERSIONS=versions)
    )
    self.assertTrue(errors)
    self.assertIn("run_benchmarks.yml", errors[0])

  def test_detects_dropped_version_in_workspace(self):
    versions = self.constants["SUPPORTED_PYTHON_VERSIONS"][:-1]
    errors = check_python_version_sync.check_workspace_requirements(
        self._drifted(SUPPORTED_PYTHON_VERSIONS=versions)
    )
    self.assertTrue(errors)
    self.assertIn("WORKSPACE", errors[0])

  def test_detects_stale_copybara_required_checks(self):
    # Pruning a version from the CPU presubmit matrix without updating
    # copy.bara.sky leaves Copybara blocking on a check that will never run.
    if not check_python_version_sync._COPYBARA_CONFIG.is_file():  # pylint: disable=protected-access
      self.skipTest("copy.bara.sky is not part of the GitHub export")
    errors = check_python_version_sync.check_copybara_required_checks(
        self._drifted(
            PRESUBMIT_CPU_PYTHON_VERSIONS=[
                self.constants["DEFAULT_PYTHON_VERSION"]
            ]
        )
    )
    self.assertTrue(errors)
    self.assertIn("copy.bara.sky", errors[0])


if __name__ == "__main__":
  unittest.main()
