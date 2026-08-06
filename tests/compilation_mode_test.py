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

import enum
from unittest import mock
import warnings
from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _loader
from torch_tpu._internal import env


class BuildType(enum.Enum):
  INTERNAL = 1
  OSS = 2


class OptMode(enum.Enum):
  OPTIMIZED = 1
  UNOPTIMIZED = 2


class WarningExpectation(enum.Enum):
  EXPECT_WARNING = 1
  EXPECT_NO_WARNING = 2


class CompilationModeTest(parameterized.TestCase):

  def test_env_exports_compilation_mode_attributes(self):
    self.assertTrue(hasattr(env, "TORCH_TPU_IS_OPTIMIZED_BUILD"))
    self.assertIsInstance(env.TORCH_TPU_IS_OPTIMIZED_BUILD, bool)

  @parameterized.parameters(
      # (build_type, opt_mode, warning_expectation)
      (
          BuildType.INTERNAL,
          OptMode.UNOPTIMIZED,
          WarningExpectation.EXPECT_NO_WARNING,
      ),
      (
          BuildType.OSS,
          OptMode.UNOPTIMIZED,
          WarningExpectation.EXPECT_WARNING,
      ),
      (
          BuildType.OSS,
          OptMode.OPTIMIZED,
          WarningExpectation.EXPECT_NO_WARNING,
      ),
  )
  def test_warn_if_unoptimized(self, build_type, opt_mode, warning_expectation):
    is_internal = build_type == BuildType.INTERNAL
    is_optimized = opt_mode == OptMode.OPTIMIZED
    expect_warning = warning_expectation == WarningExpectation.EXPECT_WARNING

    with mock.patch.multiple(
        env,
        IS_INTERNAL_TORCH_TPU=is_internal,
        TORCH_TPU_IS_OPTIMIZED_BUILD=is_optimized,
    ), warnings.catch_warnings(record=True) as recorded:
      _loader._warn_if_unoptimized()
      unopt_warnings = [
          w for w in recorded if "UNOPTIMIZED BUILD" in str(w.message)
      ]
      if expect_warning:
        self.assertLen(unopt_warnings, 1)
        self.assertIn("degraded", str(unopt_warnings[0].message).lower())
      else:
        self.assertEmpty(unopt_warnings)


if __name__ == "__main__":
  absltest.main()
