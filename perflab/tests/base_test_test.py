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

"""Tests for perflab/base_test.py."""

from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from perflab import base_test
from perflab import common
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from tests import seed_test_utils


def _factory() -> tuple[Any, ...]:
  return ("model", (), {}, None)


def _case(name: str) -> base_test.CaseParams:
  spec = registry_lib.BenchmarkSpec(
      name=name, factory=_factory, stepper=step_lib.StepperType.FORWARD
  )
  return spec, common.RunMode.EAGER_DEFAULT


def _run_benchmark_suite(
    case_params: dict[str, base_test.CaseParams],
) -> unittest.TestResult:  # UNITTEST_OK=TestResult of the fake suite.
  """Runs a fake benchmark binary whose cases all skip at run time.

  The fake is defined here rather than at module scope so that the test runner
  does not collect it.

  Args:
    case_params: The cases of the fake binary.

  Returns:
    The result of running every test method of the fake binary.
  """

  class FakeBenchmarkTest(base_test.BaseBenchmarkTest):
    framework = mode_lib.Framework.TORCH
    platform = target_lib.Platform.CPU

    def test_benchmark(self):
      self.skipTest("stands in for the generated, skipped cases")

  FakeBenchmarkTest.case_params = case_params
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(FakeBenchmarkTest)
  result = unittest.TestResult()  # UNITTEST_OK=TestResult of the fake suite.
  suite.run(result)
  return result


class CasesSelectedTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            known_failures,
            "KNOWN_FAILURES",
            (
                known_failures.KnownFailure(
                    benchmark="known_bad",
                    framework=mode_lib.Framework.TORCH,
                    reason="failure",
                ),
            ),
        )
    )
    self.enter_context(
        flagsaver.flagsaver((flags_lib.RUN_IN_SUBPROCESS, False))
    )

  def assert_fails_cases_selected(self, result: unittest.TestResult) -> None:
    self.assertFalse(result.wasSuccessful())
    self.assertLen(result.failures, 1)
    test, _ = result.failures[0]
    self.assertEqual(test.id().rpartition(".")[2], "test_cases_selected")

  def test_fails_when_every_case_is_skipped(self):
    result = _run_benchmark_suite(
        {"test_benchmark_known_bad": _case("known_bad")}
    )
    self.assert_fails_cases_selected(result)

  def test_fails_without_cases(self):
    self.assert_fails_cases_selected(_run_benchmark_suite({}))

  @flagsaver.flagsaver((flags_lib.SKIP_BEHAVIOR, "run_skipped"))
  def test_fails_when_run_skipped_selects_nothing(self):
    result = _run_benchmark_suite({"test_benchmark_healthy": _case("healthy")})
    self.assert_fails_cases_selected(result)

  def test_passes_when_a_case_is_selected(self):
    result = _run_benchmark_suite({
        "test_benchmark_known_bad": _case("known_bad"),
        "test_benchmark_healthy": _case("healthy"),
    })
    self.assertTrue(result.wasSuccessful(), result.failures)


if __name__ == "__main__":
  absltest.main()
