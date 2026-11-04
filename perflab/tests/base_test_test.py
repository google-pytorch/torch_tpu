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

import pickle
import types
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import base_test
from perflab import common
from perflab import export as export_lib
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import target as target_lib
from tests import seed_test_utils

_KNOWN_BAD = types.SimpleNamespace(
    name="known_bad", skipped_run_modes=frozenset(), microbenchmark_name=None
)
_SKIPS_COMPILED = types.SimpleNamespace(
    name="skips_compiled",
    skipped_run_modes=frozenset({"compiled"}),
    microbenchmark_name=None,
)
_HEALTHY = types.SimpleNamespace(
    name="healthy", skipped_run_modes=frozenset(), microbenchmark_name=None
)
_MICRO = types.SimpleNamespace(
    name="layer_linear_eval_batch_size_1",
    skipped_run_modes=frozenset(),
    microbenchmark_name="batch_size_1",
)


class _FakeBenchmarkTest(base_test.BaseBenchmarkTest):
  """Runs the cases below, which are skipped before a child process starts.

  The case methods do not start with "test", so the test loader does not run
  them on their own.
  """

  framework = mode_lib.Framework.TORCH
  platform = target_lib.Platform.V7_1X1X1
  case_params = {
      "known_bad_case": (_KNOWN_BAD, common.RunMode.COMPILED),
      "skipped_run_mode_case": (_SKIPS_COMPILED, common.RunMode.COMPILED),
      "healthy_case": (_HEALTHY, common.RunMode.COMPILED),
      "micro_case": (_MICRO, common.RunMode.COMPILED),
  }

  def known_bad_case(self):
    self.fail("a skipped case must not run")

  def skipped_run_mode_case(self):
    self.fail("a skipped case must not run")

  def healthy_case(self):
    self.fail("a skipped case must not run")

  def micro_case(self):
    self.fail("a skipped case must not run")


class BaseBenchmarkTestTest(seed_test_utils.RepeatableTest):

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
                    device_kind=target_lib.DeviceKind.TPU,
                    run_modes=frozenset({common.RunMode.COMPILED}),
                    reason="simulated TPU compiled failure",
                ),
            ),
        )
    )
    self.enter_context(
        flagsaver.flagsaver(
            (flags_lib.RUN_IN_SUBPROCESS, True),
            (flags_lib.SKIP_BEHAVIOR, "skip"),
            (flags_lib.DRY_RUN, False),
        )
    )
    self.mock_export_skip = self.enter_context(
        mock.patch.object(export_lib, "export_skip", autospec=True)
    )

  def _run(self, method_name: str) -> unittest.TestResult:
    result = unittest.TestResult()  # UNITTEST_OK=result of a fake test case
    _FakeBenchmarkTest(method_name).run(result)
    return result

  def test_known_failure_skip_is_exported(self):
    result = self._run("known_bad_case")

    self.assertLen(result.skipped, 1)
    self.assertEmpty(result.failures)
    self.mock_export_skip.assert_called_once_with(
        spec_name="known_bad",
        platform=target_lib.Platform.V7_1X1X1,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.COMPILED,
    )

  def test_skipped_run_mode_is_not_exported(self):
    result = self._run("skipped_run_mode_case")

    self.assertLen(result.skipped, 1)
    self.mock_export_skip.assert_not_called()

  def test_run_skipped_behavior_does_not_export_its_skips(self):
    with flagsaver.flagsaver((flags_lib.SKIP_BEHAVIOR, "run_skipped")):
      result = self._run("healthy_case")

    self.assertLen(result.skipped, 1)
    self.mock_export_skip.assert_not_called()

  def test_report_skip_does_not_export_in_dry_run(self):
    with flagsaver.flagsaver((flags_lib.DRY_RUN, True)):
      _FakeBenchmarkTest("known_bad_case").report_skip(
          _KNOWN_BAD, common.RunMode.COMPILED
      )

    self.mock_export_skip.assert_not_called()


class BaseBenchmarkTestChildOutcomeTest(seed_test_utils.RepeatableTest):
  """How the parent records and exports what the isolated child reported.

  The child process is replaced by its reported outcome; spawning it is not
  under test.
  """

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(
            (flags_lib.RUN_IN_SUBPROCESS, True),
            (flags_lib.SKIP_BEHAVIOR, "skip"),
            (flags_lib.DRY_RUN, False),
        )
    )
    self.exported = []
    self.enter_context(
        mock.patch.object(export_lib, "export", new=self.exported.append)
    )

  def _run_with_child_outcome(
      self,
      outcome: base_test.Outcome,
      detail: str | None,
      payload,
      method_name: str = "healthy_case",
  ) -> unittest.TestResult:
    self.enter_context(
        mock.patch.object(
            base_test,
            "run_isolated",
            autospec=True,
            return_value=(outcome, detail, payload),
        )
    )
    result = unittest.TestResult()  # UNITTEST_OK=result of a fake test case
    _FakeBenchmarkTest(method_name).run(result)
    return result

  @parameterized.parameters(
      base_test.Outcome.CRASHED, base_test.Outcome.TIMED_OUT
  )
  def test_child_that_died_is_exported_as_failed(self, outcome):
    result = self._run_with_child_outcome(outcome, "child died", None)

    self.assertLen(result.errors, 1)
    (data,) = self.exported
    self.assertEqual(data.spec_name, "healthy")
    self.assertEqual(data.platform, target_lib.Platform.V7_1X1X1)
    self.assertEqual(data.framework, mode_lib.Framework.TORCH)
    self.assertEqual(data.run_mode, common.RunMode.COMPILED)
    self.assertFalse(data.succeeded)
    self.assertIsNone(data.microbenchmark_name)

  def test_child_that_died_is_exported_with_microbenchmark_name(self):
    self._run_with_child_outcome(
        base_test.Outcome.CRASHED, "child died", None, method_name="micro_case"
    )

    (data,) = self.exported
    self.assertEqual(data.spec_name, "layer_linear_eval_batch_size_1")
    self.assertEqual(data.microbenchmark_name, "batch_size_1")
    self.assertFalse(data.succeeded)

  @parameterized.parameters(
      base_test.Outcome.SUCCESS, base_test.Outcome.FAILURE
  )
  def test_child_payload_is_exported(self, outcome):
    payload = export_lib.BenchmarkData(
        spec_name="healthy",
        platform=target_lib.Platform.V7_1X1X1,
        framework=mode_lib.Framework.TORCH,
        run_mode=common.RunMode.COMPILED,
        succeeded=outcome == base_test.Outcome.SUCCESS,
        metrics=metrics_lib.PerformanceMetrics(e2e_wall_time_seconds=1.0),
    )

    self._run_with_child_outcome(outcome, None, payload)

    self.assertEqual(self.exported, [payload])

  def test_child_skip_without_payload_exports_nothing(self):
    result = self._run_with_child_outcome(
        base_test.Outcome.SKIPPED, "unsupported", None
    )

    self.assertLen(result.skipped, 1)
    self.assertEmpty(self.exported)


class BaseBenchmarkTestReportResultTest(seed_test_utils.RepeatableTest):
  """What report_result exports, in process and from the isolated child."""

  def setUp(self):
    super().setUp()
    self.exported = []
    self.enter_context(
        mock.patch.object(export_lib, "export", new=self.exported.append)
    )

  def test_in_process_result_carries_microbenchmark_name(self):
    _FakeBenchmarkTest("micro_case").report_result(
        _MICRO, common.RunMode.COMPILED, succeeded=True
    )

    (data,) = self.exported
    self.assertEqual(data.spec_name, "layer_linear_eval_batch_size_1")
    self.assertEqual(data.microbenchmark_name, "batch_size_1")
    self.assertTrue(data.succeeded)

  def test_isolated_child_payload_carries_microbenchmark_name(self):
    test = _FakeBenchmarkTest("micro_case")
    test.in_isolated_child = True

    test.report_result(_MICRO, common.RunMode.COMPILED, succeeded=True)

    self.assertEmpty(self.exported)
    # The child pickles the payload to send it to the parent.
    payload = pickle.loads(pickle.dumps(test.isolated_payload))
    self.assertEqual(payload.spec_name, "layer_linear_eval_batch_size_1")
    self.assertEqual(payload.microbenchmark_name, "batch_size_1")


if __name__ == "__main__":
  absltest.main()
