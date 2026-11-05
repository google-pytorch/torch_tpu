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
from perflab import registry as registry_lib
from perflab import step_lib
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


_SWEPT = types.SimpleNamespace(
    name="swept",
    skipped_run_modes=frozenset(),
    stepper=step_lib.StepperType.FORWARD,
    batch_sweep=registry_lib.BatchSweep(),
    microbenchmark_name=None,
)
_COMPILED = common.RunMode.COMPILED
_EAGER = common.RunMode.EAGER_DEFAULT


class _FakeBatchSweepTest(base_test.BaseBenchmarkTest):
  """Runs the batch sweep cases of one benchmark; see _FakeBenchmarkTest."""

  framework = mode_lib.Framework.TORCH
  platform = target_lib.Platform.V7_1X1X1
  case_params = {
      "bs2_case": (_SWEPT, _COMPILED, 2),
      "bs4_case": (_SWEPT, _COMPILED, 4),
      "bs8_case": (_SWEPT, _COMPILED, 8),
      "eager_bs8_case": (_SWEPT, _EAGER, 8),
  }

  def bs2_case(self):
    self.fail("the child process is replaced")

  def bs4_case(self):
    self.fail("the child process is replaced")

  def bs8_case(self):
    self.fail("the child process is replaced")

  def eager_bs8_case(self):
    self.fail("the child process is replaced")


def _out_of_memory_payload(batch_size: int) -> export_lib.BenchmarkData:
  return export_lib.BenchmarkData(
      spec_name="swept",
      platform=target_lib.Platform.V7_1X1X1,
      framework=mode_lib.Framework.TORCH,
      run_mode=_COMPILED,
      succeeded=True,
      metrics=metrics_lib.PerformanceMetrics(),
      microbenchmark_name=export_lib.batch_size_key(batch_size),
      batch_size=batch_size,
      out_of_memory=True,
  )


class BaseBenchmarkTestBatchSweepTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(
            (flags_lib.RUN_IN_SUBPROCESS, True),
            (flags_lib.SKIP_BEHAVIOR, "skip"),
            (flags_lib.DRY_RUN, False),
        )
    )
    self.enter_context(
        mock.patch.dict(
            base_test._out_of_memory_batch_sizes,  # pylint: disable=protected-access
            {},
            clear=True,
        )
    )
    self.exported = []
    self.enter_context(
        mock.patch.object(export_lib, "export", new=self.exported.append)
    )
    self.run_isolated = self.enter_context(
        mock.patch.object(base_test, "run_isolated", autospec=True)
    )

  def _run(self, method_name: str) -> unittest.TestResult:
    result = unittest.TestResult()  # UNITTEST_OK=result of a fake test case
    _FakeBatchSweepTest(method_name).run(result)
    return result

  def test_larger_batch_sizes_are_skipped_after_out_of_memory(self):
    self.run_isolated.return_value = (
        base_test.Outcome.SKIPPED,
        "ran out of device memory",
        _out_of_memory_payload(4),
    )
    self._run("bs4_case")
    self.run_isolated.reset_mock()

    result = self._run("bs8_case")

    self.run_isolated.assert_not_called()
    self.assertLen(result.skipped, 1)
    self.assertIn(
        "ran out of device memory in run mode compiled at batch size 4",
        result.skipped[0][1],
    )
    # Batch size 4 was measured to run out of memory; batch size 8 was not
    # run, so it is exported as skipped rather than as out of memory.
    self.assertEqual(
        [
            (d.batch_size, d.microbenchmark_name, d.out_of_memory, d.skipped)
            for d in self.exported
        ],
        [(4, "bs4", True, False), (8, "bs8", False, True)],
    )

  def test_smaller_batch_sizes_still_run_after_out_of_memory(self):
    base_test.note_out_of_memory("swept", _COMPILED, 4)
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("bs2_case")

    self.run_isolated.assert_called_once()

  def test_other_run_modes_still_run_after_out_of_memory(self):
    base_test.note_out_of_memory("swept", _COMPILED, 4)
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("eager_bs8_case")

    self.run_isolated.assert_called_once()

  def test_crashed_case_is_exported_with_its_batch_size(self):
    self.run_isolated.return_value = (
        base_test.Outcome.CRASHED,
        "child died",
        None,
    )

    self._run("bs2_case")

    (data,) = self.exported
    self.assertFalse(data.succeeded)
    self.assertEqual(data.run_mode, _COMPILED)
    self.assertEqual(data.batch_size, 2)
    self.assertEqual(data.microbenchmark_name, "bs2")

  def test_batch_size_key_extends_the_microbenchmark_name(self):
    spec = types.SimpleNamespace(name="layer", microbenchmark_name="shape_1")
    test = _FakeBatchSweepTest("bs2_case")

    test.report_result(spec, _COMPILED, succeeded=True, batch_size=2)
    test.report_result(spec, _COMPILED, succeeded=True)

    self.assertEqual(
        [d.microbenchmark_name for d in self.exported],
        ["shape_1_bs2", "shape_1"],
    )

  def test_get_out_of_memory_skip_reason(self):
    reason = base_test.get_out_of_memory_skip_reason
    self.assertIsNone(reason("swept", _COMPILED, 8))
    base_test.note_out_of_memory("swept", _COMPILED, 8)
    base_test.note_out_of_memory("swept", _COMPILED, 16)
    self.assertIsNone(reason("swept", _COMPILED, 4))
    self.assertIsNone(reason("swept", _COMPILED, None))
    self.assertIsNone(reason("swept", _EAGER, 16))
    self.assertIn("batch size 8", reason("swept", _COMPILED, 16))


if __name__ == "__main__":
  absltest.main()
