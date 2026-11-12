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

from collections.abc import MutableSequence, Sequence
import contextlib
import os
import pickle
import types
from typing import Literal
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import base_test
from perflab import common
from perflab import device_ops as device_ops_lib
from perflab import discovery as discovery_lib
from perflab import export as export_lib
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib
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
  # Keeps the test loader from collecting the inherited imports test.
  test_benchmark_imports = None

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
      base_test.Outcome.SUCCESS,
      base_test.Outcome.FAILURE,
      base_test.Outcome.ERROR,
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


def _run_case_spec(
    name: str,
    skipped_run_modes: frozenset[str] = frozenset(),
    workload: registry_lib.WorkloadFn | None = None,
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=lambda ctx: None,
      stepper=step_lib.StepperType.FORWARD,
      presubmit=registry_lib.PresubmitPolicy.RUN,
      dtype=target_lib.DType.FP32,
      skipped_run_modes=skipped_run_modes,
      workload=workload,
  )


def _raise_workload_error(ctx):
  del ctx  # Unused.
  raise ValueError("simulated workload failure")


_MEASURED_SPEC = _run_case_spec("measured")
_FAILING_SPEC = _run_case_spec("failing")
_UNSUPPORTED_SPEC = _run_case_spec("unsupported")
_BAD_WORKLOAD_SPEC = _run_case_spec(
    "bad_workload", workload=_raise_workload_error
)
_SKIPPED_SPEC = _run_case_spec(
    "skipped", skipped_run_modes=frozenset({common.RunMode.EAGER_DEFAULT.value})
)

# The name of a setup hook of `_fake_run_case_test_class` that raises.
_FailingHook = Literal["make_device_ops", "reset_state"]


def _fake_run_case_test_class(
    calls: MutableSequence[str],
    failures: Sequence[discovery_lib.ImportFailure] = (),
    failing_hook: _FailingHook | None = None,
) -> type[base_test.BaseBenchmarkTest]:
  """Returns a test class that runs `run_case` with recording hooks.

  The class is built here rather than at module level so that the test loader
  does not run the `test_benchmark_imports` method it inherits.

  Args:
    calls: The list that the hooks append the names of their calls to.
    failures: The import failures that the class reports.
    failing_hook: The name of the setup hook that raises, if any.
  """

  class FakeRunCaseTest(base_test.BaseBenchmarkTest):
    framework = mode_lib.Framework.TORCH
    platform = target_lib.Platform.CPU
    import_failures = tuple(failures)

    def make_device_ops(self, target):
      calls.append("make_device_ops")
      if failing_hook == "make_device_ops":
        raise RuntimeError("simulated make_device_ops failure")
      return mock.create_autospec(device_ops_lib.DeviceOps, instance=True)

    def reset_state(self, target, device_ops, mode):
      calls.append("reset_state")
      if failing_hook == "reset_state":
        raise RuntimeError("simulated reset_state failure")

    @contextlib.contextmanager
    def run_context(self, mode, target):
      calls.append("enter_run_context")
      yield
      calls.append("exit_run_context")

    def make_run_step(self, spec, ctx, mode):
      calls.append("make_run_step")
      if spec.name == "failing":
        raise RuntimeError("simulated build failure")
      if spec.name == "unsupported":
        raise target_lib.UnsupportedBenchmark("simulated unsupported target")
      return mock.create_autospec(step_lib.Stepper, instance=True), None

    def after_measure(self, spec, ctx, target, case_name, metrics):
      calls.append(f"after_measure:{case_name}")

    def measured_case(self):
      self.run_case(_MEASURED_SPEC, common.RunMode.EAGER_DEFAULT)

    def failing_case(self):
      self.run_case(_FAILING_SPEC, common.RunMode.EAGER_DEFAULT)

    def unsupported_case(self):
      self.run_case(_UNSUPPORTED_SPEC, common.RunMode.EAGER_DEFAULT)

    def bad_workload_case(self):
      self.run_case(_BAD_WORKLOAD_SPEC, common.RunMode.EAGER_DEFAULT)

    def skipped_case(self):
      self.run_case(_SKIPPED_SPEC, common.RunMode.EAGER_DEFAULT)

  return FakeRunCaseTest


class BaseBenchmarkTestRunCaseTest(seed_test_utils.RepeatableTest):
  """How run_case drives the hooks of a subclass and reports the case."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(
            (flags_lib.RUN_IN_SUBPROCESS, False),
            (flags_lib.SKIP_BEHAVIOR, "skip"),
            (flags_lib.DRY_RUN, False),
            (measure_lib.POST_WARMUP_STEPS, 0),
        )
    )
    self.exported = []
    self.enter_context(
        mock.patch.object(export_lib, "export", new=self.exported.append)
    )
    self.mock_export_skip = self.enter_context(
        mock.patch.object(export_lib, "export_skip", autospec=True)
    )
    self.mock_measure = self.enter_context(
        mock.patch.object(
            measure_lib,
            "measure",
            autospec=True,
            return_value=metrics_lib.PerformanceMetrics(
                e2e_wall_time_seconds=2.0, first_step_time_seconds=1.0
            ),
        )
    )
    self.calls = []

  def _run(
      self,
      method_name: str,
      failures: Sequence[discovery_lib.ImportFailure] = (),
      failing_hook: _FailingHook | None = None,
  ) -> unittest.TestResult:
    result = unittest.TestResult()  # UNITTEST_OK=result of a fake test case
    _fake_run_case_test_class(self.calls, failures, failing_hook)(
        method_name
    ).run(result)
    return result

  def test_measured_case_is_exported_as_succeeded(self):
    result = self._run("measured_case")

    self.assertTrue(result.wasSuccessful())
    self.assertEmpty(result.skipped)
    self.assertEqual(
        self.calls,
        [
            "make_device_ops",
            "reset_state",
            "enter_run_context",
            "make_run_step",
            "exit_run_context",
            "after_measure:measured_eager_default",
        ],
    )
    self.assertEqual(
        self.mock_measure.call_args.kwargs["name"], "measured_eager_default"
    )
    (data,) = self.exported
    self.assertEqual(data.spec_name, "measured")
    self.assertEqual(data.platform, target_lib.Platform.CPU)
    self.assertEqual(data.framework, mode_lib.Framework.TORCH)
    self.assertEqual(data.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(data.succeeded)
    self.assertEqual(data.metrics.e2e_wall_time_seconds, 2.0)

  def test_unmeasured_case_fails(self):
    self.mock_measure.return_value = metrics_lib.PerformanceMetrics()

    result = self._run("measured_case")

    self.assertLen(result.failures, 1)

  def test_failed_case_is_exported_as_failed(self):
    result = self._run("failing_case")

    self.assertLen(result.errors, 1)
    self.assertIn("simulated build failure", result.errors[0][1])
    (data,) = self.exported
    self.assertEqual(data.spec_name, "failing")
    self.assertFalse(data.succeeded)

  @parameterized.parameters("make_device_ops", "reset_state")
  def test_case_whose_setup_fails_is_exported_as_failed(self, failing_hook):
    result = self._run("measured_case", failing_hook=failing_hook)

    self.assertLen(result.errors, 1)
    self.assertIn(f"simulated {failing_hook} failure", result.errors[0][1])
    (data,) = self.exported
    self.assertEqual(data.spec_name, "measured")
    self.assertFalse(data.succeeded)

  def test_case_whose_context_fails_to_resolve_is_exported_as_failed(self):
    result = self._run("bad_workload_case")

    self.assertLen(result.errors, 1)
    self.assertIn("simulated workload failure", result.errors[0][1])
    (data,) = self.exported
    self.assertEqual(data.spec_name, "bad_workload")
    self.assertFalse(data.succeeded)

  def test_isolated_child_hands_setup_failure_to_parent(self):
    test = _fake_run_case_test_class(self.calls, failing_hook="reset_state")(
        "measured_case"
    )
    test.in_isolated_child = True
    result = unittest.TestResult()  # UNITTEST_OK=result of a fake test case

    test.run(result)

    self.assertLen(result.errors, 1)
    self.assertEmpty(self.exported)
    self.assertEqual(test.isolated_payload.spec_name, "measured")
    self.assertFalse(test.isolated_payload.succeeded)

  def test_unsupported_case_is_skipped(self):
    result = self._run("unsupported_case")

    self.assertEmpty(result.errors)
    self.assertLen(result.skipped, 1)
    self.assertIn("simulated unsupported target", result.skipped[0][1])
    self.assertEmpty(self.exported)

  def test_skipped_case_is_not_run(self):
    result = self._run("skipped_case")

    self.assertLen(result.skipped, 1)
    self.assertEmpty(self.calls)
    self.assertEmpty(self.exported)
    self.mock_export_skip.assert_not_called()

  def test_run_skipped_behavior_skips_cases_that_are_not_skipped(self):
    with flagsaver.flagsaver((flags_lib.SKIP_BEHAVIOR, "run_skipped")):
      result = self._run("measured_case")

    self.assertLen(result.skipped, 1)
    self.assertEmpty(self.calls)

  def test_assert_raise_behavior_expects_skipped_case_to_raise(self):
    with flagsaver.flagsaver((flags_lib.SKIP_BEHAVIOR, "assert_raise")):
      result = self._run("skipped_case")

    # The skipped case measures fine, so assertRaises fails.
    self.assertLen(result.failures, 1)
    self.assertEmpty(self.exported)

  def test_assert_raise_behavior_reports_setup_failure_as_error(self):
    with flagsaver.flagsaver((flags_lib.SKIP_BEHAVIOR, "assert_raise")):
      result = self._run("skipped_case", failing_hook="reset_state")

    # A setup failure is not the failure that the skipped case should have.
    self.assertLen(result.errors, 1)
    self.assertIn("simulated reset_state failure", result.errors[0][1])
    self.assertNotIn("make_run_step", self.calls)

  def test_dry_run_writes_the_test_method_name(self):
    output_dir = self.create_tempdir().full_path
    self.enter_context(
        mock.patch.dict(os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": output_dir})
    )

    with flagsaver.flagsaver((flags_lib.DRY_RUN, True)):
      result = self._run("measured_case")

    self.assertTrue(result.wasSuccessful())
    self.assertEmpty(self.calls)
    self.assertEmpty(self.exported)
    with open(os.path.join(output_dir, "test_targets.txt")) as f:
      self.assertEqual(f.read(), "measured_case\n")

  def test_import_failures_fail_the_imports_test(self):
    failure = discovery_lib.ImportFailure(
        module="perflab.models.broken", error=ImportError("no module")
    )

    result = self._run("test_benchmark_imports", failures=[failure])

    self.assertLen(result.failures, 1)
    self.assertIn("perflab.models.broken", result.failures[0][1])

  def test_imports_test_passes_without_import_failures(self):
    result = self._run("test_benchmark_imports")

    self.assertTrue(result.wasSuccessful())


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
      "bs16_case": (_SWEPT, _COMPILED, 16),
      "eager_bs8_case": (_SWEPT, _EAGER, 8),
  }
  # Keeps the test loader from collecting the inherited imports test.
  test_benchmark_imports = None

  def bs2_case(self):
    self.fail("the child process is replaced")

  def bs4_case(self):
    self.fail("the child process is replaced")

  def bs8_case(self):
    self.fail("the child process is replaced")

  def bs16_case(self):
    self.fail("the child process is replaced")

  def eager_bs8_case(self):
    self.fail("the child process is replaced")


def _measured_payload(
    batch_size: int | None,
    samples_per_second: float,
    *,
    workload_batch_size: int | None = None,
    mode: common.RunMode = _COMPILED,
) -> export_lib.BenchmarkData:
  """Returns the result of a measured case with the given throughput."""
  workload_batch_size = workload_batch_size or batch_size
  return export_lib.BenchmarkData(
      spec_name="swept",
      platform=target_lib.Platform.V7_1X1X1,
      framework=mode_lib.Framework.TORCH,
      run_mode=mode,
      succeeded=True,
      metrics=metrics_lib.PerformanceMetrics(
          post_warmup_step_time_seconds=workload_batch_size
          / samples_per_second,
          throughput_step_time_seconds=workload_batch_size / samples_per_second,
          step_work=workload_lib.StepWork(
              workload=workload_lib.Workload(workload_batch_size),
              samples_per_step=workload_batch_size,
              tokens_per_step=None,
          ),
      ),
      microbenchmark_name=(
          None if batch_size is None else export_lib.batch_size_key(batch_size)
      ),
      batch_size=batch_size,
  )


def _failed_payload(batch_size: int) -> export_lib.BenchmarkData:
  return export_lib.BenchmarkData(
      spec_name="swept",
      platform=target_lib.Platform.V7_1X1X1,
      framework=mode_lib.Framework.TORCH,
      run_mode=_COMPILED,
      succeeded=False,
      metrics=metrics_lib.PerformanceMetrics(),
      microbenchmark_name=export_lib.batch_size_key(batch_size),
      batch_size=batch_size,
  )


def _unmeasured_throughput_payload(
    batch_size: int,
) -> export_lib.BenchmarkData:
  """Returns a measured case without a throughput, e.g. a pipelined OOM."""
  return export_lib.BenchmarkData(
      spec_name="swept",
      platform=target_lib.Platform.V7_1X1X1,
      framework=mode_lib.Framework.TORCH,
      run_mode=_COMPILED,
      succeeded=True,
      metrics=metrics_lib.PerformanceMetrics(
          post_warmup_step_time_seconds=0.01,
          throughput_step_time_seconds=None,
          step_work=workload_lib.StepWork(
              workload=workload_lib.Workload(batch_size),
              samples_per_step=batch_size,
              tokens_per_step=None,
          ),
      ),
      microbenchmark_name=export_lib.batch_size_key(batch_size),
      batch_size=batch_size,
  )


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
            # Every batch size counts towards the patience, unless a test sets
            # the guard against noisy small batch sizes.
            (flags_lib.BATCH_SWEEP_PATIENCE_MIN_BS, 1),
        )
    )
    for state in (
        base_test._out_of_memory_batch_sizes,  # pylint: disable=protected-access
        base_test._failed_batch_sizes,  # pylint: disable=protected-access
        base_test._throughputs,  # pylint: disable=protected-access
        base_test._stopped_batch_sizes,  # pylint: disable=protected-access
    ):
      self.enter_context(mock.patch.dict(state, {}, clear=True))
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

  def _run_measured(self, method_name: str, payload: export_lib.BenchmarkData):
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, payload)
    return self._run(method_name)

  def test_sweep_stops_after_two_batch_sizes_without_improvement(self):
    self._run_measured("bs2_case", _measured_payload(2, 100.0))
    # Less than 3% above batch size 2, twice in a row.
    self._run_measured("bs4_case", _measured_payload(4, 102.0))
    self._run_measured("bs8_case", _measured_payload(8, 101.0))
    self.run_isolated.reset_mock()

    result = self._run("bs16_case")

    self.run_isolated.assert_not_called()
    self.assertLen(result.skipped, 1)
    self.assertIn(
        "stopped improving its throughput in run mode compiled: batch sizes 4,"
        " 8 each improved on the best throughput of the smaller batch sizes by"
        " less than 3% (the best is at batch size 4), so batch size 16 is not"
        " run",
        result.skipped[0][1],
    )
    self.assertEqual(
        [(d.batch_size, d.skipped) for d in self.exported],
        [(2, False), (4, False), (8, False), (16, True)],
    )

  def test_one_batch_size_without_improvement_does_not_stop_the_sweep(self):
    self._run_measured("bs2_case", _measured_payload(2, 100.0))
    self._run_measured("bs4_case", _measured_payload(4, 102.0))
    self.run_isolated.reset_mock()
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("bs8_case")

    self.run_isolated.assert_called_once()

  def test_improvement_resets_the_patience(self):
    self._run_measured("bs2_case", _measured_payload(2, 100.0))
    self._run_measured("bs4_case", _measured_payload(4, 102.0))
    # At least 3% above the best so far (102): the count starts over.
    self._run_measured("bs8_case", _measured_payload(8, 110.0))
    self.run_isolated.reset_mock()
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("bs16_case")

    self.run_isolated.assert_called_once()

  def test_patience_of_one_stops_after_one_batch_size(self):
    with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_PATIENCE, 1)):
      base_test.note_throughput("swept", _COMPILED, 2, 100.0)
      base_test.note_throughput("swept", _COMPILED, 4, 102.0)

      self.assertIn(
          "batch size 4 improved on the best throughput of the smaller batch"
          " sizes by less than 3%",
          base_test.get_stop_reason("swept", _COMPILED, 8),
      )

  def test_small_batch_sizes_do_not_count_towards_the_patience(self):
    # E.g. bert_base_uncased_forward_gen on TPU v7: bs1 is bound by the host's
    # dispatch, and bs2 and bs3 dip below it.
    with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_PATIENCE_MIN_BS, 8)):
      for batch_size, throughput in ((1, 1716.0), (2, 1419.0), (3, 1511.0)):
        base_test.note_throughput("swept", _COMPILED, batch_size, throughput)

      self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, 4))

  def test_batch_sizes_from_the_patience_min_bs_count(self):
    with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_PATIENCE_MIN_BS, 8)):
      for batch_size, throughput in ((1, 100.0), (8, 101.0), (16, 100.5)):
        base_test.note_throughput("swept", _COMPILED, batch_size, throughput)

      self.assertIn(
          "batch sizes 8, 16 each improved on the best throughput of the"
          " smaller batch sizes by less than 3% (the best is at batch size 8),"
          " so batch size 32 is not run",
          base_test.get_stop_reason("swept", _COMPILED, 32),
      )

  def test_small_batch_size_does_not_add_to_larger_ones(self):
    with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_PATIENCE_MIN_BS, 8)):
      # Batch sizes 4, 8 and 16 each improve by less than 3%, but batch size 4
      # does not count.
      for batch_size, throughput in ((2, 100.0), (4, 101.0), (8, 101.5)):
        base_test.note_throughput("swept", _COMPILED, batch_size, throughput)
      self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, 16))

      base_test.note_throughput("swept", _COMPILED, 16, 101.0)

      self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, 16))
      self.assertIn(
          "batch sizes 8, 16 each improved",
          base_test.get_stop_reason("swept", _COMPILED, 32),
      )

  def test_patience_min_bs_of_one_counts_every_batch_size(self):
    with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_PATIENCE_MIN_BS, 1)):
      for batch_size, throughput in ((1, 1716.0), (2, 1419.0), (3, 1511.0)):
        base_test.note_throughput("swept", _COMPILED, batch_size, throughput)

      self.assertIn(
          "batch sizes 2, 3 each improved on the best throughput of the"
          " smaller batch sizes by less than 3% (the best is at batch size 1)",
          base_test.get_stop_reason("swept", _COMPILED, 4),
      )

  def test_reported_best_includes_the_last_batch_size(self):
    base_test.note_throughput("swept", _COMPILED, 2, 100.0)
    base_test.note_throughput("swept", _COMPILED, 4, 101.0)
    base_test.note_throughput("swept", _COMPILED, 8, 102.0)

    self.assertIn(
        "(the best is at batch size 8)",
        base_test.get_stop_reason("swept", _COMPILED, 16),
    )

  def test_sweep_continues_while_throughput_improves(self):
    self._run_measured("bs2_case", _measured_payload(2, 100.0))
    self._run_measured("bs4_case", _measured_payload(4, 150.0))
    self._run_measured("bs8_case", _measured_payload(8, 200.0))
    self.run_isolated.reset_mock()
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("bs16_case")

    self.run_isolated.assert_called_once()

  def test_default_case_counts_towards_its_sweep(self):
    # The default case runs at its default batch size, 2.
    self._run_measured(
        "bs2_case", _measured_payload(None, 100.0, workload_batch_size=2)
    )
    self._run_measured("bs4_case", _measured_payload(4, 101.0))
    self._run_measured("bs8_case", _measured_payload(8, 100.5))

    self.assertIn(
        "batch sizes 4, 8 each improved on the best throughput of the smaller"
        " batch sizes by less than 3% (the best is at batch size 4)",
        base_test.get_stop_reason("swept", _COMPILED, 16),
    )

  def test_later_improvement_undoes_a_stop_from_the_default_case(self):
    # The default case runs first, at its default batch size 16, and completes
    # the patience with batch size 4 before batch size 8 has run.
    for batch_size, throughput in ((16, 100.0), (2, 100.0), (4, 101.0)):
      base_test.note_throughput("swept", _COMPILED, batch_size, throughput)
    self.assertIsNotNone(base_test.get_stop_reason("swept", _COMPILED, 32))

    # Batch size 8 then improves the throughput by more than 3%, so batch size
    # 16 is the only one without improvement since and the sweep goes on.
    base_test.note_throughput("swept", _COMPILED, 8, 120.0)

    self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, 32))

  def test_case_without_throughput_does_not_count(self):
    self._run_measured("bs2_case", _measured_payload(2, 100.0))
    # E.g. its pipelined steps ran out of memory: no throughput to compare.
    self._run_measured("bs4_case", _unmeasured_throughput_payload(4))
    self._run_measured("bs8_case", _measured_payload(8, 101.0))
    self.run_isolated.reset_mock()
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("bs16_case")

    self.run_isolated.assert_called_once()

  def test_stopping_is_per_run_mode(self):
    for batch_size in (2, 4, 8):
      base_test.note_throughput("swept", _COMPILED, batch_size, 100.0)
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("eager_bs8_case")

    self.run_isolated.assert_called_once()

  def test_zero_min_improvement_measures_every_batch_size(self):
    with flagsaver.flagsaver((flags_lib.BATCH_SWEEP_MIN_IMPROVEMENT, 0.0)):
      for batch_size, throughput in ((2, 100.0), (4, 90.0), (8, 80.0)):
        base_test.note_throughput("swept", _COMPILED, batch_size, throughput)
      self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, 16))

  def test_stop_after_a_drop_keeps_the_best_batch_size(self):
    base_test.note_throughput("swept", _COMPILED, 2, 100.0)
    base_test.note_throughput("swept", _COMPILED, 4, 80.0)
    base_test.note_throughput("swept", _COMPILED, 8, 90.0)

    reason = base_test.get_stop_reason("swept", _COMPILED, 16)

    self.assertIn("batch sizes 4, 8 each improved", reason)
    self.assertIn("(the best is at batch size 2)", reason)
    self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, 8))
    self.assertIsNone(base_test.get_stop_reason("swept", _COMPILED, None))

  def test_failed_case_stops_the_sweep(self):
    self.run_isolated.return_value = (
        base_test.Outcome.FAILURE,
        "simulated failure",
        _failed_payload(4),
    )
    self._run("bs4_case")
    self.run_isolated.reset_mock()

    result = self._run("bs8_case")

    self.run_isolated.assert_not_called()
    self.assertIn(
        "failed in run mode compiled at batch size 4, so batch size 8 is not"
        " run",
        result.skipped[0][1],
    )
    self.assertEqual(
        [(d.batch_size, d.succeeded, d.skipped) for d in self.exported],
        [(4, False, False), (8, True, True)],
    )

  def test_crashed_or_timed_out_case_stops_the_sweep(self):
    for outcome in (base_test.Outcome.CRASHED, base_test.Outcome.TIMED_OUT):
      with self.subTest(outcome=outcome):
        base_test._failed_batch_sizes.clear()  # pylint: disable=protected-access
        self.run_isolated.reset_mock()
        self.run_isolated.return_value = (outcome, "child died", None)
        self._run("bs4_case")
        self.run_isolated.reset_mock()

        result = self._run("bs8_case")

        self.run_isolated.assert_not_called()
        self.assertIn(
            "failed in run mode compiled at batch size 4", result.skipped[0][1]
        )

  def test_case_that_fails_without_a_record_stops_the_sweep(self):
    self.run_isolated.return_value = (
        base_test.Outcome.ERROR,
        "failed before measuring",
        None,
    )
    self._run("bs4_case")
    self.run_isolated.reset_mock()

    result = self._run("bs8_case")

    self.run_isolated.assert_not_called()
    self.assertLen(result.skipped, 1)

  def test_failure_stops_only_larger_batch_sizes_of_its_run_mode(self):
    base_test.note_failure("swept", _COMPILED, 4)
    self.run_isolated.return_value = (base_test.Outcome.SUCCESS, None, None)

    self._run("bs2_case")
    self._run("eager_bs8_case")

    self.assertEqual(self.run_isolated.call_count, 2)

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
