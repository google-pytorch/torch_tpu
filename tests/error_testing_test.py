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

"""Tests for `error_testing.py` functions."""

from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from tests import error_testing as et
from tests import seed_test_utils


def _mock_run_in_subprocess_with_data(
    exitcode: int = 0,
    out: str = "",
    payload: tuple[str, Any] | None = None,
):
  """Returns a mock for `et._run_in_subprocess`, returning the given data."""
  data = et._SubprocessData(exitcode, out, payload)
  return mock.patch.object(et, "_run_in_subprocess", return_value=data)


class ErrorTestingTest(seed_test_utils.RepeatableTest):

  def test_append_error_test_failure_protocol(self):
    """Tests that the error test failure protocol is appended to the error."""

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        Exception,
        # Check that the appended note does not have leading spaces.
        r"\nNOTE: This test might fail depending on which PyTorch version is"
        " being used.",
    ):
      with et.assert_raises_message(
          RuntimeError,
          tpu="""TPU dummy error message for errors_test.py failure protocol""",
          gpu="""GPU dummy error message for errors_test.py failure protocol""",
          message_reviewed_by="wan",
      ):
        raise RuntimeError("Not the dummy error message")

  def test_why_tpu_only_validation(self):
    """Tests that why_tpu_only decorator requires a non-empty reason string."""
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        "non-empty",
    ):
      et.why_tpu_only("")

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        "non-empty",
    ):
      et.why_tpu_only("   ")

  def test_tpu_only_error_test_base_enforces_decorator(self):
    """Tests that TpuOnlyErrorTestBase enforces @why_tpu_only decorator."""

    class DummyTest(et.TpuOnlyErrorTestBase):

      def test_missing(self):
        pass

    test_instance = DummyTest("test_missing")
    with flagsaver.flagsaver(test_mode="tpu"):
      with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
          AssertionError,
          "TpuOnlyErrorTestBase",
      ):
        test_instance.setUp()

  def test_why_tpu_only_works_for_parameterized_tests(self):
    """Tests that @why_tpu_only decorator works for parameterized tests."""

    class DummyParameterizedTest(et.TpuOnlyErrorTestBase):

      @parameterized.named_parameters(
          {"testcase_name": "param1", "value": 1},
          {"testcase_name": "param2", "value": 2},
      )
      @et.why_tpu_only("Valid reason for parameterized test")
      def test_param(self, value):
        pass

    test_instance1 = DummyParameterizedTest("test_param_param1")
    test_instance2 = DummyParameterizedTest("test_param_param2")
    with flagsaver.flagsaver(test_mode="tpu"):
      test_instance1.setUp()
      test_instance2.setUp()

  def test_standalone_method_in_parameterized_test_enforces_decorator(self):
    """Tests standalone method in parameterized test enforces decorator."""

    class DummyParameterizedTest(et.TpuOnlyErrorTestBase):

      def test_standalone_missing(self):
        pass

    test_standalone = DummyParameterizedTest("test_standalone_missing")
    with flagsaver.flagsaver(test_mode="tpu"):
      with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
          AssertionError,
          "TpuOnlyErrorTestBase",
      ):
        test_standalone.setUp()

  def test_assert_raises_message_raises_no_exception_raised_error(self):
    """Tests that assert_raises_message raises `NoExceptionRaisedError`."""
    with self.assertRaises(et.NoExceptionRaisedError):
      with et.assert_raises_message(RuntimeError, tpu="""some msg"""):
        pass

  def test_gpu_crashes_or_raises_validation(self):
    """Tests that gpu_crashes_or_raises validates reason."""
    with self.assertRaisesRegex(ValueError, "must be specified"):
      et.gpu_crashes_or_raises("")

    with self.assertRaisesRegex(ValueError, "must be specified"):
      et.gpu_crashes_or_raises("   ")

    with self.assertRaisesRegex(ValueError, "must be specified"):
      et.gpu_crashes_or_raises(None)

  def test_gpu_crashes_or_raises_sets_attribute(self):
    """Tests that @gpu_crashes_or_raises sets metadata on the function."""

    @et.gpu_crashes_or_raises("Valid reason")
    def dummy_func():
      pass

    self.assertTrue(hasattr(dummy_func, "_gpu_crashes_or_raises_reason"))
    self.assertEqual(dummy_func._gpu_crashes_or_raises_reason, "Valid reason")

  def test_gpu_crashes_or_raises_works_for_parameterized_tests(self):
    """Tests that @gpu_crashes_or_raises is resolved for parameterized tests."""

    class DummyParameterizedCrashTest(et.ErrorTestBase):

      @parameterized.named_parameters(
          {"testcase_name": "param1", "val": 1},
          {"testcase_name": "param2", "val": 2},
      )
      @et.gpu_crashes_or_raises("Parameterized expected crash or raise")
      def test_param_crash(self, val):
        pass

    test1 = DummyParameterizedCrashTest("test_param_crash_param1")
    test2 = DummyParameterizedCrashTest("test_param_crash_param2")
    info1 = et._get_gpu_crashes_or_raises_reason(test1)
    info2 = et._get_gpu_crashes_or_raises_reason(test2)
    self.assertEqual(info1, "Parameterized expected crash or raise")
    self.assertEqual(info2, "Parameterized expected crash or raise")

  def test_tpu_only_error_test_base_rejects_gpu_crashes_or_raises(self):
    """Tests that TpuOnlyErrorTestBase disallows @gpu_crashes_or_raises decorator."""

    class DummyTpuOnlyWithGpuCrashesOrRaises(et.TpuOnlyErrorTestBase):

      @et.gpu_crashes_or_raises("Should be in ErrorTestBase")
      @et.why_tpu_only("Reason for TPU")
      def test_invalid(self):
        pass

    test_instance = DummyTpuOnlyWithGpuCrashesOrRaises("test_invalid")
    with flagsaver.flagsaver(test_mode="tpu"):
      with self.assertRaisesRegex(
          AssertionError,
          "Tests decorated with @gpu_crashes_or_raises should be placed in"
          " errors_test.py.",
      ):
        test_instance.setUp()

  def test_is_sparsecore_supported_cpu(self):
    """Tests that is_sparsecore_supported returns False when not on TPU."""
    self.addCleanup(et.is_sparsecore_supported.cache_clear)
    et.is_sparsecore_supported.cache_clear()
    with mock.patch.object(et, "is_on_tpu", return_value=False):
      self.assertFalse(et.is_sparsecore_supported())

  @parameterized.named_parameters(
      {"testcase_name": "v6e", "device_kind": "TPU v6e", "expected": True},
      {
          "testcase_name": "v6_lite",
          "device_kind": "TPU v6 lite",
          "expected": True,
      },
      {"testcase_name": "v5p", "device_kind": "TPU v5p", "expected": True},
      {"testcase_name": "v5e", "device_kind": "TPU v5e", "expected": False},
      {
          "testcase_name": "v5_lite",
          "device_kind": "TPU v5 lite",
          "expected": False,
      },
      {"testcase_name": "v4", "device_kind": "TPU v4", "expected": False},
      {
          "testcase_name": "v4_lite",
          "device_kind": "TPU v4 lite",
          "expected": False,
      },
      {"testcase_name": "v3", "device_kind": "TPU v3", "expected": False},
      {"testcase_name": "v2", "device_kind": "TPU v2", "expected": False},
      {"testcase_name": "cpu", "device_kind": "cpu", "expected": False},
      {"testcase_name": "none", "device_kind": None, "expected": False},
  )
  def test_is_sparsecore_supported_device_kinds(self, device_kind, expected):
    """Tests is_sparsecore_supported for different device kinds."""
    self.addCleanup(et.is_sparsecore_supported.cache_clear)
    et.is_sparsecore_supported.cache_clear()
    mock_module = mock.MagicMock()
    mock_module._get_local_device_attributes.return_value = {
        "device_kind": device_kind
    }
    with mock.patch.object(
        et, "is_on_tpu", return_value=True
    ), mock.patch.object(
        torch.accelerator, "is_available", return_value=True
    ), mock.patch.object(
        torch, "get_device_module", return_value=mock_module
    ):
      self.assertEqual(et.is_sparsecore_supported(), expected)

  def test_is_sparsecore_supported_exception(self):
    """Tests that is_sparsecore_supported returns False on unexpected error."""
    self.addCleanup(et.is_sparsecore_supported.cache_clear)
    et.is_sparsecore_supported.cache_clear()
    with mock.patch.object(
        et, "is_on_tpu", return_value=True
    ), mock.patch.object(
        torch.accelerator, "is_available", return_value=True
    ), mock.patch.object(
        torch, "get_device_module", side_effect=RuntimeError("PJRT not ready")
    ):
      self.assertFalse(et.is_sparsecore_supported())


class ErrorTestBaseTest(seed_test_utils.RepeatableTest):
  """Tests `ErrorTestBase` by mocking the `_run_in_subprocess()` function.

  Checks that the outcome processing translates the received payload (outcome,
  detail) as expected, taking into account whether the test was decorated with
  `@gpu_crashes_or_raises`.
  """

  def _run_mocked_test(
      self,
      outcome: str | None = None,
      detail: str | None = None,
      *,
      exitcode: int = 0,
      gpu_crashes_or_raises: bool = False,
  ) -> unittest.TestResult:  # UNITTEST_OK=TestResult type annotation.
    """Mocks test execution to verify outcome and detail handling.

    This helper mocks the execution of an `ErrorTestBase` test method under GPU
    test mode (`--test_mode=gpu`) by intercepting `_run_in_subprocess` to
    return synthetic subprocess data created from the given arguments. It is
    used to test how `ErrorTestBase.run()` and `_populate_result_on_outcome()`
    translate different subprocess outcomes, details, and exit codes into the
    returned `TestResult`.

    Args:
      outcome: The outcome status string transmitted by the child process (one
        of `_OUTCOME_*`, e.g. success, failure, error, skipped). If None,
        simulates a missing payload.
      detail: The detail string (e.g. traceback or skip reason) transmitted by
        the child process. Must be None if outcome is None.
      exitcode: The exit code of the mocked child process (0 for normal
        termination, non-zero for crashes or abrupt aborts).
      gpu_crashes_or_raises: If True, mocks execution of a test method decorated
        with `@gpu_crashes_or_raises`; if False, mocks an undecorated test
        method.

    Returns:
      The populated `unittest.TestResult` returned by `test_case.run()`.
    """
    if outcome is None and detail is not None:
      raise ValueError(
          "invalid payload for given `outcome` and `detail`: expected `detail`"
          " to be None whenever `outcome` is None. Rationale: if `outcome` is"
          " None, then the whole payload is None. `outcome` should never be"
          " None otherwise."
      )

    # Set `payload` only if `outcome` is not None.
    payload = (outcome, detail) if outcome is not None else None

    class DummyTest(et.ErrorTestBase):

      def test_undecorated(self):
        pass

      @et.gpu_crashes_or_raises("Op crashes or raises on GPU")
      def test_decorated(self):
        pass

    method_name = (
        "test_decorated" if gpu_crashes_or_raises else "test_undecorated"
    )

    test_case = DummyTest(method_name)
    res = unittest.TestResult()  # UNITTEST_OK=TestResult object for mock run

    with flagsaver.flagsaver(test_mode="gpu"):
      with _mock_run_in_subprocess_with_data(exitcode, payload=payload):
        return test_case.run(res)

  def test_subprocess_success(self):
    res = self._run_mocked_test(et._OUTCOME_SUCCESS)
    self.assertTrue(res.wasSuccessful())

  def test_subprocess_failure(self):
    outcome = et._OUTCOME_FAILURE
    detail = "Traceback:\nAssertionError: Expected error was not raised"

    res = self._run_mocked_test(outcome, detail)

    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.failures, 1)

  def test_subprocess_error(self):
    outcome = et._OUTCOME_ERROR
    detail = "Traceback:\nRuntimeError: Subprocess test errored"

    res = self._run_mocked_test(outcome, detail)

    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.errors, 1)

  def test_subprocess_crash(self):
    res = self._run_mocked_test(exitcode=1)
    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.errors, 1)

  def test_subprocess_no_payload(self):
    # Tests that an error is correctly recorded error when no payload is
    # received from the child process.
    res = self._run_mocked_test(exitcode=0)
    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.errors, 1)

  def test_gpu_crashes_or_raises_subprocess_crash(self):
    # Tests that a success is recorded when the test is expected to crash, and
    # it actually crashes.
    res = self._run_mocked_test(exitcode=1, gpu_crashes_or_raises=True)
    self.assertTrue(res.wasSuccessful())

  def test_gpu_crashes_or_raises_subprocess_arbitrary_error(self):
    # Tests that a success is recorded when the test is expected to crash, and
    # it actually raises an `AssertionError` (which gets recorded as a failure
    # by the child process).
    outcome = et._OUTCOME_FAILURE
    detail = "Traceback:\nAssertionError: Expected foo but got bar"

    res = self._run_mocked_test(outcome, detail, gpu_crashes_or_raises=True)

    self.assertTrue(res.wasSuccessful())

  def test_gpu_crashes_or_raises_subprocess_no_exception_raised(self):
    # Tests that a failure is recorded when the test is expected to crash, and it
    # runs without any issues (e.g. doesn't crash or raise any errors).
    outcome = et._OUTCOME_FAILURE
    detail = f"Traceback:\n{et.NoExceptionRaisedError.error_prefix()}error."

    res = self._run_mocked_test(outcome, detail, gpu_crashes_or_raises=True)

    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.failures, 1)

  def test_gpu_crashes_or_raises_forward_success(self):
    # Tests that the success the child process recorded is correctly forwarded,
    # when the test is expected to crash.
    res = self._run_mocked_test(et._OUTCOME_SUCCESS, gpu_crashes_or_raises=True)
    self.assertTrue(res.wasSuccessful())

  def test_gpu_crashes_or_raises_forward_error(self):
    # Tests that the error the child process recorded is correctly forwarded,
    # when the test is expected to crash.
    outcome = et._OUTCOME_ERROR
    detail = "Traceback:\nRuntimeError: Setup error in child process"

    res = self._run_mocked_test(outcome, detail, gpu_crashes_or_raises=True)

    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.errors, 1)


if __name__ == "__main__":
  absltest.main()
