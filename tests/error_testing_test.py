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

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from tests import error_testing as et
from tests import seed_test_utils


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

  def test_gpu_crashes_validation(self):
    """Tests that gpu_crashes decorator requires a non-empty reason string."""
    with self.assertRaisesRegex(ValueError, "non-empty"):
      et.gpu_crashes("")

    with self.assertRaisesRegex(ValueError, "non-empty"):
      et.gpu_crashes("   ")

    with self.assertRaisesRegex(ValueError, "non-empty"):
      et.gpu_crashes(None)

  def test_gpu_crashes_sets_attribute(self):
    """Tests that @gpu_crashes sets the _gpu_crashes_reason attribute on the function."""

    @et.gpu_crashes("Valid reason")
    def dummy_func():
      pass

    self.assertTrue(hasattr(dummy_func, "_gpu_crashes_reason"))
    self.assertEqual(dummy_func._gpu_crashes_reason, "Valid reason")

  def test_gpu_crashes_works_for_parameterized_tests(self):
    """Tests that @gpu_crashes is resolved for parameterized tests."""

    class DummyParameterizedCrashTest(et.ErrorTestBase):

      @parameterized.named_parameters(
          {"testcase_name": "param1", "val": 1},
          {"testcase_name": "param2", "val": 2},
      )
      @et.gpu_crashes("Parameterized expected crash")
      def test_param_crash(self, val):
        pass

    test1 = DummyParameterizedCrashTest("test_param_crash_param1")
    test2 = DummyParameterizedCrashTest("test_param_crash_param2")
    self.assertEqual(
        et._get_gpu_crashes_reason(test1), "Parameterized expected crash"
    )
    self.assertEqual(
        et._get_gpu_crashes_reason(test2), "Parameterized expected crash"
    )

  def test_tpu_only_error_test_base_rejects_gpu_crashes(self):
    """Tests that TpuOnlyErrorTestBase disallows @gpu_crashes decorator."""

    class DummyTpuOnlyWithGpuCrashes(et.TpuOnlyErrorTestBase):

      @et.gpu_crashes("Should be in ErrorTestBase")
      @et.why_tpu_only("Reason for TPU")
      def test_invalid(self):
        pass

    test_instance = DummyTpuOnlyWithGpuCrashes("test_invalid")
    with flagsaver.flagsaver(test_mode="tpu"):
      with self.assertRaisesRegex(
          AssertionError,
          "Tests decorated with @gpu_crashes should be placed in"
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
    mock_module.get_local_device_attributes.return_value = {
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


if __name__ == "__main__":
  absltest.main()
