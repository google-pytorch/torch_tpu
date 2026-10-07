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

"""Tests for perflab/known_failures.py."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from perflab import common
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import target as target_lib
from tests import seed_test_utils

_TPU_COMPILED_FAILURE = known_failures.KnownFailure(
    benchmark="bench",
    framework=mode_lib.Framework.TORCH,
    device_kind=target_lib.DeviceKind.TPU,
    run_modes=frozenset({common.RunMode.COMPILED}),
    reason="tpu compiled failure",
)


class KnownFailureTest(seed_test_utils.RepeatableTest):

  def test_matches_execution_coordinates(self):
    self.assertTrue(
        _TPU_COMPILED_FAILURE.matches(
            "bench",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.COMPILED,
        )
    )

  @parameterized.named_parameters(
      dict(testcase_name="other_benchmark", benchmark="other"),
      dict(
          testcase_name="other_framework",
          framework=mode_lib.Framework.TORCHAX,
      ),
      dict(
          testcase_name="other_device_kind",
          device_kind=target_lib.DeviceKind.CUDA,
      ),
      dict(
          testcase_name="other_run_mode",
          run_mode=common.RunMode.EAGER_DEFAULT,
      ),
  )
  def test_does_not_match_other_coordinates(
      self,
      benchmark="bench",
      framework=mode_lib.Framework.TORCH,
      device_kind=target_lib.DeviceKind.TPU,
      run_mode=common.RunMode.COMPILED,
  ):
    self.assertFalse(
        _TPU_COMPILED_FAILURE.matches(
            benchmark,
            framework=framework,
            device_kind=device_kind,
            run_mode=run_mode,
        )
    )

  def test_unset_filters_match_all_device_kinds_and_run_modes(self):
    failure = known_failures.KnownFailure(
        benchmark="bench",
        framework=mode_lib.Framework.TORCH,
        reason="failure",
    )
    for device_kind in target_lib.DeviceKind:
      for run_mode in common.RunMode:
        with self.subTest(device_kind=device_kind, run_mode=run_mode):
          self.assertTrue(
              failure.matches(
                  "bench",
                  framework=mode_lib.Framework.TORCH,
                  device_kind=device_kind,
                  run_mode=run_mode,
              )
          )


class GetKnownFailureReasonTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            known_failures, "KNOWN_FAILURES", (_TPU_COMPILED_FAILURE,)
        )
    )

  def test_returns_reason_of_matching_entry(self):
    self.assertEqual(
        known_failures.get_known_failure_reason(
            "bench",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.COMPILED,
        ),
        "tpu compiled failure",
    )

  def test_returns_none_without_matching_entry(self):
    self.assertIsNone(
        known_failures.get_known_failure_reason(
            "bench",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.CUDA,
            run_mode=common.RunMode.COMPILED,
        )
    )


if __name__ == "__main__":
  absltest.main()
