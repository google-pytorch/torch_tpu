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

_LAYER_PREFIX_FAILURE = known_failures.KnownFailure(
    benchmark_prefix="layer_tanh_train_",
    framework=mode_lib.Framework.TORCH,
    device_kind=target_lib.DeviceKind.TPU,
    reason="layer tanh train failure",
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

  @parameterized.named_parameters(
      dict(testcase_name="prefix_itself", benchmark="layer_tanh_train_"),
      dict(testcase_name="with_shape", benchmark="layer_tanh_train_b8_s1024"),
  )
  def test_prefix_matches_names_starting_with_prefix(self, benchmark):
    self.assertTrue(
        _LAYER_PREFIX_FAILURE.matches(
            benchmark,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.COMPILED,
        )
    )

  @parameterized.named_parameters(
      dict(testcase_name="other_family", benchmark="layer_gelu_train_b8"),
      dict(testcase_name="other_phase", benchmark="layer_tanh_eval_b8"),
      dict(
          testcase_name="prefix_not_at_start", benchmark="x_layer_tanh_train_"
      ),
      dict(testcase_name="shorter_than_prefix", benchmark="layer_tanh_train"),
  )
  def test_prefix_does_not_match_other_names(self, benchmark):
    self.assertFalse(
        _LAYER_PREFIX_FAILURE.matches(
            benchmark,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.COMPILED,
        )
    )

  def test_prefix_respects_other_coordinates(self):
    self.assertFalse(
        _LAYER_PREFIX_FAILURE.matches(
            "layer_tanh_train_b8",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.CUDA,
            run_mode=common.RunMode.COMPILED,
        )
    )

  def test_benchmark_does_not_match_by_prefix(self):
    self.assertFalse(
        _TPU_COMPILED_FAILURE.matches(
            "bench_suffix",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.COMPILED,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="both_set",
          benchmark="bench",
          benchmark_prefix="bench_",
      ),
      dict(testcase_name="neither_set"),
  )
  def test_requires_exactly_one_of_benchmark_and_prefix(
      self, benchmark=None, benchmark_prefix=None
  ):
    with self.assertRaisesRegex(
        ValueError, "Exactly one of `benchmark` and `benchmark_prefix`"
    ):
      known_failures.KnownFailure(
          benchmark=benchmark,
          benchmark_prefix=benchmark_prefix,
          framework=mode_lib.Framework.TORCH,
          reason="failure",
      )


class GetKnownFailureReasonTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            known_failures, "KNOWN_FAILURES", (_TPU_COMPILED_FAILURE,)
        )
    )
    self.enter_context(
        mock.patch.object(
            known_failures, "LAYER_KNOWN_FAILURES", (_LAYER_PREFIX_FAILURE,)
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

  def test_returns_reason_of_matching_layer_entry(self):
    self.assertEqual(
        known_failures.get_known_failure_reason(
            "layer_tanh_train_b8_s1024",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.EAGER_DEFAULT,
        ),
        "layer tanh train failure",
    )

  def test_layer_entries_do_not_affect_model_benchmarks(self):
    self.assertIsNone(
        known_failures.get_known_failure_reason(
            "bench",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
            run_mode=common.RunMode.EAGER_DEFAULT,
        )
    )


class KnownFailuresRegistryTest(seed_test_utils.RepeatableTest):

  def test_model_entries_use_exact_benchmark_names(self):
    # `model_config_test` requires every `KNOWN_FAILURES` entry to name a
    # registered model benchmark, so family-level entries belong in
    # `LAYER_KNOWN_FAILURES`.
    for failure in known_failures.KNOWN_FAILURES:
      with self.subTest(benchmark=failure.benchmark):
        self.assertIsNotNone(failure.benchmark)
        self.assertIsNone(failure.benchmark_prefix)


if __name__ == "__main__":
  absltest.main()
