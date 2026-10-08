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

"""Tests for perflab/cases.py."""

import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from perflab import cases
from perflab import common
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from tests import seed_test_utils


def _factory() -> tuple[Any, ...]:
  return ("model", (), {}, None)


def _make_spec(
    name: str, skipped_run_modes: frozenset[str] = frozenset()
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=_factory,
      stepper=step_lib.StepperType.FORWARD,
      skipped_run_modes=skipped_run_modes,
  )


class GetSkipReasonTest(seed_test_utils.RepeatableTest):

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
                    run_modes=frozenset({common.RunMode.COMPILED}),
                    reason="compiled failure",
                ),
            ),
        )
    )

  def test_returns_none_for_case_to_run(self):
    self.assertIsNone(
        cases.get_skip_reason(
            _make_spec("known_bad"),
            common.RunMode.EAGER_DEFAULT,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        )
    )

  def test_returns_known_failure_reason(self):
    self.assertEqual(
        cases.get_skip_reason(
            _make_spec("known_bad"),
            common.RunMode.COMPILED,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        ),
        "Benchmark known_bad is known to fail: compiled failure",
    )

  def test_returns_skipped_run_mode_reason(self):
    self.assertEqual(
        cases.get_skip_reason(
            _make_spec("healthy", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        ),
        "Benchmark healthy explicitly skips run mode compiled",
    )

  def test_get_cases_deterministic_across_shards_and_permuted_by_seed(self):
    fake_registry = {
        name: _make_spec(name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    }
    with mock.patch.dict(registry_lib.REGISTRY, fake_registry, clear=True):
      seed_a_first = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      seed_a_second = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      seed_b = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1002",
          )
      ]
      self.assertEqual(seed_a_first, seed_a_second)
      self.assertCountEqual(seed_a_first, seed_b)
      self.assertNotEqual(seed_a_first, seed_b)

  def test_get_cases_uses_env_seed_when_order_seed_omitted(self):
    fake_registry = {
        name: _make_spec(name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    }
    with (
        mock.patch.dict(registry_lib.REGISTRY, fake_registry, clear=True),
        mock.patch.dict(
            os.environ,  # PY_ENVIRON_OK=test
            {"TORCH_TPU_BENCHMARK_ORDER_SEED": "cl_1001"},
            clear=False,
        ),
    ):
      from_env = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
          )
      ]
      explicit = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      self.assertEqual(from_env, explicit)


if __name__ == "__main__":
  absltest.main()
