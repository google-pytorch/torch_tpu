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

"""Tests that the benchmarks that opt in to the batch sweep can be swept.

`batch_sweep_benchmarks.BATCH_SWEEP_BENCHMARKS` names benchmarks by string, so
a renamed or removed benchmark, or one that cannot be swept, would otherwise
silently drop out of the batch sweep.
"""

from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import batch_sweep_benchmarks
from perflab import cases
from perflab import common
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from tests import seed_test_utils

# Registers the benchmarks of the full run scope, including the generated ones,
# as torch_benchmark_test does.
_IMPORT_FAILURES = discovery_lib.import_submodules(models)

_NAMES = sorted(batch_sweep_benchmarks.BATCH_SWEEP_BENCHMARKS)

# The platforms of the batch sweep runs.
_PLATFORMS = (target_lib.Platform.V7_1X1X1, target_lib.Platform.B200_1)

_COMPILED = common.RunMode.COMPILED
_WAN = "wan_2_2_ti2v_5b_forward"
_TORCH = mode_lib.Framework.TORCH


class BatchSweepBenchmarksTest(seed_test_utils.RepeatableTest):

  def test_models_import(self):
    self.assertEmpty(_IMPORT_FAILURES)

  @parameterized.parameters(*_NAMES)
  def test_can_be_swept(self, name: str):
    self.assertIn(name, registry_lib.REGISTRY)
    spec = registry_lib.REGISTRY[name]
    self.assertIsNotNone(spec.workload)
    self.assertTrue(spec.batch_sweep.enabled, spec.batch_sweep.reason)
    self.assertIsNone(cases.get_batch_sweep_skip_reason(spec))

  @parameterized.parameters(*_NAMES)
  def test_is_a_forward_benchmark(self, name: str):
    # The batch sweep trial measures prefill (forward) only. Remove this check
    # when training benchmarks join the sweep.
    self.assertIs(
        registry_lib.REGISTRY[name].stepper, step_lib.StepperType.FORWARD
    )

  @parameterized.parameters(*_NAMES)
  def test_runs_compiled_on_tpus_and_gpus(self, name: str):
    # The batch sweep runs in --batch_sweep_run_modes, compiled by default.
    self.assertNotIn(
        _COMPILED.value, registry_lib.REGISTRY[name].skipped_run_modes
    )
    for platform in _PLATFORMS:
      device_kind = target_lib.make_target(platform).device_kind
      with self.subTest(device_kind=device_kind):
        self.assertIsNone(
            known_failures.get_known_failure_reason(
                name,
                framework=_TORCH,
                device_kind=device_kind,
                run_mode=_COMPILED,
            )
        )

  @parameterized.parameters(*_PLATFORMS)
  def test_batch_sweep_only_collects_the_whole_sweep_of_each_benchmark(
      self, platform: target_lib.Platform
  ):
    with flagsaver.flagsaver(
        run_scope=context_lib.RunScope.FULL,
        batch_sweep=True,
        batch_sweep_only=True,
    ):
      default_cases = list(cases.get_cases(platform, _TORCH))
      sweep_cases = list(cases.get_batch_sweep_cases(platform, _TORCH))
      batch_sizes = flags_lib.batch_sweep_flag_values().batch_sizes
    self.assertEmpty(default_cases)
    expected = []
    for name in _NAMES:
      spec = registry_lib.REGISTRY[name]
      default_workload = registry_lib.resolve_workload(
          spec,
          context_lib.Context(
              target=target_lib.make_target(platform, dtype=spec.dtype),
              run_scope=context_lib.RunScope.FULL,
          ),
      )
      max_batch_size = batch_sweep_benchmarks.BATCH_SWEEP_BENCHMARKS[
          name
      ].max_batch_size
      spec_batch_sizes = {
          b
          for b in spec.batch_sweep.batch_sizes or batch_sizes
          if max_batch_size is None or b <= max_batch_size
      }
      spec_batch_sizes.add(default_workload.batch_size)
      expected += [
          cases.batch_sweep_case_name(name, batch_size, _COMPILED)
          for batch_size in sorted(spec_batch_sizes)
      ]
    self.assertCountEqual([case[0] for case in sweep_cases], expected)
    self.assertLessEqual(
        max(b for _, spec, _, b in sweep_cases if spec.name == _WAN), 64
    )
    logging.info(
        "--batch_sweep_only collects %d cases on %s: %s",
        len(sweep_cases),
        platform.value,
        ", ".join(case[0] for case in sweep_cases),
    )


class BatchSweepOptInTest(seed_test_utils.RepeatableTest):

  def test_cap_requires_a_reason(self):
    for reason in (None, ""):
      with self.subTest(reason=reason):
        with self.assertRaisesRegex(ValueError, "if and only if"):
          batch_sweep_benchmarks.BatchSweepOptIn(
              max_batch_size=8, max_batch_size_reason=reason
          )

  def test_reason_requires_a_cap(self):
    with self.assertRaisesRegex(ValueError, "if and only if"):
      batch_sweep_benchmarks.BatchSweepOptIn(max_batch_size_reason="too big")

  def test_cap_must_be_positive(self):
    with self.assertRaisesRegex(ValueError, "positive"):
      batch_sweep_benchmarks.BatchSweepOptIn(
          max_batch_size=0, max_batch_size_reason="too big"
      )

  def test_wan_is_capped(self):
    opt_in = batch_sweep_benchmarks.BATCH_SWEEP_BENCHMARKS[_WAN]
    self.assertEqual(opt_in.max_batch_size, 64)
    self.assertTrue(opt_in.max_batch_size_reason)


if __name__ == "__main__":
  absltest.main()
