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

"""TorchAx benchmark binary.

Runs every registered single-host benchmark, once per applicable execution
mode using TorchAx.

Environment & Flags:
  BENCHMARK_PLATFORM   v5e_1x1 | v6e_1x1 | ... (default: cpu)
  --run_scope          full | presubmit         (default: full)
"""

import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from torch_tpu._internal.distributed import multiprocessing
from perflab import base_test
from perflab import cases
from perflab import common
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import flags as flags_lib
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab.torchax import torchax_device_ops
from perflab.torchax import torchax_step

# Pre-import module_registry so torchvision initializes before XLADispatchMode is enabled.
from tests import module_registry  # pylint: disable=unused-import # noqa: F401
import torch_xla2 as torchax  # pylint: disable=unused-import # noqa: F401
from torchax import interop  # pylint: disable=unused-import # noqa: F401

# Framework definition for this binary.
_FRAMEWORK = mode_lib.Framework.TORCHAX

# Resolved once at collection time.
_PLATFORM = target_lib.platform_from_env()

# Import all models and trigger registration of benchmarks and steppers.
failures = discovery_lib.import_submodules(models)

torchax.enable_globally()
torchax.enable_performance_mode()


_CASES = list(cases.get_cases(_PLATFORM, _FRAMEWORK))


def _make_run_step(
    spec: registry_lib.BenchmarkSpec,
    ctx: context_lib.Context,
    mode: common.RunMode,
) -> step_lib.Stepper:
  common.seed_rngs()
  runner = torchax_step.resolve_torchax_stepper(
      spec.stepper, **spec.stepper_kwargs
  )
  runner.init_with_benchmark_args(spec, ctx)

  if common.is_torch_compile(mode):
    compile_config = spec.compile_config or compile_lib.CompileConfig()
    runner.compile(compile_config, ctx.target)

  return runner


def reset_state() -> None:
  """Resets the JAX compilation cache."""
  jax.clear_caches()


class TorchaxBenchmarkTest(
    base_test.BaseBenchmarkTest, parameterized.TestCase  # ABSLTEST_OK=benchmark
):
  """One test method, parameterized over the registry x mode matrix for TorchAx."""

  framework = _FRAMEWORK
  platform = _PLATFORM
  case_params = base_test.case_params(_CASES)

  def test_benchmark_imports(self):
    if failures:
      self.fail(
          "benchmark module(s) failed to import:\n  "
          + "\n  ".join(str(f) for f in failures)
      )

  @parameterized.named_parameters(*_CASES)
  def test_benchmark(
      self, spec: registry_lib.BenchmarkSpec, mode: common.RunMode
  ):
    # TODO - b/534438865: Add support for decoder only decode stepper.
    if spec.stepper == step_lib.StepperType.DECODER_ONLY_DECODE:
      self.skipTest(
          f"TorchAx benchmark does not support stepper {spec.stepper}"
      )

    skip_reason = cases.get_skip_reason(
        spec,
        mode,
        framework=_FRAMEWORK,
        device_kind=target_lib.make_target(_PLATFORM).device_kind,
    )

    if skip_reason is not None and flags_lib.SKIP_BEHAVIOR.value == "skip":
      self.skipTest(skip_reason)
    if skip_reason is None and flags_lib.SKIP_BEHAVIOR.value == "run_skipped":
      self.skipTest(
          f"Benchmark {spec.name} is not skipped for run mode {mode.value},"
          " skipping due to skip_behavior=run_skipped"
      )

    if flags_lib.DRY_RUN.value:
      self._dry_run_test()
      return

    target = target_lib.make_target(_PLATFORM, dtype=spec.dtype)
    device_ops = torchax_device_ops.TorchaxDeviceOps(target)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    reset_state()

    if (
        skip_reason is not None
        and flags_lib.SKIP_BEHAVIOR.value == "assert_raise"
    ):
      with self.assertRaises(Exception):  # ASSERT_RAISES_OK=benchmark
        self._run_and_measure(spec, mode, device_ops, ctx)
      return

    try:
      metrics = self._run_and_measure(spec, mode, device_ops, ctx)
    except unittest.SkipTest:
      raise
    except Exception:
      self.report_result(spec, mode, succeeded=False)
      raise

    self.report_result(spec, mode, succeeded=True, metrics=metrics)

  def _run_and_measure(
      self, spec, mode, device_ops, ctx
  ) -> metrics_lib.PerformanceMetrics:
    try:
      run_step = _make_run_step(spec, ctx, mode)
      metrics = measure_lib.measure(
          run_step,
          device_ops,
          name=f"{spec.name}_{mode.value}",
          enable_xprof=flags_lib.ENABLE_XPROF.value,
      )
    except target_lib.UnsupportedBenchmark as e:
      self.skipTest(f"{spec.name}: {e}")

    logging.info("Metrics for %s_%s:\n%s", spec.name, mode.value, metrics)
    self._assert_measured(metrics)
    return metrics

  def _assert_measured(self, metrics: metrics_lib.PerformanceMetrics) -> None:
    self.assertGreater(metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(metrics.first_step_time_seconds, 0.0)
    if measure_lib.POST_WARMUP_STEPS.value > 0:
      self.assertGreater(metrics.post_warmup_step_time_seconds, 0.0)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
