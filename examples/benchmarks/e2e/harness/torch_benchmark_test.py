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

"""Torch benchmark binary.

Runs every registered single-host benchmark, once per applicable execution
mode, on the platform named by BENCHMARK_PLATFORM. This assembles a step
function according to the benchmark spec, resolves the run-scope, and
calls measure on the step function to get metrics.

Environment & Flags:
  BENCHMARK_PLATFORM   cpu | b200_1 | b200_8 | v6e_1x1 | ...   (default: cpu)
  --run_scope          full | presubmit                        (default: full)
"""

from typing import Iterator, Tuple
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import cases
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import discovery as discovery_lib
from examples.benchmarks.e2e.harness import measure as measure_lib
from examples.benchmarks.e2e.harness import metrics as metrics_lib
from examples.benchmarks.e2e.harness import mode as mode_lib
from examples.benchmarks.e2e.harness import models
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import steps
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops

# This binary IS the torch run. Not a flag -- see module docstring.
_FRAMEWORK = mode_lib.Framework.TORCH

# Resolved once at import/collection time without acquiring device handles.
_PLATFORM = target_lib.platform_from_env()

# Some benchmarks may be registered but known to fail. This can be used to
# define the desired behavior when encountering such benchmarks.
_SKIP_BEHAVIOR = flags.DEFINE_enum(
    "skip_behavior",
    "skip",
    ["skip", "assert_raise", "ignore", "run_skipped"],
    "Behavior for run modes that are marked as skipped in the config.",
)


# Import all models and trigger registration of benchmarks.
failures = discovery_lib.import_submodules(models)
failures += discovery_lib.import_submodules(steps)


def _cases() -> (
    Iterator[Tuple[str, registry_lib.BenchmarkSpec, common.RunMode]]
):
  return cases.get_cases(_PLATFORM, _FRAMEWORK)


def _make_run_step(
    spec: registry_lib.BenchmarkSpec,
    ctx: context_lib.Context,
    mode: common.RunMode,
) -> step_lib.Stepper:
  """Collapse every axis into one bound zero-arg callable.

  Seeding lives here before construction so deterministic init is a harness
  guarantee. Everything the runner shouldn't know about is closed over rather
  than passed through.
  """
  common.seed_rngs()
  step_factory = step_lib.resolve_stepper(spec.step, **spec.step_kwargs)
  runner = step_factory(*spec.factory(ctx))

  if common.is_torch_compile(mode):
    compile_config = spec.compile_config or compile_lib.CompileConfig()
    runner.compile(compile_config, ctx.target)

  return runner


class BenchmarkTest(parameterized.TestCase):
  """One test method, parameterised over the registry x mode matrix."""

  def setUp(self):
    super().setUp()
    logging._log_counter_per_token.clear()  # pylint: disable=protected-access

  def test_benchmark_imports(self):
    if failures:
      self.fail(
          "benchmark module(s) failed to import:\n  "
          + "\n  ".join(str(f) for f in failures)
      )

  @parameterized.named_parameters(*_cases())
  def test_benchmark(
      self, spec: registry_lib.BenchmarkSpec, mode: common.RunMode
  ):
    is_skipped = mode.value in spec.skipped_run_modes

    if is_skipped and _SKIP_BEHAVIOR.value == "skip":
      self.skipTest(
          f"Benchmark {spec.name} explicitly skips run mode {mode.value}"
      )
    if not is_skipped and _SKIP_BEHAVIOR.value == "run_skipped":
      self.skipTest(
          f"Benchmark {spec.name} is not skipped for run mode {mode.value},"
          " skipping due to skip_behavior=run_skipped"
      )

    target = target_lib.make_target(_PLATFORM, dtype=spec.dtype)
    device_ops = torch_device_ops.TorchDeviceOps(target)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    if is_skipped and _SKIP_BEHAVIOR.value == "assert_raise":
      with self.assertRaises(Exception):
        self._run_and_measure(spec, mode, target, device_ops, ctx)
    else:
      self._run_and_measure(spec, mode, target, device_ops, ctx)

  def _run_and_measure(self, spec, mode, target, device_ops, ctx):
    try:
      with mode_lib.run_mode_context(mode, target):
        run_step = _make_run_step(spec, ctx, mode)
        metrics = measure_lib.measure(
            run_step,
            device_ops,
            name=f"{spec.name}_{mode.value}",
        )
    except target_lib.UnsupportedBenchmark as e:
      self.skipTest(f"{spec.name}: {e}")

    logging.info("Metrics for %s_%s:\n%s", spec.name, mode.value, metrics)
    self._assert_measured(metrics)

  def _assert_measured(self, metrics: metrics_lib.PerformanceMetrics) -> None:
    self.assertGreater(metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(metrics.first_step_time_seconds, 0.0)
    if measure_lib.POST_WARMUP_STEPS.value > 0:
      self.assertGreater(metrics.post_warmup_step_time_seconds, 0.0)


if __name__ == "__main__":
  absltest.main()
