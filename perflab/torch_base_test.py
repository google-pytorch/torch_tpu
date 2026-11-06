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

"""Test base class for the single-host torch benchmark binaries.

This is a library, not an entry point: it has no `main`. The binaries, such as
`torch_benchmark_test.py`, subclass `BaseTorchBenchmarkTest` and run it.

Runs one registered benchmark case (a spec and a run mode) on the platform of
the binary. This assembles a step function according to the benchmark spec,
resolves the run-scope, and calls measure on the step function to get metrics.

This module does not discover benchmarks: each binary imports the benchmark
modules it runs, collects its cases, and parameterizes a subclass of
`BaseTorchBenchmarkTest` over them.
"""

import contextlib
from typing import Sequence
import unittest
from absl import logging
import torch
from perflab import base_test
from perflab import cases
from perflab import common
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import fake_memory
from perflab import flags as flags_lib
from perflab import flops as flops_lib
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def make_run_step(
    spec: registry_lib.BenchmarkSpec,
    ctx: context_lib.Context,
    mode: common.RunMode,
) -> tuple[step_lib.Stepper, workload_lib.StepWork | None]:
  """Collapse every axis into one bound zero-arg callable.

  Seeding lives here before construction so deterministic init is a harness
  guarantee. Everything the runner shouldn't know about is closed over rather
  than passed through.

  Args:
    spec: The benchmark to build.
    ctx: The context to build the benchmark for, with its resolved workload.
    mode: The run mode to build the benchmark for.

  Returns:
    The stepper of the benchmark and the work of one of its steps, or None if
    the benchmark declares no workload.

  Raises:
    ValueError: In the FakeTensorMode CPU smoke test, if the factory did not
      build its inputs at the workload of the context.
  """
  common.seed_rngs()
  runner = step_lib.resolve_stepper(spec.stepper, **spec.stepper_kwargs)
  if mode_lib.fake_tensor_smoke_enabled():
    # FakeTensorMode allocates nothing, so check here that building the
    # benchmark would fit in the memory of the device.
    with fake_memory.TensorMemoryTracker() as tracker:
      benchmark_args = spec.factory(ctx)
    fake_memory.check_construction_peak(tracker, name=spec.name)
    # The smoke test builds every benchmark, so it also checks that each builds
    # its inputs at the workload that the harness exports.
    if ctx.workload is not None:
      mismatch = workload_lib.check_built_inputs(ctx.workload, benchmark_args)
      if mismatch is not None:
        raise ValueError(
            f"Benchmark {spec.name} did not build its inputs at its workload:"
            f" {mismatch}. Build them from `ctx.require_workload()`."
        )
  else:
    benchmark_args = spec.factory(ctx)
  common.prepare_inputs(benchmark_args[0], benchmark_args[1:])
  runner.init_with_benchmark_args(*benchmark_args)

  if common.is_torch_compile(mode):
    compile_config = spec.compile_config or compile_lib.CompileConfig()
    runner.compile(compile_config, ctx.target)

  step_work = (
      None
      if ctx.workload is None
      else workload_lib.step_work(
          ctx.workload, spec.stepper, spec.stepper_kwargs
      )
  )
  return runner, step_work


def reset_state(
    target: target_lib.Target,
    device: torch.device,
    run_mode: common.RunMode,
) -> None:
  """Resets framework and hardware compilation/runtime state."""
  if target.device_kind.value == "tpu":
    getattr(torch, "tpu")._clear_cache()  # pylint: disable=protected-access
    if device.type == "tpu":
      from torch_tpu._internal import testing as tt_testing  # pylint: disable=g-import-not-at-top

      tt_testing.reset_eager_state()
  if common.is_torch_compile(run_mode):
    torch._dynamo.reset()


class BaseTorchBenchmarkTest(base_test.BaseBenchmarkTest):
  """Runs and reports one torch benchmark case per test method.

  A binary subclasses this together with the `TestCase` of absl's
  `parameterized` module, sets the `framework`, `platform`, and `case_params`
  attributes of `base_test.BaseBenchmarkTest` and `import_failures`, and defines
  a test method parameterized over its cases that calls `run_case`.

  This is a `TestCase` base class with a test method of its own,
  `test_benchmark_imports`. Import this module rather than the class, so
  that test loaders do not collect the class itself.

  Attributes:
    import_failures: The benchmark modules that the binary failed to import;
      `test_benchmark_imports` fails if there are any.
  """

  import_failures: Sequence[discovery_lib.ImportFailure] = ()

  def test_benchmark_imports(self):
    if self.import_failures:
      self.fail(
          "benchmark module(s) failed to import:\n  "
          + "\n  ".join(str(f) for f in self.import_failures)
      )

  def run_case(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      batch_size: int | None = None,
  ) -> None:
    """Runs, measures, and reports the case of `spec` in run mode `mode`.

    Args:
      spec: The benchmark to run.
      mode: The run mode to run it in.
      batch_size: The batch size of a batch sweep case, or None to measure the
        benchmark at its own batch size.
    """
    skip_reason = cases.get_skip_reason(
        spec,
        mode,
        framework=self.framework,
        device_kind=target_lib.make_target(self.platform).device_kind,
    )

    if skip_reason is not None and flags_lib.SKIP_BEHAVIOR.value == "skip":
      self.report_skip(spec, mode)
      self.skipTest(skip_reason)
    if skip_reason is None and flags_lib.SKIP_BEHAVIOR.value == "run_skipped":
      self.skipTest(
          f"Benchmark {spec.name} is not skipped for run mode {mode.value},"
          " skipping due to skip_behavior=run_skipped"
      )

    if flags_lib.DRY_RUN.value:
      self._dry_run_test()
      return

    # Batch sizes above one that ran out of memory; the parent skips them
    # before starting a child, so this only applies to in-process runs.
    out_of_memory_reason = base_test.get_out_of_memory_skip_reason(
        spec.name, mode, batch_size
    )
    if batch_size is not None and out_of_memory_reason is not None:
      self.report_out_of_memory_skip(spec, mode, batch_size)
      self.skipTest(out_of_memory_reason)

    target = target_lib.make_target(self.platform, dtype=spec.dtype)
    device_ops = torch_device_ops.TorchDeviceOps(target)
    ctx = registry_lib.resolve_context(
        spec,
        context_lib.Context(
            target=target, run_scope=context_lib.RUN_SCOPE.value
        ),
        batch_size,
    )

    reset_state(target, device_ops.device, mode)

    if (
        skip_reason is not None
        and flags_lib.SKIP_BEHAVIOR.value == "assert_raise"
    ):
      with self.assertRaises(Exception):  # ASSERT_RAISES_OK=benchmark
        self._run_and_measure(spec, mode, target, device_ops, ctx, batch_size)
      return

    try:
      metrics = self._run_and_measure(
          spec, mode, target, device_ops, ctx, batch_size
      )
    except unittest.SkipTest:
      raise
    except Exception as e:
      if batch_size is not None and common.is_out_of_memory(e):
        # The batch size bounds what fits, rather than failing the sweep.
        self.report_out_of_memory(spec, mode, batch_size)
        self.skipTest(
            f"{spec.name} ran out of device memory at batch size"
            f" {batch_size}: {e}"
        )
      self.report_result(spec, mode, succeeded=False, batch_size=batch_size)
      raise

    self.report_result(
        spec, mode, succeeded=True, metrics=metrics, batch_size=batch_size
    )

  def _run_and_measure(
      self, spec, mode, target, device_ops, ctx, batch_size
  ) -> metrics_lib.PerformanceMetrics:
    case_name = (
        f"{spec.name}_{mode.value}"
        if batch_size is None
        else cases.batch_sweep_case_name(spec.name, batch_size, mode)
    )
    try:
      with (
          mode_lib.run_mode_context(mode, target),
          (
              mode_lib.fake_tensor_context()
              if mode_lib.fake_tensor_smoke_enabled()
              else contextlib.nullcontext()
          ),
      ):
        run_step, step_work = make_run_step(spec, ctx, mode)
        metrics = measure_lib.measure(
            run_step,
            device_ops,
            name=case_name,
            enable_xprof=flags_lib.ENABLE_XPROF.value,
        )
    except target_lib.UnsupportedBenchmark as e:
      self.skipTest(f"{spec.name}: {e}")
    metrics.step_work = step_work

    if flags_lib.COUNT_FLOPS.value:
      # After measuring and outside the run mode, so that counting changes
      # neither the measured times nor the state of the measured device.
      flop_count = flops_lib.count_step_flops(
          spec, ctx.run_scope, workload=ctx.workload
      )
      if flop_count is not None:
        metrics.model_flops_per_step = float(flop_count.total)
        metrics.peak_flops_per_second = target_lib.peak_flops_per_device(target)
        flops_lib.check_utilization(case_name, metrics.metric_map())

    logging.info("Metrics for %s:\n%s", case_name, metrics)
    self._assert_measured(metrics)
    return metrics

  def _assert_measured(self, metrics: metrics_lib.PerformanceMetrics) -> None:
    self.assertGreater(metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(metrics.first_step_time_seconds, 0.0)
    if measure_lib.POST_WARMUP_STEPS.value > 0:
      self.assertGreater(metrics.post_warmup_step_time_seconds, 0.0)
