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

"""Counts the model FLOPs of one benchmark step on a fake CPU copy.

The model FLOPs of a step are a property of the benchmark and its run scope,
not of the device or the run mode it is measured on. So instead of counting on
the measured device, which `torch.compile` and the single-trace trainer hide
from a `TorchDispatchMode`, `count_step_flops` builds a second copy of the
benchmark for a CPU target under the FakeTensorMode of the CPU smoke test
(`mode.fake_tensor_context`), which allocates no storage, and runs one eager
step under `torch.utils.flop_counter.FlopCounterMode`.

What is counted is what `FlopCounterMode` has formulas for: matmuls (`mm`,
`addmm`, `bmm`, `baddbmm`, `_scaled_mm`), convolutions and scaled dot product
attention, forward and backward. Elementwise ops, normalizations and the
optimizer step count zero, as is usual for model FLOPs utilization (MFU). Note
that the attention formula ignores causal masking, and that ops without a
formula or a decomposition into ops with one (e.g. `_grouped_mm`) count zero.
"""

from collections.abc import Mapping, Sequence
import dataclasses

from absl import logging
from torch.utils import flop_counter
from perflab import common
from perflab import context as context_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib

# The key of `FlopCounterMode.get_flop_counts()` that holds the counts of the
# whole step rather than of one module.
_GLOBAL_MODULE = "Global"


@dataclasses.dataclass(frozen=True)
class FlopCount:
  """The FLOPs counted for one benchmark step.

  Attributes:
    total: The FLOPs of the step.
    by_op: The FLOPs of the step by op overload packet, e.g. `aten.mm`.
  """

  total: int
  by_op: Mapping[str, int]


def _count(
    spec: registry_lib.BenchmarkSpec, ctx: context_lib.Context
) -> FlopCount:
  """Builds `spec` for `ctx` and counts the FLOPs of one step."""
  common.seed_rngs()
  stepper = step_lib.resolve_stepper(spec.stepper, **spec.stepper_kwargs)
  benchmark_args = spec.factory(ctx)
  common.prepare_inputs(benchmark_args[0], benchmark_args[1:])
  stepper.init_with_benchmark_args(*benchmark_args)
  # Not compiled: the FLOPs are those of the eager step, which a dispatch mode
  # sees op by op. E.g. the decode prefill runs here, uncounted.
  stepper.pre_warmup_init()
  step_fn = stepper.get_step_fn()
  with flop_counter.FlopCounterMode(display=False) as counter:
    step_fn()
  by_op = counter.get_flop_counts().get(_GLOBAL_MODULE, {})
  return FlopCount(
      total=int(counter.get_total_flops()),
      by_op={str(op): int(n) for op, n in by_op.items()},
  )


def count_step_flops(
    spec: registry_lib.BenchmarkSpec,
    run_scope: context_lib.RunScope,
    *,
    workload: workload_lib.Workload | None = None,
) -> FlopCount | None:
  """Counts the model FLOPs of one step of `spec` at `run_scope`.

  Builds the benchmark for a CPU target with the same dtype and run scope under
  `mode.fake_tensor_context()`, so it neither allocates nor touches the device
  being measured, and counts one eager step. Call it after measuring, outside
  `mode.run_mode_context`.

  Args:
    spec: The benchmark to count.
    run_scope: The run scope to build the benchmark at, which sets e.g. its
      batch size, sequence length and depth.
    workload: The workload of the measured case (`Context.workload`), which the
      copy is built at. Passed rather than resolved for the CPU target, so that
      the FLOPs are those of the measured batch, also for benchmarks whose
      workload depends on the device kind.

  Returns:
    The FLOPs of one step, or None if the benchmark could not be built or
    stepped under FakeTensorMode on CPU. Counting never fails the benchmark.
  """
  ctx = context_lib.Context(
      target=target_lib.make_target(target_lib.Platform.CPU, dtype=spec.dtype),
      run_scope=run_scope,
      workload=workload,
  )
  try:
    with mode_lib.fake_tensor_context():
      flop_count = _count(spec, ctx)
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("Could not count the FLOPs of %s.", spec.name)
    return None
  logging.info(
      "Counted %.4g TFLOPs per step for %s: %s",
      flop_count.total / 1e12,
      spec.name,
      {op: f"{n / 1e12:.4g}" for op, n in flop_count.by_op.items()},
  )
  if flop_count.total == 0:
    logging.warning(
        "Counted 0 FLOPs for %s: its ops may have no FlopCounterMode formula.",
        spec.name,
    )
  return flop_count


def check_utilization(
    name: str, metric_map: Mapping[str, float]
) -> Sequence[str]:
  """Logs a warning for each implausible FLOPs metric of a benchmark case.

  A model FLOPs utilization (MFU) above 1 means that the FLOPs were counted
  wrong or the step was timed wrong, and FLOPs of 0 mean that the ops of the
  benchmark have no `FlopCounterMode` formula, so its MFU is meaningless.

  Args:
    name: The name of the benchmark case, for the warnings.
    metric_map: The exported metrics of the case (`metric_map()`).

  Returns:
    The warnings logged, empty if the metrics are plausible or not counted.
  """
  problems = []
  if metric_map.get("model_flops_per_step") == 0:
    problems.append("counted 0 model FLOPs per step, so its MFU is meaningless")
  for key in ("mfu", "device_mfu"):
    utilization = metric_map.get(key)
    if utilization is not None and utilization > 1.0:
      problems.append(
          f"{key} is {utilization:.3f} > 1: the FLOPs or the step time are"
          " wrong"
      )
  for problem in problems:
    logging.warning("Implausible utilization for %s: %s.", name, problem)
  return problems
