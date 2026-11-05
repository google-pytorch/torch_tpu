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

"""The benchmark registry containing a spec for each benchmark.

A benchmark wrapped in a @register_benchmark decorator is added to a global
dictionary. The spec holds only properties of a benchmark:

    factory, step (+ kwargs), topology, requires_kind

Things that are properties of the run like target/platform, run scope, eager
mode vs compiled, profiling are chosen per invocation. This lets one
registered benchmark fan out across modes and platforms.

Two rules the shape of this file encodes:

A benchmark is expected to run every mode in its matrix cell like only compiled
for torchax, eager/eager_optimized/compiled for TorchTPU.

Every benchmark must also declare how it runs in presubmit scope
(`PresubmitPolicy`), so that presubmit coverage is an explicit, reviewed
decision. The run scope itself is still chosen per invocation.
"""

from collections.abc import Callable, Mapping, Set
import dataclasses
import enum
from typing import Any

from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib

# factory returns variable number of outputs (e.g., model, input_args, ...)
Factory = Callable[..., tuple[Any, ...]]

# Returns the workload that a benchmark runs at by default in a context (its
# target and run scope). See `BenchmarkSpec.workload`.
WorkloadFn = Callable[[context_lib.Context], workload_lib.Workload]


def scoped_workload(
    full: workload_lib.Workload,
    presubmit: workload_lib.Workload | None = None,
) -> WorkloadFn:
  """Returns a `WorkloadFn` with one workload per run scope.

  Args:
    full: The workload in the full run scope.
    presubmit: The workload in the presubmit run scope; defaults to `full`.
  """
  presubmit = full if presubmit is None else presubmit

  def workload_fn(ctx: context_lib.Context) -> workload_lib.Workload:
    if ctx.run_scope is context_lib.RunScope.PRESUBMIT:
      return presubmit
    return full

  return workload_fn


class PresubmitPolicy(enum.Enum):
  """How a benchmark runs in presubmit scope (`--run_scope=presubmit`).

  Presubmit scope runs every benchmark once to catch breakages, so each
  benchmark must say whether and how it fits there. Each policy runs the
  benchmark in a different set of presubmit-scope runs; the harness, not the
  factory, applies it.
  """

  # Runs in every presubmit-scope run. The factory must keep the workload small
  # enough for presubmit, e.g. by building fewer layers, a shorter sequence or a
  # smaller batch when `ctx.run_scope` is `RunScope.PRESUBMIT`.
  RUN = "run"
  # Too large to build for real in presubmit scope. Runs in presubmit scope only
  # in the FakeTensorMode CPU smoke (`--fake_tensor_mode`), which allocates no
  # storage. Requires a `presubmit_reason`.
  CPU_SMOKE_ONLY = "cpu_smoke_only"
  # Never runs in presubmit scope. Requires a `presubmit_reason`.
  SKIP = "skip"


@dataclasses.dataclass(frozen=True)
class BatchSweep:
  """How a benchmark runs in the batch sweep (`--batch_sweep`).

  The batch sweep measures a benchmark at more batch sizes than its default
  workload's, in each run mode, e.g. to find the batch size with the best
  throughput on each platform. Each case builds the benchmark at its workload
  with another batch size (`Context.workload`), so only benchmarks that declare
  a workload are swept.

  Attributes:
    enabled: Whether the batch sweep measures the benchmark.
    batch_sizes: The batch sizes to measure the benchmark at, or None for those
      of `--batch_sweep_sizes`.
    reason: Why the batch sweep does not measure the benchmark. Required if and
      only if it is not enabled.
  """

  enabled: bool = True
  batch_sizes: tuple[int, ...] | None = None
  reason: str | None = None

  def __post_init__(self):
    if self.enabled == bool(self.reason):
      raise ValueError(
          "BatchSweep requires a reason if and only if it is disabled, got"
          f" enabled={self.enabled}, reason={self.reason!r}"
      )
    if self.batch_sizes is not None:
      if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
        raise ValueError(
            "BatchSweep.batch_sizes must be positive batch sizes, got"
            f" {self.batch_sizes!r}"
        )

  @classmethod
  def disabled(cls, reason: str) -> "BatchSweep":
    """Returns a BatchSweep that does not measure the benchmark, and why."""
    return cls(enabled=False, reason=reason)


# Policies that keep a benchmark out of all or part of presubmit, and so must
# say why.
_POLICIES_WITH_REASON = frozenset(
    {PresubmitPolicy.CPU_SMOKE_ONLY, PresubmitPolicy.SKIP}
)


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
  """Defines a benchmark.

  Run-scoped choices live elsewhere (see module docstring).

  `microbenchmark_name` optionally identifies the sub-configuration (e.g. a
  layer shape) and is exported with the results.

  Attributes:
    workload: Returns the workload that the benchmark runs at by default, e.g.
      `scoped_workload(full=Workload(16, 2048), presubmit=Workload(1, 128))`.
      The harness passes the workload of each case to the factory as
      `Context.workload` and exports its throughput. None for a benchmark
      without a batch, e.g. a host-overhead benchmark.
    batch_sweep: How the benchmark runs in the batch sweep.
  """

  name: str
  factory: Factory
  stepper: step_lib.StepperType
  presubmit: PresubmitPolicy
  stepper_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  dtype: target_lib.DType = target_lib.DType.BF16
  compile_config: compile_lib.CompileConfig | None = None
  skipped_run_modes: frozenset[str] = dataclasses.field(
      default_factory=frozenset
  )
  presubmit_reason: str | None = None
  workload: WorkloadFn | None = None
  batch_sweep: BatchSweep = dataclasses.field(default_factory=BatchSweep)
  microbenchmark_name: str | None = dataclasses.field(
      default=None, kw_only=True
  )

  def __post_init__(self):
    if not isinstance(self.presubmit, PresubmitPolicy):
      raise TypeError(
          f"Benchmark {self.name!r}: presubmit must be a PresubmitPolicy, got"
          f" {self.presubmit!r}"
      )
    if self.presubmit in _POLICIES_WITH_REASON:
      if not self.presubmit_reason:
        raise ValueError(
            f"Benchmark {self.name!r}: presubmit policy"
            f" {self.presubmit.name} requires a presubmit_reason"
        )
    elif self.presubmit_reason is not None:
      allowed = sorted(p.name for p in _POLICIES_WITH_REASON)
      raise ValueError(
          f"Benchmark {self.name!r}: presubmit_reason is only allowed with"
          f" presubmit policies {allowed}, got {self.presubmit.name}"
      )


REGISTRY: dict[str, BenchmarkSpec] = {}


def add_benchmark(
    factory: Factory,
    stepper: step_lib.StepperType,
    stepper_kwargs: Mapping[str, Any] | None = None,
    dtype: target_lib.DType = target_lib.DType.BF16,
    compile_config: compile_lib.CompileConfig | None = None,
    name: str | None = None,
    skipped_run_modes: Set[str] | None = None,
    *,
    presubmit: PresubmitPolicy,
    workload: WorkloadFn | None,
    presubmit_reason: str | None = None,
    batch_sweep: BatchSweep | None = None,
    microbenchmark_name: str | None = None,
) -> None:
  """Adds a benchmark spec to REGISTRY.

  Args:
    factory: The factory function for the benchmark.
    stepper: The step function type to use for this benchmark.
    stepper_kwargs: Optional keyword arguments to pass when resolving step.
    dtype: The data type for running the benchmark (defaults to BF16).
    compile_config: Optional compile configuration (defaults to None).
    name: Optional explicit name for the benchmark. If not provided, the
      factory's name is used.
    skipped_run_modes: Optional set of run modes to skip for this benchmark.
    presubmit: How the benchmark runs in presubmit scope. Required.
    presubmit_reason: Why the benchmark does not run (everywhere) in presubmit
      scope. Required for `CPU_SMOKE_ONLY` and `SKIP`, and not allowed
      otherwise.
    workload: The workload that the benchmark runs at by default, or None for a
      benchmark without a batch (see `BenchmarkSpec.workload`). Required.
    batch_sweep: How the benchmark runs in the batch sweep. Defaults to
      measuring it at the batch sizes of `--batch_sweep_sizes`.
    microbenchmark_name: Optional microbenchmark key exported with every result
      of this benchmark (defaults to None). See
      `BenchmarkSpec.microbenchmark_name`.
  """
  if stepper_kwargs is None:
    stepper_kwargs = {}
  if batch_sweep is None:
    batch_sweep = BatchSweep()

  if skipped_run_modes is None:
    skipped_run_modes = frozenset()
  elif not isinstance(skipped_run_modes, frozenset):
    skipped_run_modes = frozenset(skipped_run_modes)

  key = name if name is not None else getattr(factory, "__name__", str(factory))
  if key in REGISTRY:
    raise ValueError(
        f"duplicate benchmark name {key!r} (already registered from "
        f"{REGISTRY[key].factory.__module__})"
    )
  REGISTRY[key] = BenchmarkSpec(
      name=key,
      factory=factory,
      stepper=stepper,
      stepper_kwargs=dict(stepper_kwargs),
      dtype=dtype,
      compile_config=compile_config,
      skipped_run_modes=skipped_run_modes,
      presubmit=presubmit,
      presubmit_reason=presubmit_reason,
      workload=workload,
      batch_sweep=batch_sweep,
      microbenchmark_name=microbenchmark_name,
  )


def register_benchmark(
    stepper: step_lib.StepperType,
    stepper_kwargs: Mapping[str, Any] | None = None,
    dtype: target_lib.DType = target_lib.DType.BF16,
    compile_config: compile_lib.CompileConfig | None = None,
    skipped_run_modes: Set[str] | None = None,
    *,
    presubmit: PresubmitPolicy,
    workload: WorkloadFn | None,
    presubmit_reason: str | None = None,
    batch_sweep: BatchSweep | None = None,
    microbenchmark_name: str | None = None,
) -> Callable[[Factory], Factory]:
  """Decorator to wrap a benchmark factory and add its spec to REGISTRY.

  Args:
    stepper: The stepper type to use for this benchmark.
    stepper_kwargs: Optional keyword arguments to pass when resolving stepper.
    dtype: The data type for running the benchmark (defaults to BF16).
    compile_config: Optional compile configuration (defaults to None).
    skipped_run_modes: Optional set of run modes to skip for this benchmark.
    presubmit: How the benchmark runs in presubmit scope. Required.
    presubmit_reason: Why the benchmark does not run (everywhere) in presubmit
      scope. Required for `CPU_SMOKE_ONLY` and `SKIP`, and not allowed
      otherwise.
    workload: The workload that the benchmark runs at by default, or None for a
      benchmark without a batch (see `BenchmarkSpec.workload`). Required.
    batch_sweep: How the benchmark runs in the batch sweep. Defaults to
      measuring it at the batch sizes of `--batch_sweep_sizes`.
    microbenchmark_name: Optional microbenchmark key exported with every result
      of this benchmark (defaults to None). See
      `BenchmarkSpec.microbenchmark_name`.

  Returns:
    A decorator that registers the factory function and returns it unchanged.
  """
  if stepper_kwargs is None:
    stepper_kwargs = {}

  def deco(factory: Factory) -> Factory:
    add_benchmark(
        factory=factory,
        stepper=stepper,
        stepper_kwargs=dict(stepper_kwargs),
        dtype=dtype,
        compile_config=compile_config,
        skipped_run_modes=skipped_run_modes,
        presubmit=presubmit,
        presubmit_reason=presubmit_reason,
        workload=workload,
        batch_sweep=batch_sweep,
        microbenchmark_name=microbenchmark_name,
    )
    return factory

  return deco


def resolve_workload(
    spec: BenchmarkSpec,
    ctx: context_lib.Context,
    batch_size: int | None = None,
) -> workload_lib.Workload | None:
  """Returns the workload of a case of a benchmark.

  Args:
    spec: The benchmark.
    ctx: The context of the case; its own workload is ignored.
    batch_size: Another batch size to run the benchmark at (e.g. that of a batch
      sweep case), or None for the default workload of the benchmark.

  Returns:
    The default workload of the benchmark in `ctx`, at `batch_size` if set, or
    None if the benchmark declares no workload.

  Raises:
    ValueError: If `batch_size` is set for a benchmark without a workload.
  """
  if spec.workload is None:
    if batch_size is not None:
      raise ValueError(
          f"Benchmark {spec.name} declares no workload, so it cannot run at"
          f" batch size {batch_size}"
      )
    return None
  workload = spec.workload(ctx)
  return (
      workload if batch_size is None else workload.with_batch_size(batch_size)
  )


def resolve_context(
    spec: BenchmarkSpec,
    ctx: context_lib.Context,
    batch_size: int | None = None,
) -> context_lib.Context:
  """Returns `ctx` with the workload of a case of a benchmark.

  Args:
    spec: The benchmark.
    ctx: The context of the case.
    batch_size: Another batch size to run the benchmark at (e.g. that of a batch
      sweep case), or None for the default workload of the benchmark.
  """
  return dataclasses.replace(
      ctx, workload=resolve_workload(spec, ctx, batch_size)
  )
