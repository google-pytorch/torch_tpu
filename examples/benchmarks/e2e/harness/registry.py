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
"""

import dataclasses
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import target as target_lib

# factory returns (model, input_args, input_kwargs, optimizer | None)
Factory = Callable[
    ..., Tuple[Any, Sequence[Any], Mapping[str, Any], Any | None]
]


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
  """Defines a benchmark.

  Run-scoped choices live elsewhere (see module docstring).
  """

  name: str
  factory: Factory
  step: step_lib.StepperType
  step_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  dtype: target_lib.DType = target_lib.DType.BF16
  compile_config: compile_lib.CompileConfig | None = None
  skipped_run_modes: frozenset[str] = dataclasses.field(
      default_factory=frozenset
  )


REGISTRY: Dict[str, BenchmarkSpec] = {}


def add_benchmark(
    factory: Factory,
    step: step_lib.StepperType,
    step_kwargs: Mapping[str, Any] | None = None,
    dtype: target_lib.DType = target_lib.DType.BF16,
    compile_config: compile_lib.CompileConfig | None = None,
    name: str | None = None,
    skipped_run_modes: frozenset[str] | set[str] | None = None,
) -> None:
  """Adds a benchmark spec to REGISTRY.

  Args:
    factory: The factory function for the benchmark.
    step: The step function type to use for this benchmark.
    step_kwargs: Optional keyword arguments to pass when resolving step.
    dtype: The data type for running the benchmark (defaults to BF16).
    compile_config: Optional compile configuration (defaults to None).
    name: Optional explicit name for the benchmark. If not provided, the
      factory's name is used.
  """
  if step_kwargs is None:
    step_kwargs = {}

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
      step=step,
      step_kwargs=dict(step_kwargs),
      dtype=dtype,
      compile_config=compile_config,
      skipped_run_modes=skipped_run_modes,
  )


def register_benchmark(
    step: step_lib.StepperType,
    step_kwargs: Mapping[str, Any] | None = None,
    dtype: target_lib.DType = target_lib.DType.BF16,
    compile_config: compile_lib.CompileConfig | None = None,
    skipped_run_modes: frozenset[str] | set[str] | None = None,
) -> Callable[[Factory], Factory]:
  """Decorator to wrap a benchmark factory and add its spec to REGISTRY.

  Args:
    step: The step function type to use for this benchmark.
    step_kwargs: Optional keyword arguments to pass when resolving step.
    dtype: The data type for running the benchmark (defaults to BF16).
    compile_config: Optional compile configuration (defaults to None).

  Returns:
    A decorator that registers the factory function and returns it unchanged.
  """
  if step_kwargs is None:
    step_kwargs = {}

  def deco(factory: Factory) -> Factory:
    add_benchmark(
        factory=factory,
        step=step,
        step_kwargs=dict(step_kwargs),
        dtype=dtype,
        compile_config=compile_config,
        skipped_run_modes=skipped_run_modes,
    )
    return factory

  return deco
