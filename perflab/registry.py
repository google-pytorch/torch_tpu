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
from typing import Any, Callable, Dict, Mapping, Tuple

from perflab import compile as compile_lib
from perflab import step_lib
from perflab import target as target_lib

# factory returns variable number of outputs (e.g., model, input_args, ...)
Factory = Callable[..., Tuple[Any, ...]]


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
  """Defines a benchmark.

  Run-scoped choices live elsewhere (see module docstring).
  """

  name: str
  factory: Factory
  stepper: step_lib.StepperType
  stepper_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  dtype: target_lib.DType = target_lib.DType.BF16
  compile_config: compile_lib.CompileConfig | None = None
  skipped_run_modes: frozenset[str] = dataclasses.field(
      default_factory=frozenset
  )


REGISTRY: Dict[str, BenchmarkSpec] = {}


def add_benchmark(
    factory: Factory,
    stepper: step_lib.StepperType,
    stepper_kwargs: Mapping[str, Any] | None = None,
    dtype: target_lib.DType = target_lib.DType.BF16,
    compile_config: compile_lib.CompileConfig | None = None,
    name: str | None = None,
    skipped_run_modes: frozenset[str] | set[str] | None = None,
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
  """
  if stepper_kwargs is None:
    stepper_kwargs = {}

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
  )


def register_benchmark(
    stepper: step_lib.StepperType,
    stepper_kwargs: Mapping[str, Any] | None = None,
    dtype: target_lib.DType = target_lib.DType.BF16,
    compile_config: compile_lib.CompileConfig | None = None,
    skipped_run_modes: frozenset[str] | set[str] | None = None,
) -> Callable[[Factory], Factory]:
  """Decorator to wrap a benchmark factory and add its spec to REGISTRY.

  Args:
    stepper: The stepper type to use for this benchmark.
    stepper_kwargs: Optional keyword arguments to pass when resolving stepper.
    dtype: The data type for running the benchmark (defaults to BF16).
    compile_config: Optional compile configuration (defaults to None).
    skipped_run_modes: Optional set of run modes to skip for this benchmark.

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
    )
    return factory

  return deco
