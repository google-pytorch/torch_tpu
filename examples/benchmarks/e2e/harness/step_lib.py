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

"""Stepper types and protocols for benchmarking."""

import enum
from typing import Any, Callable, Dict, Protocol, TypeAlias, TypeVar, runtime_checkable

# The callable that is actually benchmarked. Nullary as all inputs/model are bound within the factory.
StepFn: TypeAlias = Callable[[], Any]


# Relates one-to-one with the actual stepper implementations registered in steps/....
# Individual benchmarks specify desired stepper via this enum upon registration.
class StepperType(enum.Enum):
  """The closed set of step types."""

  # Single trivial forward pass per benchmark step.
  FORWARD = "forward"

  # Training loop with fwd+bwd+loss+opt.step(). Configurable grad acc & loss function.
  TRAINING = "training"

  # Training loop traced with functorch. Only works with compiled.
  SINGLE_TRACE_TRAINING = "single_trace_training"

  # For decoder-only models (e.g. LLama) that have a distinct prefill and decode step.
  # This stepper facilitates an unmeasured prefill, followed by `n` measured decode steps.
  DECODER_ONLY_DECODE = "decoder_only_decode"


# Basic stepper functionality, single step fn and compile hook.
@runtime_checkable
class Stepper(Protocol):

  # Initialize the stepper with the arguments returned from a benchmark spec's factory.
  def init_with_benchmark_args(self, *args: Any, **kwargs: Any) -> None:
    ...

  # Get actual step callable usable for benchmarking. Any stepper-specific args
  # are bound within this method and the returned callable is nullary.
  def get_step_fn(self) -> StepFn:
    ...

  # Apply compilation to the internals based on config.
  def compile(self, compile_config: Any, target: Any) -> None:
    ...

  # Optional initialization hook called before warmup (e.g. for prefill)
  def pre_warmup_init(self) -> None:
    ...

  # Optional hook called after warmup (e.g. for cache resets)
  def post_warmup_hook(self) -> None:
    ...


# Convenience aliases removed, simple forward and training share same Stepper protocol.

# A factory of a specific stepper type.
StepperT = TypeVar("StepperT", bound=Stepper)
StepperFactory: TypeAlias = Callable[..., StepperT]

STEPPERS: Dict[StepperType, Callable[..., Stepper]] = {}


def register_stepper(name: StepperType):
  """Register a step factory under `name`."""

  def deco(factory):
    if name in STEPPERS:
      raise ValueError(f"duplicate step: {name}")
    STEPPERS[name] = factory
    return factory

  return deco


def resolve_stepper(name: StepperType, **kwargs) -> Stepper:
  """Build the stepper with the provided kwargs."""
  return STEPPERS[name](**kwargs)
