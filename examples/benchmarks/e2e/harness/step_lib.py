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

  # For decoder-only models (e.g. LLama) that have a distinct prefill and decode step.
  # This stepper facilitates an unmeasured prefill, followed by `n` measured decode steps.
  DECODER_ONLY_DECODE = "decoder_only_decode"


# Basic stepper functionality, single step fn and compile hook.
@runtime_checkable
class Stepper(Protocol):

  # The actual model leveraged, saved as state to facilitate potential compilation.
  _model: Any

  # Step function saved as state to facilitate potential compilation.
  # TODO(@lukeboyer): Refine type annotations for various callables with types from torch.
  _inner_stepper: Callable[..., Any]

  # Step function core logic. This should take in the model, args and kwargs that are user provided, as well as any params that will be bound
  # to the step callable POST torch.compile.
  @staticmethod
  def _step_function(*args: Any, **kwargs: Any) -> Any:
    ...

  # Get actual step callable usable for benchmarking. This is the inner_stepper bound with model, inputs, kwargs, and any extra POST torch.compile args.
  def get_step_fn(self) -> StepFn:
    ...

  # Apply compilation to the internals based on config.
  def compile(self, compile_config: Any, target: Any) -> None:
    ...


# Additional stepper functionality for decoder only models that have a distinct prefill and decode step.
@runtime_checkable
class DecoderOnlyDecodeStepper(Stepper, Protocol):

  # Perform the prefill step. First new token output saved in state.
  def prefill(self) -> None:
    ...

  # Reset the KV cache after warmup so facilitate decode steps w/ clean compilation.
  def post_warmup_cache_reset(self) -> None:
    ...


# Convenience aliases, simple forward and training share same Stepper protocol.
ForwardStepper: TypeAlias = Stepper
TrainingStepper: TypeAlias = Stepper

# A factory of a specific stepper type.
StepperT = TypeVar("StepperT", bound=Stepper)
StepperFactory: TypeAlias = Callable[..., StepperT]

STEPPERS: Dict[StepperType, Callable[..., Callable[..., Stepper]]] = {}


def register_stepper(name: StepperType):
  """Register a step factory under `name`."""

  def deco(factory):
    if name in STEPPERS:
      raise ValueError(f"duplicate step: {name}")
    STEPPERS[name] = factory
    return factory

  return deco


def resolve_stepper(name: StepperType, **kwargs) -> Callable:
  """Build the step callable with the provided kwargs."""
  return STEPPERS[name](**kwargs)
