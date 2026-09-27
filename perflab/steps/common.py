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

import abc
import enum
import typing
from typing import Any
from perflab import target as target_lib


class Scope(enum.Enum):
  """Compilation scope for a compiled mode benchmark."""

  MODEL = "model"
  STEP = "step"


class CompileConfig(typing.Protocol):
  """Protocol representing the compile configuration interface."""

  scope: Scope

  def apply(self, obj: Any, target: target_lib.Target) -> Any:
    ...


class BaseStepper(abc.ABC):
  """Base class providing default trivial compile hook for the step function."""

  def __init__(self):
    self._model = None
    self._inner_stepper = self._step_function

  def init_with_benchmark_args(
      self, model: Any, *args: Any, **kwargs: Any
  ) -> None:
    # args and kwargs are required to satisfy the Stepper protocol
    # but unused in the base class.
    del args, kwargs
    self._model = model

  @staticmethod
  @abc.abstractmethod
  def _step_function(*args: Any, **kwargs: Any) -> Any:
    """The actual step function to be executed."""
    pass

  def compile(self, compile_config: CompileConfig, target: target_lib.Target):
    """Default trivial compile application"""
    if compile_config.scope == Scope.MODEL:
      self._model = compile_config.apply(self._model, target)
    elif compile_config.scope == Scope.STEP:
      self._inner_stepper = compile_config.apply(self._inner_stepper, target)
    else:
      raise ValueError(f"Unsupported compile scope: {compile_config.scope}")

  def pre_warmup_init(self) -> None:
    pass

  def post_warmup_hook(self) -> None:
    pass
