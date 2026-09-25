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

import functools
import torch
from perflab import step_lib
from perflab.steps import common


class ForwardStepper(common.BaseStepper):
  """Forward only stepper."""

  def __init__(self):
    super().__init__()
    self._input_args = ()
    self._input_kwargs = {}

  def init_with_benchmark_args(
      self, model, input_args, input_kwargs, *args, **kwargs
  ) -> None:
    super().init_with_benchmark_args(model, *args, **kwargs)
    self._input_args = input_args
    self._input_kwargs = input_kwargs

  @staticmethod
  def _step_function(m, args, kwargs):
    with torch.no_grad():
      return m(*args, **kwargs)

  def get_step_fn(self):
    return functools.partial(
        self._inner_stepper, self._model, self._input_args, self._input_kwargs
    )


@step_lib.register_stepper(step_lib.StepperType.FORWARD)
def forward() -> step_lib.Stepper:
  """Forward only, no grad."""
  return ForwardStepper()
