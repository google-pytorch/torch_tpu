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
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness.steps import common


@step_lib.register_stepper(step_lib.StepperType.FORWARD)
def forward() -> step_lib.StepperFactory[step_lib.ForwardStepper]:
  """Forward only, no grad."""

  def step_factory(model, input_args, input_kwargs) -> step_lib.ForwardStepper:
    class ForwardStepper(common.BaseStepper):

      @staticmethod
      def _step_function(m, args, kwargs):
        with torch.no_grad():
          return m(*args, **kwargs)

      def get_step_fn(self):
        return functools.partial(
            self._inner_stepper, self._model, input_args, input_kwargs
        )

    return ForwardStepper(model)

  return step_factory
