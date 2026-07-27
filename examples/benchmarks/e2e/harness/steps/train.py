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
from examples.benchmarks.e2e.harness import loss as loss_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness.steps import common


@step_lib.register_stepper(step_lib.StepperType.TRAINING)
def training(
    accum_steps: int = 1, compute_loss=loss_lib.real_loss
) -> step_lib.StepperFactory[step_lib.TrainingStepper]:
  """Forward+backward over accum_steps micro-batches, then one optimizer step.

  Args:
    accum_steps: The number of micro-batches to accumulate gradients over before
      taking an optimizer step.
    compute_loss: A callable to compute the loss from the model output. Defaults
      to `real_loss`. Override the compute_loss function if you want to measure
      a different backward graph than the one measured by the default.
  """
  if accum_steps < 1:
    raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")

  def step_factory(
      model, input_args, input_kwargs, optimizer
  ) -> step_lib.TrainingStepper:
    class TrainingStepper(common.BaseStepper):

      @staticmethod
      def _step_function(m, args, kwargs, opt):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
          # Scale by accum_steps: backward sums gradients across micro-batches, so
          # without this the gradient is silently scaled by accum_steps.
          loss = compute_loss(m, args, kwargs) / accum_steps
          loss.backward()
        opt.step()
        return

      def get_step_fn(self):
        return functools.partial(
            self._inner_stepper,
            self._model,
            input_args,
            input_kwargs,
            optimizer,
        )

    return TrainingStepper(model)

  return step_factory
