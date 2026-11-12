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

"""Standard training stepper.

Kept free of torch_tpu C++ dependencies so CPU-only users can reuse it.
"""

from collections.abc import Mapping
import functools
import torch
from perflab import loss as loss_lib
from perflab import step_lib
from perflab.steps import common

# Prefix that `torch.compile` adds to parameter names of the wrapped module.
_COMPILED_MODULE_PREFIX = "_orig_mod."


class StandardTrainingStepper(common.BaseStepper):
  """Standard training stepper for eager and fallback execution."""

  def __init__(
      self,
      accum_steps: int = step_lib.DEFAULT_ACCUM_STEPS,
      compute_loss=loss_lib.real_loss,
  ):
    super().__init__()
    self._accum_steps = accum_steps
    self._compute_loss = compute_loss
    self._input_args = ()
    self._input_kwargs = {}
    self._optimizer = None

  def init_with_benchmark_args(
      self, model, input_args, input_kwargs, optimizer, *args, **kwargs
  ) -> None:
    super().init_with_benchmark_args(model, *args, **kwargs)
    self._input_args = input_args
    self._input_kwargs = input_kwargs
    self._optimizer = optimizer

  @staticmethod
  def _step_function(accum_steps, compute_loss, m, args, kwargs, opt):
    if opt is not None:
      opt.zero_grad(set_to_none=True)
    else:
      # Gradient-only step: zero the grads so they don't accumulate across
      # steps.
      m.zero_grad(set_to_none=True)
    for _ in range(accum_steps):
      # Scale by accum_steps: backward sums gradients across micro-batches, so
      # without this the gradient is silently scaled by accum_steps.
      loss = compute_loss(m, args, kwargs) / accum_steps
      loss.backward()
    if opt is not None:
      opt.step()
    # Return the last micro-batch loss so measure() can check that the step's
    # executions succeeded. A failed `opt.step()` leaves the params failed, so
    # the next step's loss fails; only a failure in the final step's
    # `opt.step()` goes unnoticed.
    return loss

  def get_params(self) -> Mapping[str, torch.Tensor]:
    """Returns the current model parameters keyed by their original names.

    A model compiled with `torch.compile` (e.g. under `Scope.MODEL`) prefixes
    parameter names with `_orig_mod.`; the prefix is stripped so the names match
    the uncompiled model and `SingleTraceTrainingStepper.get_params`.
    """
    if self._model is None:
      return {}
    return {
        name.removeprefix(_COMPILED_MODULE_PREFIX): param
        for name, param in self._model.named_parameters()
    }

  def get_step_fn(self):
    return functools.partial(
        self._inner_stepper,
        self._accum_steps,
        self._compute_loss,
        self._model,
        self._input_args,
        self._input_kwargs,
        self._optimizer,
    )
