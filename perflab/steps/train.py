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
from absl import logging
import torch
from perflab import loss as loss_lib
from perflab import single_trace_trainer_adapter
from perflab import step_lib
from perflab import target as target_lib
from perflab.steps import common


class _StandardTrainingStepper(common.BaseStepper):
  """Standard training stepper for eager and fallback execution."""

  def __init__(self, accum_steps: int = 1, compute_loss=loss_lib.real_loss):
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
    for _ in range(accum_steps):
      # Scale by accum_steps: backward sums gradients across micro-batches, so
      # without this the gradient is silently scaled by accum_steps.
      loss = compute_loss(m, args, kwargs) / accum_steps
      loss.backward()
    if opt is not None:
      opt.step()
    return

  def get_params(self) -> dict[str, torch.Tensor]:
    """Returns the current model parameters."""
    if self._model is not None:
      return dict(self._model.named_parameters())
    return {}

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


def _to_single_trace_optimizer(
    optimizer, use_bfloat16_moments: bool = False
) -> single_trace_trainer_adapter.Optimizer:
  """Converts a torch.optim.Optimizer into a SingleTraceTrainer Optimizer."""
  if isinstance(optimizer, single_trace_trainer_adapter.Optimizer):
    return optimizer
  defaults = getattr(optimizer, "defaults", {})
  if isinstance(optimizer, torch.optim.SGD):
    return single_trace_trainer_adapter.TorchSgd(
        lr=defaults.get("lr", 1e-3),
        momentum=defaults.get("momentum", 0.0),
        dampening=defaults.get("dampening", 0.0),
        weight_decay=defaults.get("weight_decay", 0.0),
        nesterov=defaults.get("nesterov", False),
        maximize=defaults.get("maximize", False),
        foreach=defaults.get("foreach", False),
        fused=defaults.get("fused", True),
    )
  beta1, beta2 = defaults.get("betas", (0.9, 0.999))
  return single_trace_trainer_adapter.ReferenceAdamW(
      lr=defaults.get("lr", 1e-4),
      weight_decay=defaults.get("weight_decay", 1e-2),
      beta1=beta1,
      beta2=beta2,
      eps=defaults.get("eps", 1e-8),
      use_bfloat16_moments=use_bfloat16_moments,
  )


class SingleTraceTrainingStepper(common.BaseStepper):
  """Training stepper using SingleTraceTrainer under the hood."""

  def __init__(self, accum_steps: int = 1, compute_loss=loss_lib.real_loss):
    super().__init__()
    if accum_steps != 1:
      raise ValueError(
          "SingleTraceTrainingStepper only supports accum_steps=1, got"
          f" {accum_steps}"
      )
    self._accum_steps = accum_steps
    self._compute_loss = compute_loss
    self._input_args = ()
    self._input_kwargs = {}
    self._trainer = None
    self._step_fn = None

  def init_with_benchmark_args(
      self, model, input_args, input_kwargs, optimizer=None, *args, **kwargs
  ) -> None:
    super().init_with_benchmark_args(model, *args, **kwargs)
    assert self._model is not None
    self._input_args = input_args
    self._input_kwargs = input_kwargs
    self._step_fn = None
    first_param = next(self._model.parameters(), None)
    use_bfloat16_moments = (
        first_param is not None and first_param.dtype == torch.bfloat16
    )
    self._trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model=self._model,
        optimizer=_to_single_trace_optimizer(
            optimizer, use_bfloat16_moments=use_bfloat16_moments
        ),
        # Keep the built-in loss for the default `real_loss`. Both pick the
        # same loss when `real_loss` succeeds, and the built-in loss falls back
        # to averaging `logits` or the first output tensor where `real_loss`
        # raises.
        compute_loss_fn=(
            None
            if self._compute_loss is loss_lib.real_loss
            else self._compute_loss
        ),
    )

  def compile(self, compile_config, target):
    # SingleTraceTrainerAdapter always compiles using StaticCompiler with input donation.
    del compile_config, target

  @staticmethod
  def _step_function(step_fn, inputs):
    return step_fn(inputs)

  def get_params(self) -> dict[str, torch.Tensor]:
    """Returns the current model parameters from the trainer."""
    if self._trainer is not None:
      return self._trainer.params
    if self._model is not None:
      return dict(self._model.named_parameters())
    return {}

  def get_step_fn(self):
    assert self._trainer is not None
    if self._step_fn is None:
      inputs = self._input_kwargs if self._input_kwargs else self._input_args
      step_fn = self._trainer.make_compiled_train_step(example_inputs=inputs)
      self._step_fn = functools.partial(self._step_function, step_fn, inputs)
    return self._step_fn


class TrainingStepper:
  """Composite training stepper dispatching to SingleTraceTrainer when compiled."""

  def __init__(
      self,
      accum_steps: int = 1,
      compute_loss=loss_lib.real_loss,
      use_single_trace: bool = True,
  ):
    self._accum_steps = accum_steps
    self._compute_loss = compute_loss
    self._use_single_trace = use_single_trace and accum_steps == 1
    self._init_args = ()
    self._init_kwargs = {}
    self._compile_args = ()
    self._standard_stepper = _StandardTrainingStepper(
        accum_steps=accum_steps, compute_loss=compute_loss
    )
    self._single_trace_stepper: SingleTraceTrainingStepper | None = None

  def init_with_benchmark_args(
      self, model, input_args, input_kwargs, optimizer, *args, **kwargs
  ) -> None:
    self._init_args = (model, input_args, input_kwargs, optimizer, *args)
    self._init_kwargs = kwargs
    self._single_trace_stepper = None
    self._standard_stepper.init_with_benchmark_args(
        model, input_args, input_kwargs, optimizer, *args, **kwargs
    )

  def compile(self, compile_config, target) -> None:
    self._compile_args = (compile_config, target)
    # SingleTraceTrainer compiles through the torch_tpu StaticCompiler, so it
    # only applies to TPU targets. Other devices keep the standard stepper.
    if (
        self._use_single_trace
        and target.device_kind == target_lib.DeviceKind.TPU
    ):
      self._single_trace_stepper = SingleTraceTrainingStepper(
          accum_steps=self._accum_steps, compute_loss=self._compute_loss
      )
      self._single_trace_stepper.init_with_benchmark_args(
          *self._init_args, **self._init_kwargs
      )
      self._single_trace_stepper.compile(compile_config, target)
    else:
      self._standard_stepper.compile(compile_config, target)

  @property
  def _active_stepper(
      self,
  ) -> _StandardTrainingStepper | SingleTraceTrainingStepper:
    return self._single_trace_stepper or self._standard_stepper

  @property
  def _trainer(
      self,
  ) -> single_trace_trainer_adapter.SingleTraceTrainerAdapter | None:
    if self._single_trace_stepper is not None:
      return self._single_trace_stepper._trainer
    return None

  def get_params(self) -> dict[str, torch.Tensor]:
    """Returns the current model parameters from the active stepper."""
    return self._active_stepper.get_params()

  def pre_warmup_init(self) -> None:
    self._active_stepper.pre_warmup_init()

  def post_warmup_hook(self) -> None:
    self._active_stepper.post_warmup_hook()

  def _fall_back_to_standard(self, reason: BaseException) -> None:
    """Disables single-tracing for this benchmark and compiles the normal path."""
    logging.warning(
        "SingleTraceTrainer could not trace this model, falling back to the"
        " standard compiled stepper. Numbers for this benchmark are NOT"
        " single-trace. Reason: %s: %s",
        type(reason).__name__,
        reason,
    )
    self._single_trace_stepper = None
    self._standard_stepper.compile(*self._compile_args)

  def get_step_fn(self):
    if self._single_trace_stepper is None:
      return self._standard_stepper.get_step_fn()
    try:
      return self._single_trace_stepper.get_step_fn()
    except Exception as e:  # pylint: disable=broad-except
      self._fall_back_to_standard(e)
      return self._standard_stepper.get_step_fn()


@step_lib.register_stepper(step_lib.StepperType.TRAINING)
def training(
    accum_steps: int = 1,
    compute_loss=loss_lib.real_loss,
    use_single_trace: bool = True,
) -> step_lib.Stepper:
  """Forward+backward over accum_steps micro-batches, then one optimizer step.

  Args:
    accum_steps: The number of micro-batches to accumulate gradients over before
      taking an optimizer step.
    compute_loss: A callable to compute the loss from the model output. Defaults
      to `real_loss`. Override the compute_loss function if you want to measure
      a different backward graph than the one measured by the default.
    use_single_trace: Whether to use SingleTraceTrainer for compiled execution.

  Returns:
    A composite TrainingStepper instance.
  """
  if accum_steps < 1:
    raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
  return TrainingStepper(
      accum_steps=accum_steps,
      compute_loss=compute_loss,
      use_single_trace=use_single_trace,
  )


@step_lib.register_stepper(step_lib.StepperType.SINGLE_TRACE_TRAINING)
def single_trace_training(
    accum_steps: int = 1,
    compute_loss=loss_lib.real_loss,
) -> step_lib.Stepper:
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
  return SingleTraceTrainingStepper(
      accum_steps=accum_steps,
      compute_loss=compute_loss,
  )
