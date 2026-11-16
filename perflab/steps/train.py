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

from collections.abc import Mapping
import functools
from typing import Any
from absl import logging
import torch
from perflab import loss as loss_lib
from perflab import single_trace_trainer_adapter
from perflab import step_lib
from perflab import target as target_lib
from perflab.steps import common
from perflab.steps import standard_train


def _to_single_trace_optimizer(
    optimizer: Any = None, use_bfloat16_moments: bool = False
) -> single_trace_trainer_adapter.Optimizer | None:
  """Converts a torch.optim.Optimizer into a SingleTraceTrainer Optimizer.

  Args:
    optimizer: The optimizer from the benchmark factory, or None for a
      gradient-only step.
    use_bfloat16_moments: Whether AdamW keeps its moments in bfloat16.

  Returns:
    The SingleTraceTrainer optimizer, or None if `optimizer` is None.
  """
  if optimizer is None:
    return None
  if isinstance(optimizer, single_trace_trainer_adapter.Optimizer):
    return optimizer
  defaults = getattr(optimizer, "defaults", {})
  param_groups = getattr(optimizer, "param_groups", ())
  if isinstance(optimizer, torch.optim.SGD):
    return single_trace_trainer_adapter.Sgd(
        lr=defaults.get("lr", 1e-3),
        momentum=defaults.get("momentum", 0.0),
        dampening=defaults.get("dampening", 0.0),
        weight_decay=defaults.get("weight_decay", 0.0),
        nesterov=defaults.get("nesterov", False),
        maximize=defaults.get("maximize", False),
        foreach=defaults.get("foreach", False),
        fused=defaults.get("fused", True),
    )
  if defaults.get("amsgrad", False) or defaults.get("maximize", False):
    raise ValueError(
        "Unsupported Adam/AdamW options (amsgrad or maximize) for"
        f" SingleTraceTrainer: {type(optimizer).__name__}"
    )
  if isinstance(optimizer, torch.optim.AdamW):
    beta1, beta2 = defaults.get("betas", (0.9, 0.999))
    return single_trace_trainer_adapter.ReferenceAdamW(
        lr=defaults.get("lr", 1e-4),
        weight_decay=defaults.get("weight_decay", 1e-2),
        beta1=beta1,
        beta2=beta2,
        eps=defaults.get("eps", 1e-8),
        use_bfloat16_moments=use_bfloat16_moments,
    )
  if (
      isinstance(optimizer, torch.optim.Adam)
      and defaults.get("weight_decay", 0.0) == 0.0
      and all(
          isinstance(g, dict) and g.get("weight_decay", 0.0) == 0.0
          for g in param_groups
      )
  ):
    beta1, beta2 = defaults.get("betas", (0.9, 0.999))
    return single_trace_trainer_adapter.ReferenceAdamW(
        lr=defaults.get("lr", 1e-3),
        weight_decay=0.0,
        beta1=beta1,
        beta2=beta2,
        eps=defaults.get("eps", 1e-8),
        use_bfloat16_moments=use_bfloat16_moments,
    )
  if isinstance(optimizer, torch.optim.Adam):
    raise ValueError(
        "SingleTraceTrainer does not support torch.optim.Adam with"
        f" weight_decay={defaults.get('weight_decay')!r} (expected 0.0 or"
        " AdamW with decoupled weight decay)"
    )
  raise ValueError(
      "Unsupported torch optimizer for SingleTraceTrainer:"
      f" {type(optimizer).__name__} (expected AdamW, SGD, or Adam with"
      " weight_decay=0.0)"
  )


class SingleTraceTrainingStepper(common.BaseStepper):
  """Training stepper using SingleTraceTrainer under the hood."""

  def __init__(
      self,
      accum_steps: int = step_lib.DEFAULT_ACCUM_STEPS,
      compute_loss=loss_lib.real_loss,
  ):
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
    params = list(self._model.parameters())
    use_bfloat16_moments = bool(params) and all(
        p.dtype == torch.bfloat16 for p in params
    )
    if self._input_args and self._input_kwargs:
      compute_loss = self._compute_loss

      def compute_loss_fn(m, a, _kw):
        return compute_loss(m, a[0], a[1])

    elif self._compute_loss is loss_lib.real_loss:
      compute_loss_fn = None
    else:
      compute_loss_fn = self._compute_loss
    self._trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
        model=self._model,
        optimizer=_to_single_trace_optimizer(
            optimizer, use_bfloat16_moments=use_bfloat16_moments
        ),
        # Keep the built-in loss for the default `real_loss` unless both
        # positional and keyword inputs are packed together. Both pick the
        # same loss when `real_loss` succeeds, and the built-in loss falls back
        # to averaging `logits` or the first output tensor where `real_loss`
        # raises.
        compute_loss_fn=compute_loss_fn,
    )

  def compile(self, compile_config, target):
    # SingleTraceTrainerAdapter always compiles using StaticCompiler with input donation.
    del compile_config, target

  @staticmethod
  def _step_function(step_fn, inputs):
    return step_fn(inputs)

  def get_params(self) -> Mapping[str, torch.Tensor]:
    """Returns the current model parameters from the trainer."""
    if self._trainer is not None:
      return self._trainer.params
    if self._model is not None:
      return dict(self._model.named_parameters())
    return {}

  def get_step_fn(self):
    assert self._trainer is not None
    if self._step_fn is None:
      if self._input_args and self._input_kwargs:
        inputs = (self._input_args, self._input_kwargs)
      elif self._input_kwargs:
        inputs = self._input_kwargs
      else:
        inputs = self._input_args
      step_fn = self._trainer.make_compiled_train_step(example_inputs=inputs)
      self._step_fn = functools.partial(self._step_function, step_fn, inputs)
    return self._step_fn


class TrainingStepper:
  """Composite training stepper dispatching to SingleTraceTrainer when compiled."""

  def __init__(
      self,
      accum_steps: int = step_lib.DEFAULT_ACCUM_STEPS,
      compute_loss=loss_lib.real_loss,
      use_single_trace: bool = True,
  ):
    self._accum_steps = accum_steps
    self._compute_loss = compute_loss
    self._use_single_trace = use_single_trace and accum_steps == 1
    self._init_args = ()
    self._init_kwargs = {}
    self._compile_args = ()
    self._standard_stepper = standard_train.StandardTrainingStepper(
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
      try:
        single_trace_stepper = SingleTraceTrainingStepper(
            accum_steps=self._accum_steps, compute_loss=self._compute_loss
        )
        single_trace_stepper.init_with_benchmark_args(
            *self._init_args, **self._init_kwargs
        )
        single_trace_stepper.compile(compile_config, target)
        self._single_trace_stepper = single_trace_stepper
      except Exception as e:  # pylint: disable=broad-except
        self._fall_back_to_standard(e)
    else:
      self._standard_stepper.compile(compile_config, target)

  @property
  def supports_pipelined_steps(self) -> bool:
    return self._active_stepper.supports_pipelined_steps

  @property
  def _active_stepper(
      self,
  ) -> standard_train.StandardTrainingStepper | SingleTraceTrainingStepper:
    return self._single_trace_stepper or self._standard_stepper

  @property
  def _trainer(
      self,
  ) -> single_trace_trainer_adapter.SingleTraceTrainerAdapter | None:
    if self._single_trace_stepper is not None:
      return self._single_trace_stepper._trainer
    return None

  def get_params(self) -> Mapping[str, torch.Tensor]:
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
    accum_steps: int = step_lib.DEFAULT_ACCUM_STEPS,
    compute_loss=loss_lib.real_loss,
    use_single_trace: bool = True,
) -> step_lib.Stepper:
  """Forward+backward over accum_steps micro-batches, then one optimizer step.

  If the benchmark factory returns `None` as the optimizer, the step is
  gradient-only: it calls `model.zero_grad(set_to_none=True)`, runs
  forward+backward, and takes no optimizer step, so the params stay unchanged
  and `param.grad` holds the gradients of the last step. The single-trace path
  builds `SingleTraceTrainer(optimizer=None)`, which computes the gradients in
  the compiled step and writes them to `param.grad`.

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
    accum_steps: int = step_lib.DEFAULT_ACCUM_STEPS,
    compute_loss=loss_lib.real_loss,
) -> step_lib.Stepper:
  """Forward+backward over accum_steps micro-batches, then one optimizer step.

  If the benchmark factory returns `None` as the optimizer, the step is
  gradient-only: `SingleTraceTrainer(optimizer=None)` computes the gradients in
  the compiled step and writes them to `param.grad`, and the params stay
  unchanged.

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
