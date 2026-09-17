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

# Backend performance benchmark functions database.
"""Database of benchmark execution functions for backend performance tests."""

import contextlib
import functools
from typing import Any, Callable

import torch
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.e2e import single_trace_trainer_adapter
from examples.tracing import optimizers


# Per-step callers get a no-op context manager when the benchmark runner has
# already established these backends for the whole run. See
# `common.sdpa_kernel_if_not_cuda`.
_sdpa_kernel_if_not_cuda = common.sdpa_kernel_if_not_cuda


@contextlib.contextmanager
def _eval_benchmark_context():
  if torch.is_inference_mode_enabled():
    with _sdpa_kernel_if_not_cuda():
      yield
  else:
    with torch.inference_mode(), _sdpa_kernel_if_not_cuda():
      yield


@contextlib.contextmanager
def _no_grad_benchmark_context():
  with torch.no_grad(), _sdpa_kernel_if_not_cuda():
    yield


@contextlib.contextmanager
def _train_benchmark_context():
  with _sdpa_kernel_if_not_cuda():
    yield


def assert_grad_accumulation_steps(grad_accumulation_steps: int) -> None:
  """Asserts that gradient accumulation steps is 1."""
  # Grad accumulation is not supported by torchax so we don't allow it for the
  # purposes of running the same code across all backends.
  assert grad_accumulation_steps == 1


# ==============================================================================
# 1. HUGGING FACE LLM FACTORIES & RUNNERS
# ==============================================================================


def huggingface_eval_factory() -> Callable[..., Any]:
  """Creates a benchmark function for Hugging Face model evaluation.

  Returns:
    A callable step function that executes a forward pass on the model under
    `torch.no_grad` and returns the output logits.
  """

  def step_fn(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused
    # TODO(bbahl): Change to inference mode after fixing error in
    # torch.inference_mode(). Currently it raises: RuntimeError: Cannot set
    # version_counter for inference tensor in torch.embedding.
    with torch.no_grad():
      with _sdpa_kernel_if_not_cuda():
        result = model(**inputs)
      if hasattr(result, "logits"):
        result = result.logits
      return result

  setattr(step_fn, "benchmark_context", _no_grad_benchmark_context)
  return step_fn


def huggingface_llm_train_factory(
    grad_accumulation_steps: int,
    use_math_attention_fallback: bool = False,
) -> Callable[[torch.nn.Module, Any, torch.optim.Optimizer], torch.Tensor]:
  """Returns the benchmark function for training a Hugging Face LLM.

  Args:
    grad_accumulation_steps: The number of gradient accumulation steps.
    use_math_attention_fallback: Whether to use fallback to math attention.

  Returns:
    A callable step function that executes training iterations with gradient
    accumulation and returns the average step loss as a tensor.
  """
  assert_grad_accumulation_steps(grad_accumulation_steps)

  def train_step(
      model: torch.nn.Module, inputs: Any, optimizer: torch.optim.Optimizer
  ) -> torch.Tensor:
    if optimizer is None:
      raise ValueError("Optimizer must be provided for training.")
    accumulated_losses = []
    optimizer.zero_grad()
    for _ in range(grad_accumulation_steps):
      # Dynamic attention kernel overrides for HuggingFace training.
      with _sdpa_kernel_if_not_cuda(use_math_attention_fallback):
        output = model(**inputs)
        output.loss.backward()
      accumulated_losses.append(output.loss.detach())
    optimizer.step()
    step_loss = torch.sum(torch.stack(accumulated_losses))
    return step_loss

  return train_step


# ==============================================================================
# 2. META LLAMA FACTORIES & RUNNERS
# ==============================================================================


def meta_llama_eval_factory() -> Callable[..., Any]:
  """Creates a benchmark function for Meta Llama model evaluation.

  Returns:
    A callable step function that executes inference on the Llama model using
    (tokens, start_pos) inputs and returns the output predictions.
  """

  def step_fn(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused
    tokens, start_pos = inputs
    with torch.no_grad():
      with _sdpa_kernel_if_not_cuda():
        result = model(tokens, start_pos)
    return result

  return step_fn


# ==============================================================================
# 3. HUGGINGFACE DIFFUSER FACTORIES & RUNNERS
# ==============================================================================


def huggingface_diffuser_train_factory(
    grad_accumulation_steps: int,
) -> Callable[[torch.nn.Module, Any, torch.optim.Optimizer], torch.Tensor]:
  """Returns the benchmark function for training a Hugging Face Diffuser.

  Args:
    grad_accumulation_steps: The number of gradient accumulation steps.

  Returns:
    A callable step function that executes training iterations with gradient
    accumulation and returns the average step loss as a tensor.
  """
  assert_grad_accumulation_steps(grad_accumulation_steps)

  def train_step(
      model: torch.nn.Module, inputs: Any, optimizer: torch.optim.Optimizer
  ) -> torch.Tensor:
    if optimizer is None:
      raise ValueError("Optimizer must be provided for training.")
    accumulated_losses = []
    optimizer.zero_grad()
    for _ in range(grad_accumulation_steps):
      with _sdpa_kernel_if_not_cuda():
        output = model(**inputs)
      if hasattr(output, "sample"):
        loss = torch.mean(output.sample)
      elif isinstance(output, tuple):
        loss = torch.mean(output[0])
      else:
        loss = torch.mean(output)
      loss.backward()
      accumulated_losses.append(loss.detach())
    optimizer.step()
    step_loss = torch.sum(torch.stack(accumulated_losses))
    return step_loss

  return train_step


# ==============================================================================
# 3. TIMM FACTORIES & RUNNERS
# ==============================================================================


def timm_eval_factory() -> Callable[..., Any]:
  """Creates a benchmark function for TIMM model evaluation.

  Returns:
    A callable step function that executes inference on the TIMM model under
    `torch.inference_mode` and returns the output predictions.
  """
  cached_model = None
  cached_call = None

  def step_fn(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused
    nonlocal cached_model, cached_call
    if model is not cached_model:
      cached_model = model
      if not (
          getattr(model, "_forward_hooks", None)
          or getattr(model, "_forward_pre_hooks", None)
      ):
        cached_call = model.forward
      else:
        cached_call = model

    if torch.is_inference_mode_enabled() and common.is_default_sdpa_active():
      return cached_call(inputs)

    with torch.inference_mode():
      with _sdpa_kernel_if_not_cuda():
        out = cached_call(inputs)
    return out

  setattr(step_fn, "benchmark_context", _eval_benchmark_context)
  return step_fn


# ==============================================================================
# 4. SIMPLE / LAYER FACTORIES & RUNNERS
# ==============================================================================


def simple_eval_factory() -> Callable[..., Any]:
  """Creates a benchmark function for simple layer/model evaluation.

  Returns:
    A callable step function that executes inference on the model under
    `torch.inference_mode` supporting unpacked tuple inputs.
  """
  cached_model = None
  cached_call = None

  def step_fn(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused
    nonlocal cached_model, cached_call
    if model is not cached_model:
      cached_model = model
      if not (
          getattr(model, "_forward_hooks", None)
          or getattr(model, "_forward_pre_hooks", None)
      ):
        cached_call = model.forward
      else:
        cached_call = model

    if torch.is_inference_mode_enabled() and common.is_default_sdpa_active():
      if isinstance(inputs, tuple):
        return cached_call(*inputs)
      return cached_call(inputs)

    with torch.inference_mode():
      with _sdpa_kernel_if_not_cuda():
        if isinstance(inputs, tuple):
          return cached_call(*inputs)
        return cached_call(inputs)

  setattr(step_fn, "benchmark_context", _eval_benchmark_context)
  return step_fn


def simple_train_factory() -> (
    Callable[[torch.nn.Module, Any, torch.optim.Optimizer | None], torch.Tensor]
):
  """Returns the benchmark function for training a simple layer/model.

  Returns:
    A callable step function that executes a forward pass, computes a mean
    loss, executes a backward pass, and returns the detached loss.
  """

  def train_step(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused in simple training step
    model.zero_grad(set_to_none=True)
    with _sdpa_kernel_if_not_cuda():
      if isinstance(inputs, tuple):
        y_pred = model(*inputs)
      else:
        y_pred = model(inputs)
    if isinstance(y_pred, tuple):
      y_pred = y_pred[0]
    loss = torch.mean(y_pred)
    loss.backward()
    return loss.detach()

  setattr(train_step, "benchmark_context", _train_benchmark_context)
  return train_step


# TODO(lukeboyer): Refactor generic_train_factory later to improve failure debugging.
def generic_train_factory(
    grad_accumulation_steps: int,
) -> Callable[[torch.nn.Module, Any, torch.optim.Optimizer], torch.Tensor]:
  """A fully generic training step runner supporting NLP, Diffusers, and TIMM.

  Args:
    grad_accumulation_steps: The number of gradient accumulation steps.

  Returns:
    A callable step function that executes training iterations with gradient
    accumulation, optimizer updates, and returns the average step loss.
  """
  assert_grad_accumulation_steps(grad_accumulation_steps)

  def train_step(
      model: torch.nn.Module, inputs: Any, optimizer: torch.optim.Optimizer
  ) -> torch.Tensor:
    if optimizer is None:
      raise ValueError("Optimizer must be provided for training.")
    accumulated_losses = []
    optimizer.zero_grad()
    for _ in range(grad_accumulation_steps):
      # Forward pass based on input format
      with _sdpa_kernel_if_not_cuda():
        if isinstance(inputs, dict):
          output = model(**inputs)
        elif isinstance(inputs, tuple):
          output = model(*inputs)
        else:
          output = model(inputs)

      # Resolve loss based on output format
      if hasattr(output, "loss") and output.loss is not None:
        loss = output.loss
      elif hasattr(output, "logits"):
        loss = torch.mean(output.logits)
      elif hasattr(output, "sample"):
        loss = torch.mean(output.sample)
      elif isinstance(output, tuple):
        loss = torch.mean(output[0])
      else:
        loss = torch.mean(output)

      loss.backward()
      accumulated_losses.append(loss.detach())
    optimizer.step()
    step_loss = torch.sum(torch.stack(accumulated_losses))
    return step_loss

  return train_step


def functional_train_factory(
    grad_accumulation_steps: int = 1,
    optimizer: Any = None,
) -> Callable[
    [torch.nn.Module, Any, torch.optim.Optimizer | None], torch.Tensor
]:
  """Returns a functional training step runner using SingleTraceTrainer.

  Args:
    grad_accumulation_steps: Ignored (SingleTraceTrainer doesn't support
      accumulation yet).
    optimizer: SingleTraceTrainer Optimizer instance.

  Returns:
    A callable step function that executes functional FWD + BWD + OPT in one
    graph using SingleTraceTrainer.
  """

  step_fn: Callable[..., Any] | None = None
  del grad_accumulation_steps  # Ignored; SingleTraceTrainer doesn't support accumulation yet.

  def train_step(
      model: torch.nn.Module,
      inputs: Any,
      opt_arg: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del opt_arg  # Ignored; use SingleTraceTrainer's internal optimizer.
    nonlocal step_fn
    if step_fn is not None and common.is_default_sdpa_active():
      return step_fn(inputs)
    with _sdpa_kernel_if_not_cuda():
      if step_fn is None:
        trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
            model=model,
            optimizer=optimizer,
        )
        step_fn = trainer.make_compiled_train_step(
            example_inputs=inputs,
        )

      return step_fn(inputs)

  setattr(train_step, "is_pre_compiled", True)
  setattr(train_step, "benchmark_context", _train_benchmark_context)
  return train_step


def get_train_factory(
    run_mode: common.RunMode,
    eager_fact: Callable[..., Any],
    **eager_kwargs: Any,
) -> Callable[..., Any]:
  """Returns a training step runner factory for the given run mode."""
  if run_mode == common.RunMode.COMPILED:
    # Layer benchmarks (simple_train_factory) evaluate pure FWD + BWD without
    # an optimizer, while model benchmarks use ReferenceAdamw.
    optim = (
        None
        if eager_fact == simple_train_factory
        else optimizers.ReferenceAdamw()
    )
    return functools.partial(
        functional_train_factory,
        grad_accumulation_steps=1,
        optimizer=optim,
    )
  if eager_fact == simple_train_factory:
    return eager_fact
  return functools.partial(
      eager_fact,
      grad_accumulation_steps=1,
      **eager_kwargs,
  )


def functional_eval_factory() -> Callable[..., Any]:
  """Returns a functional evaluation step runner using SingleTraceTrainer."""
  step_fn: Callable[..., Any] | None = None

  def eval_step(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> Any:
    del optimizer
    nonlocal step_fn
    # Fast path: when called from within benchmark_utils._execution_run or
    # _warmup_run, _eval_benchmark_context has already activated inference_mode
    # and default SDPA kernel selection outside the timing loop.
    if (
        step_fn is not None
        and torch.is_inference_mode_enabled()
        and common.is_default_sdpa_active()
    ):
      return step_fn(inputs)
    with torch.inference_mode():
      with _sdpa_kernel_if_not_cuda():
        if step_fn is None:
          try:
            # Prefer SingleTraceTrainerAdapter.make_compiled_eval_step to trace
            # and compile the model once via torch.export, bypassing Dynamo's
            # per-step guard evaluation overhead during steady-state execution.
            trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
                model=model,
                optimizer=None,
            )
            step_fn = trainer.make_compiled_eval_step(
                example_inputs=inputs,
            )
          except Exception:  # pylint: disable=broad-except
            # Fallback to standard torch.compile if torch.export fails on
            # dynamic control flow or unsupported Python constructs.
            param_or_buf = next(model.parameters(), next(model.buffers(), None))
            if param_or_buf is not None:
              device_type = param_or_buf.device.type
            elif isinstance(inputs, torch.Tensor):
              device_type = inputs.device.type
            elif (
                isinstance(inputs, (tuple, list))
                and inputs
                and isinstance(inputs[0], torch.Tensor)
            ):
              device_type = inputs[0].device.type
            else:
              device_type = "tpu"
            compiled_model = device_utils.torch_compile(
                model, device_type, dynamic=False
            )
            if isinstance(inputs, tuple):
              step_fn = lambda inp: compiled_model(*inp)
            elif isinstance(inputs, dict):
              step_fn = lambda inp: compiled_model(**inp)
            else:
              step_fn = compiled_model

        return step_fn(inputs)

  setattr(eval_step, "is_pre_compiled", True)
  setattr(eval_step, "benchmark_context", _eval_benchmark_context)
  return eval_step


def get_eval_factory(
    run_mode: common.RunMode,
    eager_fact: Callable[..., Any],
    *,
    is_bounded_dynamic: bool = False,
) -> Callable[..., Any]:
  """Returns an evaluation step runner factory for the given run mode."""
  # Use functional_eval_factory for static compiled layer benchmarks on TPU/CPU
  # to eliminate per-step Dynamo guard and context manager overhead.
  if (
      run_mode == common.RunMode.COMPILED
      and not is_bounded_dynamic
      and eager_fact == simple_eval_factory
      and common.get_current_platform()
      not in (
          common.Platform.B200_8,
          common.Platform.B200_4,
          common.Platform.B200_1,
      )
  ):
    return functional_eval_factory
  return eager_fact
