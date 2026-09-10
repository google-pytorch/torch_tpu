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
from examples.benchmarks.e2e import single_trace_trainer_adapter
from examples.tracing import optimizers


@contextlib.contextmanager
def _sdpa_kernel_if_not_cuda(use_math_attention_fallback: bool = False):
  device = common.get_torch_device(common.PLATFORM.value)
  is_cuda = device.type == "cuda"

  if is_cuda:
    yield
  else:
    attention_kernels = [
        torch.nn.attention.SDPBackend.OVERRIDEABLE,
    ]
    if use_math_attention_fallback:
      attention_kernels.append(torch.nn.attention.SDPBackend.MATH)
    with torch.nn.attention.sdpa_kernel(attention_kernels):
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

  def step_fn(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused
    with torch.inference_mode():
      with _sdpa_kernel_if_not_cuda():
        out = model(inputs)
    return out

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

  def step_fn(
      model: torch.nn.Module,
      inputs: Any,
      optimizer: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    del optimizer  # Unused
    with torch.inference_mode():
      with _sdpa_kernel_if_not_cuda():
        if isinstance(inputs, tuple):
          return model(*inputs)
        return model(inputs)

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

  state: dict[str, Any] = {"step_fn": None}
  del grad_accumulation_steps  # Ignored; SingleTraceTrainer doesn't support accumulation yet.

  def train_step(
      model: torch.nn.Module,
      inputs: Any,
      opt_arg: torch.optim.Optimizer | None = None,
  ) -> torch.Tensor:
    with _sdpa_kernel_if_not_cuda():
      del opt_arg  # Ignored; use SingleTraceTrainer's internal optimizer.
      if state["step_fn"] is None:
        trainer = single_trace_trainer_adapter.SingleTraceTrainerAdapter(
            model=model,
            optimizer=optimizer,
        )
        state["step_fn"] = trainer.make_compiled_train_step(
            example_inputs=inputs,
        )

      return state["step_fn"](inputs)

  setattr(train_step, "is_pre_compiled", True)
  return train_step


def get_train_factory(
    run_mode: common.RunMode,
    eager_fact: Callable[..., Any],
    **eager_kwargs: Any,
) -> Callable[..., Any]:
  """Returns a training step runner factory for the given run mode."""
  if run_mode == common.RunMode.COMPILED:
    return functools.partial(
        functional_train_factory,
        grad_accumulation_steps=1,
        optimizer=optimizers.ReferenceAdamw(),
    )
  if eager_fact == simple_train_factory:
    return eager_fact
  return functools.partial(
      eager_fact,
      grad_accumulation_steps=1,
      **eager_kwargs,
  )
