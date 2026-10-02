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

"""Gemma model benchmark definitions."""

from typing import Any
import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops


def _load_hf_gemma(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    seq_len: int = 2048,
    batch_size: int = 2,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads the HuggingFace Gemma model and generates sample inputs for benchmarking."""

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if num_layers:
      if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = num_layers
      if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_layers
    return config

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  return model_helpers.load_hf_llm(
      model_name,
      dtype=dtype,
      device_str=device_str,
      is_training=is_training,
      seq_len=seq_len,
      batch_size=batch_size,
      modify_config_hook=_modify_config_hook,
  )


# --- Gemma 3 270M ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gemma_3_270m_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-3 270M forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 6
  else:
    batch_size = 16
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-3-270m",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def gemma_3_270m_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-3 270M training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 6
  else:
    batch_size = 4
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-3-270m",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


# --- Gemma 2 2B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gemma_2_2b_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-2 2B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 2
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-2-2b",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def gemma_2_2b_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-2 2B training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 2
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-2-2b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


# --- Gemma 4 31B (6 Layers) ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gemma_4_31b_6_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 31B 6 layers forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 2
    seq_len = 1024
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-31b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=6,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def gemma_4_31b_6_layers_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 31B 6 layers training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    raise target_lib.UnsupportedBenchmark(
        "Gemma 4 31B training uses too much memory to run on TPUs with smaller"
        " memory."
    )
  batch_size = 2
  seq_len = 1024
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-31b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=6,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


# --- Gemma 4 E2B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gemma_4_e2b_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 E2B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    # Covers the all the sliding window attention + full attention layers.
    num_layers = 5
  else:
    batch_size = 2
    seq_len = 1024
    num_layers = None
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-e2b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def gemma_4_e2b_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 E2B training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    raise target_lib.UnsupportedBenchmark(
        "Gemma 4 E2B training uses too much memory to run on TPUs with smaller"
        " memory."
    )
  batch_size = 2
  seq_len = 1024
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-e2b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


# --- Gemma 4 26B A4B Ragged MoE ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gemma_4_26b_a4b_6_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B Ragged MoE forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 2
    seq_len = 2048
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-26B-A4B",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=6,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def gemma_4_26b_a4b_6_layers_training(ctx: context_lib.Context):
  """Benchmark factory for Gemma-4 26B-A4B Ragged MoE training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    raise target_lib.UnsupportedBenchmark(
        "Gemma 4 26B-A4B training uses too much memory to run on TPUs with"
        " smaller memory."
    )
  batch_size = 2
  seq_len = 2048
  model, inputs = _load_hf_gemma(
      ctx,
      "google/gemma-4-26B-A4B",
      is_training=True,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=6,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt
