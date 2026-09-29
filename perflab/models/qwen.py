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

"""Qwen model benchmark definitions."""

from typing import Any
import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops


def _load_hf_qwen(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    seq_len: int = 2048,
    batch_size: int = 1,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads the HuggingFace Qwen model and generates sample inputs for benchmarking."""

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if hasattr(config, "get_text_config"):
      text_config = config.get_text_config()
    else:
      text_config = config

    if num_layers is not None:
      text_config.num_hidden_layers = num_layers
      if hasattr(text_config, "layer_types"):
        text_config.layer_types = text_config.layer_types[:num_layers]
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


# --- Qwen3 0.6B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def qwen3_0_6b_forward(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-0.6B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 8
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-0.6B",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def qwen3_0_6b_training(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-0.6B training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 8
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-0.6B",
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


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE
)
def qwen3_0_6b_decode(ctx):
  """Benchmark factory for Qwen3-0.6B decode."""
  model, inputs = _load_hf_qwen(ctx, "Qwen/Qwen3-0.6B", is_training=False)
  return model, (), inputs


# --- Qwen3 1.7B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def qwen3_1_7b_forward(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-1.7B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 4
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-1.7B",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def qwen3_1_7b_training(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-1.7B training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 4
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-1.7B",
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


# --- Qwen3 4B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def qwen3_4b_forward(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-4B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 2
    seq_len = 1024
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-4B",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def qwen3_4b_training(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-4B training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 2
    seq_len = 1024
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-4B",
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


# --- Qwen3-Coder-30B MoE ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def qwen3_coder_30b_moe_forward(ctx: context_lib.Context):
  """Benchmark factory for Qwen3-Coder-30B-A3B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_layers = None

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3-Coder-30B-A3B-Instruct",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=num_layers,
  )
  return model, (), inputs


# --- Qwen3.5-397B MoE (4 Layers) ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    # TODO(b/567589153): Re-enable once OOM on TPU v5e and view decomposition
    # error in compiled mode are resolved.
    skipped_run_modes={
        "compiled",
        "eager_default",
        "eager_optimized",
    },
)
def qwen3_5_397b_4_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for Qwen3.5-397B-A17B (4 Layers) MoE forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 1
    seq_len = 2048

  model, inputs = _load_hf_qwen(
      ctx,
      "Qwen/Qwen3.5-397B-A17B",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=4,
  )
  return model, (), inputs
