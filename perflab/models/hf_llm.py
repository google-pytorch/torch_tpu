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

"""HuggingFace causal LM benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops


def _load_hf_causal_lm(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    batch_size: int,
    seq_len: int,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace causal LM and generates token inputs.

  Args:
    ctx: The benchmark context.
    model_name: HuggingFace model identifier.
    is_training: Whether to prepare the model and labels for training.
    batch_size: Number of sequences per step.
    seq_len: Tokens per sequence.
    num_layers: If set, overrides the number of decoder layers.

  Returns:
    A tuple of (model, inputs dictionary).
  """

  def _modify_config_hook(config: Any) -> Any:
    config.use_cache = False
    if num_layers is not None:
      config.num_hidden_layers = num_layers
    return config

  return model_helpers.load_hf_llm(
      model_name,
      dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
      device_str=ctx.device_kind.value,
      is_training=is_training,
      seq_len=seq_len,
      batch_size=batch_size,
      modify_config_hook=_modify_config_hook,
  )


# --- StarCoder2 3B ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
)
def starcoder2_3b_forward(ctx: context_lib.Context):
  """Benchmark factory for StarCoder2-3B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_layers = None
  model, inputs = _load_hf_causal_lm(
      ctx,
      "bigcode/starcoder2-3b",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
)
def starcoder2_3b_decode(ctx: context_lib.Context):
  """Benchmark factory for StarCoder2-3B decode."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_layers = None
  model, inputs = _load_hf_causal_lm(
      ctx,
      "bigcode/starcoder2-3b",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


# --- Phi-3 Mini 4K Instruct ---


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
)
def phi_3_mini_4k_instruct_forward(ctx: context_lib.Context):
  """Benchmark factory for Phi-3-mini-4k-instruct forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_layers = None
  model, inputs = _load_hf_causal_lm(
      ctx,
      "microsoft/Phi-3-mini-4k-instruct",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.RUN,
)
def phi_3_mini_4k_instruct_training(ctx: context_lib.Context):
  """Benchmark factory for Phi-3-mini-4k-instruct training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_layers = None
  model, inputs = _load_hf_causal_lm(
      ctx,
      "microsoft/Phi-3-mini-4k-instruct",
      is_training=True,
      batch_size=batch_size,
      seq_len=seq_len,
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
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.RUN,
)
def phi_3_mini_4k_instruct_decode(ctx: context_lib.Context):
  """Benchmark factory for Phi-3-mini-4k-instruct decode."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
    num_layers = 2
  else:
    batch_size = 1
    seq_len = 2048
    num_layers = None
  model, inputs = _load_hf_causal_lm(
      ctx,
      "microsoft/Phi-3-mini-4k-instruct",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs
