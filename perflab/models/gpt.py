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

"""GPT model benchmark definitions."""

from typing import Any
import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import torch_device_ops


def _load_hf_gpt(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    seq_len: int = 512,
    batch_size: int = 1,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads the HuggingFace GPT model and generates sample inputs for benchmarking."""

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if num_layers is not None:
      config.num_hidden_layers = num_layers
      if hasattr(config, "layer_types"):
        config.layer_types = config.layer_types[:num_layers]
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


# --- GPT-OSS-20B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gpt_oss_20b_forward(ctx: context_lib.Context):
  """Benchmark factory for GPT-OSS 20B forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    raise target_lib.UnsupportedBenchmark(
        "GPT-OSS 20B uses too much memory to run on TPUs with smaller memory."
    )
  batch_size = 4
  seq_len = 4096
  model, inputs = _load_hf_gpt(
      ctx,
      "openai/gpt-oss-20b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def gpt_oss_20b_4_layers_training(ctx: context_lib.Context):
  """Benchmark factory for GPT-OSS 20B (4 Layers) training."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    raise target_lib.UnsupportedBenchmark(
        "GPT-OSS 20B training uses too much memory to run on TPUs with smaller"
        " memory."
    )
  batch_size = 1
  seq_len = 1024
  model, inputs = _load_hf_gpt(
      ctx,
      "openai/gpt-oss-20b",
      is_training=True,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=4,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


# --- GPT-OSS-120B ---


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def gpt_oss_120b_4_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for GPT-OSS 120B (4 Layers) forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    raise target_lib.UnsupportedBenchmark(
        "GPT-OSS 120B uses too much memory to run on TPUs with smaller memory."
    )
  batch_size = 16
  seq_len = 2048
  model, inputs = _load_hf_gpt(
      ctx,
      "openai/gpt-oss-120b",
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      num_layers=4,
  )
  return model, (), inputs
