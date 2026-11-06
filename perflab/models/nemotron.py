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

"""Nemotron model benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_hf_nemotron(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    seq_len: int = 512,
    batch_size: int = 1,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads the HuggingFace Nemotron model and generates sample inputs."""

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if num_layers is not None:
      # NemotronHConfig derives num_hidden_layers from layers_block_type.
      config.layers_block_type = config.layers_block_type[:num_layers]
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


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "Nemotron-3-Nano-30B uses too much memory to run on TPUs with smaller"
        " memory."
    ),
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1, 512)),
)
def nemotron_3_nano_30b_6_layers_forward(ctx: context_lib.Context):
  """Benchmark factory for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (6 Layers) forward."""
  workload = ctx.require_workload()
  model, inputs = _load_hf_nemotron(
      ctx,
      "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
      is_training=False,
      seq_len=workload.require_seq_len(),
      batch_size=workload.batch_size,
      num_layers=6,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "Nemotron-3-Nano-30B uses too much memory to run on TPUs with smaller"
        " memory."
    ),
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1, 512)),
)
def nemotron_3_nano_30b_6_layers_decode(ctx: context_lib.Context):
  """Benchmark factory for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (6 Layers) decode."""
  workload = ctx.require_workload()
  model, inputs = _load_hf_nemotron(
      ctx,
      "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
      is_training=False,
      seq_len=workload.require_seq_len(),
      batch_size=workload.batch_size,
      num_layers=6,
  )
  return model, (), inputs
