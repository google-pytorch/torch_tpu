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

"""Single-device HuggingFace Llama model benchmark definitions.

Multi-device FairScale tensor-parallel Llama benchmarks (such as
`meta_llama_8b_forward`) live in `distributed_models/llama.py` so they are
discovered by `distributed_runner.py` rather than single-device `base_test.py`.
"""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops


def _load_hf_llama(
    ctx: context_lib.Context, model_name: str, is_training: bool
):
  """Loads the HuggingFace Llama model and generates sample inputs for benchmarking."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 2
    seq_len = 2048

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
      config.num_hidden_layers = 2
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


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def llama_1b_forward(ctx):
  """Benchmark factory for Llama 3.2 1B inference."""
  model, inputs = _load_hf_llama(
      ctx, "meta-llama/Llama-3.2-1B", is_training=False
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.DECODER_ONLY_DECODE
)
def llama_1b_decode(ctx):
  """Benchmark factory for Llama 3.2 1B decode."""
  model, inputs = _load_hf_llama(
      ctx, "meta-llama/Llama-3.2-1B", is_training=False
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
)
def llama_1b_training(ctx):
  """Benchmark factory for Llama 3.2 1B training."""
  model, inputs = _load_hf_llama(
      ctx, "meta-llama/Llama-3.2-1B", is_training=True
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def llama_3_1_8b_forward(ctx: context_lib.Context):
  """Benchmark factory for Llama 3.1 8B forward."""
  model, inputs = _load_hf_llama(
      ctx, "meta-llama/Llama-3.1-8B", is_training=False
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
def llama_3_2_3b_forward(ctx: context_lib.Context):
  """Benchmark factory for Llama 3.2 3B forward."""
  model, inputs = _load_hf_llama(
      ctx, "meta-llama/Llama-3.2-3B", is_training=False
  )
  return model, (), inputs


@registry_lib.register_benchmark(stepper=step_lib.StepperType.TRAINING)
def llama_3_2_3b_training(ctx: context_lib.Context):
  """Benchmark factory for Llama 3.2 3B training."""
  model, inputs = _load_hf_llama(
      ctx, "meta-llama/Llama-3.2-3B", is_training=True
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt
