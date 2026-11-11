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

"""HuggingFace vision-language model benchmark definitions."""

from collections.abc import Callable
from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_hf_vlm(
    ctx: context_lib.Context,
    model_name: str,
    batch_size: int,
    seq_len: int,
    modify_config_hook: Callable[[Any], Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace vision-language model and generates image+text inputs.

  Args:
    ctx: The benchmark context.
    model_name: HuggingFace model identifier.
    batch_size: Number of image-text pairs per step.
    seq_len: Text tokens per sequence, including the image tokens.
    modify_config_hook: Optional hook to modify the model configuration.

  Returns:
    A tuple of (model, inputs dictionary).
  """

  def _modify_config_hook(config: Any) -> Any:
    config.use_cache = False
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
      text_config.use_cache = False
    if modify_config_hook is not None:
      config = modify_config_hook(config)
    return config

  return model_helpers.load_hf_llm(
      model_name,
      dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
      device_str=ctx.device_kind.value,
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      modify_config_hook=_modify_config_hook,
      keep_vision_inputs=True,
  )


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    # The image alone takes 576 tokens.
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1, 1024)),
)
def llava_1_5_7b_forward(ctx: context_lib.Context):
  """Benchmark factory for LLaVA-1.5-7B (liuhaotian/llava-v1.5-7b) forward."""
  presubmit = ctx.run_scope == context_lib.RunScope.PRESUBMIT

  def _modify_config_hook(config: Any) -> Any:
    # The LLaVA-1.5 checkpoint resizes the embedding table to 32064 tokens, so
    # sample token ids may reach 32063.
    config.vocab_size = max(getattr(config, "vocab_size", 32000), 32064)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
      text_config.vocab_size = max(
          getattr(text_config, "vocab_size", 32000), 32064
      )
      if presubmit:
        text_config.num_hidden_layers = 2
    vision_config = getattr(config, "vision_config", None)
    if presubmit and vision_config is not None:
      # The vision feature comes from the second-to-last vision layer.
      vision_config.num_hidden_layers = 2
    return config

  workload = ctx.require_workload()
  model, inputs = _load_hf_vlm(
      ctx,
      "liuhaotian/llava-v1.5-7b",
      batch_size=workload.batch_size,
      seq_len=workload.require_seq_len(),
      modify_config_hook=_modify_config_hook,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
    presubmit_reason=(
        "Llama 3.2 11B Vision uses too much memory to run on TPUs with"
        " smaller memory, and its cross-attention layer indices prevent"
        " truncating it."
    ),
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(1, 512),
        presubmit=workload_lib.Workload(
            model_helpers.SMOKE_BATCH_SIZE, model_helpers.SMOKE_SEQ_LEN
        ),
    ),
)
def mllama_11b_forward(ctx: context_lib.Context):
  """Benchmark factory for Llama-3.2-11B-Vision (mLlama) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  model, inputs = _load_hf_vlm(
      ctx,
      "meta-llama/Llama-3.2-11B-Vision",
      batch_size=batch_size,
      seq_len=seq_len,
  )
  return model, (), inputs
