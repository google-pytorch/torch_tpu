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

"""HuggingFace vision and vision-language encoder benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_hf_vision_model(
    ctx: context_lib.Context,
    model_name: str,
    batch_size: int,
    seq_len: int = 1,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace vision model and generates image (and text) inputs.

  Args:
    ctx: The benchmark context.
    model_name: HuggingFace model identifier.
    batch_size: Number of images (and texts) per step.
    seq_len: Text sequence length of vision-language models. The image size
      comes from the model config.

  Returns:
    A tuple of (model, inputs dictionary).
  """

  def _modify_config_hook(config: Any) -> Any:
    if hasattr(config, "use_cache"):
      config.use_cache = False
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


# Full-scope batch sizes are the smallest powers of two (at most 64) at which
# compiled device time is at least 85% of the step time on v7_1x1x1 (80% if 85%
# is not reached), as for the generated benchmarks.


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(32), presubmit=workload_lib.Workload(1)
    ),
)
def clip_base_forward(ctx: context_lib.Context):
  """Benchmark factory for CLIP ViT-B/16 (openai/clip-vit-base-patch16) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  # 77 is the maximum number of CLIP text positions.
  model, inputs = _load_hf_vision_model(
      ctx, "openai/clip-vit-base-patch16", batch_size=batch_size, seq_len=77
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(32), presubmit=workload_lib.Workload(1)
    ),
)
def vit_base_forward(ctx: context_lib.Context):
  """Benchmark factory for ViT-B/16 (google/vit-base-patch16-224) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, inputs = _load_hf_vision_model(
      ctx, "google/vit-base-patch16-224", batch_size=batch_size
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(32), presubmit=workload_lib.Workload(1)
    ),
)
def siglip_base_forward(ctx: context_lib.Context):
  """Benchmark factory for SigLIP-B/16 (google/siglip-base-patch16-224) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  # 64 is the maximum number of SigLIP text positions.
  model, inputs = _load_hf_vision_model(
      ctx, "google/siglip-base-patch16-224", batch_size=batch_size, seq_len=64
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(16), presubmit=workload_lib.Workload(1)
    ),
)
def dinov2_base_forward(ctx: context_lib.Context):
  """Benchmark factory for DINOv2-B (facebook/dinov2-base) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, inputs = _load_hf_vision_model(
      ctx, "facebook/dinov2-base", batch_size=batch_size
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(64), presubmit=workload_lib.Workload(1)
    ),
)
def deit_tiny_forward(ctx: context_lib.Context):
  """Benchmark factory for DeiT-Ti (facebook/deit-tiny-patch16-224) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, inputs = _load_hf_vision_model(
      ctx, "facebook/deit-tiny-patch16-224", batch_size=batch_size
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(32), presubmit=workload_lib.Workload(1)
    ),
)
def segformer_b2_forward(ctx: context_lib.Context):
  """Benchmark factory for SegFormer-B2 (ADE20K, 512x512) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, inputs = _load_hf_vision_model(
      ctx,
      "nvidia/segformer-b2-finetuned-ade-512-512",
      batch_size=batch_size,
  )
  return model, (), inputs
