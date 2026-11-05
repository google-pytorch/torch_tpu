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

"""V-JEPA2 video model benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_hf_vjepa2(
    ctx: context_lib.Context,
    model_name: str,
    batch_size: int = 1,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace V-JEPA2 encoder and generates video tensor inputs."""
  registry = model_helpers.get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
  )
  config = module_spec.config
  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  input_shape = (
      batch_size,
      getattr(config, "frames_per_clip", 64),
      getattr(config, "in_chans", 3),
      getattr(config, "crop_size", 256),
      getattr(config, "crop_size", 256),
  )

  with torch.device(device_str):
    model = module_spec.module_factory(dtype=dtype)
  model.apply(model_helpers.init_small_random_weights)
  model.eval()

  inputs: dict[str, Any] = {
      "pixel_values_videos": torch.randn(
          input_shape, dtype=dtype, device=device_str
      ),
      # Encoder only, as in the legacy benchmark.
      "skip_predictor": True,
  }
  return model, inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "V-JEPA2 ViT-G uses too much memory to run on TPUs with smaller memory."
    ),
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1)),
)
def vjepa2_vitg_forward(ctx: context_lib.Context):
  """Benchmark factory for V-JEPA2 ViT-G (facebook/vjepa2-vitg-fpc64-384) forward."""
  model, inputs = _load_hf_vjepa2(
      ctx,
      "facebook/vjepa2-vitg-fpc64-384",
      batch_size=ctx.require_workload().batch_size,
  )
  return model, (), inputs
