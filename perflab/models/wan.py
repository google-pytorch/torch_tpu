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

"""Wan2.2 Diffusers model benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_diffusers_wan(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    height: int = 704,
    width: int = 1280,
    num_frames: int = 5,
    text_seq_len: int = 512,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a Wan2.2 Diffusers transformer model and generates latent inputs.

  The batch size comes from the workload of the context.
  """
  batch_size = ctx.require_workload().batch_size
  registry = model_helpers.get_module_registry()
  module_spec = registry.get_module_spec(
      "diffusers", model_name, load_weights=False, subfolder="transformer"
  )

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  with torch.device(device_str):
    model = module_spec.module_factory(dtype=dtype)
  model.apply(model_helpers.init_small_random_weights)

  if is_training:
    model.train()
  else:
    model.eval()

  latent_frames = (num_frames - 1) // 4 + 1
  latent_height = height // 16
  latent_width = width // 16

  _, inputs = module_spec.sample_inputs_factory(
      (batch_size, text_seq_len, latent_frames, latent_height, latent_width),
      device_str,
  )
  for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor) and v.is_floating_point():
      inputs[k] = v.to(dtype=dtype)

  return model, inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "Wan2.2-TI2V-5B uses too much memory to run on TPUs with smaller"
        " memory."
    ),
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1)),
)
def wan_2_2_ti2v_5b_forward(ctx: context_lib.Context):
  """Benchmark factory for Wan2.2-TI2V-5B forward."""
  model, inputs = _load_diffusers_wan(
      ctx,
      "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
      is_training=False,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    # The legacy suite measured the 256x256 variants in compiled mode only;
    # eager modes are covered by the full-resolution specs.
    skipped_run_modes=frozenset({"eager_default", "eager_optimized"}),
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "Wan2.2-TI2V-5B uses too much memory to run on TPUs with smaller"
        " memory."
    ),
    # Batch 1 at 256x256 leaves the step dominated by host overhead.
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(4)),
)
def wan_2_2_ti2v_5b_256x256_forward(ctx: context_lib.Context):
  """Benchmark factory for Wan2.2-TI2V-5B (256x256) forward."""
  model, inputs = _load_diffusers_wan(
      ctx,
      "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
      is_training=False,
      height=256,
      width=256,
      num_frames=1,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "Wan2.2-TI2V-5B uses too much memory to run on TPUs with smaller"
        " memory."
    ),
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1)),
)
def wan_2_2_ti2v_5b_training(ctx: context_lib.Context):
  """Benchmark factory for Wan2.2-TI2V-5B training."""
  model, inputs = _load_diffusers_wan(
      ctx,
      "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
      is_training=True,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    # The legacy suite measured the 256x256 variants in compiled mode only;
    # eager modes are covered by the full-resolution specs.
    skipped_run_modes=frozenset({"eager_default", "eager_optimized"}),
    presubmit=registry_lib.PresubmitPolicy.SKIP,
    presubmit_reason=(
        "Wan2.2-TI2V-5B uses too much memory to run on TPUs with smaller"
        " memory."
    ),
    workload=registry_lib.scoped_workload(full=workload_lib.Workload(1)),
)
def wan_2_2_ti2v_5b_256x256_training(ctx: context_lib.Context):
  """Benchmark factory for Wan2.2-TI2V-5B (256x256) training."""
  model, inputs = _load_diffusers_wan(
      ctx,
      "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
      is_training=True,
      height=256,
      width=256,
      num_frames=1,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, (), inputs, opt
