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

"""TIMM vision model benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_timm_model(
    ctx: context_lib.Context,
    model_name: str,
    input_shape: tuple[int, int, int, int],
    is_training: bool,
) -> tuple[torch.nn.Module, tuple[Any, ...]]:
  """Loads a TIMM model and generates sample positional inputs."""
  registry = model_helpers.get_module_registry()
  module_spec = registry.get_module_spec("timm", model_name, load_weights=False)

  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  with torch.device(device_str):
    model = module_spec.module_factory(dtype=dtype)

  if is_training:
    model.train()
  else:
    model.eval()

  input_args, _ = module_spec.sample_inputs_factory(input_shape, device_str)
  example_inputs = (input_args[0].to(dtype=dtype),)
  return model, example_inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(128), presubmit=workload_lib.Workload(2)
    ),
)
def timm_resnet_50_forward(ctx: context_lib.Context):
  """Benchmark factory for TIMM ResNet-50 (resnet50d) forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, input_args = _load_timm_model(
      ctx,
      "resnet50d",
      input_shape=(batch_size, 3, 224, 224),
      is_training=False,
  )
  return model, input_args, {}


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(64), presubmit=workload_lib.Workload(2)
    ),
)
def timm_resnet_50_training(ctx: context_lib.Context):
  """Benchmark factory for TIMM ResNet-50 (resnet50d) training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, input_args = _load_timm_model(
      ctx,
      "resnet50d",
      input_shape=(batch_size, 3, 224, 224),
      is_training=True,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, input_args, {}, opt


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(8), presubmit=workload_lib.Workload(1)
    ),
)
def timm_vit_small_dino_forward(ctx: context_lib.Context):
  """Benchmark factory for TIMM ViT-Small DINO forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, input_args = _load_timm_model(
      ctx,
      "vit_small_patch8_224.dino",
      input_shape=(batch_size, 3, 224, 224),
      is_training=False,
  )
  return model, input_args, {}


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.TRAINING,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(8), presubmit=workload_lib.Workload(1)
    ),
)
def timm_vit_small_dino_training(ctx: context_lib.Context):
  """Benchmark factory for TIMM ViT-Small DINO training."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  model, input_args = _load_timm_model(
      ctx,
      "vit_small_patch8_224.dino",
      input_shape=(batch_size, 3, 224, 224),
      is_training=True,
  )
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=1e-4,
      capturable=True,
      fused=True,
  )
  return model, input_args, {}, opt
