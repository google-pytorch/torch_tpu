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

"""Whisper audio model benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib


def _load_hf_whisper(
    ctx: context_lib.Context,
    model_name: str,
    is_training: bool,
    batch_size: int = 16,
    seq_len: int = 448,
    num_layers: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace Whisper model and generates spectrogram/decoder inputs."""

  def _modify_config_hook(config: Any) -> Any:
    setattr(config, "use_cache", False)
    if num_layers is not None:
      if hasattr(config, "encoder_layers"):
        config.encoder_layers = num_layers
      if hasattr(config, "decoder_layers"):
        config.decoder_layers = num_layers
    return config

  registry = model_helpers.get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
      modify_config_hook=_modify_config_hook,
  )
  config = module_spec.config
  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device_str = ctx.device_kind.value

  with torch.device(device_str):
    model = module_spec.module_factory(dtype=dtype)

  if is_training:
    model.train()
  else:
    model.eval()

  num_mel_bins = getattr(config, "num_mel_bins", 128)
  # Whisper's convolutional front end requires exactly 30 s of audio:
  # 3000 log-mel frames at 100 frames/s.
  feature_length = 3000
  decoder_start = getattr(config, "decoder_start_token_id", 50258)

  input_features = torch.randn(
      (batch_size, num_mel_bins, feature_length),
      dtype=dtype,
      device=device_str,
  )
  decoder_input_ids = torch.full(
      (batch_size, seq_len),
      decoder_start,
      dtype=torch.long,
      device=device_str,
  )

  inputs = {
      "input_features": input_features,
      "decoder_input_ids": decoder_input_ids,
  }
  return model, inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(16, 448),
        presubmit=workload_lib.Workload(1, 64),
    ),
)
def whisper_large_v3_forward(ctx: context_lib.Context):
  """Benchmark factory for Whisper-Large-v3 forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    num_layers = 2
  else:
    num_layers = None

  model, inputs = _load_hf_whisper(
      ctx,
      "openai/whisper-large-v3",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
      num_layers=num_layers,
  )
  return model, (), inputs


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
    workload=registry_lib.scoped_workload(
        full=workload_lib.Workload(16, 448),
        presubmit=workload_lib.Workload(1, 64),
    ),
)
def whisper_base_forward(ctx: context_lib.Context):
  """Benchmark factory for Whisper-Base forward."""
  workload = ctx.require_workload()
  batch_size = workload.batch_size
  seq_len = workload.require_seq_len()

  model, inputs = _load_hf_whisper(
      ctx,
      "openai/whisper-base",
      is_training=False,
      batch_size=batch_size,
      seq_len=seq_len,
  )
  return model, (), inputs
