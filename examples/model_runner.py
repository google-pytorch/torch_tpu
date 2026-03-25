# Copyright 2025 Google LLC
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

"""Model runner script for benchmarking PyTorch models on different devices."""

from absl import app
from absl import flags
import torch
from torch_tpu import api
from examples import model_runner_lib
from examples.qwen3.impl import qwen3


model_runner = model_runner_lib.model_runner
models = {
    'qwen3': qwen3.Qwen3Model,
}

model_configs = {
    'qwen3': qwen3.configs,
}

_MODEL_TYPE = flags.DEFINE_string('model_type', 'qwen3', 'Model type to use.')
_MODEL_FLAVOR = flags.DEFINE_string(
    'model_flavor', '0.6B', 'Model flavor to use.'
)
_PROMPT_LEN = flags.DEFINE_integer(
    'prompt_len', 8, 'Length of the prefill prompt.'
)
_DECODE_STEPS = flags.DEFINE_integer(
    'decode_steps', 2, 'Number of decode steps to benchmark.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'Batch size.')
_DEVICE = flags.DEFINE_string('device', 'tpu', 'Device to run the model on.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_type = _MODEL_TYPE.value
  if model_type not in models:
    raise ValueError(
        f'Unknown model type: {model_type}. Available: {list(models.keys())}'
    )
  model = models[model_type]
  configs = model_configs[model_type]
  model_flavor = _MODEL_FLAVOR.value
  if model_flavor not in configs:
    raise ValueError(
        f'Unknown model flavor: {model_flavor}. Available:'
        f' {list(configs.keys())}'
    )
  cfg = configs[model_flavor]

  if _PROMPT_LEN.value > cfg['context_length']:
    raise ValueError(
        f'Prompt length {_PROMPT_LEN.value} is larger than the context length'
        f' {cfg["context_length"]}. Please use a smaller prompt length.'
    )

  # Adjust dtype if needed, e.g., to torch.bfloat16 for TPU
  # cfg['dtype'] = torch.bfloat16
  model = model(cfg)

  device = _DEVICE.value
  if device == 'tpu':
    device = api.tpu_device()
  else:
    device = torch.device(device)

  model_runner(
      model,
      device,
      prompt_len=_PROMPT_LEN.value,
      decode_steps=_DECODE_STEPS.value,
      batch_size=_BATCH_SIZE.value,
  )


if __name__ == '__main__':
  app.run(main)
