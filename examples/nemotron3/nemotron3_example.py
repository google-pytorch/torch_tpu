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

"""Example of running Nemotron3 on TPU."""

import sys
from absl import app
from absl import flags
import torch
from torch_tpu._internal.utils import utils
from examples.nemotron3 import nemotron3

CheckValueMode = utils.CheckValueMode

_MODEL = flags.DEFINE_string(
    "model",
    "Toy",
    "Model flavor (one of: Toy, 30B)",
)
_N_LAYERS = flags.DEFINE_integer(
    "n_layers",
    None,
    "Number of layers. If not specified, uses the default for the model"
    " flavor.",
)


# pylint: disable=unused-argument
def main(argv):
  torch.manual_seed(123)

  model_flavor = _MODEL.value
  if model_flavor not in nemotron3.configs:
    raise ValueError(f"Unknown model flavor: {model_flavor}")
  cfg_params = nemotron3.configs[model_flavor].copy()

  if _N_LAYERS.value is not None:
    default_pattern = cfg_params.get(
        "layers_block_type", ["mamba", "moe", "mamba", "attention", "moe"]
    )
    layers_block_type = [
        default_pattern[i % len(default_pattern)]
        for i in range(_N_LAYERS.value)
    ]
    cfg_params["layers_block_type"] = layers_block_type

  config = nemotron3.NemotronHConfig(**cfg_params)

  # Make model
  model = nemotron3.NemotronHForCausalLM(config)

  # Dummy input (batch_size=1, seq_len=16)
  tensor_in = torch.randint(0, config.vocab_size, (1, 16))

  print("CPU Run...", file=sys.stderr)
  output_cpu = model(tensor_in)
  print("CPU Output shape:", output_cpu.shape, file=sys.stderr)

  print("TPU Run...", file=sys.stderr)
  tpu_device = torch.device("tpu")
  model.to(tpu_device)
  tensor_in_tpu = tensor_in.to(tpu_device)
  output_tpu = model(tensor_in_tpu).to("cpu")
  print("TPU Output shape:", output_tpu.shape, file=sys.stderr)

  print("TPU Compiled Run...", file=sys.stderr)
  compiled_model = torch.compile(model, backend="tpu")
  output_compiled = compiled_model(tensor_in_tpu).to("cpu")
  print("TPU Compiled Output shape:", output_compiled.shape, file=sys.stderr)

  # Verify outputs are close
  n_layers = len(config.layers_block_type)
  if n_layers > 5:
    rtol = 1.0
    atol = 5e-1
  else:
    rtol = 1.5e-1
    atol = 2e-2

  utils.assert_close(
      actual=output_tpu,
      expected=output_cpu,
      check_value=CheckValueMode.LOOSE,
      rtol=rtol,
      atol=atol,
      preamble="Comparing CPU and TPU eager outputs",
  )

  utils.assert_close(
      actual=output_compiled,
      expected=output_cpu,
      check_value=CheckValueMode.LOOSE,
      rtol=rtol,
      atol=atol,
      preamble="Comparing CPU and TPU compiled outputs",
  )


if __name__ == "__main__":
  app.run(main)
