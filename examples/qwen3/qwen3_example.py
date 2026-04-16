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

"""Example use of Qwen3 model."""
import sys

from absl import app
from absl import flags
import torch
from torch_tpu._internal.utils import utils
from examples.qwen3.impl import qwen3

CheckValueMode = utils.CheckValueMode
Qwen3Model = qwen3.Qwen3Model
configs = qwen3.configs

_MODEL = flags.DEFINE_string(
    "model",
    "Integration_Test",
    "Model flavor (one of: Integration_Test, MoEmini, MoE, 0.6B, 1.7B, 4B, 8B,"
    " 14B, 32B)",
)


# pylint: disable=unused-argument
def main(argv):
  torch.manual_seed(123)
  model_flavor = _MODEL.value
  print(f"Qwen3, flavor: '{model_flavor}'", file=sys.stderr)
  if model_flavor not in configs.keys():
    raise ValueError("Unknown model flavor: " + model_flavor)
  cfg = configs[model_flavor]

  # Make model:
  tensor_in = torch.tensor([1, 2, 3]).unsqueeze(0)
  model = Qwen3Model(cfg)
  print(utils.format_model(model, tensor_in, pt=True, params=True))

  # CPU:
  print("CPU")
  output_cpu = model(tensor_in)
  print(output_cpu)

  print("TPU")
  tpu_device = torch.device("tpu")
  model.to(tpu_device)
  tensor_in = tensor_in.to(tpu_device)
  output_tpu = model(tensor_in).to("cpu")
  print(output_tpu)

  print("TPU Compiled AOT")
  compiled = torch.compile(model, backend="tpu")
  aot_compiled_output_tpu = compiled(tensor_in).to("cpu")
  print(aot_compiled_output_tpu)

  utils.assert_close(
      actual=output_tpu,
      expected=output_cpu,
      check_value=CheckValueMode.LOOSE,
      rtol=1e-3,
      atol=6.6e-3,
      preamble="Comparing CPU and TPU",
  )

  utils.assert_close(
      actual=aot_compiled_output_tpu,
      expected=output_cpu,
      check_value=CheckValueMode.LOOSE,
      rtol=1e-3,
      # TODO: debug why AOT version produces result with larger difference.
      atol=6.6e-3,
      preamble="Comparing CPU and TPU with AOT compilation",
  )


if __name__ == "__main__":
  app.run(main)
