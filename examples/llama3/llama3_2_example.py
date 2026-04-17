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

"""Example use of Llama 3.2 model."""

import sys

from absl import app
from absl import flags
import torch
from torch_tpu._internal.utils import utils
from examples.llama3.impl import llama3_2


CheckValueMode = utils.CheckValueMode
Llama32Model = llama3_2.Llama32Model
configs = llama3_2.configs

_MODEL = flags.DEFINE_string("model", "1B", "Model flavor (one of: 1B, 3B)")
_TPU = flags.DEFINE_bool("tpu", True, "Run on TPU.")
_CPU = flags.DEFINE_bool("cpu", True, "Run on CPU.")


# pylint: disable=unused-argument
def main(argv):
  torch.manual_seed(123)
  model_flavor = _MODEL.value
  print(f"Llama 3.2, flavor: '{model_flavor}'", file=sys.stderr)
  if model_flavor not in configs.keys():
    raise ValueError("Unknown model flavor: " + model_flavor)
  cfg = configs[model_flavor]

  # Make model:
  tensor_in = torch.tensor([3]).unsqueeze(0)
  model = Llama32Model(cfg)
  print(utils.format_model(model, tensor_in, pt=True, params=True))

  # CPU:
  logits = None
  if _CPU.value:
    print("CPU")
    with torch.no_grad():
      logits = model(tensor_in)
    print(logits)

  # TPU:
  logits_tpu = None
  if _TPU.value:
    print("TPU")
    tpu_device = torch.device("tpu")
    model.to(tpu_device)
    tpu_tensor = tensor_in.to(tpu_device)
    with torch.no_grad():
      logits_tpu = model(tpu_tensor).to("cpu")
    print(logits_tpu)

  # Compare:
  if _CPU.value and _TPU.value:
    utils.assert_close(
        logits_tpu,
        logits,
        check_value=CheckValueMode.LOOSE,
        rtol=1e-3,
        atol=1e-5,
    )


if __name__ == "__main__":
  app.run(main)
