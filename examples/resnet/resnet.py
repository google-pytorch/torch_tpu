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

"""Example use of ResNet model."""

import sys

from absl import app
from absl import flags
import torch
from torch_tpu._internal.utils import utils
from examples.resnet.impl import resnet

Block = resnet.Block
ResNet = resnet.ResNet

rand = torch.rand
_MODEL = flags.DEFINE_integer("model", 18, "Model flavor. Can be 18 or 34.")
_PRINT = flags.DEFINE_bool("print", False, "Print model after construction.")
_PARAMS = flags.DEFINE_bool("params", False, "Print # parameters and memory.")
_TPU = flags.DEFINE_bool("tpu", True, "Run on TPU and compare output to CPU.")
_ATEN = flags.DEFINE_bool("aten", False, "Print the aten kernels.")
_SHLO = flags.DEFINE_bool("shlo", False, "Print the generated SHLO.")
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None, "Output file for program dumps."
)


def main(argv):
  del argv
  flavor = _MODEL.value
  print(f"PyTorch ResNet: {flavor}", file=sys.stderr)

  model = ResNet(
      img_channels=3, num_layers=flavor, block=Block, num_classes=1000
  )
  model.eval()

  # Run CPU Model.
  input_tensor = rand([1, 3, 224, 224])
  with utils.open_output(_OUTPUT_FILE.value) as f:
    print(
        utils.format_model(
            model,
            input_tensor,
            params=_PARAMS.value,
            pt=_PRINT.value,
            aten=_ATEN.value,
            shlo=_SHLO.value,
        ),
        file=f,
    )
  output = model(input_tensor)

  # Run TPU Model.
  if _TPU.value:
    tpu_device = torch.device("tpu")
    model.to(tpu_device)
    output_tpu = model(input_tensor.to(tpu_device)).to("cpu")
    utils.assert_close(
        actual=output_tpu,
        expected=output,
        preamble="Resnet output",
        rtol=2.1e-4,
        atol=3.4e-1,
        check_value=utils.CheckValueMode.LOOSE,
    )


if __name__ == "__main__":
  app.run(main)
