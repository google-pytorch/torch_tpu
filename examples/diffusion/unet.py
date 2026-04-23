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

"""Building up an implementation of a denoising Unet.

///////////////////////////////////////////////////////////////
NOTE: This file/model is meant to DEBUG and TRIAGE remaining
      issues in the torch_tpu implementation. It will contain
      some experimental things and lots of TODOs to bypass
      existing problems.
//////////////////////////////////////////////////////////////

Currently needed aten ops:
  torch.ops.aten.convolution.default
  torch.ops.aten.detach.default
  torch.ops.aten.empty.memory_format
  torch.ops.aten.max_pool2d_with_indices.default
  torch.ops.aten._native_batch_norm_legit_no_training.default
  torch.ops.aten.relu_.default
"""

import sys

from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.utils import utils


class UNet(nn.Module):
  """A simple denoising UNet."""

  def __init__(self, in_channels=1, out_channels=1):
    super().__init__()

    def conv_block(in_f, out_f, pool=False):
      layers = [
          nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_f),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_f),
          nn.ReLU(inplace=True),
      ]
      if pool:
        layers.insert(0, nn.MaxPool2d(2))
      return nn.Sequential(*layers)

    self.conv1 = conv_block(in_channels, 64)
    self.conv2 = conv_block(64, 128, pool=True)
    self.conv3 = conv_block(128, 256, pool=True)
    self.conv4 = conv_block(256, 512, pool=True)

  def forward(self, x: torch.Tensor):
    # Encoder
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x4 = self.conv4(x3)
    return x4


def main(argv):
  torch.manual_seed(123)
  print("Unet / simple.", file=sys.stderr)

  # Make model:
  t_tensor = torch.full((3, 3, 28, 28), 0, device="cpu", dtype=torch.float)
  model = UNet(3, 3).eval()
  print(utils.format_model(model, t_tensor, aten=True, pt=True, params=True))
  result = model(t_tensor)

  device = torch.accelerator.current_accelerator()
  model.to(device)
  t_tensor_tpu = t_tensor.to(device)
  result_tpu = model(t_tensor_tpu).to("cpu")

  utils.assert_close(
      result,
      result_tpu,
      rtol=4e-3,
      atol=1e-4,
      check_value=utils.CheckValueMode.LOOSE,
  )


if __name__ == "__main__":
  app.run(main)
