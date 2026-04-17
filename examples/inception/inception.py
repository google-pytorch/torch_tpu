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

"""Example of an Inception (GoogLeNet) model."""

from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.utils import utils

_TPU = flags.DEFINE_bool("tpu", True, "Run on TPU and compare output to CPU.")


class Inception(nn.Module):
  """GoogLeNet Inception Module."""

  def __init__(
      self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj
  ):
    super().__init__()

    # 1x1 Branch
    self.branch1 = nn.Sequential(
        nn.Conv2d(in_channels, n1x1, kernel_size=1), nn.ReLU(True)
    )

    # 3x3 Branch (with 1x1 reduction)
    self.branch2 = nn.Sequential(
        nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1),
        nn.ReLU(True),
        nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
        nn.ReLU(True),
    )

    # 5x5 Branch (with 1x1 reduction)
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1),
        nn.ReLU(True),
        nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
        nn.ReLU(True),
    )

    # Pooling Branch (with 1x1 projection)
    self.branch4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        nn.ReLU(True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Concatenate the outputs of the four branches along the channel dimension
    # (batch_size, channels, height, width)
    return torch.cat(
        [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
    )


class GoogLeNet(nn.Module):
  """The full GoogLeNet model."""

  def __init__(self, num_classes=1000, init_weights=True):
    super().__init__()

    # Initial Layers
    self.pre_layers = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(True),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(True),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
    )

    # Inception Modules (The parameters correspond to the paper's specification)
    #
    # Note that the original Inception paper introduced a local response layer
    # after the first and second maxpool layers. We omit this here as this layer
    # is also mostly omitted in later versions. For example:
    # Create an LRN layer with a window size of 5
    #   lrn_layer = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)

    self.section3 = nn.Sequential(
        # Inception 3a (input 192, output 256)
        Inception(192, 64, 96, 128, 16, 32, 32),
        # Inception 3b (input 256, output 480)
        Inception(256, 128, 128, 192, 32, 96, 64),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
    )

    self.section4 = nn.Sequential(
        # Inception 4a (input 480, output 512)
        Inception(480, 192, 96, 208, 16, 48, 64),
        # Inception 4b (input 512, output 512)
        Inception(512, 160, 112, 224, 24, 64, 64),
        # Inception 4c (input 512, output 512)
        Inception(512, 128, 128, 256, 24, 64, 64),
        # Inception 4d (input 512, output 528)
        Inception(512, 112, 144, 288, 32, 64, 64),
        # Inception 4e (input 528, output 832)
        Inception(528, 256, 160, 320, 32, 128, 128),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
    )

    self.section5 = nn.Sequential(
        # Inception 5a (input 832, output 832)
        Inception(832, 256, 160, 320, 32, 128, 128),
        # Inception 5b (input 832, output 1024)
        Inception(832, 384, 192, 384, 48, 128, 128),
    )

    # Classifier Layers
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(1024, num_classes)  # Final output channels is 1024

    if init_weights:
      self._initialize_weights()

  def _initialize_weights(self):
    # A simple weight initialization for demonstration
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Initial Block
    x = self.pre_layers(x)

    # Inception Blocks
    x = self.section3(x)
    x = self.section4(x)
    x = self.section5(x)

    # Final Classification
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.fc(x)

    return x


def main(argv: list[str]) -> None:
  del argv

  # Instantiate the model (e.g., for 10 classes like CIFAR-10)
  model = GoogLeNet(num_classes=10)
  model.eval()

  # Create a dummy input tensor (Batch size 1, 3 color channels, 224x224 image)
  torch.manual_seed(0)
  dummy_input = torch.randn(1, 3, 224, 224)

  # Get the output
  output = model(dummy_input)
  print(f"CPU: {output}")

  # Run TPU Model.
  if not _TPU.value:
    return
  tpu_device = torch.device("tpu")
  model.to(tpu_device)
  output_tpu = model(dummy_input.to(tpu_device)).to("cpu")
  utils.assert_close(
      actual=output_tpu,
      expected=output,
      preamble="Inception output",
      rtol=5e-2,
      atol=6.1e-1,
      check_value=utils.CheckValueMode.LOOSE,
  )
  print(f"TPU: {output_tpu}")


if __name__ == "__main__":
  app.run(main)
