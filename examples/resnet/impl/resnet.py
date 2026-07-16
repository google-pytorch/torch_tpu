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

"""ResNet 18 / 34 in PyTorch.

Paper:
  Deep Residual Learning for Image Recognition.
  https://arxiv.org/pdf/1512.03385v1.pdf

This implementation was created with help of the instructions in:
  https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
"""

import torch
from torch import nn

flatten = torch.flatten
Tensor = torch.Tensor


class Block(nn.Module):
  """Standard basic block in the resnet architecture."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      stride: int = 1,
      expansion: int = 1,
      downsample: nn.Module | None = None,
  ) -> None:
    super().__init__()

    # Multiplicative factor for the subsequent conv2d layer's output channels.
    # It is 1 for ResNet18 and ResNet34.
    self.expansion = expansion
    self.downsample = downsample
    self.conv1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        out_channels,
        out_channels * self.expansion,
        kernel_size=3,
        padding=1,
        bias=False,
    )
    self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

  def forward(self, x: Tensor) -> Tensor:
    identity = x

    res = self.conv1(x)
    res = self.bn1(res)
    res = self.relu(res)

    res = self.conv2(res)
    res = self.bn2(res)

    if self.downsample is not None:
      identity = self.downsample(x)

    res += identity
    res = self.relu(res)
    return res


class ResNet(nn.Module):
  """Resnet Model itself."""

  def __init__(
      self,
      img_channels: int,
      num_layers: int,
      block: type[Block],
      num_classes: int = 1000,
  ) -> None:
    super().__init__()
    assert num_layers == 18 or num_layers == 34
    layers = [2, 2, 2, 2] if num_layers == 18 else [3, 4, 6, 3]
    self.expansion = 1
    self.in_channels = 64

    # All variants have Conv2d => BN => ReLU for the first three layers.
    # Kernel size is 7 for these models, according to the paper. Similar
    # for stride and padding.
    kernel_size = 7
    stride = 2
    padding = 3

    self.conv1 = nn.Conv2d(
        in_channels=img_channels,
        out_channels=self.in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(
        kernel_size=kernel_size, stride=stride, padding=padding
    )

    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=stride)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=stride)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=stride)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * self.expansion, num_classes)

  def _make_layer(
      self,
      block: type[Block],
      out_channels: int,
      blocks: int,
      stride: int = 1,
  ) -> nn.Sequential:
    downsample = (
        nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        if stride != 1
        else None
    )
    layers = []
    layers.append(
        block(
            self.in_channels, out_channels, stride, self.expansion, downsample
        )
    )
    self.in_channels = out_channels * self.expansion

    for _ in range(1, blocks):
      layers.append(
          block(self.in_channels, out_channels, expansion=self.expansion)
      )
    return nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    assert x.shape[2] == 7
    assert x.shape[3] == 7
    print("Dimensions of the last convolutional feature map: ", x.shape)

    x = self.avgpool(x)
    x = flatten(x, 1)
    x = self.fc(x)

    return x
