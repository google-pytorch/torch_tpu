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

"""Example of a quantized linear layer for training."""

from absl import app
import torch
from torch import nn
import torch.nn.functional as F
from torch_tpu._internal.utils import test_utils

# Uncomment this line in a future exercise.
# torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access


def quantize(values):
  """Returns signs and a scale as a quantized representation of the values."""

  # Set the scale to be the mean of the magnitudes of the weights.
  scale = abs(values).mean()
  # AKA scale = torch.abs(values).mean()

  # Determine the sign. Force zeros to positive.
  signs = torch.sign(values)
  signs[signs == 0.0] = 1.0

  # Now, all values should be -1 or 1.
  test_utils.assert_close(torch.abs(signs), torch.ones_like(signs))

  return signs, scale


def dequantize(signs, scale):
  return signs * scale


def _pack32(values):
  """Packs a vector of 32 quantized values (signs) into a uint32"""
  assert values.size() == (32,)
  assert values.dtype == torch.float32
  test_utils.assert_close(torch.abs(values), torch.ones_like(values))

  # Translate values from {-1, 1} to {0, 1}
  bits = (values == 1.0).to(torch.uint32)
  # AKA bits = torch.eq(values, 1.0).to(torch.uint32)

  # Create all bit patterns from 0000, 0001, 0010, etc.
  bit_patterns = 2 ** torch.arange(32, dtype=torch.uint32)
  # AKA bit_patterns = torch.pow(2, torch.arange(32, dtype=torch.uint32))

  # Summing the bit patterns is a bitwise_or reduction on a vector.
  return torch.sum(bits * bit_patterns, dim=0, dtype=torch.uint32)


def _unpack32(packed):
  """Unpacks a uint32 into a vector of 32 values"""
  assert packed.dtype == torch.uint32

  # Shift each of the 32 bits by zero to 31 positions.
  shifted = packed >> torch.arange(32, dtype=torch.uint32)
  # AKA shifted = packed.bitwise_right_shift(torch.arange(32, dtype=torch.uint32))

  # Mask out the upper bits.
  values = shifted & 1
  # AKA values = shifted.bitwise_and(1)

  # Translate values from {0, 1} to {-1, 1}
  return values.float() * 2.0 - 1.0


def pack(values):
  return torch.vmap(_pack32)(values)


def unpack(values):
  return torch.vmap(_unpack32)(values)


@torch.compile(backend="aot_eager")
def qat_linear(weight, data):
  """Performs a forward pass of a linear layer with Quantization Aware Training."""
  weight_qat = dequantize(*quantize(weight))

  # This trick implements Quantization Aware Training (QAT):
  # during training the master weights are quantized but not packed
  # in the forward pass. The backward pass still sees the master weights.
  return F.linear(data, (weight_qat - weight).detach() + weight)


class QATLinear(nn.Module):
  """A linear layer that performs Quantization Aware Training."""

  def __init__(self):
    super().__init__()

    self.weight = nn.Parameter(torch.empty((64, 64), dtype=torch.float32))
    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight, a=0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return qat_linear(self.weight, x)


@torch.compile(backend="aot_eager")
def fwd_bwd(model, data):
  output = model(data)
  output.sum().backward()


def main(argv):
  del argv
  data = torch.randn(32, 64)
  qat_linear_layer = QATLinear()
  fwd_bwd(qat_linear_layer, data)
  print("Success.")


if __name__ == "__main__":
  app.run(main)
