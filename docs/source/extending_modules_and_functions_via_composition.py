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

# pylint: skip-file

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
      # Extending modules and functions via composition

      ### Builtin functionality: modules, functions

      There are two types of builtin callables in PyTorch: modules and functions.

      *   Functions, such as `torch.nn.functional.linear`, are designed to be
          stateless. Stateful values, such as weights, are provided as explicit
          arguments to these functions.
      *   Modules, such as `nn.Linear`, hold state such as trainable weights. The
          computation of a module is a function of both the input and the weights.
          Modules are always subclasses of `torch.nn.Module`.

      A common pattern in the spirit of separation of concerns is for code in modules
      to focus on state management and hooks. The actual computation is delegated to a
      function.

      For example, `torch.nn.Linear`'s `forward()` method delegates the core logic to
      `torch.nn.functional.linear`:

      ```python
      def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
      ```

      [PyTorch source](https://github.com/pytorch/pytorch/blob/446033e622b25fea5fb72c3b1c5d9f9da2a11ef3/torch/nn/modules/linear.py#L130-L134)

      The rest of the torch.nn.Linear class handles creation, saving, and loading
      weights via the built-in functionality of `torch.nn.Module` and
      `torch.nn.Parameter`.

      ```python
      def __init__(...) -> None:
        ...
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
      ```

      [PyTorch source](https://github.com/pytorch/pytorch/blob/446033e622b25fea5fb72c3b1c5d9f9da2a11ef3/torch/nn/modules/linear.py#L96)

      ### Extending modules and functions via composition

      The most basic way to use PyTorch to build your own custom functionality is to
      combine PyTorch-provided functions into larger functions, aka composition. When
      you need to hold state, such as trainable weights, you create a subclass of
      `torch.nn.Module`.

      As a toy problem, suppose you want to create a custom module that handles
      quantization aware training (QAT). In QAT, the master weights are held in a high
      precision format, but the effect of quantization is simulated during training.
      For simplicity, the quantization scheme will quantize to one bit, encoding only
      +1 and -1, along with a block scaling factor.

      > This guide's focus is on extending PyTorch functionality, so its discussion of
      > quantization should not be considered realistic.

      Look at the sample implementation below. Notice that:

      *   `QATLinear` is a subclass of `torch.nn.Module`.
      *   `QATLinear` holds state in the form of trainable parameters, e.g. weights.
      *   The core computation in `forward()` is delegated to `qat_linear()`.
      *   `qat_linear` itself calls `quantize` and `dequantize` to implement QAT.
      """)
  return


@app.cell()
def _():
  from absl import app as absl_app
  import torch
  from torch import nn
  import torch.nn.functional as F

  # Uncomment this line in a future exercise.
  # torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access
  return (F, absl_app, nn, torch)


@app.cell()
def _(torch):
  def quantize(values):
    """Returns signs and a scale as a quantized representation of the values."""

    # Set the scale to be the mean of the magnitudes of the weights.
    scale = abs(values).mean()
    # AKA scale = torch.abs(values).mean()

    # Determine the sign. Force zeros to positive.
    signs = torch.sign(values)
    signs[signs == 0.0] = 1.0

    # Now, all values should be -1 or 1.
    torch.testing.assert_close(torch.abs(signs), torch.ones_like(signs))

    return signs, scale

  def dequantize(signs, scale):
    return signs * scale

  return (dequantize, quantize)


@app.cell()
def _(torch):
  def pack32(values):
    """Packs a vector of 32 quantized values (signs) into a uint32"""
    assert values.size() == (32,)
    assert values.dtype == torch.float32
    torch.testing.assert_close(torch.abs(values), torch.ones_like(values))

    # Translate values from {-1, 1} to {0, 1}
    bits = (values == 1.0).to(torch.int32)
    # AKA bits = torch.eq(values, 1.0).to(torch.uint32)

    # Create all bit patterns from 0000, 0001, 0010, etc.
    bit_patterns = 2 ** torch.arange(32, dtype=torch.int32)
    # AKA bit_patterns = torch.pow(2, torch.arange(32, dtype=torch.uint32))

    # Summing the bit patterns is a bitwise_or reduction on a vector.
    return torch.sum(bits * bit_patterns, dim=0, dtype=torch.int32).to(
        torch.uint32
    )

  def unpack32(packed):
    """Unpacks a uint32 into a vector of 32 values"""
    assert packed.dtype == torch.uint32

    # Cast to int64 to perform bitwise operations safely on CPU
    packed_int = packed.to(torch.int64)

    # Shift each of the 32 bits by zero to 31 positions.
    shifted = packed_int >> torch.arange(32, dtype=torch.int32)
    # AKA shifted = packed.bitwise_right_shift(torch.arange(32, dtype=torch.uint32))

    # Mask out the upper bits.
    values = shifted & 1
    # AKA values = shifted.bitwise_and(1)

    # Translate values from {0, 1} to {-1, 1}
    return values.float() * 2.0 - 1.0

  def pack(values):
    return torch.vmap(pack32)(values)

  def unpack(values):
    return torch.vmap(unpack32)(values)

  return (pack32, unpack32, pack, unpack)


@app.cell()
def _(F, dequantize, quantize, torch):
  @torch.compile(backend="aot_eager")
  def qat_linear(weight, data):
    """Performs a forward pass of a linear layer with Quantization Aware Training."""
    weight_qat = dequantize(*quantize(weight))

    # This trick implements Quantization Aware Training (QAT):
    # during training the master weights are quantized but not packed
    # in the forward pass. The backward pass still sees the master weights.
    return F.linear(data, (weight_qat - weight).detach() + weight)

  return (qat_linear,)


@app.cell()
def _(nn, qat_linear, torch):
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

  return (QATLinear,)


@app.cell()
def _(torch):
  @torch.compile(backend="aot_eager")
  def fwd_bwd(model, data):
    output = model(data)
    output.sum().backward()

  return (fwd_bwd,)


@app.cell()
def _(QATLinear, fwd_bwd, torch):
  def main():
    data = torch.randn(32, 64)
    qat_linear_layer = QATLinear()
    fwd_bwd(qat_linear_layer, data)
    print("Success.")

  main()
  return (main,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
      ## 🔍 Interactive Step-by-Step Visualization

      Let's walk through exactly how quantization, dequantization, and bit-packing modify a sample tensor of 32 weights.
      """)
  return


@app.cell()
def _(dequantize, pack32, quantize, torch, unpack32):
  # 1. Create sample raw weights (32 elements)
  sample_weights = torch.randn(32)
  print("1. Original Weights (First 10):")
  print(sample_weights[:10])

  # 2. Quantize
  signs, scale = quantize(sample_weights)
  print("\n2. Quantized signs (First 10 in {-1, 1}):")
  print(signs[:10])
  print(f"Calculated scale (mean magnitude): {scale.item():.4f}")

  # 3. Dequantize (Simulating QAT weights during forward pass)
  simulated_weights = dequantize(signs, scale)
  print("\n3. Simulated QAT weights (dequantized, First 10):")
  print(simulated_weights[:10])

  # 4. Pack to a single uint32 integer directly
  packed_val = pack32(signs)
  print(f"\n4. Packed uint32 representation (32 signs -> 1 integer):")
  print(
      f"Integer value: {packed_val.item()} | Hex: {packed_val.item():#010x} |"
      f" Binary: {packed_val.item():032b}"
  )

  # 5. Unpack back to signs directly
  unpacked_signs = unpack32(packed_val)
  print("\n5. Unpacked signs from uint32 (First 10):")
  print(unpacked_signs[:10])

  # Verify parity
  torch.testing.assert_close(signs, unpacked_signs)
  print(
      "\n✅ Parity Check: Unpacked signs match original binarized signs"
      " perfectly!"
  )
  return (packed_val, sample_weights, signs, simulated_weights, unpacked_signs)


if __name__ == "__main__":
  app.run()
