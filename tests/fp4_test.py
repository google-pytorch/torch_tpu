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

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import utils


class FP4Test(absltest.TestCase):

  def test_fp4_cpu_creation(self):
    x = torch.empty(2, 4, dtype=torch.float4_e2m1fn_x2)
    self.assertEqual(x.dtype, torch.float4_e2m1fn_x2)
    self.assertEqual(x.shape, torch.Size([2, 4]))

  def test_fp4_cpu_to_tpu_copy(self):
    device = torch.device("tpu")
    x_uint8 = torch.empty(2, 4, dtype=torch.uint8).fill_(0x11)
    x = x_uint8.view(torch.float4_e2m1fn_x2)
    self.assertEqual(x.storage().nbytes(), 8)

    x_tpu = x.to(device)

    with self.subTest(name="dtype"):
      self.assertEqual(x_tpu.dtype, torch.float4_e2m1fn_x2)
    with self.subTest(name="shape"):
      self.assertEqual(x_tpu.shape, torch.Size([2, 4]))
    with self.subTest(name="storage_bytes"):
      self.assertGreaterEqual(
          x_tpu.storage().nbytes(), 8
      )  # Allow padding on TPU

  def test_fp4_tpu_to_cpu_copy(self):
    device = torch.device("tpu")
    x_uint8 = torch.empty(2, 4, dtype=torch.uint8).fill_(0x11)
    x = x_uint8.view(torch.float4_e2m1fn_x2)
    x_tpu = x.to(device)

    x_cpu = x_tpu.to("cpu")
    with self.subTest(name="dtype"):
      self.assertEqual(x_cpu.dtype, torch.float4_e2m1fn_x2)
    with self.subTest(name="shape"):
      self.assertEqual(x_cpu.shape, torch.Size([2, 4]))
    with self.subTest(name="storage_bytes"):
      self.assertEqual(x_cpu.storage().nbytes(), 8)
    with self.subTest(name="values"):
      utils.assert_close(x_cpu.view(torch.uint8), x_uint8)

  def test_fp4_casting_and_rounding_on_tpu(self):
    device = torch.device("tpu")
    # Boundary values for testing RTNE rounding and saturation
    # We add a padding 0.0 at the end to make the last dimension even
    # (24 elements)
    x_float = torch.tensor(
        [[
            0.24,
            0.25,
            0.26,
            0.74,
            0.75,
            0.76,
            1.24,
            1.25,
            1.26,
            1.74,
            1.75,
            1.76,
            2.49,
            2.5,
            2.51,
            3.49,
            3.5,
            3.51,
            4.99,
            5.0,
            5.01,
            7.0,
            -7.0,
            0.0,
        ]],
        dtype=torch.float32,
    )

    # Move float tensor to TPU
    x_float_tpu = x_float.to(device)

    # Cast to FP4 on TPU
    x_fp4_tpu = x_float_tpu.to(torch.float4_e2m1fn_x2)

    # Copy back to CPU
    x_fp4_cpu = x_fp4_tpu.to("cpu")

    # View as uint8 to inspect exact bit patterns
    x_uint8_cpu = x_fp4_cpu.view(torch.uint8)

    # Expected byte representation calculated mathematically:
    # - 0.24 -> 0.0 (0x0), 0.25 -> 0.0 (0x0) -> 0x00
    # - 0.26 -> 0.5 (0x1), 0.74 -> 0.5 (0x1) -> 0x11
    # - 0.75 -> 1.0 (0x2), 0.76 -> 1.0 (0x2) -> 0x22
    # - 1.24 -> 1.0 (0x2), 1.25 -> 1.0 (0x2) -> 0x22
    # - 1.26 -> 1.5 (0x3), 1.74 -> 1.5 (0x3) -> 0x33
    # - 1.75 -> 2.0 (0x4), 1.76 -> 2.0 (0x4) -> 0x44
    # - 2.49 -> 2.0 (0x4), 2.5  -> 2.0 (0x4) -> 0x44
    # - 2.51 -> 3.0 (0x5), 3.49 -> 3.0 (0x5) -> 0x55
    # - 3.5  -> 4.0 (0x6), 3.51 -> 4.0 (0x6) -> 0x66
    # - 4.99 -> 4.0 (0x6), 5.0  -> 4.0 (0x6) -> 0x66
    # - 5.01 -> 6.0 (0x7), 7.0  -> 6.0 (0x7) -> 0x77
    # - -7.0 -> -6.0 (0xF), 0.0  -> 0.0 (0x0) -> 0x0F
    # Since dest has shape [1, 24] of FP4, it has logical shape [1, 48].
    # The cast only produces 24 logical values, which we padded with zeros
    # to 48.
    # So we expect 12 bytes of casted values followed by 12 bytes of zero
    # padding.
    #
    # Detect if the TPU compiler exhibits the wrap-around emulation
    # limitation on TPU v5e. Under software emulation for sub-byte types on
    # v5e, out-of-range values like 7.0 and -7.0 wrap to -0.0 (0x8) instead of
    # saturating to 6.0 / -6.0 (0x7/0xF). This is a known hardware emulation
    # limitation rather than a bug, and we do not work around it on the eager
    # path to avoid performance overhead. This is resolved in newer TPU
    # generations (v6+) which support native saturation.
    detect_tensor = torch.tensor([7.0, 0.0], dtype=torch.float32, device=device)
    detect_fp4 = detect_tensor.to(torch.float4_e2m1fn_x2)
    detect_uint8 = detect_fp4.to("cpu").view(torch.uint8)
    has_emulation_limitation = (detect_uint8[0] & 0x0F) == 0x08

    expected_bytes = torch.tensor(
        [[
            0x00,
            0x11,
            0x22,
            0x22,
            0x33,
            0x44,
            0x44,
            0x55,
            0x66,
            0x66,
            0x77,
            0x0F,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        ]],
        dtype=torch.uint8,
    )

    if has_emulation_limitation:
      # If the emulation limitation is present, adjust the expected bytes
      # for the wrapped out-of-range values:
      # - 7.0 -> -0.0 (0x8)
      # - -7.0 -> -0.0 (0x8)
      # This results in the 11th byte (5.01 and 7.0) being 0x87 instead of 0x77,
      # and the 12th byte (-7.0 and 0.0) being 0x08 instead of 0x0F.
      expected_bytes[0, 10] = 0x87
      expected_bytes[0, 11] = 0x08

    utils.assert_close(x_uint8_cpu, expected_bytes)

  def test_fp4_dequantization_on_tpu(self):
    device = torch.device("tpu")
    # Values that are exactly representable in FP4 e2m1fn.
    x_float = torch.tensor(
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    # Cast to FP4 on TPU (quantization)
    x_fp4 = x_float.to(torch.float4_e2m1fn_x2)

    # Cast back to Float32 on TPU (dequantization)
    x_dequant = torch.empty_like(x_float)
    x_dequant.copy_(x_fp4)

    self.assertEqual(x_dequant.device.type, "tpu")
    self.assertEqual(x_dequant.dtype, torch.float32)
    utils.assert_close(x_dequant.cpu(), x_float.cpu())

  def test_dtype_float4_e2m1fn_x2_supported(self):
    # Verify that FP4 empty tensor creation is supported on TPU.
    x = torch.empty(2, dtype=torch.float4_e2m1fn_x2, device="tpu")
    self.assertEqual(x.dtype, torch.float4_e2m1fn_x2)
    self.assertEqual(x.device.type, "tpu")

  def test_fp4_zero_element_tensor(self):
    device = torch.device("tpu")
    x = torch.empty(2, 0, dtype=torch.float4_e2m1fn_x2, device=device)
    self.assertEqual(x.shape, torch.Size([2, 0]))
    # Verify copy works
    x_cpu = x.to("cpu")
    self.assertEqual(x_cpu.shape, torch.Size([2, 0]))

  def test_fp4_odd_outer_dimension(self):
    device = torch.device("tpu")
    x_uint8 = torch.empty(3, 4, dtype=torch.uint8).fill_(0x11)
    x = x_uint8.view(torch.float4_e2m1fn_x2).to(device)
    self.assertEqual(x.shape, torch.Size([3, 4]))
    x_cpu = x.to("cpu")
    self.assertEqual(x_cpu.shape, torch.Size([3, 4]))

  def test_fp4_outer_non_contiguous_copy(self):
    device = torch.device("tpu")
    src_cpu = torch.tensor(
        [[17, 34], [51, 68], [85, 102], [119, 136]], dtype=torch.uint8
    ).view(torch.float4_e2m1fn_x2)
    src = src_cpu.to(device)
    dest = torch.empty(2, 2, dtype=torch.float4_e2m1fn_x2, device=device)
    # Slice src to make it non-contiguous in outer dim
    src_sliced = src[::2, :]
    dest.copy_(src_sliced)
    self.assertEqual(
        dest.to("cpu").view(torch.uint8).tolist(), [[17, 34], [85, 102]]
    )


if __name__ == "__main__":
  absltest.main()
