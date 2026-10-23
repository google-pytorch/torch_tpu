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
from torch_tpu._internal.utils import test_utils as utils
from tests import quantize_utils
from tests import seed_test_utils


class QuantizeUtilsTest(seed_test_utils.RepeatableTest):

  def test_quantize_to_float4_e2m1fn_x2(self):
    # Boundary values for testing Round-to-Nearest-Even and saturation.
    # Add a padding 0.0 at the end to make the last dimension even.
    t = torch.tensor(
        [[
            0.24, 0.25, 0.26, 0.74, 0.75, 0.76, 1.24, 1.25, 1.26, 1.74, 1.75,
            1.76, 2.49, 2.5, 2.51, 3.49, 3.5, 3.51, 4.99, 5.0, 5.01, 7.0,
            -7.0, 0.0,
        ]],
        dtype=torch.float32,
    )  # pyformat: disable
    fp4 = quantize_utils.quantize_to_float4_e2m1fn_x2(t)
    self.assertEqual(fp4.dtype, torch.float4_e2m1fn_x2)
    self.assertEqual(fp4.shape, t.shape)

    # Explanation of expected byte representation:
    # FP4 (E2M1) 3-bit positive magnitude encoding:
    #   0x0: 0.0,  0x1: 0.5,  0x2: 1.0,  0x3: 1.5,
    #   0x4: 2.0,  0x5: 3.0,  0x6: 4.0,  0x7: 6.0
    # Bit 3 is the sign bit (e.g. 0xF is -6.0, 0x8 is -0.0).
    #
    # Two FP4 nibbles are packed into each uint8 byte in little-endian order:
    #   byte = (nibble_0 & 0x0F) | ((nibble_1 & 0x0F) << 4)
    #
    # Element-by-element quantization:
    # - t[0]=0.24 -> 0.0 (0x0), t[1]=0.25 -> 0.0 (0x0 via RTNE) -> Byte 0: 0x00
    # - t[2]=0.26 -> 0.5 (0x1), t[3]=0.74 -> 0.5 (0x1)          -> Byte 1: 0x11
    # - t[4]=0.75 -> 1.0 (0x2 via RTNE), t[5]=0.76 -> 1.0 (0x2) -> Byte 2: 0x22
    # - t[6]=1.24 -> 1.0 (0x2), t[7]=1.25 -> 1.0 (0x2 via RTNE) -> Byte 3: 0x22
    # - t[8]=1.26 -> 1.5 (0x3), t[9]=1.74 -> 1.5 (0x3)          -> Byte 4: 0x33
    # - t[10]=1.75 -> 2.0 (0x4), t[11]=1.76 -> 2.0 (0x4)        -> Byte 5: 0x44
    # - t[12]=2.49 -> 2.0 (0x4), t[13]=2.5 -> 2.0 (0x4)         -> Byte 6: 0x44
    # - t[14]=2.51 -> 3.0 (0x5), t[15]=3.49 -> 3.0 (0x5)        -> Byte 7: 0x55
    # - t[16]=3.5 -> 4.0 (0x6), t[17]=3.51 -> 4.0 (0x6)         -> Byte 8: 0x66
    # - t[18]=4.99 -> 4.0 (0x6), t[19]=5.0 -> 4.0 (0x6 via RTNE)-> Byte 9: 0x66
    # - t[20]=5.01 -> 6.0 (0x7), t[21]=7.0 -> 6.0 (0x7 sat)     -> Byte 10: 0x77
    # - t[22]=-7.0 -> -6.0 (0xF sat), t[23]=0.0 -> 0.0 (0x0)    -> Byte 11: 0x0F
    #
    # The [1, 24] tensor has 24 logical elements packed into 12 bytes, followed
    # by 12 zero-padded bytes to match container allocation size.
    expected_bytes = torch.tensor(
        [
            [0x00, 0x11, 0x22, 0x22, 0x33, 0x44, 0x44, 0x55, 0x66, 0x66, 0x77,
             0x0F] + [0x00] * 12
        ],
        dtype=torch.uint8,
    )  # pyformat: disable
    utils.assert_close(fp4.view(torch.uint8), expected_bytes)

  def test_quantize_to_float4_e2m1fn_x2_non_finite_and_signed_zero(self):
    t = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), -0.0],
        dtype=torch.float32,
    )
    fp4 = quantize_utils.quantize_to_float4_e2m1fn_x2(t)
    self.assertEqual(fp4.dtype, torch.float4_e2m1fn_x2)
    self.assertEqual(fp4.shape, t.shape)

    # - t[0]=NaN  -> +0.0 (0x0), t[1]=+inf -> +6.0 (0x7) -> Byte 0: 0x70
    # - t[2]=-inf -> -6.0 (0xF), t[3]=-0.0 -> -0.0 (0x8) -> Byte 1: 0x8F
    expected_bytes = torch.tensor([0x70, 0x8F, 0x00, 0x00], dtype=torch.uint8)
    utils.assert_close(fp4.view(torch.uint8), expected_bytes)

  def test_quantize_to_int4(self):
    t = torch.tensor([-100, -9, -8, -1, 0, 7, 8, 100], dtype=torch.int8)
    i4 = quantize_utils.quantize_to_int4(t)
    self.assertEqual(i4.dtype, torch.int4)
    self.assertEqual(i4.shape, t.shape)

    # Signed int4 clamps numerics to the range [-8, 7]. Elements outside the
    # range saturate to -8 and 7 respectively.
    expected = torch.tensor([-8, -8, -8, -1, 0, 7, 7, 7], dtype=torch.int8)
    utils.assert_close(i4.view(torch.int8), expected)

  def test_decode_sub_byte_tensor_fp4(self):
    data = torch.tensor([[0x10, 0x32]], dtype=torch.uint8)
    fp4 = data.view(torch.float4_e2m1fn_x2)
    decoded = quantize_utils.decode_sub_byte_tensor(fp4)
    self.assertEqual(decoded.dtype, torch.float32)
    self.assertEqual(decoded.shape, torch.Size([1, 4]))

    # Decoded float32 values match the 4 unpacked nibbles in order:
    # - Byte 0 (0x10): low nibble 0x0 (0.0), high nibble 0x1 (0.5)
    # - Byte 1 (0x32): low nibble 0x2 (1.0), high nibble 0x3 (1.5)
    expected = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.float32)
    utils.assert_close(decoded, expected)

  def test_decode_sub_byte_tensor_int4(self):
    data = torch.tensor([-8, 0, 7], dtype=torch.int8).view(torch.int4)
    decoded = quantize_utils.decode_sub_byte_tensor(data)
    self.assertEqual(decoded.dtype, torch.int8)
    utils.assert_close(decoded, torch.tensor([-8, 0, 7], dtype=torch.int8))


if __name__ == "__main__":
  absltest.main()
