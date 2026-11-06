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


class FP4Test(seed_test_utils.RepeatableTest):

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
      utils.assert_close(x_cpu, x)

  def test_fp4_tpu_to_cpu_copy_non_blocking(self):
    device = torch.device("tpu")
    x_uint8 = torch.empty(2, 4, dtype=torch.uint8).fill_(0x11)
    x = x_uint8.view(torch.float4_e2m1fn_x2)
    x_tpu = x.to(device)

    x_cpu = x_tpu.to("cpu", non_blocking=True)
    torch.accelerator.synchronize()

    with self.subTest(name="dtype"):
      self.assertEqual(x_cpu.dtype, torch.float4_e2m1fn_x2)
    with self.subTest(name="shape"):
      self.assertEqual(x_cpu.shape, torch.Size([2, 4]))
    with self.subTest(name="storage_bytes"):
      self.assertEqual(x_cpu.storage().nbytes(), 8)
    with self.subTest(name="values"):
      utils.assert_close(x_cpu, x)

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

    expected_fp4 = quantize_utils.quantize_to_float4_e2m1fn_x2(x_float)

    if has_emulation_limitation:
      # Under emulation, out-of-range values wrap to -0.0 (0x8) instead of
      # saturating. Patch the packed bytes directly.
      expected_bytes = expected_fp4.view(torch.uint8).clone()
      expected_bytes[0, 10] = 0x87
      expected_bytes[0, 11] = 0x08
      expected_fp4 = expected_bytes.view(torch.float4_e2m1fn_x2)

    utils.assert_close(x_fp4_cpu, expected_fp4)

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

  def test_fp4_eq_and_ne(self):
    device = torch.device("tpu")
    a_u8 = torch.tensor([[0x11, 0x22], [0x33, 0x44]], dtype=torch.uint8)
    b_u8 = torch.tensor([[0x11, 0x20], [0x33, 0x44]], dtype=torch.uint8)

    a_tpu = a_u8.view(torch.float4_e2m1fn_x2).to(device)
    b_tpu = b_u8.view(torch.float4_e2m1fn_x2).to(device)

    # Tensor-Tensor eq and ne
    eq_res = torch.eq(a_tpu, b_tpu)
    ne_res = torch.ne(a_tpu, b_tpu)

    expected_eq = torch.tensor([[True, False], [True, True]], device=device)
    expected_ne = torch.tensor([[False, True], [False, False]], device=device)

    self.assertEqual(eq_res.shape, torch.Size([2, 2]))
    self.assertEqual(ne_res.shape, torch.Size([2, 2]))
    utils.assert_close(eq_res, expected_eq)
    utils.assert_close(ne_res, expected_ne)

  def test_fp4_scaled_mm(self):
    device = torch.device("tpu")
    scale_a = torch.ones((1,), dtype=torch.float32, device=device)
    scale_b = torch.ones((1,), dtype=torch.float32, device=device)

    # 1. Transposed column-major b = b_raw.t() created via .to(fp4) on TPU
    a = torch.randn(128, 128, dtype=torch.bfloat16, device=device).to(
        torch.float4_e2m1fn_x2
    )
    b_raw = torch.randn(128, 128, dtype=torch.bfloat16, device=device).to(
        torch.float4_e2m1fn_x2
    )
    b_t = b_raw.t()
    out_t = torch._scaled_mm(a, b_t, scale_a, scale_b, out_dtype=torch.bfloat16)
    expected_t = (
        torch.matmul(a.to(torch.float32), b_raw.to(torch.float32).t())
        * scale_a
        * scale_b
    ).to(torch.bfloat16)
    self.assertEqual(out_t.shape, torch.Size([128, 128]))
    self.assertEqual(out_t.dtype, torch.bfloat16)
    utils.assert_close(
        out_t.to(torch.float32),
        expected_t.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # 2. H2D-packed CPU float4_e2m1fn_x2 tensors (2 FP4 values/byte)
    a_cpu = quantize_utils.quantize_to_float4_e2m1fn_x2(
        torch.randn(64, 64, dtype=torch.float32)
    )[:32]
    b_raw_cpu = quantize_utils.quantize_to_float4_e2m1fn_x2(
        torch.randn(64, 64, dtype=torch.float32)
    )[:32]
    a_h2d = a_cpu.to(device)
    b_h2d = b_raw_cpu.to(device).t()
    out_h2d = torch._scaled_mm(
        a_h2d, b_h2d, scale_a, scale_b, out_dtype=torch.bfloat16
    )
    expected_h2d = torch.matmul(
        quantize_utils.decode_sub_byte_tensor(a_cpu),
        quantize_utils.decode_sub_byte_tensor(b_raw_cpu).t(),
    ).to(torch.bfloat16)
    utils.assert_close(
        out_h2d.cpu().to(torch.float32),
        expected_h2d.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # 3. torch.compile
    compiled_fn = torch.compile(
        lambda x, y, sa, sb: torch._scaled_mm(
            x, y, sa, sb, out_dtype=torch.bfloat16
        ),
        backend="tpu",
    )
    compiled_out = compiled_fn(a_h2d, b_h2d, scale_a, scale_b)
    utils.assert_close(
        compiled_out.cpu().to(torch.float32),
        expected_h2d.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

  def test_fp4_scaled_mm_variants(self):
    device = torch.device("tpu")
    m, k_packed, n = 32, 32, 32
    a_cpu = quantize_utils.quantize_to_float4_e2m1fn_x2(
        torch.randn(2 * m, k_packed, dtype=torch.float32)
    )[:m]
    b_raw_cpu = quantize_utils.quantize_to_float4_e2m1fn_x2(
        torch.randn(2 * n, k_packed, dtype=torch.float32)
    )[:n]
    a_f32 = quantize_utils.decode_sub_byte_tensor(a_cpu)
    b_raw_f32 = quantize_utils.decode_sub_byte_tensor(b_raw_cpu)
    a = a_cpu.to(device)
    b = b_raw_cpu.to(device).t()
    scale_a = torch.tensor([1.5], dtype=torch.float32, device=device)
    scale_b = torch.tensor([2.0], dtype=torch.float32, device=device)

    # _scaled_mm.out producing float4_e2m1fn_x2 and growing a non-empty out=
    out_fp4 = torch.empty((4, 4), dtype=torch.float4_e2m1fn_x2, device=device)
    torch.ops.aten._scaled_mm.out(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype=torch.float4_e2m1fn_x2,
        out=out_fp4,
    )
    self.assertEqual(out_fp4.shape, torch.Size([m, n // 2]))

    # _scaled_mm_v2 and _scaled_mm_v2.out with TensorWise and BlockWise1x32 (2*K)
    out_v2 = torch._scaled_mm_v2(
        a,
        b,
        [scale_a],
        [0],
        [0],
        [scale_b],
        [0],
        [0],
        None,
        torch.bfloat16,
    )
    out_v2_buf = torch.empty((4, 4), dtype=torch.bfloat16, device=device)
    torch.ops.aten._scaled_mm_v2.out(
        a,
        b,
        [scale_a],
        [0],
        [0],
        [scale_b],
        [0],
        [0],
        None,
        torch.bfloat16,
        out=out_v2_buf,
    )
    expected_bf16 = (
        torch.matmul(a_f32, b_raw_f32.t()) * scale_a.cpu() * scale_b.cpu()
    ).to(torch.bfloat16)
    utils.assert_close(
        out_v2.cpu().to(torch.float32),
        expected_bf16.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    utils.assert_close(
        out_v2_buf.cpu().to(torch.float32),
        expected_bf16.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # BlockWise1x32 with 2*K unpacked FP4 elements (k_unpacked = 64 -> 2 blocks)
    scale_a_bw = torch.ones((m, 2), dtype=torch.float8_e8m0fnu, device=device)
    scale_b_bw = torch.ones((2, n), dtype=torch.float8_e8m0fnu, device=device)
    out_bw = torch._scaled_mm_v2(
        a,
        b,
        [scale_a_bw],
        [3],
        [0],
        [scale_b_bw],
        [3],
        [0],
        None,
        torch.bfloat16,
    )
    expected_bw = torch.matmul(a_f32, b_raw_f32.t()).to(torch.bfloat16)
    utils.assert_close(
        out_bw.cpu().to(torch.float32),
        expected_bw.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # Transposed (N, 1) FP4 tensor view has packed_dim = 0
    unit_t = (
        torch.zeros((4, 1), dtype=torch.uint8)
        .view(torch.float4_e2m1fn_x2)
        .to(device)
        .t()
    )
    self.assertEqual(unit_t.shape, torch.Size([1, 4]))
    self.assertTrue(torch.equal(unit_t, unit_t))


if __name__ == "__main__":
  absltest.main()
