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

"""Unit tests for TorchTPU tensor factory operators."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from tests import seed_test_utils


class TensorFactoriesTest(seed_test_utils.RepeatableTest):
  """Tests for zeros, ones, full, *_like, and new_* factory operators on TPU."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _assert_tensor_equal(
      self, actual: torch.Tensor, expected: torch.Tensor
  ) -> None:
    self.assertEqual(actual.device.type, "tpu")
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertEqual(actual.shape, expected.shape)
    if expected.dtype == torch.float8_e8m0fnu:
      self.assertTrue(
          torch.equal(
              actual.cpu().view(torch.uint8), expected.view(torch.uint8)
          )
      )
    else:
      self.assertTrue(torch.equal(actual.cpu(), expected))

  def test_privateuse1_kernel_registered(self):
    """Verifies that all 9 tensor factory schemas have PrivateUse1 kernels."""
    op_names = (
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::zeros_like",
        "aten::ones_like",
        "aten::full_like",
        "aten::new_zeros",
        "aten::new_ones",
        "aten::new_full",
    )
    for op_name in op_names:
      with self.subTest(op_name=op_name):
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                op_name, "PrivateUse1"
            ),
            msg=f"{op_name} does not have a direct PrivateUse1 kernel",
        )

  @parameterized.parameters(
      (),
      (5,),
      (2, 3, 4),
      (0, 3),
      (2, 0, 4),
  )
  def test_shapes_and_dtypes(self, *shape):
    """Tests factory ops across shapes and dtypes against CPU reference."""
    dtypes = [
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
        torch.complex64,
    ]
    for dtype in dtypes:
      if dtype == torch.bool:
        fill_val = True
      elif dtype == torch.complex64:
        fill_val = 1.5 + 2.5j
      elif dtype.is_floating_point:
        fill_val = 3.5
      else:
        fill_val = 7

      with self.subTest(dtype=dtype):
        tpu_proto = torch.empty(shape, dtype=dtype, device=self.device)
        cpu_proto = torch.empty(shape, dtype=dtype, device="cpu")

        self._assert_tensor_equal(
            torch.zeros(shape, dtype=dtype, device=self.device),
            torch.zeros(shape, dtype=dtype, device="cpu"),
        )
        self._assert_tensor_equal(
            torch.ones(shape, dtype=dtype, device=self.device),
            torch.ones(shape, dtype=dtype, device="cpu"),
        )
        self._assert_tensor_equal(
            torch.full(shape, fill_val, dtype=dtype, device=self.device),
            torch.full(shape, fill_val, dtype=dtype, device="cpu"),
        )
        self._assert_tensor_equal(
            torch.zeros_like(tpu_proto),
            torch.zeros_like(cpu_proto),
        )
        self._assert_tensor_equal(
            torch.ones_like(tpu_proto),
            torch.ones_like(cpu_proto),
        )
        self._assert_tensor_equal(
            torch.full_like(tpu_proto, fill_val),
            torch.full_like(cpu_proto, fill_val),
        )
        self._assert_tensor_equal(
            tpu_proto.new_zeros(shape),
            cpu_proto.new_zeros(shape),
        )
        self._assert_tensor_equal(
            tpu_proto.new_ones(shape),
            cpu_proto.new_ones(shape),
        )
        self._assert_tensor_equal(
            tpu_proto.new_full(shape, fill_val),
            cpu_proto.new_full(shape, fill_val),
        )

    if hasattr(torch, "float8_e8m0fnu"):
      fp8_dtype = torch.float8_e8m0fnu
      with self.subTest(dtype=fp8_dtype):
        tpu_proto = torch.empty(shape, dtype=fp8_dtype, device=self.device)
        cpu_proto = torch.empty(shape, dtype=fp8_dtype, device="cpu")
        self._assert_tensor_equal(
            torch.zeros(shape, dtype=fp8_dtype, device=self.device),
            torch.zeros(shape, dtype=fp8_dtype, device="cpu"),
        )
        self._assert_tensor_equal(
            torch.ones(shape, dtype=fp8_dtype, device=self.device),
            torch.ones(shape, dtype=fp8_dtype, device="cpu"),
        )
        self._assert_tensor_equal(
            torch.zeros_like(tpu_proto),
            torch.zeros_like(cpu_proto),
        )
        self._assert_tensor_equal(
            torch.ones_like(tpu_proto),
            torch.ones_like(cpu_proto),
        )
        self._assert_tensor_equal(
            tpu_proto.new_zeros(shape),
            cpu_proto.new_zeros(shape),
        )
        self._assert_tensor_equal(
            tpu_proto.new_ones(shape),
            cpu_proto.new_ones(shape),
        )

  def test_sequential_shapes_no_cache_collision(self):
    """Tests sequential calls with different shapes to ensure no cache collision."""
    shapes = [(2, 3), (4, 5)]
    base_tpu = torch.empty((1,), dtype=torch.float32, device=self.device)
    base_cpu = torch.empty((1,), dtype=torch.float32, device="cpu")
    for shape in shapes:
      self._assert_tensor_equal(
          torch.zeros(shape, device=self.device),
          torch.zeros(shape, device="cpu"),
      )
      self._assert_tensor_equal(
          torch.ones(shape, device=self.device),
          torch.ones(shape, device="cpu"),
      )
      self._assert_tensor_equal(
          torch.full(shape, 4.25, device=self.device),
          torch.full(shape, 4.25, device="cpu"),
      )
      proto_tpu = torch.empty(shape, dtype=torch.float32, device=self.device)
      proto_cpu = torch.empty(shape, dtype=torch.float32, device="cpu")
      self._assert_tensor_equal(
          torch.zeros_like(proto_tpu),
          torch.zeros_like(proto_cpu),
      )
      self._assert_tensor_equal(
          torch.ones_like(proto_tpu),
          torch.ones_like(proto_cpu),
      )
      self._assert_tensor_equal(
          torch.full_like(proto_tpu, 4.25),
          torch.full_like(proto_cpu, 4.25),
      )
      self._assert_tensor_equal(
          base_tpu.new_zeros(shape),
          base_cpu.new_zeros(shape),
      )
      self._assert_tensor_equal(
          base_tpu.new_ones(shape),
          base_cpu.new_ones(shape),
      )
      self._assert_tensor_equal(
          base_tpu.new_full(shape, 4.25),
          base_cpu.new_full(shape, 4.25),
      )

  def test_default_dtype_inference(self):
    """Tests default dtype inference when dtype=None."""
    default_dtype = torch.get_default_dtype()
    z = torch.zeros((2, 3), device=self.device)
    o = torch.ones((2, 3), device=self.device)
    self.assertEqual(z.dtype, default_dtype)
    self.assertEqual(o.dtype, default_dtype)

    f_bool = torch.full((2, 3), True, device=self.device)
    self.assertEqual(f_bool.dtype, torch.bool)
    self._assert_tensor_equal(f_bool, torch.full((2, 3), True, device="cpu"))

    f_int = torch.full((2, 3), 42, device=self.device)
    self.assertEqual(f_int.dtype, torch.int64)
    self._assert_tensor_equal(f_int, torch.full((2, 3), 42, device="cpu"))

    f_float = torch.full((2, 3), 3.5, device=self.device)
    self.assertEqual(f_float.dtype, default_dtype)
    self._assert_tensor_equal(f_float, torch.full((2, 3), 3.5, device="cpu"))

    f_complex = torch.full((2, 3), 1.0 + 2.0j, device=self.device)
    self.assertEqual(f_complex.dtype, torch.complex64)
    self._assert_tensor_equal(
        f_complex, torch.full((2, 3), 1.0 + 2.0j, device="cpu")
    )

    proto_i16 = torch.empty((2, 3), dtype=torch.int16, device=self.device)
    self.assertEqual(torch.zeros_like(proto_i16).dtype, torch.int16)
    self.assertEqual(torch.ones_like(proto_i16).dtype, torch.int16)
    self.assertEqual(torch.full_like(proto_i16, 5).dtype, torch.int16)
    self.assertEqual(proto_i16.new_zeros((2, 3)).dtype, torch.int16)
    self.assertEqual(proto_i16.new_ones((2, 3)).dtype, torch.int16)
    self.assertEqual(proto_i16.new_full((2, 3), 5).dtype, torch.int16)

    self.assertEqual(
        torch.zeros_like(proto_i16, dtype=torch.bfloat16).dtype, torch.bfloat16
    )
    self.assertEqual(
        torch.ones_like(proto_i16, dtype=torch.bfloat16).dtype, torch.bfloat16
    )
    self.assertEqual(
        torch.full_like(proto_i16, 5.0, dtype=torch.bfloat16).dtype,
        torch.bfloat16,
    )
    self.assertEqual(
        proto_i16.new_zeros((2, 3), dtype=torch.bfloat16).dtype, torch.bfloat16
    )
    self.assertEqual(
        proto_i16.new_ones((2, 3), dtype=torch.bfloat16).dtype, torch.bfloat16
    )
    self.assertEqual(
        proto_i16.new_full((2, 3), 5.0, dtype=torch.bfloat16).dtype,
        torch.bfloat16,
    )

  def test_like_memory_format_and_strides(self):
    """Tests stride preservation and memory_format handling for *_like ops."""
    # 1. Non-overlapping and dense non-contiguous input with default preserve_format.
    x_tpu = torch.empty((3, 4), device=self.device).t()
    x_cpu = torch.empty((3, 4), device="cpu").t()
    for like_fn in (
        torch.zeros_like,
        torch.ones_like,
        lambda t, **kw: torch.full_like(t, 2.5, **kw),
    ):
      out_tpu = like_fn(x_tpu)
      out_cpu = like_fn(x_cpu)
      self.assertEqual(out_tpu.stride(), x_tpu.stride())
      self.assertEqual(out_tpu.stride(), out_cpu.stride())
      self._assert_tensor_equal(out_tpu, out_cpu)

    x4d_tpu = torch.empty(
        (2, 3, 4, 5), device=self.device, memory_format=torch.channels_last
    )
    x4d_cpu = torch.empty(
        (2, 3, 4, 5), device="cpu", memory_format=torch.channels_last
    )
    for like_fn in (
        torch.zeros_like,
        torch.ones_like,
        lambda t, **kw: torch.full_like(t, 2.5, **kw),
    ):
      out_tpu = like_fn(x4d_tpu)
      out_cpu = like_fn(x4d_cpu)
      self.assertEqual(out_tpu.stride(), x4d_tpu.stride())
      self.assertEqual(out_tpu.stride(), out_cpu.stride())
      self._assert_tensor_equal(out_tpu, out_cpu)

    # 2. Sliced view or expanded view with default memory_format=None produces
    # dense non-overlapping strides matching CPU.
    sliced_tpu = torch.empty((4, 6), device=self.device)[:, :3]
    sliced_cpu = torch.empty((4, 6), device="cpu")[:, :3]
    expanded_tpu = torch.empty((1, 4), device=self.device).expand(3, 4)
    expanded_cpu = torch.empty((1, 4), device="cpu").expand(3, 4)
    for t_tpu, t_cpu in (
        (sliced_tpu, sliced_cpu),
        (expanded_tpu, expanded_cpu),
    ):
      for like_fn in (
          torch.zeros_like,
          torch.ones_like,
          lambda t, **kw: torch.full_like(t, 2.5, **kw),
      ):
        out_tpu = like_fn(t_tpu)
        out_cpu = like_fn(t_cpu)
        self.assertEqual(out_tpu.stride(), out_cpu.stride())
        self._assert_tensor_equal(out_tpu, out_cpu)

    # 3. Explicit memory_format=torch.contiguous_format and torch.channels_last.
    for like_fn in (
        torch.zeros_like,
        torch.ones_like,
        lambda t, **kw: torch.full_like(t, 2.5, **kw),
    ):
      contig_tpu = like_fn(x4d_tpu, memory_format=torch.contiguous_format)
      contig_cpu = like_fn(x4d_cpu, memory_format=torch.contiguous_format)
      self.assertTrue(contig_tpu.is_contiguous())
      self.assertEqual(contig_tpu.stride(), contig_cpu.stride())
      self._assert_tensor_equal(contig_tpu, contig_cpu)

      cl_in_tpu = torch.empty((2, 3, 4, 5), device=self.device)
      cl_in_cpu = torch.empty((2, 3, 4, 5), device="cpu")
      cl_tpu = like_fn(cl_in_tpu, memory_format=torch.channels_last)
      cl_cpu = like_fn(cl_in_cpu, memory_format=torch.channels_last)
      self.assertEqual(cl_tpu.stride(), cl_cpu.stride())
      self._assert_tensor_equal(cl_tpu, cl_cpu)

  def test_cross_device_fallback_for_like_and_new(self):
    """Tests calling *_like and new_* on a TPU tensor with device='cpu'."""
    tpu_tensor = torch.empty((2, 3), dtype=torch.float32, device=self.device)

    z_like_cpu = torch.zeros_like(tpu_tensor, device="cpu")
    self.assertEqual(z_like_cpu.device.type, "cpu")
    self.assertTrue(torch.equal(z_like_cpu, torch.zeros((2, 3), device="cpu")))

    o_like_cpu = torch.ones_like(tpu_tensor, device="cpu")
    self.assertEqual(o_like_cpu.device.type, "cpu")
    self.assertTrue(torch.equal(o_like_cpu, torch.ones((2, 3), device="cpu")))

    f_like_cpu = torch.full_like(tpu_tensor, 7.0, device="cpu")
    self.assertEqual(f_like_cpu.device.type, "cpu")
    self.assertTrue(
        torch.equal(f_like_cpu, torch.full((2, 3), 7.0, device="cpu"))
    )

    new_z_cpu = tpu_tensor.new_zeros((2, 3), device="cpu")
    self.assertEqual(new_z_cpu.device.type, "cpu")
    self.assertTrue(torch.equal(new_z_cpu, torch.zeros((2, 3), device="cpu")))

    new_o_cpu = tpu_tensor.new_ones((2, 3), device="cpu")
    self.assertEqual(new_o_cpu.device.type, "cpu")
    self.assertTrue(torch.equal(new_o_cpu, torch.ones((2, 3), device="cpu")))

    new_f_cpu = tpu_tensor.new_full((2, 3), 7.0, device="cpu")
    self.assertEqual(new_f_cpu.device.type, "cpu")
    self.assertTrue(
        torch.equal(new_f_cpu, torch.full((2, 3), 7.0, device="cpu"))
    )


if __name__ == "__main__":
  absltest.main()
