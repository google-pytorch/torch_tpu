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

"""Tests error handling on TPU only."""

import re
from typing import Any
import unittest
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import dynamism
from torch_tpu._internal import execution_mode
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.pallas import tpu_torch_pallas
from tests import error_testing as et


def setUpModule():
  """Called by absltest after flags are parsed and before tests are run."""
  et.set_up_module()


def _get_aminmax_outputs(
    op: Any, device: str, dtype: torch.dtype
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Creates the output tensors for `op` of `dtype`, on `device`.

  Args:
    op: a torch function. Should be one of: amax, amin, or aminmax.
    device: the device of the output tensors
    dtype: the dtype of the output tensors

  Returns:
    The out object to be used as a kwarg when calling `op`.
  """

  assert op in (torch.amax, torch.amin, torch.aminmax)

  def scalar_tensor() -> torch.Tensor:
    return torch.tensor(0, device=device, dtype=dtype)

  return (
      scalar_tensor()
      if op != torch.aminmax
      else (scalar_tensor(), scalar_tensor())
  )


def _make_lu_unpack_outputs(
    p: tuple[int, ...], l: tuple[int, ...], u: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
  """Creates a 3-tuple of tensors for `lu_unpack()` op."""
  return (
      torch.empty(p, device=et.device()),
      torch.empty(l, device=et.device()),
      torch.empty(u, device=et.device()),
  )


class TpuOnlyErrorTest(et.TpuOnlyErrorTestBase, parameterized.TestCase):
  """Tests error messages on TPU."""

  def test_ones_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        # PyTorch implements ones() in terms of empty() automatically, so we
        # don't have a good way to know that the op in question is "ones"
        # instead of "empty".
        tpu="""empty(): TorchTPU does not yet support dtype int1""",
        message_reviewed_by="wan",
    ):
      # This succeeds on CPU but is unimplemented on TPU.
      torch.ones(2, 3, device=et.device(), dtype=torch.int1)

  def test_prod_with_op_dispatch_failure(self):
    """Tests that prod() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("prod", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prod(): my error""",
    ):
      torch.prod(t1).to("cpu")

  def test_prod_out_with_op_dispatch_failure(self):
    """Tests that prod() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("prod", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prod(): my error""",
    ):
      torch.prod(t1, dim=1, out=torch.zeros_like(t1)).to("cpu")

  def test_nonzero_with_op_dispatch_failure(self):
    """Tests that nonzero() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""nonzero(): my error""",
    ):
      out = torch.nonzero(t1)
      out.to("cpu")

  def test_nonzero_out_with_op_dispatch_failure(self):
    """Tests that nonzero() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""nonzero(): my error""",
    ):
      out = torch.zeros(1, 3, device="tpu", dtype=torch.long)
      torch.nonzero(t1, out=out)
      out.to("cpu")

  def test_nonzero_size_with_op_dispatch_failure(self):
    """Tests that nonzero_size() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero_size", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""nonzero(): my error""",
    ):
      out = torch.nonzero(t1)
      out.to("cpu")

  def test_nonzero_out_size_with_op_dispatch_failure(self):
    """Tests that nonzero_size() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("nonzero_size", "my error")
    t1 = torch.ones(2, 3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""nonzero(): my error""",
    ):
      out = torch.zeros(1, 3, device="tpu", dtype=torch.long)
      torch.nonzero(t1, out=out)
      out.to("cpu")

  def test_topk_with_op_dispatch_failure(self):
    """Tests that topk() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("topk", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""topk(): my error""",
    ):
      values, indices = torch.topk(t1, 2)
      values.to("cpu")
      indices.to("cpu")

  # Why do we run this test only on TPU (and not on CPU)?
  # TorchTPU does not support `sorted=False` for `_unique2()`.
  # TODO: add support for `_unique2()` with `sorted=False`
  def test_unique2_unsupported_sorted(self):
    """Tests unique2 fails if sorted=False."""
    t = torch.ones(2, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""unique2(): sorted=False is not yet supported""",
    ):
      torch.ops.aten._unique2(
          t, sorted=False, return_inverse=True, return_counts=True
      ).to("cpu")

  def test_index_put_with_op_dispatch_failure(self):
    """Tests that index_put() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("index_put_", "my error")
    t = torch.tensor([0, 1], device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_put_(): my error""",
    ):
      torch.index_put_(
          t,
          (torch.tensor([0], device="tpu"),),
          torch.tensor([0], device="tpu"),
      )
      t.to("cpu")

  def test_index_put_bool_mask_with_op_dispatch_failure(self):
    """Tests that index_put() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("index_put_", "my error")
    t = torch.tensor([0, 1], device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_put_(): my error""",
    ):
      torch.index_put_(
          t,
          (torch.tensor([True, False], device="tpu"),),
          torch.tensor(1, device="tpu"),
      )
      t.to("cpu")

  def test_index_put_with_assign_buffer_to_at_tensor_failure(self):
    """Tests that index_put() bubbles up the error from AssignBufferToAtTensor.

    The test uses overlapping views to trigger the error. The tensor 't' below
    has overlapping views. This test passes on CPU but is undefined behavior.
    On TPU, an error is raised.
    """

    t = torch.arange(5.0, device=et.device()).as_strided((3, 3), (1, 1))
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_put_(): inplace writes to overlapping views are undefined behavior and are not supported.
Because multiple logical tensor indices point to the same buffer elements, writes from multiple indices may overwrite each other.
Please use clone() or contiguous() to copy the tensor before writing""",
    ):
      torch.index_put_(
          t,
          (
              torch.tensor([0], device=et.device(), dtype=torch.long),
              torch.tensor([0], device=et.device(), dtype=torch.long),
          ),
          torch.tensor(0.0, device=et.device()),
      )

  def test_index_copy_with_assign_buffer_to_at_tensor_failure(self):
    """Tests that index_copy() bubbles up the error from AssignBufferToAtTensor.

    The test uses overlapping views to trigger the error. The tensor 't' below
    has overlapping views. This test passes on CPU but is undefined behavior.
    On TPU, an error is raised.
    """

    t = torch.arange(5.0, device=et.device()).as_strided((3, 3), (1, 1))
    self_tensor = torch.zeros(3, 3, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.long)
    source = torch.ones(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_copy(): inplace writes to overlapping views are undefined behavior and are not supported.
Because multiple logical tensor indices point to the same buffer elements, writes from multiple indices may overwrite each other.
Please use clone() or contiguous() to copy the tensor before writing""",
    ):
      torch.index_copy(self_tensor, 0, index, source, out=t)

  def test_mm_with_oom_result(self):
    """Tests that mm with a large result that OOMs fails with expected error."""

    # Each input tensor is 4 MB.
    t1 = torch.ones(2**20, 1, device=et.device(), dtype=torch.float32)
    t2 = torch.ones(1, 2**20, device=et.device(), dtype=torch.float32)
    # The result tensor would need 4 TB, which is impossible to allocate.
    t3 = torch.mm(t1, t2)
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r"""to_copy\(\): the TPU ran out of memory while awaiting the materialization of value float32\[1048576, 1048576\]:(.|\n)*"""
        ),
    ):
      t3.to("cpu")

  def test_add_tpu_and_cpu(self):
    tpu_tensor = torch.tensor([1, 2, 3], device="tpu")
    cpu_tensor = torch.tensor([4, 5, 6], device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Expected all tensors to be on the same device, but found at least two devices, tpu:0 and cpu!""",
    ):
      tpu_tensor + cpu_tensor  # pylint: disable=pointless-statement

  def test_add_cpu_and_tpu(self):
    cpu_tensor = torch.tensor([4, 5, 6], device="cpu")
    tpu_tensor = torch.tensor([1, 2, 3], device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Expected all tensors to be on the same device, but found at least two devices, tpu:0 and cpu!""",
    ):
      cpu_tensor + tpu_tensor  # pylint: disable=pointless-statement

  def test_add_out_cpu(self):
    tpu_tensor1 = torch.tensor([1, 2, 3], device="tpu")
    tpu_tensor2 = torch.tensor([4, 5, 6], device="tpu")
    cpu_tensor = torch.tensor([7, 8, 9], device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): the out tensor is expected to be on tpu, got cpu""",
    ):
      torch.add(tpu_tensor1, tpu_tensor2, out=cpu_tensor)

  def test_dtype_complex32_unsupported(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""empty(): TorchTPU does not yet support dtype complex32""",
        message_reviewed_by="wan",
    ):
      torch.empty(2, dtype=torch.complex32, device="tpu")

  def test_dtype_float4_e2m1fn_x2_unsupported(self):
    # The float4_e2m1fn_x2 dtype represents 2x f4e2m1fn values packed into 8bits
    # which is different from XLA's supported single-value f4e2m1fn dtype.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""empty(): TorchTPU does not yet support dtype float4_e2m1fn_x2""",
        message_reviewed_by="wan",
    ):
      torch.empty(2, dtype=torch.float4_e2m1fn_x2, device="tpu")

  def test_empty_strided(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""empty_strided(): TorchTPU does not yet support dtype complex32""",
        message_reviewed_by="wan",
    ):
      torch.empty_strided((2,), (1,), dtype=torch.complex32, device="tpu")

  def test_empty_unsupported_layout(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""empty(): only layout=torch.strided is supported by TorchTPU for now, got torch.jagged""",
    ):
      torch.empty(2, layout=torch.jagged, device="tpu")

  def test_histc_bounds_unsupported_dtype(self):
    """Tests that torch.histc() fails when the bounds have an unsupported dtype."""
    t = torch.tensor([0, 0], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""histc(): expected min and max to be float or int type, got Bool and Bool""",
    ):
      torch.histc(t, min=False, max=True)

  def test_is_nonzero_with_more_than_one_value(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Boolean value of Tensor with more than one value is ambiguous""",
    ):
      torch.is_nonzero(torch.tensor([1, 3, 5], device="tpu"))

  def test_is_nonzero_with_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This is the error message we get from pytorch cpu.
        tpu="""Boolean value of Tensor with no values is ambiguous""",
    ):
      torch.is_nonzero(torch.tensor([], device="tpu"))

  def test_is_nonzero_with_nested_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This is the error message we get from pytorch cpu.
        tpu="""Boolean value of Tensor with no values is ambiguous""",
    ):
      torch.is_nonzero(torch.tensor([[]], device="tpu"))

  def test_masked_select_out_on_different_device(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device="tpu", dtype=torch.float32)
    mask = torch.ones(5, device="tpu", dtype=torch.bool)
    out = torch.ones(5, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): the out tensor is expected to be on tpu, got cpu""",
    ):
      torch.masked_select(t, mask, out=out)

  def test_masked_select_out_on_different_device2(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device="cpu", dtype=torch.float32)
    mask = torch.ones(5, device="cpu", dtype=torch.bool)
    out = torch.ones(5, device="tpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): tensor is expected to be on tpu, got cpu""",
    ):
      torch.masked_select(t, mask, out=out)

  def test_set_invalid_metadata(self):
    t = torch.zeros(1, device="tpu", dtype=torch.float32)
    source = torch.arange(8, device="tpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""set_(): expected the number of bytes required by the given arguments to be <= 32 (actual storage size), got 64""",
        message_reviewed_by="wan",
    ):
      t.set_(source.untyped_storage(), storage_offset=0, size=[16], stride=[1])

  def test_resize_materialization_error(self):
    """Materializing a tensor that has been resized down throws an error."""
    # This is not a supported operation on CPU either, but if and what message
    # gets thrown depends on how the elements are accessed.
    tensor = torch.arange(8, dtype=torch.float32, device="tpu")
    nbytes = tensor.untyped_storage().nbytes()
    tensor.untyped_storage().resize_(nbytes // 2)
    with et.assert_raises_message(
        IndexError,
        tpu="""to_copy(): cannot read 32 bytes (8 elements of type float32 with an offset of 0 elements) from a storage buffer with 16 bytes""",
    ):
      tensor.to("cpu")

  # Confirmed this test does NOT fail on cpu.
  def test_addmm_on_complex_input(self):
    complex_val = torch.complex(torch.tensor(1.0), torch.tensor(1.0))
    complex_val = complex_val.tile((2, 2))
    complex_val = complex_val.to(et.device())

    input_ = complex_val.clone()
    mat1 = complex_val.clone()
    mat2 = complex_val.clone()
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""addmm(): complex dtypes are not yet supported""",
    ):
      torch.addmm(input_, mat1, mat2)

  def test_cumsum_with_unsupported_boolean_dtype(self):
    """Tests that cumsum with a boolean out tensor throws the correct NotImplementedError."""
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""cumsum(): invalid output dtype bool""",
    ):
      t = torch.ones(2, 2, device="tpu")
      res = torch.cumsum(t, dim=1, dtype=torch.bool)
      res.to("cpu")

    x_cpu = torch.tensor([True, False], dtype=torch.bool)
    x_tpu = x_cpu.to(device="tpu")

    # Test out-of-place with out=
    with et.assert_raises_message(
        NotImplementedError,
        tpu=""""cumsum_out_cpu" not implemented for 'Bool'""",
    ):
      torch.cumsum(x_cpu, dim=0, out=x_cpu)
    with et.assert_raises_message(
        NotImplementedError, tpu="""cumsum(): invalid output dtype bool"""
    ):
      torch.cumsum(x_tpu, dim=0, out=x_tpu)

    # Test in-place
    with et.assert_raises_message(
        NotImplementedError,
        tpu=""""cumsum_out_cpu" not implemented for 'Bool'""",
    ):
      x_cpu.cumsum_(dim=0)
    with et.assert_raises_message(
        NotImplementedError, tpu="""cumsum(): invalid output dtype bool"""
    ):
      x_tpu.cumsum_(dim=0)

  def test_cummax_dimension_size_limit(self):
    """Tests cummax fails if dimension size has > 2^31-1 elements."""
    # Create an empty tensor with shape 2**31
    t = torch.empty(2**31, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""cummax_helper(): expected dimension size to be less than or equal to 2147483647, got 2147483648""",
    ):
      y, _ = torch.cummax(t, dim=0)
      y.cpu()

  def test_cummin_dimension_size_limit(self):
    """Tests cummin fails if dimension size has > 2^31-1 elements."""
    # Create an empty tensor with shape 2**31
    t = torch.empty(2**31, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""cummin_helper(): expected dimension size to be less than or equal to 2147483647, got 2147483648""",
    ):
      y, _ = torch.cummin(t, dim=0)
      y.cpu()

  def test_cummin_with_unsupported_complex_dtype(self):
    """Tests cummin fails with unsupported complex dtypes."""
    with et.assert_raises_message(
        RuntimeError,
        tpu="""cummin_helper(): expected supported element type, got ComplexFloat""",
    ):
      t = torch.ones(2, 2, device="tpu", dtype=torch.complex64)
      torch.cummin(t, dim=1)

  def test_index_add_with_assign_buffer_to_at_tensor_failure(self):
    """Tests that index_add() bubbles up the error from AssignBufferToAtTensor.

    The test uses overlapping views to trigger the error. The tensor 't' below
    has overlapping views. This test passes on CPU but is undefined behavior.
    On TPU, an error is raised.
    """

    t = torch.arange(5.0, device=et.device()).as_strided((3, 3), (1, 1))
    self_tensor = torch.zeros(3, 3, device=et.device())
    index = torch.tensor([0], device=et.device(), dtype=torch.long)
    source = torch.ones(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index_add(): inplace writes to overlapping views are undefined behavior and are not supported.
Because multiple logical tensor indices point to the same buffer elements, writes from multiple indices may overwrite each other.
Please use clone() or contiguous() to copy the tensor before writing""",
    ):
      torch.index_add(self_tensor, 0, index, source, out=t)

  # TODO: b/480225714 remove this after the corresponding PyTorch#173995 bug
  # is fixed.
  @unittest.skip("PyTorch upstream bug #173995 crashes this test.")
  def test_unsafe_masked_index_error(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""index(): at least one index tensor must be defined""",
    ):
      torch.ops.aten._unsafe_masked_index(
          torch.tensor([1.0], device=et.device()),
          torch.tensor([True], device=et.device()),
          [],  # indices
          0.0,  # fill
      )

  def test_max_pool2d_with_indices_input_too_large(self):
    """Tests max_pool2d_with_indices fails if input has > 2^31-1 elements."""
    # h * w = 2^15 * 2^16 = 2^31, exceeding the ui32 index limit 2^31 - 1
    t = torch.empty(1, 1, 2**15, 2**16, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r""".*tpu doesn't support max_pool2d_with_indices on inputs with more than 2147483647 spatial elements due to int32 indices limitation for now, got 2147483648.*""",
            re.DOTALL,
        ),
    ):
      y, _ = torch.nn.functional.max_pool2d(
          t,
          kernel_size=1,
          stride=1,
          padding=0,
          dilation=1,
          return_indices=True,
      )
      y.cpu()

  # TODO: remove this test once we support complex alpha on TPU.
  def test_add_complex_alpha(self):
    """Tests add with an alpha of complex dtype."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.complex64)
    s = torch.ones(4, device=device, dtype=torch.complex64)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add(): complex128 alpha value is not yet supported""",
    ):
      torch.add(t, s, alpha=1j)

  # TODO: remove this test once we support complex alpha on TPU.
  def test__add_relu_Scalar_complex_alpha(self):
    """Tests _add_relu.Scalar with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu.Scalar(t, 2.0, 1j)

  def test__add_relu_Tensor_complex_alpha(self):
    """Tests _add_relu.Tensor with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    s = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu.Tensor(t, s, alpha=1j)

  def test__add_relu_out_complex_alpha(self):
    """Tests _add_relu.out with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    s = torch.ones(4, device=device, dtype=torch.float32)
    out = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu.out(t, s, alpha=1j, out=out)

  def test__add_relu__Scalar_complex_alpha(self):
    """Tests _add_relu_.Scalar with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu_(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu_.Scalar(t, 2.0, alpha=1j)

  def test__add_relu__Tensor_complex_alpha(self):
    """Tests _add_relu_.Tensor with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    s = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu_(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu_.Tensor(t, s, alpha=1j)

  def test_threshold_unsupported_bool_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold(): threshold is not implemented for bool type""",
    ):
      torch.threshold(torch.tensor([True, False], device=et.device()), 0.5, 0.0)

  def test_threshold_unsupported_complex_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold(): threshold is not implemented for complex types""",
    ):
      torch.threshold(
          torch.tensor([1 + 1j, 2 + 2j], device=et.device()), 0.5, 0.0
      )

  # Why do we run this test only on TPU (and not on CPU)?
  # There are no other available devices on CPU runs, other than CPU.
  @parameterized.named_parameters(
      {"testcase_name": "amin", "op_name": "amin", "op": torch.amin},
      {"testcase_name": "amax", "op_name": "amax", "op": torch.amax},
      {"testcase_name": "aminmax", "op_name": "aminmax", "op": torch.aminmax},
  )
  def test_aminmax_invalid_output_device(self, op_name: str, op: Any):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = _get_aminmax_outputs(op, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected output tensor to be on tpu, got cpu""",
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # This test should be run only on TPU because there are no other available
  # devices on CPU runs, other than CPU.
  @parameterized.named_parameters(
      {"testcase_name": "argmin", "op_name": "argmin", "op": torch.argmin},
      {"testcase_name": "argmax", "op_name": "argmax", "op": torch.argmax},
  )
  def test_argmin_argmax_invalid_output_device(self, op_name: str, op: Any):
    tensor = torch.ones(5, 2, device=et.device(), dtype=torch.float32)
    out = torch.empty(5, 1, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected output tensor to be on tpu, got cpu""",
        message_reviewed_by="wan",
    ):
      op(tensor, dim=0, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch core implementations don't check `pivots` rank.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_unpack_pivots_invalid_rank(self):
    data = torch.ones(2, 4, 4, device=et.device())

    # TODO: b/485613841 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): lu_pivots must have at least 1 dimension, got 0""",
    ):
      pivots = torch.tensor(1, device=et.device(), dtype=torch.int32)
      torch.lu_unpack(data, pivots)

    # TODO: b/483972819 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): expected the first output tensor to be a 3D tensor (pivots dimension + 1), got 1D of shape [4]""",
    ):
      pivots = torch.ones(2, 2, device=et.device(), dtype=torch.int32)
      out = _make_lu_unpack_outputs(p=(4,), l=(2, 4, 4), u=(2, 4, 4))
      torch.lu_unpack(data, pivots, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch core implementations don't check `pivots` dimensions.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_unpack_pivots_invalid_dimension(self):
    data = torch.ones(2, 4, 4, device=et.device())

    # TODO: b/485613841 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): pivots size must be less than or equal to the size of the matrix, got 5 and 4""",
    ):
      pivots = torch.ones(2, 5, device=et.device(), dtype=torch.int32)
      torch.lu_unpack(data, pivots)

    # TODO: b/485613841 remove this test when the divergence is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): pivots and tensor must have the same batch dimensions, got [2] and [3]""",
    ):
      pivots = torch.ones(3, 4, device=et.device(), dtype=torch.int32)
      torch.lu_unpack(data, pivots)

  # Why do we run this test only on TPU (and not on CPU)?
  # CPU kernel raises an `IndexError`, instead of a `RuntimeError`, because it
  # tries to get `pivots.size(-1)` of a 0-dim tensor.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_solve_pivots_rank_too_low(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.tensor(0, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, 4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, device=et.device())

    # TODO: b/485628812 also test CPU when the TPU kernel is fixed, raising an
    # `IndexError`, instead of an `RuntimeError`.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): pivots must have at least 1 dimension, got 0""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # CPU kernel runs successfully, broadcasting the inputs.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_lu_solve_rank_mismatch(self):
    lu = torch.ones(4, 4, device=et.device())
    pivots = torch.ones(4, device=et.device(), dtype=torch.int32)
    b = torch.ones(4, 4, 4, device=et.device())

    # Call the out overload of linalg.lu_solve() op.
    out = torch.empty(4, 4, 4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_lu_solve(): the rank of b must be equal to the rank of lu, got rank(b) = 3 and rank(lu) = 2""",
    ):
      torch.linalg.lu_solve(lu, pivots, b, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # Currently, generator is not yet supportted in TorchTPU.
  def test_multinomial_generator(self):
    inp = torch.randn(2, device=et.device())
    gen = torch.Generator()

    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): generator is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.multinomial(inp, num_samples=1, generator=gen)

  def test_index_multiple_bool_indices(self):
    inp = torch.ones(2, 2, device=et.device())
    indices = [
        torch.tensor([True, False], device=et.device()),
        torch.tensor([False, True], device=et.device()),
    ]

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""index(): indexing with more than one bool tensor is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.index(inp, indices)

  # Why do we run this test only on TPU (and not on CPU)?
  # TorchTPU behaves differently from CPU/GPU kernels. Instead of resizing the
  # given output, TorchTPU raises an error.
  #
  # TODO: b/487653209 remove this test when the divergence is resolved.
  def test_linalg_inv_ex_output_rank_mismatch(self):
    a = torch.ones(4, 4, device=et.device())

    # Call the out overload of linalg.inv_ex() op.
    out = (
        torch.ones(4, 4, 4, device=et.device()),
        torch.ones(4, 4, 4, device=et.device()),
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""linalg_inv_ex(): expected the inverse output shape to match the input tensor of shape [4, 4], got [4, 4, 4]""",
        message_reviewed_by="wan",
    ):
      torch.linalg.inv_ex(a, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # TorchTPU doesn't support setting the `sparse_grad` parameter.
  def test_gather_with_sparse_grad(self):
    inp = torch.ones(2, 3, 4, device=et.device())
    dim = 0
    index = torch.ones(2, 3, 4, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""gather(): sparse_grad is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.gather(inp, dim, index, sparse_grad=True)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch runs successfully.
  def test_lerp_complex_double(self):
    t = torch.tensor([1.0], dtype=torch.complex128, device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""lerp(): complex128 dtype is not yet supported""",
    ):
      torch.lerp(t, t, t)

  # Why do we run this test only on TPU (and not on CPU)?
  # The machine that runs the CPU test has no other device other than the CPU
  # itself for testing this.
  def test_copy_from_other_device_inputs(self):
    dk = torch._C._parse_dispatch_key("PrivateUse1")
    t_src = torch.ones(5, device="cpu")
    t_tgt = torch.zeros(5, device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""copy_from(): expected at least one of the inputs to be on 'tpu' device, got 'cpu' (source) and 'cpu' (destination)""",
        message_reviewed_by="wan",
    ):
      # Dispatch to `_copy_from()` TPU kernel with CPU inputs.
      # Otherwise, can't reach the error.
      torch.ops.aten._copy_from.default.redispatch(
          torch._C.DispatchKeySet(dk), t_src, t_tgt
      )

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch runs successfully.
  # BUG: TorchTPU should have similar behavior w.r.t. PyTorch native devices.
  def test_local_scalar_dense_too_many_elements(self):
    inp = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""local_scalar_dense(): expected the input tensor to have 1 element, got 2""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._local_scalar_dense(inp)

  # Why do we run this test only on TPU (and not on CPU)?
  # CPU kernel runs successfully.
  # BUG: TPU kernels should mimic native devices behavior, including bugs.
  def test_cumprod_bool_dtype(self):
    inp = torch.ones(2, 2, device=et.device(), dtype=torch.bool)

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""cumprod(): materialization failed with: the dtype argument cannot be bool""",
    ):
      # cpu() is needed because the error is triggered inside the op builder.
      torch.cumprod(inp, dim=0, dtype=torch.bool).cpu()

  # Why do we run this test only on TPU (and not on CPU)?
  # CustomKernel exists only on TorchTPU.
  def test_custom_kernel_not_registered(self):
    name = "not_registered_kernel_name"
    kernel_key = "not_registered_kernel_name-(f32[2]):(f32[2])"

    inputs = [torch.ones(2, device=et.device())]
    output_shapes = [torch.ones(2, device=et.device())]

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""custom_kernel(): materialization failed with: unknown custom kernel; call torch_tpu._internal.pallas.tpu_torch_pallas.register_custom_kernel("{name}", "{kernel_key}", ...) to register the kernel before calling it""",
    ):
      outputs = tpu_torch_pallas.call_custom_kernel(
          name, kernel_key, inputs=inputs, output_shapes=output_shapes
      )

      # cpu() is needed because the error is triggered inside the op builder.
      outputs[0].cpu()

  # Why do we run this test only on TPU (and not on CPU)?
  # The notion of 'dynamic dimensions' does not exist in eager PyTorch.
  def test_embedding_bag_dynamic_shape_arg(self):
    with self.subTest(arg="indices"):
      indices = torch.ones(10, 10, device=et.device())
      weight = torch.ones(10, 10, device=et.device())

      # Mark dimension 1 of `indices` as dynamic.
      dynamism.mark_dynamic(indices, 1, 5, 20)

      with et.assert_raises_message(
          RuntimeError,
          tpu="""embedding_bag_forward_only(): expected all dimensions of the indices tensor to be static, got 1 dynamic dimension in the underlying tensor behind a view of shape [100]""",
          message_reviewed_by="wan",
      ):
        torch.nn.functional.embedding_bag(indices, weight)

    with self.subTest(arg="weight"):
      indices = torch.ones(10, 10, device=et.device())
      weight = torch.ones(10, 10, device=et.device())

      # Mark the dimension 1 of `weight` as dynamic.
      dynamism.mark_dynamic(weight, 1, 5, 20)

      with et.assert_raises_message(
          RuntimeError,
          tpu="""embedding_bag_forward_only(): expected all dimensions of the weight tensor to be static, got 1 dynamic dimension within shape [10, 10 (up to 20)]""",
          message_reviewed_by="wan",
      ):
        torch.nn.functional.embedding_bag(indices, weight)

    with self.subTest(arg="offsets"):
      indices = torch.ones(10 * 10, device=et.device())
      weight = torch.ones(10, 10, device=et.device())
      offsets = torch.arange(0, 100, 10, device=et.device(), dtype=torch.int64)

      # Mark the dimension 1 of `offsets` as dynamic.
      dynamism.mark_dynamic(offsets, 0, 5, 20)

      with et.assert_raises_message(
          RuntimeError,
          tpu="""embedding_bag_forward_only(): expected all dimensions of the offsets tensor to be static, got 1 dynamic dimension within shape [10 (up to 20)]""",
          message_reviewed_by="wan",
      ):
        torch.nn.functional.embedding_bag(indices, weight, offsets)

  @parameterized.named_parameters(
      {"testcase_name": "2d", "op": torch.grid_sampler_2d, "dims": 2},
      {"testcase_name": "3d", "op": torch.grid_sampler_3d, "dims": 3},
  )
  def test_grid_sampler_invalid_padding_mode(self, op, dims: int):
    inp_shape = [1, 1] + [2] * dims
    inp = torch.ones(inp_shape, device=et.device(), dtype=torch.complex64)

    grid_shape = [1] + [2] * dims + [dims]
    grid = torch.zeros(grid_shape, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""grid_sampler_{dims}d(): expected the padding mode to be 0 (zeros), 1 (border), or 2 (reflection), got 3""",
        message_reviewed_by="wan",
    ):
      padding_mode = 3
      op(inp, grid, 0, padding_mode, False)

  def test_grid_sampler_2d_invalid_interpolation_mode(self):
    inp = torch.ones(1, 1, 2, 2, device=et.device())
    grid = torch.zeros(1, 2, 2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_2d(): expected the interpolation mode to be 0 (bilinear), 1 (nearest), or 2 (bicubic), got 3""",
        message_reviewed_by="wan",
    ):
      interpolation_mode = 3
      torch.grid_sampler(inp, grid, interpolation_mode, 0, False)

  def test_grid_sampler_3d_invalid_interpolation_mode(self):
    inp = torch.ones(1, 1, 2, 2, 2, device=et.device())
    grid = torch.zeros(1, 2, 2, 2, 3, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_3d(): expected the interpolation mode to be 0 (bilinear) or 1 (nearest), got 2""",
        message_reviewed_by="wan",
    ):
      interpolation_mode = 2
      torch.grid_sampler(inp, grid, interpolation_mode, 0, False)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch `elu_backward` implementation doesn't error on this check.
  # BUG: TorchTPU should have similar behavior w.r.t. PyTorch native devices.
  @parameterized.named_parameters(
      ("bool", torch.bool, "bool"),
      ("int64", torch.int64, "int64"),
  )
  def test_elu_backward_unsupported_dtypes(
      self, dtype: torch.dtype, tpu_dtype_str: str
  ):
    grad_output = torch.ones(4, device=et.device(), dtype=dtype)

    alpha = 1.0
    scale = 1.0
    input_scale = 1.0
    is_result = False
    self_or_result = torch.ones(4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""elu_backward(): expected the grad output dtype to be floating"""
        f""" point, got {tpu_dtype_str}""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.elu_backward(
          grad_output, alpha, scale, input_scale, is_result, self_or_result
      )

  # Why do we run this test only on TPU (and not on CPU)?
  # The notion of 'dynamic dimensions' does not exist in eager PyTorch.
  def test_fft_r2c_dynamic_shape(self):
    inp = torch.ones(10, device=et.device(), dtype=torch.float32)

    # Mark the dimension 0 as dynamic.
    dynamism.mark_dynamic(inp, 0, 5, 20)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fft_r2c(): expected all dimensions of the input tensor to be static, got 1 dynamic dimension within shape [10 (up to 20)]""",
        message_reviewed_by="wan",
    ):
      torch.fft.rfftn(inp)

  def test_mark_dynamic_multiple_dimensions_failure(self):
    inp = torch.ones(10, 10, device=et.device(), dtype=torch.float32)

    # Mark the dimension 0 as dynamic.
    dynamism.mark_dynamic(inp, 0, 5, 20)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""only one dynamic dimension is supported per tensor""",
    ):
      # Try to mark dimension 1 as dynamic too.
      dynamism.mark_dynamic(inp, 1, 5, 20)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch `native_group_norm_backward` implementation doesn't check the
  # dimensions of `grad` and `inp`.
  # TODO: TorchTPU should have similar behavior w.r.t. PyTorch native devices.
  def test_native_group_norm_backward_mismatch_grad_shape(self):
    n = 5
    c = 5
    h_w = 5
    group = 5

    grad_out = torch.ones(n * c * h_w, device=et.device())
    inp = torch.ones(n, c, h_w, device=et.device())
    mean = torch.ones(n, group, device=et.device())
    rstd = torch.ones(n, group, device=et.device())
    weight = torch.ones(c, device=et.device())

    output_mask = [True, True, True]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_group_norm_backward(): grad_out and input must have the same dimensions, got grad_out size [125], input size [5, 5, 5]""",
    ):
      torch.ops.aten.native_group_norm_backward(
          grad_out, inp, mean, rstd, weight, n, c, h_w, group, output_mask
      )

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch implementation promotes `input` and `target` dtypes, instead of
  # raising an error.
  # TODO: TorchTPU should have similar behavior w.r.t. PyTorch native devices.
  def test_mse_loss_dtype_mismatch(self):
    inp = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(2, 2, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): input and target must have the same dtype""",
    ):
      torch.nn.functional.mse_loss(inp, target)

  # Why do we run this test only on TPU (and not on CPU)?
  # Limitation is TorchTPU specific.
  # TODO: add support to other memory formats for `resize_()` op.
  def test_resize_unsupported_memory_format(self):
    t = torch.ones(1, 2, 3, 4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""resize_(): non-contiguous memory formats are not yet supported""",
        message_reviewed_by="wan",
    ):
      t.resize_((1, 2, 3, 4), memory_format=torch.channels_last)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch native devices don't error on bool input dtype.
  # TODO: add support to threshold_backward() for bool input dtype.
  def test_threshold_backward_unsupported_dtype_bool(self):
    grad_output = torch.ones(2, device=et.device())
    self_tensor = torch.tensor([True, False], device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""threshold_backward(): bool input dtype is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten.threshold_backward(grad_output, self_tensor, 0.5)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch native devices don't error on sorted=False.
  # TODO: add support to topk() on sorted=False.
  def test_topk_unsupported_sorted_false(self):
    t = torch.ones(2, 2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""topk(): sorted=False is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.topk(t, k=1, sorted=False)

  # Why do we run this test only on TPU (and not on CPU)?
  # There are no other available devices on CPU runs, other than CPU.
  def test_acos_out_on_cpu(self):
    t = torch.ones(5, device=et.device())
    out = torch.ones(5, device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""acos(): expected the output tensor to be on tpu, got cpu""",
        message_reviewed_by="wan",
    ):
      torch.acos(t, out=out)

  def test_set_dimension_logical_size_size_not_0d(self):
    inp = torch.ones(2, 2, device=et.device())
    size = torch.tensor([5], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""set_dimension_logical_size(): expected a 0-dimensional tensor for size, got 1-dimensional tensor""",
        message_reviewed_by="wan",
    ):
      torch.ops.torch_tpu.set_dimension_logical_size(inp, 0, size)

  def test_set_dimension_logical_size_size_not_int(self):
    inp = torch.ones(2, 2, device=et.device())
    size = torch.tensor(5.0, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""set_dimension_logical_size(): expected an int32 tensor for size, got float32""",
        message_reviewed_by="wan",
    ):
      torch.ops.torch_tpu.set_dimension_logical_size(inp, 0, size)

  def test_leaky_relu_backward_negative_slope_with_self_is_result(self):
    grad_output = torch.ones(2, device=et.device())
    self_or_result = torch.ones(2, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. This is caused by calling in-place forward function with a negative slope, please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do require supporting in-place leakRelu backward calculation with negative slope""",
    ):
      torch.ops.aten.leaky_relu_backward(
          grad_output, self_or_result, negative_slope=-1.0, self_is_result=True
      )

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch CPU also supports float16.
  # TODO: add support to float16 dtype for `view_as_complex()` op.
  def test_view_as_complex_unsupported_dtypes_float16(self):
    t1 = torch.ones(2, 3, 2, device=et.device(), dtype=torch.float16)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): float16 dtype is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t1)

  def test_get_pad_module_mlir_mismatched_bounds_size(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([1], [8, 16])]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""dimension indices and upper bounds must have the same size, got 1 and 2""",
    ):
      tpu_torch_compile.get_pad_module_mlir(tensor_info, bounds_list)

  def test_get_pad_module_mlir_dim_out_of_bounds(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([2], [8])]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""dimension index must be within bounds [0, 1], got 2 for input tensor 0 with shape [1, 4]""",
    ):
      tpu_torch_compile.get_pad_module_mlir(tensor_info, bounds_list)

  def test_get_pad_module_mlir_invalid_upper_bound(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([1], [2])]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""upper bound must be greater than or equal to the static shape's dimension size, got upper bound 2 for dimension 1 for input tensor 0 with shape [1, 4]""",
    ):
      tpu_torch_compile.get_pad_module_mlir(tensor_info, bounds_list)

  def test_execute_output_shapes_too_many(self):
    with execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(10, device="cpu").to(device=et.device())
      y = torch.ones(10, device="cpu").to(device=et.device())
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""output shapes must be specified for all outputs or none, got 2 output shapes for 1 output tensors""",
    ):
      tpu_torch_compile.execute(executable, [x, y], [[5], [5]])

  def test_execute_output_shapes_rank_mismatch(self):
    with execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(10, device="cpu").to(device=et.device())
      y = torch.ones(10, device="cpu").to(device=et.device())
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""output shape number of dimensions must match the statically inferred dimensions, got output shape dimensions 2 and inferred dimensions 1 for output tensor 0""",
    ):
      tpu_torch_compile.execute(executable, [x, y], [[5, 2]])

  def test_execute_output_shapes_exceeds_bound(self):
    with execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(10, device="cpu").to(device=et.device())
      y = torch.ones(10, device="cpu").to(device=et.device())
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""output shape dimension must not exceed the statically inferred bound, got output shape [15] and inferred shape [10]""",
    ):
      tpu_torch_compile.execute(executable, [x, y], [[15]])

  def test_mm_dtype_outdtype_mismatch(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.int32)
    rhs = torch.ones(4, 5, device=et.device(), dtype=torch.int32)
    out = torch.ones(3, 5, device=et.device(), dtype=torch.int64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected out_dtype to be the same as the dtype of the out tensor, got out_dtype int32 vs out tensor dtype int64""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.mm.dtype_out(lhs, rhs, out_dtype=torch.int32, out=out)

  def test_mm_dtype_output_dtype_mismatch_fp32(self):
    lhs = torch.ones(3, 4, device=et.device(), dtype=torch.float16)
    rhs = torch.ones(4, 5, device=et.device(), dtype=torch.float16)
    out = torch.ones(3, 5, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mm(): expected out_dtype to be the same as input dtype or fp32 for fp16/bf16 inputs, got out_dtype float64 vs inputs dtype float16""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.mm.dtype_out(lhs, rhs, out_dtype=torch.float64, out=out)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch CPU does not raise an error when g size does not match the size of
  # weight in dim 0.
  # TODO: TorchTPU should have similar behavior w.r.t. PyTorch native devices.
  def test_weight_norm_interface_g_size_mismatch(self):
    v = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    g = torch.ones(3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected the size of the weight magnitude (g) to be 2, which is the size of the weight at dimension 0, got 3""",
        message_reviewed_by="wan",
    ):
      torch._weight_norm(v, g, 0)

  # Why do we run this test only on TPU (and not on CPU)?
  # PyTorch CPU does not raise an error when g has more than 1 dimension.
  # TODO: TorchTPU should have similar behavior w.r.t. PyTorch native devices.
  def test_weight_norm_interface_g_rank_too_large(self):
    v = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    g = torch.ones(3, 3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected the weight magnitude (g) to be a scalar or a 1D tensor, got a tensor of shape [3, 3]""",
        message_reviewed_by="wan",
    ):
      torch._weight_norm(v, g, 1)

  # Why do we run this test only on TPU (and not on CPU)?
  # The notion of 'dynamic dimensions' does not exist in eager PyTorch.
  def test_bitcast_dynamic_shape(self):
    inp = torch.ones(5, 2, device=et.device(), dtype=torch.int32)

    # Mark dimension 0 of `inp` as dynamic.
    dynamism.mark_dynamic(inp, 0, 4, 10)

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""to_copy(): materialization failed with: expected all dimensions of the bitcast input tensor to be static, got 1 dynamic dimension within shape [dyn, 2]; calling ViewPrimitiveShlo() with input shape=[dyn, 2] and primitive=bitcast(from_type=int32, to_type=int64)""",
        message_reviewed_by="wan",
    ):
      out = inp.view(torch.int64)
      out.cpu()


if __name__ == "__main__":
  absltest.main()
