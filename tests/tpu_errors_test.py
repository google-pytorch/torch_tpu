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
from typing import Any, TypeAlias
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

EagerMode: TypeAlias = execution_mode.EagerMode


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

  @et.why_tpu_only("TODO: support int1 dtype on TPU")
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

  @et.why_tpu_only("TODO: support complex128 dtype on TPU")
  def test_fill_complex128(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fill_(): complex128 dtype is not supported""",
        message_reviewed_by="wan",
    ):
      t = torch.empty((2, 2), dtype=torch.complex128, device=et.device())
      t.fill_(1.0)

  @et.why_tpu_only("For testing TPU op dispatching.")
  def test_prod_with_op_dispatch_failure(self):
    """Tests that prod() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("prod", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prod(): my error""",
    ):
      torch.prod(t1).to("cpu")

  @et.why_tpu_only("Can only test device mismatch in a TPU test.")
  def test_add_out_device_mismatch(self):
    a = torch.ones(4, device=et.device())
    b = torch.ones(4, device=et.device())
    out = torch.empty(4, device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): expected output tensor to be on 'tpu' device, got 'cpu'""",
        message_reviewed_by="wan",
    ):
      torch.add(a, b, alpha=2, out=out)

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
  def test_prod_out_with_op_dispatch_failure(self):
    """Tests that prod() bubbles up errors in op dispatching."""

    tt_testing.set_op_dispatch_failure("prod", "my error")
    t1 = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prod(): my error""",
    ):
      torch.prod(t1, dim=1, out=torch.zeros_like(t1)).to("cpu")

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("TODO: support _unique2(sorted=False) on TPU.")
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

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("Can only test op dispatch failure in a TPU test.")
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

  @et.why_tpu_only("The behavior is undefined on CPU.")
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

  @et.why_tpu_only("The behavior is undefined on CPU.")
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

  @et.why_tpu_only("Testing TPU OOM behavior.")
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

  @et.why_tpu_only("Can only test device mismatch errors on TPU.")
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

  @et.why_tpu_only("Can only test device mismatch errors on TPU.")
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

  @et.why_tpu_only("Can only test device mismatch errors on TPU.")
  def test_add_out_cpu(self):
    tpu_tensor1 = torch.tensor([1, 2, 3], device="tpu")
    tpu_tensor2 = torch.tensor([4, 5, 6], device="tpu")
    cpu_tensor = torch.tensor([7, 8, 9], device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""add(): the out tensor is expected to be on tpu, got cpu""",
    ):
      torch.add(tpu_tensor1, tpu_tensor2, out=cpu_tensor)

  @et.why_tpu_only("TODO: support complex32 on TPU.")
  def test_dtype_complex32_unsupported(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""empty(): TorchTPU does not yet support dtype complex32""",
        message_reviewed_by="wan",
    ):
      torch.empty(2, dtype=torch.complex32, device="tpu")

  @et.why_tpu_only("TODO: support complex32 on TPU.")
  def test_empty_strided(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""empty_strided(): TorchTPU does not yet support dtype complex32""",
        message_reviewed_by="wan",
    ):
      torch.empty_strided((2,), (1,), dtype=torch.complex32, device="tpu")

  @et.why_tpu_only("TODO: support jagged layout on TPU.")
  def test_empty_unsupported_layout(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""empty(): only layout=torch.strided is supported by TorchTPU for now, got torch.jagged""",
    ):
      torch.empty(2, layout=torch.jagged, device="tpu")

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_histc_bounds_unsupported_dtype(self):
    """Tests that torch.histc() fails when the bounds have an unsupported dtype."""
    t = torch.tensor([0, 0], device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""histc(): expected min and max to be float or int type, got Bool and Bool""",
    ):
      torch.histc(t, min=False, max=True)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_dynamic_arange_unsupported_dtype(self):
    """Tests that torch.ops.tpu.dynamic_arange() fails with bool."""
    device = et.device()
    start = torch.tensor(0, device=device, dtype=torch.int32)
    end = torch.tensor(5, device=device, dtype=torch.int32)
    step = torch.tensor(1, device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_arange(): expected float or int dtype, got bool""",
    ):
      torch.ops.tpu.dynamic_arange(start, end, step, 5, torch.bool)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_dynamic_arange_invalid_start_dim(self):
    """Tests that torch.ops.tpu.dynamic_arange() fails if start is 1D."""
    device = et.device()
    start = torch.tensor([0], device=device, dtype=torch.int32)
    end = torch.tensor(5, device=device, dtype=torch.int32)
    step = torch.tensor(1, device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_arange(): expected a 0-dimensional tensor for start, got 1-dimensional tensor""",
    ):
      torch.ops.tpu.dynamic_arange(start, end, step, 5, torch.int32)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_dynamic_arange_invalid_end_dim(self):
    """Tests that torch.ops.tpu.dynamic_arange() fails if end is 1D."""
    device = et.device()
    start = torch.tensor(0, device=device, dtype=torch.int32)
    end = torch.tensor([5], device=device, dtype=torch.int32)
    step = torch.tensor(1, device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_arange(): expected a 0-dimensional tensor for end, got 1-dimensional tensor""",
    ):
      torch.ops.tpu.dynamic_arange(start, end, step, 5, torch.int32)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_dynamic_arange_invalid_step_dim(self):
    """Tests that torch.ops.tpu.dynamic_arange() fails if step is 1D."""
    device = et.device()
    start = torch.tensor(0, device=device, dtype=torch.int32)
    end = torch.tensor(5, device=device, dtype=torch.int32)
    step = torch.tensor([1], device=device, dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_arange(): expected a 0-dimensional tensor for step, got 1-dimensional tensor""",
    ):
      torch.ops.tpu.dynamic_arange(start, end, step, 5, torch.int32)

  @et.why_tpu_only(
      "The op dynamic_broadcast is TPU only for internal use in"
      " torch.compile()."
  )
  def test_dynamic_broadcast_shape_size_mismatch(self):
    device = et.device()
    x = torch.tensor([1.0, 2.0], device=device)
    shape = [torch.tensor(3, device=device, dtype=torch.int32)]
    broadcast_dims = [1]
    static_shape = [3, 2]
    is_dynamic = [True, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_broadcast(): shape list size must match static_shape size, got shape list size 1 and static_shape size 2""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only(
      "The op dynamic_broadcast is TPU only for internal use in"
      " torch.compile()."
  )
  def test_dynamic_broadcast_is_dynamic_size_mismatch(self):
    device = et.device()
    x = torch.tensor([1.0, 2.0], device=device)
    shape = [
        torch.tensor(3, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [1]
    static_shape = [3, 2]
    is_dynamic = [True]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_broadcast(): is_dynamic size must match static_shape size, got is_dynamic size 1 and static_shape size 2""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only(
      "The op dynamic_broadcast is TPU only for internal use in"
      " torch.compile()."
  )
  def test_dynamic_broadcast_shape_not_0d(self):
    device = et.device()
    x = torch.tensor([1.0, 2.0], device=device)
    shape = [
        torch.tensor([3], device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [1]
    static_shape = [3, 2]
    is_dynamic = [True, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_broadcast(): shape tensor at index 0 must be a 0-D (scalar) tensor, got 1-D tensor""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only(
      "The op dynamic_broadcast is TPU only for internal use in"
      " torch.compile()."
  )
  def test_dynamic_broadcast_shape_not_int32(self):
    device = et.device()
    x = torch.tensor([1.0, 2.0], device=device)
    shape = [
        torch.tensor(3.0, device=device, dtype=torch.float32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [1]
    static_shape = [3, 2]
    is_dynamic = [True, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_broadcast(): shape must be a list of int32 tensors, got float32 tensor at index 0""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only(
      "The op dynamic_broadcast is TPU only for internal use in"
      " torch.compile()."
  )
  def test_dynamic_broadcast_broadcast_dims_size_mismatch(self):
    device = et.device()
    x = torch.tensor([1.0, 2.0], device=device)
    shape = [
        torch.tensor(3, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [1, 0]
    static_shape = [3, 2]
    is_dynamic = [True, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_broadcast(): broadcast_dims size must match input rank, got broadcast_dims size 2 and input rank 1""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only(
      "The op dynamic_broadcast is TPU only for internal use in"
      " torch.compile()."
  )
  def test_dynamic_broadcast_broadcast_dims_out_of_range(self):
    device = et.device()
    x = torch.tensor([1.0, 2.0], device=device)
    shape = [
        torch.tensor(3, device=device, dtype=torch.int32),
        torch.tensor(2, device=device, dtype=torch.int32),
    ]
    broadcast_dims = [2]
    static_shape = [3, 2]
    is_dynamic = [True, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_broadcast(): broadcast_dims must be in range [0, 2), got 2 for broadcast dim at index 0""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_is_nonzero_with_more_than_one_value(self):
    with et.assert_raises_message(
        RuntimeError,
        # This error is generated by pytorch. We don't have a good way to
        # replace it.
        tpu="""Boolean value of Tensor with more than one value is ambiguous""",
    ):
      torch.is_nonzero(torch.tensor([1, 3, 5], device="tpu"))

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_is_nonzero_with_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This is the error message we get from pytorch cpu.
        tpu="""Boolean value of Tensor with no values is ambiguous""",
    ):
      torch.is_nonzero(torch.tensor([], device="tpu"))

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_is_nonzero_with_nested_empty_tensor(self):
    with et.assert_raises_message(
        RuntimeError,
        # This is the error message we get from pytorch cpu.
        tpu="""Boolean value of Tensor with no values is ambiguous""",
    ):
      torch.is_nonzero(torch.tensor([[]], device="tpu"))

  @et.why_tpu_only("Can only test device mismatch errors on TPU.")
  def test_masked_select_out_on_different_device(self):
    """Masked select function fails when self and out have different devices."""
    t = torch.ones(5, device="tpu", dtype=torch.float32)
    mask = torch.ones(5, device="tpu", dtype=torch.bool)
    out = torch.ones(5, device="cpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_select(): expected out tensor to be on tpu, got cpu""",
    ):
      torch.masked_select(t, mask, out=out)

  @et.why_tpu_only("Can only test device mismatch errors on TPU.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_set_invalid_metadata(self):
    t = torch.zeros(1, device="tpu", dtype=torch.float32)
    source = torch.arange(8, device="tpu", dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""set_(): expected the number of bytes required by the given arguments to be <= 32 (actual storage size), got 64""",
        message_reviewed_by="wan",
    ):
      t.set_(source.untyped_storage(), storage_offset=0, size=[16], stride=[1])

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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
  @et.why_tpu_only("TODO: support complex dtypes in addmm on TPU.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_cummin_with_unsupported_complex_dtype(self):
    """Tests cummin fails with unsupported complex dtypes."""
    with et.assert_raises_message(
        RuntimeError,
        tpu="""cummin_helper(): expected supported element type, got ComplexFloat""",
    ):
      t = torch.ones(2, 2, device="tpu", dtype=torch.complex64)
      torch.cummin(t, dim=1)

  @et.why_tpu_only("logcumsumexp dtype validation lives in the TPU kernel.")
  def test_logcumsumexp_with_unsupported_integer_dtype(self):
    """Tests logcumsumexp rejects non-floating-point inputs.

    Only the functional path is exercised. The out= path decomposes through the
    functional _logcumsumexp under functionalization, so AtenLogcumsumexpOut's
    identical check is unreachable (marked ERROR_COV_INFEASIBLE in the kernel).
    """
    t = torch.ones(2, 2, device="tpu", dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""logcumsumexp(): expected the input dtype to be floating point, got int32""",
    ):
      torch.logcumsumexp(t, dim=1)

  @et.why_tpu_only("The behavior is undefined on CPU.")
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

  @et.why_tpu_only(
      "TODO: b/480225714 remove this after the corresponding PyTorch#173995 bug"
      " is fixed."
  )
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

  @et.why_tpu_only("TODO: lift int32 index limit on TPU.")
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

  @et.why_tpu_only("TODO: support complex128 alpha in add on TPU.")
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
  @et.why_tpu_only("TODO: support complex128 alpha in add_relu on TPU.")
  def test__add_relu_Scalar_complex_alpha(self):
    """Tests _add_relu.Scalar with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu.Scalar(t, 2.0, 1j)

  @et.why_tpu_only("TODO: support complex128 alpha in add_relu on TPU.")
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

  @et.why_tpu_only("TODO: support complex128 alpha in add_relu on TPU.")
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

  @et.why_tpu_only("TODO: support complex128 alpha in add_relu on TPU.")
  def test__add_relu__Scalar_complex_alpha(self):
    """Tests _add_relu_.Scalar with an alpha of complex type."""
    device = et.device()
    t = torch.ones(4, device=device, dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""add_relu_(): complex128 alpha value is not yet supported""",
    ):
      torch.ops.aten._add_relu_.Scalar(t, 2.0, alpha=1j)

  @et.why_tpu_only("TODO: support complex128 alpha in add_relu on TPU.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_threshold_unsupported_bool_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold(): threshold is not implemented for bool type""",
    ):
      torch.threshold(torch.tensor([True, False], device=et.device()), 0.5, 0.0)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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
  @et.why_tpu_only("Can only test device mismatch on TPU.")
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
  @et.why_tpu_only("Can only test device mismatch on TPU.")
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
  # This test should be run only on TPU because there are no other available
  # devices on CPU runs, other than CPU.
  @parameterized.named_parameters(
      {"testcase_name": "min", "op_name": "min", "op": torch.min},
      {"testcase_name": "max", "op_name": "max", "op": torch.max},
  )
  @et.why_tpu_only("Can only test device mismatch on TPU.")
  def test_min_max_unary_invalid_output_device(self, op_name: str, op: Any):
    tensor = torch.ones(5, device=et.device(), dtype=torch.float32)
    out = torch.tensor(0.0, device="cpu", dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu=f"""{op_name}(): expected output tensor to be on tpu, got cpu""",
        message_reviewed_by="gunhyun",
    ):
      op(tensor, out=out)

  # CPU kernel raises an `IndexError`, instead of a `RuntimeError`, because it
  # tries to get `pivots.size(-1)` of a 0-dim tensor.
  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
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

  # CPU kernel raises an `IndexError`, instead of a `RuntimeError`, because it
  # tries to get `pivots.size(-1)` of a 0-dim tensor.
  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
  def test_lu_unpack_pivots_rank_too_low(self):
    lu_data = torch.ones(4, 4, device=et.device())
    lu_pivots = torch.tensor(0, device=et.device(), dtype=torch.int32)

    # Call the out overload of lu_unpack() op.
    out = _make_lu_unpack_outputs(p=(4, 4), l=(4, 4), u=(4, 4))

    # TODO: b/485613841 also test CPU when the TPU kernel is fixed, raising an
    # `IndexError`, instead of an `RuntimeError`.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lu_unpack(): lu_pivots must have at least 1 dimension, got 0""",
    ):
      torch.lu_unpack(lu_data, lu_pivots, out=out)

  # CPU kernel runs successfully, broadcasting the inputs.
  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
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

  @et.why_tpu_only("TODO: support generator in multinomial on TPU.")
  def test_multinomial_generator(self):
    inp = torch.randn(2, device=et.device())
    gen = torch.Generator()

    with et.assert_raises_message(
        RuntimeError,
        tpu="""multinomial(): generator is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.multinomial(inp, num_samples=1, generator=gen)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  @et.why_tpu_only("TODO: support sparse_grad in gather() on TPU.")
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

  @et.why_tpu_only("TODO: support complex128 in lerp() on TPU.")
  def test_lerp_complex_double(self):
    t = torch.tensor([1.0], dtype=torch.complex128, device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""lerp(): complex128 dtype is not yet supported""",
    ):
      torch.lerp(t, t, t)

  @et.why_tpu_only("Can only test device mismatch on TPU.")
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

  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
  def test_local_scalar_dense_too_many_elements(self):
    inp = torch.ones(2, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""local_scalar_dense(): expected the input tensor to have 1 element, got 2""",
        message_reviewed_by="wan",
    ):
      torch.ops.aten._local_scalar_dense(inp)

  @et.why_tpu_only("Custom kernel is a TPU-specific feature.")
  def test_custom_kernel_not_registered(self):
    name = "not_registered_kernel_name"
    kernel_key = "not_registered_kernel_name-(f32[2]):(f32[2])"

    inputs = [torch.ones(2, device=et.device())]
    output_shapes = [torch.ones(2, device=et.device())]

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""custom_kernel(): materialization failed with: unknown custom kernel; call torch_tpu._internal.pallas.tpu_torch_pallas.register_custom_kernel() to register the kernel before calling it""",
    ):
      outputs = tpu_torch_pallas.call_custom_kernel(
          name, kernel_key, inputs=inputs, output_shapes=output_shapes
      )

      # cpu() is needed because the error is triggered inside the op builder.
      outputs[0].cpu()

  @et.why_tpu_only("Bounded dynamism is a TPU-specific feature.")
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
  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

    inp_backward = torch.randn(
        2, 3, 4, 4, device=et.device(), dtype=torch.float32
    )
    grid_backward = torch.randn(
        2, 5, 5, 2, device=et.device(), dtype=torch.float32
    )
    grad_output = torch.randn(
        2, 3, 5, 5, device=et.device(), dtype=torch.float32
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_2d_backward(): expected the interpolation mode to be 0 (bilinear), 1 (nearest), or 2 (bicubic), got 3""",
    ):
      interpolation_mode = 3
      torch.ops.aten.grid_sampler_2d_backward(
          grad_output,
          inp_backward,
          grid_backward,
          interpolation_mode,
          0,
          False,
          [True, True],
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

    grad_output = torch.randn(
        2, 3, 5, 5, 5, device=et.device(), dtype=torch.float32
    )
    inp_backward = torch.randn(
        2, 3, 4, 4, 4, device=et.device(), dtype=torch.float32
    )
    grid_backward = torch.randn(
        2, 5, 5, 5, 3, device=et.device(), dtype=torch.float32
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu="""grid_sampler_3d_backward(): expected the interpolation mode to be 0 (bilinear) or 1 (nearest), got 2""",
    ):
      interpolation_mode = 2
      torch.ops.aten.grid_sampler_3d_backward(
          grad_output,
          inp_backward,
          grid_backward,
          interpolation_mode,
          0,
          False,
          [True, True],
      )

  @parameterized.named_parameters(
      ("bool", torch.bool, "bool"),
      ("int64", torch.int64, "int64"),
  )
  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
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

  @et.why_tpu_only("Bounded dynamism is a TPU-specific feature.")
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

  @et.why_tpu_only("Bounded dynamism is a TPU-specific feature.")
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

  @et.why_tpu_only("Bounded dynamism is a TPU-specific feature.")
  def test_slice_dynamic_dim_on_dynamic_tensor_error(self):
    inp = torch.ones(5, 2, device=et.device())

    # Mark dimension 0 of `inp` as dynamic.
    dynamism.mark_dynamic(inp, 0, 4, 10)

    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r".*slicing dynamic dimension 0 for input .* is not supported"
        ),
    ):
      out = inp[1:]
      out.cpu()

  # PyTorch `native_group_norm_backward` implementation doesn't check the
  # dimensions of `grad` and `inp`.
  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
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

  # PyTorch implementation promotes `input` and `target` dtypes, instead of
  # raising an error.
  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
  def test_mse_loss_dtype_mismatch(self):
    inp = torch.ones(2, 2, device=et.device(), dtype=torch.float32)
    target = torch.ones(2, 2, device=et.device(), dtype=torch.float64)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""mse_loss(): input and target must have the same dtype""",
    ):
      torch.nn.functional.mse_loss(inp, target)

  @et.why_tpu_only("TODO: support more memory formats in resize_() on TPU.")
  def test_resize_unsupported_memory_format(self):
    t = torch.ones(1, 2, 3, 4, device=et.device())

    with et.assert_raises_message(
        RuntimeError,
        tpu="""resize_(): non-contiguous memory formats are not yet supported""",
        message_reviewed_by="wan",
    ):
      t.resize_((1, 2, 3, 4), memory_format=torch.channels_last)

  @et.why_tpu_only("TODO: make the behavior consistent between TPU and CPU.")
  def test_threshold_backward_unsupported_dtype_bool(self):
    grad_output = torch.ones(2, device=et.device())
    self_tensor = torch.tensor([True, False], device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold_backward(): threshold is not implemented for bool type""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.threshold_backward(grad_output, self_tensor, 0.5)

  @et.why_tpu_only(
      "TODO: support complex dtype for threshold_backward() on TPU."
  )
  def test_threshold_backward_unsupported_dtype_complex(self):
    grad_output = torch.ones(2, device=et.device())
    self_tensor = torch.tensor([1 + 1j, 2 + 2j], device=et.device())

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""threshold_backward(): threshold is not implemented for complex types""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.threshold_backward(grad_output, self_tensor, 0.5)

  @et.why_tpu_only("Can only test device mismatch on TPU.")
  def test_acos_out_on_cpu(self):
    t = torch.ones(5, device=et.device())
    out = torch.ones(5, device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""acos(): expected the output tensor to be on tpu, got cpu""",
        message_reviewed_by="wan",
    ):
      torch.acos(t, out=out)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_set_dimension_logical_size_size_not_0d(self):
    inp = torch.ones(2, 2, device=et.device())
    size = torch.tensor([5], device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""set_dimension_logical_size(): expected a 0-dimensional tensor for size, got 1-dimensional tensor""",
        message_reviewed_by="wan",
    ):
      torch.ops.tpu.set_dimension_logical_size(inp, 0, size)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_set_dimension_logical_size_size_not_int(self):
    inp = torch.ones(2, 2, device=et.device())
    size = torch.tensor(5.0, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""set_dimension_logical_size(): expected an int32 tensor for size, got float32""",
        message_reviewed_by="wan",
    ):
      torch.ops.tpu.set_dimension_logical_size(inp, 0, size)

  @et.why_tpu_only(
      "The op dynamic_reshape is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_reshape_shape_size_mismatch(self):
    inp = torch.ones(2, 2, device=et.device())
    shape = [torch.tensor(2, device=et.device(), dtype=torch.int32)]
    static_shape = [2, 2]
    is_dynamic = [False, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_reshape(): shape list size must match static_shape size, got shape list size 1 and static_shape size 2""",
    ):
      torch.ops.tpu.dynamic_reshape(inp, shape, static_shape, is_dynamic)

  @et.why_tpu_only(
      "The op dynamic_reshape is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_reshape_is_dynamic_size_mismatch(self):
    inp = torch.ones(2, 2, device=et.device())
    shape = [
        torch.tensor(2, device=et.device(), dtype=torch.int32),
        torch.tensor(2, device=et.device(), dtype=torch.int32),
    ]
    static_shape = [2, 2]
    is_dynamic = [False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_reshape(): is_dynamic size must match static_shape size, got is_dynamic size 1 and static_shape size 2""",
    ):
      torch.ops.tpu.dynamic_reshape(inp, shape, static_shape, is_dynamic)

  @et.why_tpu_only(
      "The op dynamic_reshape is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_reshape_shape_not_0d(self):
    inp = torch.ones(2, 2, device=et.device())
    shape = [
        torch.tensor([2], device=et.device(), dtype=torch.int32),
        torch.tensor(2, device=et.device(), dtype=torch.int32),
    ]
    static_shape = [2, 2]
    is_dynamic = [False, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_reshape(): shape tensor at index 0 must be a 0-D (scalar) tensor, got 1-D tensor""",
    ):
      torch.ops.tpu.dynamic_reshape(inp, shape, static_shape, is_dynamic)

  @et.why_tpu_only(
      "The op dynamic_reshape is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_reshape_shape_not_int32(self):
    inp = torch.ones(2, 2, device=et.device())
    shape = [
        torch.tensor(2.0, device=et.device(), dtype=torch.float32),
        torch.tensor(2, device=et.device(), dtype=torch.int32),
    ]
    static_shape = [2, 2]
    is_dynamic = [False, False]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_reshape(): shape must be a list of int32 tensors, got float32 tensor at index 0""",
    ):
      torch.ops.tpu.dynamic_reshape(inp, shape, static_shape, is_dynamic)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  @et.why_tpu_only(
      "TODO: support float16 dtype for `view_as_complex()` on TPU."
  )
  def test_view_as_complex_unsupported_dtypes_float16(self):
    t1 = torch.ones(2, 3, 2, device=et.device(), dtype=torch.float16)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""view_as_complex(): float16 dtype is not yet supported""",
        message_reviewed_by="wan",
    ):
      torch.view_as_complex(t1)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_get_or_compile_pad_module_invalid_bounds(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([1], [8, 16])]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""dimension indices and upper bounds must have the same size, got 1 and 2""",
    ):
      tpu_torch_compile.get_or_compile_pad_module(
          tensor_info,
          bounds_list,
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_get_or_compile_pad_module_dim_out_of_bounds(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([2], [8])]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""dimension index must be within bounds [0, 1], got 2 for input tensor 0 with shape [1, 4]""",
    ):
      tpu_torch_compile.get_or_compile_pad_module(
          tensor_info,
          bounds_list,
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_get_or_compile_pad_module_invalid_upper_bound(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([1], [2])]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""upper bound must be greater than or equal to the static shape's dimension size, got upper bound 2 for dimension 1 for input tensor 0 with shape [1, 4]""",
    ):
      tpu_torch_compile.get_or_compile_pad_module(
          tensor_info,
          bounds_list,
      )

  @et.why_tpu_only("Dynamic compilation is TPU-only.")
  def test_get_dynamic_pad_module_empty_shapes(self):
    expected = re.compile(r".*DynamicPadModule requires at least one shape\..*")
    with et.assert_raises_message(
        RuntimeError,
        tpu=expected,
    ):
      tpu_torch_compile.get_dynamic_pad_module([], [])

  @parameterized.named_parameters(
      dict(
          testcase_name="mismatched_padded_shapes_size",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 8], [1, 16]],
          input_scalar_types=[torch.float32],
          expected_error_message=(
              "target shapes and padded shapes must have the same size, got 1"
              " and 2"
          ),
      ),
      dict(
          testcase_name="mismatched_input_scalar_types_size",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 8]],
          input_scalar_types=[torch.float32, torch.float32],
          expected_error_message=(
              "target shapes and input scalar types must have the same size,"
              " got 1 and 2"
          ),
      ),
      dict(
          testcase_name="empty_target_shapes",
          target_shapes=[],
          padded_shapes=[],
          input_scalar_types=[],
          expected_error_message="expected at least one target shape, got none",
      ),
      dict(
          testcase_name="mismatched_dimensions_size",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 8, 16]],
          input_scalar_types=[torch.float32],
          expected_error_message=(
              "target shape and padded shape must have the same number of"
              " dimensions, got 2 and 3 for tensor index 0"
          ),
      ),
      dict(
          testcase_name="invalid_padded_shape_bound",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 2]],
          input_scalar_types=[torch.float32],
          expected_error_message=(
              "padded shape dimension size must be greater than or equal to"
              " target shape dimension size, got padded shape [1, 2] and target"
              " shape [1, 4] for tensor index 0"
          ),
      ),
  )
  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_get_or_compile_slice_module_error_conditions(
      self,
      target_shapes,
      padded_shapes,
      input_scalar_types,
      expected_error_message,
  ):
    with et.assert_raises_message(
        RuntimeError,
        tpu=expected_error_message,
    ):
      tpu_torch_compile.get_or_compile_slice_module(
          target_shapes,
          padded_shapes,
          input_scalar_types,
          build_mlir_module=True,
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_execute_output_shapes_too_many(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_execute_output_shapes_rank_mismatch(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_execute_output_shapes_exceeds_bound(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
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

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_traverse_and_compile_invalid_layout_size(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(2, 3, device="cpu").to(device=et.device())
      z = x + x

    # 1 argument, but 2 layouts provided. Should fail.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""number of argument_layouts must match with the number of argument_tensors, got number of argument_layouts 2 and number of argument_tensors 1""",
    ):
      tpu_torch_compile.traverse_and_compile(
          [z], [x], argument_layouts=[[1, 0], [0, 1]]
      )

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_traverse_and_compile_invalid_layout_values(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(2, 3, device="cpu").to(device=et.device())
      z = x + x

    # Rank mismatch: shape [2, 3] (rank 2), layout [0] (rank 1)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""invalid layout for argument 0, got layout [0] for shape [2, 3]""",
    ):
      tpu_torch_compile.traverse_and_compile([z], [x], argument_layouts=[[0]])

    # Out of bounds index: shape [2, 3], layout [2, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""invalid layout for argument 0, got layout [2, 0] for shape [2, 3]""",
    ):
      tpu_torch_compile.traverse_and_compile(
          [z], [x], argument_layouts=[[2, 0]]
      )

    # Duplicate index: shape [2, 3], layout [0, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""invalid layout for argument 0, got layout [0, 0] for shape [2, 3]""",
    ):
      tpu_torch_compile.traverse_and_compile(
          [z], [x], argument_layouts=[[0, 0]]
      )

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_missing_input_to_build_mlir(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(10, device="cpu").to(device=et.device())
      y = torch.ones(10, device="cpu").to(device=et.device())
      z = x + y
    result_tensors = [z]
    argument_tensors = [x]
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r"compile_mlir\(\): failed to validate and reorder inputs: "
            r"identified an argument that was not provided:"
            r" DeviceBufferRef:[\s\S]*"
        ),
    ):
      tpu_torch_compile.build_mlir(result_tensors, argument_tensors)

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_compile_mlir_invalid_layout_size(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(2, 3, device="cpu").to(device=et.device())
      z = x + x
    mlir = tpu_torch_compile.build_mlir([z], [x])

    # 1 argument, but 2 layouts provided. Should fail.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): number of argument layouts (2) does not match number of arguments in MLIR main function (1)""",
    ):
      tpu_torch_compile.compile_mlir(mlir, argument_layouts=[[1, 0], [0, 1]])

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_compile_mlir_invalid_layout_values(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(2, 3, device="cpu").to(device=et.device())
      z = x + x
    mlir = tpu_torch_compile.build_mlir([z], [x])

    # Rank mismatch: shape [2, 3] (rank 2), layout [0] (rank 1)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): invalid layout for argument 0, got layout [0] for shape [2, 3]""",
    ):
      tpu_torch_compile.compile_mlir(mlir, argument_layouts=[[0]])

    # Out of bounds index: shape [2, 3], layout [2, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): invalid layout for argument 0, got layout [2, 0] for shape [2, 3]""",
    ):
      tpu_torch_compile.compile_mlir(mlir, argument_layouts=[[2, 0]])

    # Duplicate index: shape [2, 3], layout [0, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): invalid layout for argument 0, got layout [0, 0] for shape [2, 3]""",
    ):
      tpu_torch_compile.compile_mlir(mlir, argument_layouts=[[0, 0]])

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_compile_mlir_missing_main(self):
    mlir_text = """
module {
  func.func @not_main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    return %arg0 : tensor<2x3xf32>
  }
}
"""
    module = tpu_torch_compile.parse_mlir_text(mlir_text)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): could not find 'main' function in MLIR module""",
    ):
      tpu_torch_compile.compile_mlir(module, argument_layouts=[[1, 0]])

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_compile_mlir_invalid_argument_type(self):
    mlir_text = """
module {
  func.func @main(%arg0: tensor<2x3xi17>) {
    return
  }
}
"""
    module = tpu_torch_compile.parse_mlir_text(mlir_text)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): failed to convert MLIR type to XLA shape""",
    ):
      tpu_torch_compile.compile_mlir(module, argument_layouts=[[1, 0]])

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
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

  # PyTorch CPU does not raise an error when 1D g size does not match the size
  # of weight in the normalization dimension. TPU validates this strictly to
  # avoid division shape mismatches during StableHLO compilation.
  @et.why_tpu_only(
      "TPU enforces strict shape validation to avoid StableHLO compilation"
      " failures, whereas CPU/CUDA behavior is unsafe or undefined."
  )
  def test_weight_norm_interface_g_size_mismatch(self):
    v = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    g = torch.ones(3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected weight magnitude (g) size 0 to match weight size 2 at dimension 0, got 3""",
        message_reviewed_by="wan",
    ):
      torch._weight_norm(v, g, 0)

  # PyTorch CPU does not raise an error when g has invalid shape for same rank
  # (the CPU fused kernel silently ignores extra elements). However, TPU must
  # validate this strictly because incompatible shapes will cause division and
  # broadcasting failures during StableHLO compilation.
  @et.why_tpu_only(
      "TPU enforces strict shape validation to avoid StableHLO compilation"
      " failures, whereas CPU/CUDA behavior is unsafe or undefined."
  )
  def test_weight_norm_interface_g_shape_mismatch_same_rank(self):
    v = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    g = torch.ones(3, 3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected the size of the weight magnitude (g) at dimension 0 to be 1, got 3""",
        message_reviewed_by="wan",
    ):
      torch._weight_norm(v, g, 1)

  @et.why_tpu_only(
      "TPU enforces strict shape validation to avoid StableHLO compilation"
      " failures, whereas CPU/CUDA behavior is unsafe or undefined."
  )
  def test_weight_norm_interface_g_rank_mismatch(self):
    v = torch.ones(2, 3, 4, device=et.device(), dtype=torch.float32)
    g = torch.ones(2, 3, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected the weight magnitude (g) to be a scalar, a 1D tensor, or have the same rank as v, got a tensor of shape [2, 3]""",
        message_reviewed_by="wan",
    ):
      torch._weight_norm(v, g, 0)

  @et.why_tpu_only(
      "TPU enforces strict shape validation to avoid StableHLO compilation"
      " failures, whereas CPU/CUDA behavior is unsafe or undefined."
  )
  def test_weight_norm_interface_g_shape_mismatch_norm_dim(self):
    v = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    g = torch.ones(1, 2, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""weight_norm_interface(): expected the size of the weight magnitude (g) at dimension 1 to be 3, got 2""",
        message_reviewed_by="wan",
    ):
      torch._weight_norm(v, g, 1)

  @et.why_tpu_only("Bounded dynamism is a TPU-specific feature.")
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

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_ctc_loss_backward_targets_1d_or_2d(self):
    log_probs = torch.randn(5, 2, 3, device=et.device())
    targets = torch.randint(
        1, 3, (2, 3, 4), dtype=torch.int32, device=et.device()
    )
    input_lengths = torch.tensor([5, 5], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    neg_log_likelihood = torch.randn(2, device=et.device())
    log_alpha = torch.randn(2, 5, 7, device=et.device())
    grad_out = torch.randn(2, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""ctc_loss_backward(): expected targets to be 1-D or 2-D, got 3-D""",
    ):
      torch.ops.aten._ctc_loss_backward.Tensor(
          grad_out,
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          neg_log_likelihood,
          log_alpha,
          0,
          False,
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_ctc_loss_backward_input_lengths_size_match_batch_size(self):
    grad_out = torch.randn(2, device=et.device())
    log_probs = torch.randn(5, 2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor(
        [5, 5, 5], dtype=torch.int32, device=et.device()
    )
    target_lengths = torch.tensor([3, 3], dtype=torch.int32, device=et.device())
    neg_log_likelihood = torch.randn(2, device=et.device())
    log_alpha = torch.randn(2, 5, 7, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""ctc_loss_backward(): expected input_lengths to have batch_size (2) elements, got 3""",
    ):
      torch.ops.aten._ctc_loss_backward.Tensor(
          grad_out,
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          neg_log_likelihood,
          log_alpha,
          0,
          False,
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_ctc_loss_backward_target_lengths_size_match_batch_size(self):
    grad_out = torch.randn(2, device=et.device())
    log_probs = torch.randn(5, 2, 3, device=et.device())
    targets = torch.randint(1, 3, (2, 3), dtype=torch.int32, device=et.device())
    input_lengths = torch.tensor([5, 5], dtype=torch.int32, device=et.device())
    target_lengths = torch.tensor(
        [3, 3, 3], dtype=torch.int32, device=et.device()
    )
    neg_log_likelihood = torch.randn(2, device=et.device())
    log_alpha = torch.randn(2, 5, 7, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""ctc_loss_backward(): expected target_lengths to have batch_size (2) elements, got 3""",
    ):
      torch.ops.aten._ctc_loss_backward.Tensor(
          grad_out,
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          neg_log_likelihood,
          log_alpha,
          0,
          False,
      )

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_rng_validate_device_index_error(self):
    self.assertEqual(torch.tpu.current_device(), 0)

    with et.assert_raises_message(
        ValueError,
        tpu="""expected local device index 0, got 1: accessing RNG state of a non-current TPU device is not supported""",
    ):
      torch.tpu.get_rng_state(1)

  @et.why_tpu_only(
      "TPU enforces scale orientation ([M, 1] / [1, N]); CPU has no equivalent"
      " check (it rejects all non-scalar scales outright)."
  )
  def test_scaled_mm_invalid_scale_a_orientation(self):
    """Tests that scaled_mm rejects a 2-D row-wise scale_a with wrong orientation.

    TPU-only: scale_a must be [M, 1] (per-row); a [1, M] scale is rejected so a
    transposed scale is a clean error rather than a silent wrong-axis broadcast.
    There is no comparable CPU error to assert -- CPU has no scale-orientation
    check; it rejects all non-scalar scales outright (covered by
    test_scaled_mm_invalid_scale_a_size in errors_test.py).
    """
    device = et.device()
    # Generate in F32 and cast to FP8 to avoid randn failure on TPU!
    mat1 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 32, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    # scale_a should be [16, 1]; [1, 16] is the wrong orientation.
    scale_a = torch.ones(1, 16, dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_mm(): expected row-wise scale_a to have shape [16, 1], got [1, 16]""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  @et.why_tpu_only(
      "TPU enforces scale orientation ([M, 1] / [1, N]); CPU has no equivalent"
      " check (it rejects all non-scalar scales outright)."
  )
  def test_scaled_mm_invalid_scale_b_orientation(self):
    """Tests that scaled_mm rejects a 2-D per-channel scale_b with wrong orientation.

    TPU-only: scale_b must be [1, N] (per-output-channel); a [N, 1] scale is
    rejected so a transposed scale is a clean error rather than a silent
    wrong-axis broadcast. There is no comparable CPU error to assert -- CPU has
    no scale-orientation check; it rejects all non-scalar scales outright
    (covered by test_scaled_mm_invalid_scale_b_size in errors_test.py).
    """
    device = et.device()
    # Generate in F32 and cast to FP8 to avoid randn failure on TPU!
    mat1 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 32, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    # scale_b should be [1, 32]; [32, 1] is the wrong orientation.
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.ones(32, 1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_mm(): expected per-channel scale_b to have shape [1, 32], got [32, 1]""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_scaled_mm_use_fast_accum_unsupported(self):
    """Tests that scaled_mm fails if use_fast_accum is True."""
    # Note: use_fast_accum=true is not supported yet on TPU.
    # TPU and GPU match on this.
    device = et.device()
    mat1 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""scaled_mm(): use_fast_accum=true is not supported yet on TPU""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b, use_fast_accum=True)

  @et.why_tpu_only("TODO: make the behavior consistent with CPU.")
  def test_pdist_backward_negative_p(self):
    grad = torch.randn(1, device=et.device())
    self_tensor = torch.randn(2, 2, device=et.device())
    pdist = torch.randn(1, device=et.device())

    # CPU does not perform a negative p check (relies on forward validation)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""pdist_backward(): expected the p value to be >= 0, got -1""",
    ):
      torch.ops.aten._pdist_backward(grad, self_tensor, -1.0, pdist)

  @et.why_tpu_only("Testing TPU device generator initialization failure.")
  def test_default_generator_init_failure(self):
    tt_testing.reset_default_device_generators()
    tt_testing.set_init_default_generator_failure("Simulated init failure")
    try:
      # The first call to default_generators[] should fail.
      with et.assert_raises_message(
          RuntimeError,
          tpu="""Simulated init failure""",
      ):
        _ = torch.tpu.default_generators[0]
      # Subsequent calls to default_generators[] should fail too.
      with et.assert_raises_message(
          RuntimeError,
          tpu="""Simulated init failure""",
      ):
        _ = torch.tpu.default_generators[0]
    finally:
      tt_testing.set_init_default_generator_failure("")
      tt_testing.reset_default_device_generators()

  @et.why_tpu_only("Can only test device mismatch in a TPU test.")
  def test_prod_mixed_device_error(self):
    """Tests prod() with a TPU input and CPU out tensor."""
    t = torch.tensor([1.0, 2.0, 3.0], device=et.device())
    out = torch.empty(1, device="cpu")

    with et.assert_raises_message(
        RuntimeError,
        tpu="""prod(): expected the output tensor to be on 'tpu', got 'cpu'""",
    ):
      torch.prod(t, dim=0, out=out)

  @et.why_tpu_only(
      "GPU AdamW error handling differences for unsupported dtypes."
  )
  def test_fused_adamw_default_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got int32""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for unsupported dtypes."
  )
  def test_fused_adamw_default_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got complex64""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for unsupported dtypes."
  )
  def test_fused_adamw_tensor_lr_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got int32""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.tensor_lr(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for unsupported dtypes."
  )
  def test_fused_adamw_tensor_lr_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adamw_(): expected the input dtype to be floating-point, got complex64""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.tensor_lr(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for mismatched list sizes."
  )
  def test_fused_adamw_mismatched_grads_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adamw_(): expected grads to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p1, p2],
          [g1],
          [ea1, ea1],
          [eas1, eas1],
          [],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for mismatched list sizes."
  )
  def test_fused_adamw_mismatched_exp_avgs_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adamw_(): expected exp_avgs to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p1, p2],
          [g1, g1],
          [ea1],
          [eas1, eas1],
          [],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for mismatched list sizes."
  )
  def test_fused_adamw_mismatched_exp_avg_sqs_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adamw_(): expected exp_avg_sqs to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p1, p2],
          [g1, g1],
          [ea1, ea1],
          [eas1],
          [],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for mismatched list sizes."
  )
  def test_fused_adamw_mismatched_state_steps_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adamw_(): expected state_steps to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p1, p2],
          [g1, g1],
          [ea1, ea1],
          [eas1, eas1],
          [],
          [step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU AdamW error handling differences for mismatched list sizes."
  )
  def test_fused_adamw_mismatched_max_exp_avg_sqs_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adamw_(): expected max_exp_avg_sqs to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adamw_.default(
          [p1, p2],
          [g1, g1],
          [ea1, ea1],
          [eas1, eas1],
          [eas1],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=True,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for unsupported dtypes."
  )
  def test_fused_adam_default_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got int32""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for unsupported dtypes."
  )
  def test_fused_adam_default_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got complex64""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for unsupported dtypes."
  )
  def test_fused_adam_tensor_lr_int32_dtype(self):
    device = et.device()
    p_int = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g_int = torch.tensor([1, 1], dtype=torch.int32, device=device)
    ea_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    eas_int = torch.tensor([0, 0], dtype=torch.int32, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got int32""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.tensor_lr(
          [p_int],
          [g_int],
          [ea_int],
          [eas_int],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for unsupported dtypes."
  )
  def test_fused_adam_tensor_lr_complex64_dtype(self):
    device = et.device()
    p_cplx = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g_cplx = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    ea_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    eas_cplx = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)
    step = torch.tensor(1.0, dtype=torch.float32, device=device)
    lr_tensor = torch.tensor(0.001, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_adam_(): expected the input dtype to be floating-point, got complex64""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.tensor_lr(
          [p_cplx],
          [g_cplx],
          [ea_cplx],
          [eas_cplx],
          [],
          [step],
          lr=lr_tensor,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for mismatched list sizes."
  )
  def test_fused_adam_mismatched_grads_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adam_(): expected grads to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p1, p2],
          [g1],
          [ea1, ea1],
          [eas1, eas1],
          [],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for mismatched list sizes."
  )
  def test_fused_adam_mismatched_exp_avgs_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adam_(): expected exp_avgs to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p1, p2],
          [g1, g1],
          [ea1],
          [eas1, eas1],
          [],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for mismatched list sizes."
  )
  def test_fused_adam_mismatched_exp_avg_sqs_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adam_(): expected exp_avg_sqs to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p1, p2],
          [g1, g1],
          [ea1, ea1],
          [eas1],
          [],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for mismatched list sizes."
  )
  def test_fused_adam_mismatched_state_steps_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adam_(): expected state_steps to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p1, p2],
          [g1, g1],
          [ea1, ea1],
          [eas1, eas1],
          [],
          [step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=False,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adam error handling differences for mismatched list sizes."
  )
  def test_fused_adam_mismatched_max_exp_avg_sqs_size(self):
    device = et.device()
    p1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    p2 = torch.tensor([2.0], dtype=torch.float32, device=device)
    g1 = torch.tensor([0.1], dtype=torch.float32, device=device)
    ea1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    eas1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    step1 = torch.tensor(1.0, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adam_(): expected max_exp_avg_sqs to have the same number of tensors as self, got 1""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten._fused_adam_.default(
          [p1, p2],
          [g1, g1],
          [ea1, ea1],
          [eas1, eas1],
          [eas1],
          [step1, step1],
          lr=0.001,
          beta1=0.9,
          beta2=0.999,
          weight_decay=0.01,
          eps=1e-8,
          amsgrad=True,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adagrad error handling differences for mismatched list sizes."
  )
  def test_fused_adagrad_mismatched_grads_size(self):
    device = et.device()
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [torch.tensor([0.1], dtype=torch.float32, device=device)]
    v = [
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
    ]
    s = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adagrad_(): expected grads to have size 2, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.default(
          p,
          g,
          v,
          s,
          lr=0.1,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adagrad error handling differences for mismatched list sizes."
  )
  def test_fused_adagrad_tensor_lr_mismatched_grads_size(self):
    device = et.device()
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [torch.tensor([0.1], dtype=torch.float32, device=device)]
    v = [
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
    ]
    s = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adagrad_(): expected grads to have size 2, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.tensor_lr(
          p,
          g,
          v,
          s,
          lr=lr,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adagrad error handling differences for mismatched list sizes."
  )
  def test_fused_adagrad_mismatched_state_sums_size(self):
    device = et.device()
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [
        torch.tensor([0.1], dtype=torch.float32, device=device),
        torch.tensor([0.2], dtype=torch.float32, device=device),
    ]
    v = [torch.tensor([0.0], dtype=torch.float32, device=device)]
    s = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adagrad_(): expected state_sums to have size 2, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.default(
          p,
          g,
          v,
          s,
          lr=0.1,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adagrad error handling differences for mismatched list sizes."
  )
  def test_fused_adagrad_tensor_lr_mismatched_state_sums_size(self):
    device = et.device()
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [
        torch.tensor([0.1], dtype=torch.float32, device=device),
        torch.tensor([0.2], dtype=torch.float32, device=device),
    ]
    v = [torch.tensor([0.0], dtype=torch.float32, device=device)]
    s = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adagrad_(): expected state_sums to have size 2, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.tensor_lr(
          p,
          g,
          v,
          s,
          lr=lr,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adagrad error handling differences for mismatched list sizes."
  )
  def test_fused_adagrad_mismatched_state_steps_size(self):
    device = et.device()
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [
        torch.tensor([0.1], dtype=torch.float32, device=device),
        torch.tensor([0.2], dtype=torch.float32, device=device),
    ]
    v = [
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
    ]
    s = [torch.tensor([1.0], dtype=torch.float32, device=device)]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adagrad_(): expected state_steps to have size 2, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.default(
          p,
          g,
          v,
          s,
          lr=0.1,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  @et.why_tpu_only(
      "GPU Adagrad error handling differences for mismatched list sizes."
  )
  def test_fused_adagrad_tensor_lr_mismatched_state_steps_size(self):
    device = et.device()
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [
        torch.tensor([0.1], dtype=torch.float32, device=device),
        torch.tensor([0.2], dtype=torch.float32, device=device),
    ]
    v = [
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
    ]
    s = [torch.tensor([1.0], dtype=torch.float32, device=device)]
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_adagrad_(): expected state_steps to have size 2, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_adagrad_.tensor_lr(
          p,
          g,
          v,
          s,
          lr=lr,
          lr_decay=0.0,
          weight_decay=0.01,
          eps=1e-10,
          maximize=False,
      )

  @et.why_tpu_only(
      "Verifying TPU-specific error formatting and type validation for"
      " fused_sgd"
  )
  def test_fused_sgd_default_int32_dtype(self):
    device = "tpu"
    p = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g = torch.tensor([1, 2], dtype=torch.int32, device=device)
    mb = torch.tensor([1, 2], dtype=torch.int32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_sgd_(): expected the input dtype to be floating-point, got int32""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.default(
          [p],
          [g],
          [mb],
          weight_decay=0.01,
          momentum=0.9,
          lr=0.1,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  @et.why_tpu_only(
      "Verifying TPU-specific error formatting and type validation for"
      " fused_sgd"
  )
  def test_fused_sgd_default_complex64_dtype(self):
    device = "tpu"
    p = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
    g = torch.tensor([0.1 + 0.1j], dtype=torch.complex64, device=device)
    mb = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_sgd_(): expected the input dtype to be floating-point, got complex64""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.default(
          [p],
          [g],
          [mb],
          weight_decay=0.01,
          momentum=0.9,
          lr=0.1,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  @et.why_tpu_only(
      "Verifying TPU-specific error formatting and type validation for"
      " fused_sgd"
  )
  def test_fused_sgd_tensor_lr_int32_dtype(self):
    device = "tpu"
    p = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g = torch.tensor([1, 2], dtype=torch.int32, device=device)
    mb = torch.tensor([1, 2], dtype=torch.int32, device=device)
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fused_sgd_(): expected the input dtype to be floating-point, got int32""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.tensor_lr(
          [p],
          [g],
          [mb],
          weight_decay=0.01,
          momentum=0.9,
          lr=lr,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  @et.why_tpu_only(
      "Verifying TPU-specific error formatting and list size validation for"
      " fused_sgd"
  )
  def test_fused_sgd_mismatched_grads_size(self):
    device = "tpu"
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [torch.tensor([0.1], dtype=torch.float32, device=device)]
    mb = [
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
    ]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_sgd_(): expected grads to have the same number of tensors as self, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.default(
          p,
          g,
          mb,
          weight_decay=0.01,
          momentum=0.9,
          lr=0.1,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  @et.why_tpu_only(
      "Verifying TPU-specific error formatting and list size validation for"
      " fused_sgd"
  )
  def test_fused_sgd_mismatched_momentum_buffer_size(self):
    device = "tpu"
    p = [
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([2.0], dtype=torch.float32, device=device),
    ]
    g = [
        torch.tensor([0.1], dtype=torch.float32, device=device),
        torch.tensor([0.2], dtype=torch.float32, device=device),
    ]
    mb = [torch.tensor([0.0], dtype=torch.float32, device=device)]

    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_sgd_(): expected momentum_buffer_list to have the same number of tensors as self, got 1""",
        message_reviewed_by="adivinpatel",
    ):
      torch.ops.aten._fused_sgd_.default(
          p,
          g,
          mb,
          weight_decay=0.01,
          momentum=0.9,
          lr=0.1,
          dampening=0.0,
          nesterov=False,
          maximize=False,
          is_first_step=False,
      )

  @et.why_tpu_only(
      "Custom op sparse_dense_matmul_grad_with_adagrad is TPU only."
  )
  def test_sparse_dense_matmul_grad_with_adagrad_invalid_accumulator_dim(self):
    device = et.device()
    row_pointers = torch.tensor([0, 1], dtype=torch.int32, device=device)
    embedding_ids = torch.tensor([0], dtype=torch.int32, device=device)
    sample_ids = torch.tensor([0], dtype=torch.int32, device=device)
    gains = torch.tensor([1.0], dtype=torch.float32, device=device)
    embedding_table = torch.ones(10, 8, dtype=torch.float32, device=device)
    accumulator_3d = torch.ones(10, 8, 1, dtype=torch.float32, device=device)
    activations_grad = torch.ones(1, 8, dtype=torch.float32, device=device)
    learning_rate = torch.tensor(0.01, dtype=torch.float32, device=device)
    epsilon = torch.tensor(1e-10, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_dense_matmul_grad_with_adagrad(): materialization failed with: Accumulator must be 1D (row-wise) or 2D (standard)""",
    ):
      torch.ops.tpu.sparse_dense_matmul_grad_with_adagrad(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          embedding_table,
          accumulator_3d,
          activations_grad,
          learning_rate,
          epsilon,
          device_batch_size=1,
          max_ids_per_partition=1,
          max_unique_ids_per_partition=1,
          computation_name="test_acc_3d",
      )

    accumulator_wrong_dim = torch.ones(
        10, 4, dtype=torch.float32, device=device
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_dense_matmul_grad_with_adagrad(): materialization failed with: Accumulator dimension 1 must match embedding dimension; expected 8, got 4""",
    ):
      torch.ops.tpu.sparse_dense_matmul_grad_with_adagrad(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          embedding_table,
          accumulator_wrong_dim,
          activations_grad,
          learning_rate,
          epsilon,
          device_batch_size=1,
          max_ids_per_partition=1,
          max_unique_ids_per_partition=1,
          computation_name="test_acc_dim",
      )

  @et.why_tpu_only(
      "Upstream CUDA crashes with illegal memory access without host check"
  )
  def test_fused_moving_avg_obs_fq_helper_invalid_quant_min_max(self):
    dev = et.device()
    self_t = torch.ones((2, 3), dtype=torch.float32, device=dev)
    observer_on = torch.tensor([1], dtype=torch.int32, device=dev)
    fake_quant_on = torch.tensor([1], dtype=torch.int32, device=dev)
    running_min = torch.empty((0,), dtype=torch.float32, device=dev)
    running_max = torch.empty((0,), dtype=torch.float32, device=dev)
    scale = torch.empty((0,), dtype=torch.float32, device=dev)
    zero_point = torch.empty((0,), dtype=torch.int32, device=dev)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""fused_moving_avg_obs_fq_helper(): expected quant_min to be strictly less than quant_max, got quant_min=10 and quant_max=0""",
    ):
      torch.ops.aten._fused_moving_avg_obs_fq_helper(
          self_t,
          observer_on,
          fake_quant_on,
          running_min,
          running_max,
          scale,
          zero_point,
          0.01,
          10,
          0,
          0,
          False,
          False,
      )

  @et.why_tpu_only(
      "sparse_dense_matmul_grad_with_adam is a TPU specific Custom Op"
  )
  def test_sparse_dense_matmul_grad_with_adam_velocity_shape(self):
    device = "tpu"
    vocab_size = 10
    embedding_dim = 8

    row_pointers = torch.zeros(vocab_size + 1, dtype=torch.int32, device=device)
    col_indices = torch.zeros(10, dtype=torch.int32, device=device)
    values = torch.ones(10, dtype=torch.float32, device=device)
    gains = torch.ones(10, dtype=torch.float32, device=device)

    embedding_table = torch.ones(
        vocab_size, embedding_dim, dtype=torch.float32, device=device
    )
    momentum = torch.ones(
        vocab_size, embedding_dim, dtype=torch.float32, device=device
    )

    activations_grad = torch.zeros(
        vocab_size, dtype=torch.float32, device=device
    )
    alpha_t = torch.tensor(0.1, dtype=torch.float32, device=device)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    device_batch_size = 1
    max_ids_per_partition = 1
    max_unique_ids_per_partition = 1
    computation_name = "test"

    # 3D velocity - should trigger Line 91
    velocity_3d = torch.ones(
        vocab_size, embedding_dim, 2, dtype=torch.float32, device=device
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_dense_matmul_grad_with_adam(): materialization failed with: expected velocity tensor to be 1D or 2D, got rank 3""",
        message_reviewed_by="songbaie",
    ):
      torch.ops.tpu.sparse_dense_matmul_grad_with_adam(
          row_pointers,
          col_indices,
          values,
          gains,
          embedding_table,
          momentum,
          velocity_3d,
          activations_grad,
          alpha_t,
          beta_1,
          beta_2,
          epsilon,
          device_batch_size,
          max_ids_per_partition,
          max_unique_ids_per_partition,
          computation_name,
      )

    # 2D velocity with wrong embedding dim - should trigger Line 94
    velocity_wrong_dim = torch.ones(
        vocab_size, embedding_dim + 1, dtype=torch.float32, device=device
    )

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_dense_matmul_grad_with_adam(): materialization failed with: expected velocity tensor dimension 1 to be 8, got 9""",
        message_reviewed_by="songbaie",
    ):
      torch.ops.tpu.sparse_dense_matmul_grad_with_adam(
          row_pointers,
          col_indices,
          values,
          gains,
          embedding_table,
          momentum,
          velocity_wrong_dim,
          activations_grad,
          alpha_t,
          beta_1,
          beta_2,
          epsilon,
          device_batch_size,
          max_ids_per_partition,
          max_unique_ids_per_partition,
          computation_name,
      )

  @et.why_tpu_only("TPU-specific C++ kernel argument validation")
  def test_scaled_mm_v2_invalid_contraction_dim_size(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_mm_v2(): expected contraction_dim list to contain exactly 2 elements, got 1""",
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
          contraction_dim=[1],
      )

  @et.why_tpu_only("TPU-specific C++ kernel argument validation")
  def test_scaled_mm_v2_non_standard_contraction_dim(self):
    (
        self_tpu,
        mat2_tpu,
        scale_a_tpu,
        recipe_a,
        swizzle_a,
        scale_b_tpu,
        recipe_b,
        swizzle_b,
    ) = et.get_scaled_mm_v2_default_inputs()
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""scaled_mm_v2(): expected contraction_dim to be [1, 0], got [0, 1]""",
        message_reviewed_by="wan",
    ):
      torch._scaled_mm_v2(
          self_tpu,
          mat2_tpu,
          scale_a_tpu,
          recipe_a,
          swizzle_a,
          scale_b_tpu,
          recipe_b,
          swizzle_b,
          None,
          None,
          contraction_dim=[0, 1],
      )


if __name__ == "__main__":
  absltest.main()
