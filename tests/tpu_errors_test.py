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


class TpuOnlyErrorTest(et.TpuOnlyErrorTestBase):
  """Tests error messages on TPU."""

  @et.why_tpu_only("TODO: support complex128 dtype on TPU")
  def test_fill_complex128(self):
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""fill_(): complex128 dtype is not supported""",
        message_reviewed_by="wan",
    ):
      t = torch.empty((2, 2), dtype=torch.complex128, device=et.device())
      t.fill_(1.0)

  @et.why_tpu_only(
      "TPU masked_scatter uses int32 source offsets. GPU supports larger"
      " offsets."
  )
  def test_masked_scatter_input_too_large(self):
    """Tests masked_scatter_ rejects a broadcasted input larger than int32."""
    t = torch.empty(2**31, device=et.device(), dtype=torch.float32)
    mask = torch.zeros(1, device=et.device(), dtype=torch.bool)
    source = torch.ones(1, device=et.device(), dtype=torch.float32)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""masked_scatter_(): expected the broadcasted input to contain at most 2147483647 elements because masked_scatter_ uses int32 source offsets, got 2147483648""",
    ):
      torch.masked_scatter(t, mask, source)

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
        tpu="""dynamic_broadcast(): expected shape list size to match static_shape size, got shape list size 1 and static_shape size 2""",
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
        tpu="""dynamic_broadcast(): expected is_dynamic size to match static_shape size, got is_dynamic size 1 and static_shape size 2""",
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
        tpu="""dynamic_broadcast(): expected shape tensor at index 0 to be a 0-D (scalar) tensor, got 1-D tensor""",
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
        tpu="""dynamic_broadcast(): expected shape to be a list of int32 tensors, got float32 tensor at index 0""",
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
        tpu="""dynamic_broadcast(): expected broadcast_dims size to match input rank, got broadcast_dims size 2 and input rank 1""",
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
        tpu="""dynamic_broadcast(): expected broadcast_dims to be in range [0, 2), got 2 for broadcast dim at index 0""",
    ):
      torch.ops.tpu.dynamic_broadcast(
          x, shape, broadcast_dims, static_shape, is_dynamic
      )

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_start_indices_size_mismatch(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [torch.tensor(0, device=device, dtype=torch.int32)]
    slice_sizes = [1, 2]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected start_indices list size to match input number of dimensions, got start_indices size 1 and input number of dimensions 2""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_slice_sizes_size_mismatch(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [
        torch.tensor(0, device=device, dtype=torch.int32),
        torch.tensor(0, device=device, dtype=torch.int32),
    ]
    slice_sizes = [1]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected slice_sizes size to match input number of dimensions, got slice_sizes size 1 and input number of dimensions 2""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_start_indices_not_0d(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.tensor(0, device=device, dtype=torch.int32),
    ]
    slice_sizes = [1, 2]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected start_indices tensor at index 0 to be a 0-D (scalar) tensor, got 1-D tensor""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_start_indices_not_int32_or_int64(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [
        torch.tensor(0.0, device=device, dtype=torch.float32),
        torch.tensor(0, device=device, dtype=torch.int32),
    ]
    slice_sizes = [1, 2]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected start_indices to be a list of int32 or int64 tensors, got float32 tensor at index 0""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_start_indices_mixed_dtypes(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [
        torch.tensor(0, device=device, dtype=torch.int32),
        torch.tensor(0, device=device, dtype=torch.int64),
    ]
    slice_sizes = [1, 2]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected all start_indices to have the same dtype, got int32 at index 0 but int64 at index 1""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_slice_sizes_negative(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [
        torch.tensor(0, device=device, dtype=torch.int32),
        torch.tensor(0, device=device, dtype=torch.int32),
    ]
    slice_sizes = [-1, 2]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected slice_sizes at index 0 to be in range [0, 2], got -1""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

  @et.why_tpu_only(
      "The op dynamic_slice is TPU only for internal use in torch.compile()."
  )
  def test_dynamic_slice_slice_sizes_exceeds_input_size(self):
    device = et.device()
    x = torch.ones((2, 3), device=device)
    start_indices = [
        torch.tensor(0, device=device, dtype=torch.int32),
        torch.tensor(0, device=device, dtype=torch.int32),
    ]
    slice_sizes = [5, 2]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""dynamic_slice(): expected slice_sizes at index 0 to be in range [0, 2], got 5""",
    ):
      torch.ops.tpu.dynamic_slice(x, start_indices, slice_sizes)

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

  @et.why_tpu_only("TODO: lift int32 index limit on TPU.")
  def test_max_pool2d_with_indices_input_too_large(self):
    """Tests max_pool2d_with_indices fails if input has > 2^31-1 elements."""
    # h * w = 2^15 * 2^16 = 2^31, exceeding the ui32 index limit 2^31 - 1
    t = torch.empty(1, 1, 2**15, 2**16, dtype=torch.float32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r""".*expected max_pool2d_with_indices input to have at most 2147483647 spatial elements due to int32 indices limitation \(for now\), got 2147483648.*""",
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
        tpu="""linalg_lu_solve(): expected b to have the same number of dimensions as lu (2), got 3""",
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
    kernel_key = "some_kernel_key"

    inputs = [torch.ones(2, device=et.device())]
    output_shapes = [torch.ones(2, device=et.device())]

    # TODO: Error eagerly, i.e. without having to call the op builder.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""custom_kernel(): materialization failed with: unknown custom kernel "not_registered_kernel_name" with key "some_kernel_key"; call torch_tpu._internal.pallas.tpu_torch_pallas.register_custom_kernel() to register the kernel before calling it""",
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
        tpu="""native_group_norm_backward(): expected grad_out and input to have the same dimensions, got grad_out size [125], input size [5, 5, 5]""",
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
        tpu="""mse_loss(): expected input and target to have the same dtype, got float32 and float64""",
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
        tpu="""threshold_backward(): expected the input dtype to be non-bool and non-complex, got bool""",
        message_reviewed_by="gunhyun",
    ):
      torch.ops.aten.threshold_backward(grad_output, self_tensor, 0.5)

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
        tpu="""dynamic_reshape(): expected shape list size to match static_shape size, got shape list size 1 and static_shape size 2""",
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
        tpu="""dynamic_reshape(): expected is_dynamic size to match static_shape size, got is_dynamic size 1 and static_shape size 2""",
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
        tpu="""dynamic_reshape(): expected shape tensor at index 0 to be a 0-D (scalar) tensor, got 1-D tensor""",
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
        tpu="""dynamic_reshape(): expected shape to be a list of int32 tensors, got float32 tensor at index 0""",
    ):
      torch.ops.tpu.dynamic_reshape(inp, shape, static_shape, is_dynamic)

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
        tpu="""get_or_compile_pad_module(): expected dimension indices and upper bounds to have the same size, got 1 and 2""",
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
        tpu="""get_or_compile_pad_module(): expected dimension index to be within bounds [0, 1], got 2 for input tensor 0 with shape [1, 4]""",
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
        tpu="""get_or_compile_pad_module(): expected upper bound to be >= the static shape's dimension size, got upper bound 2 for dimension 1 for input tensor 0 with shape [1, 4]""",
    ):
      tpu_torch_compile.get_or_compile_pad_module(
          tensor_info,
          bounds_list,
      )

  @et.why_tpu_only("Dynamic compilation is TPU-only.")
  def test_get_dynamic_pad_module_empty_shapes(self):
    expected = re.compile(
        r".*get_dynamic_pad_module\(\): dynamic_pad_module requires at least"
        r" one shape\..*"
    )
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
              "get_or_compile_slice_module(): expected target shapes and padded"
              " shapes to have the same size, got 1 and 2"
          ),
      ),
      dict(
          testcase_name="mismatched_input_scalar_types_size",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 8]],
          input_scalar_types=[torch.float32, torch.float32],
          expected_error_message=(
              "get_or_compile_slice_module(): expected target shapes and input"
              " scalar types to have the same size, got 1 and 2"
          ),
      ),
      dict(
          testcase_name="empty_target_shapes",
          target_shapes=[],
          padded_shapes=[],
          input_scalar_types=[],
          expected_error_message=(
              "get_or_compile_slice_module(): expected at least one target"
              " shape, got none"
          ),
      ),
      dict(
          testcase_name="mismatched_dimensions_size",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 8, 16]],
          input_scalar_types=[torch.float32],
          expected_error_message=(
              "get_or_compile_slice_module(): expected target shape and padded"
              " shape to have the same number of dimensions, got 2 and 3 for"
              " tensor index 0"
          ),
      ),
      dict(
          testcase_name="invalid_padded_shape_bound",
          target_shapes=[[1, 4]],
          padded_shapes=[[1, 2]],
          input_scalar_types=[torch.float32],
          expected_error_message=(
              "get_or_compile_slice_module(): expected padded shape dimension"
              " sizes to be >= target shape dimension"
              " sizes, got padded shape [1, 2] and target shape [1, 4] for"
              " tensor index 0"
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
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_COMPILE_FX_GRAPH):
      x = torch.ones(10, device="cpu").to(device=et.device())
      y = torch.ones(10, device="cpu").to(device=et.device())
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""execute(): expected output shapes to be specified for all outputs or none, got 2 output shapes for 1 output tensors""",
    ):
      tpu_torch_compile.execute(
          executable,
          [x, y],
          [
              tpu_torch_compile.OutputShape([5]),
              tpu_torch_compile.OutputShape([5]),
          ],
      )

  @et.why_tpu_only("For testing compiled mode argument count validation.")
  def test_execute_argument_count_mismatch(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_COMPILE_FX_GRAPH):
      x = torch.ones(10, device="cpu").to(device=et.device())
      z = x + x

    compile_result = tpu_torch_compile.traverse_and_compile([z], [x, x])
    with et.assert_raises_message(
        RuntimeError,
        tpu="""execute(): failed to prepare compiled mode arguments: number of argument tensors (1) does not match compiled argument indices (2)""",
    ):
      tpu_torch_compile.execute(compile_result.executable, [x])

  @et.why_tpu_only("TODO: investigate why this is TPU-only.")
  def test_execute_output_shapes_rank_mismatch(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_COMPILE_FX_GRAPH):
      x = torch.ones(10, device="cpu").to(device=et.device())
      y = torch.ones(10, device="cpu").to(device=et.device())
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""execute(): expected output shape number of dimensions to match the statically inferred dimensions, got output shape dimensions 2 and inferred dimensions 1 for output tensor 0""",
    ):
      tpu_torch_compile.execute(
          executable, [x, y], [tpu_torch_compile.OutputShape([5, 2])]
      )

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
        tpu="""execute(): expected output shape dimension not to exceed the statically inferred bound, got output shape [15] and inferred shape [10]""",
    ):
      tpu_torch_compile.execute(
          executable, [x, y], [tpu_torch_compile.OutputShape([15])]
      )

  @et.why_tpu_only("For testing TPU compile API validation.")
  def test_traverse_and_compile_invalid_layout_size(self):
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL):
      x = torch.ones(2, 3, device="cpu").to(device=et.device())
      z = x + x

    # 1 argument, but 2 layouts provided. Should fail.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""traverse_and_compile(): expected number of argument_layouts to match the number of argument_tensors, got number of argument_layouts 2 and number of argument_tensors 1""",
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
        tpu="""traverse_and_compile(): expected valid layout for argument 0, got layout [0] for shape [2, 3]""",
    ):
      tpu_torch_compile.traverse_and_compile([z], [x], argument_layouts=[[0]])

    # Out of bounds index: shape [2, 3], layout [2, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""traverse_and_compile(): expected valid layout for argument 0, got layout [2, 0] for shape [2, 3]""",
    ):
      tpu_torch_compile.traverse_and_compile(
          [z], [x], argument_layouts=[[2, 0]]
      )

    # Duplicate index: shape [2, 3], layout [0, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""traverse_and_compile(): expected valid layout for argument 0, got layout [0, 0] for shape [2, 3]""",
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
        tpu="""build_mlir(): failed to validate and reorder inputs: identified an argument that was not provided: 10xfloat32""",
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
        tpu="""compile_mlir(): expected valid layout for argument 0, got layout [0] for shape [2, 3]""",
    ):
      tpu_torch_compile.compile_mlir(mlir, argument_layouts=[[0]])

    # Out of bounds index: shape [2, 3], layout [2, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): expected valid layout for argument 0, got layout [2, 0] for shape [2, 3]""",
    ):
      tpu_torch_compile.compile_mlir(mlir, argument_layouts=[[2, 0]])

    # Duplicate index: shape [2, 3], layout [0, 0]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""compile_mlir(): expected valid layout for argument 0, got layout [0, 0] for shape [2, 3]""",
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

  # TODO(marcosyukio): check if this test passes on CUDA with CC >= 8.9.
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

  # TODO(marcosyukio): check if this test passes on CUDA with CC >= 8.9.
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

  # TODO(marcosyukio): check if this test passes on CUDA with CC >= 8.9.
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
        tpu="""sparse_dense_matmul_grad_with_adagrad(): materialization failed with: expected accumulator to be 1D (row-wise) or 2D (standard), got 3D""",
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
        tpu="""sparse_dense_matmul_grad_with_adagrad(): materialization failed with: expected accumulator dimension 1 to match embedding dimension (8), got 4""",
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

  @et.why_tpu_only("Custom op sparse_gather is TPU only.")
  def test_sparse_gather_invalid_row_pointers_dim(self):
    device = et.device()
    row_pointers_2d = torch.tensor([[0, 8]], dtype=torch.int32, device=device)
    indices = torch.tensor([0] * 8, dtype=torch.int32, device=device)
    operand = torch.ones(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather(): expected row_pointers to be a 1D tensor, got a 2D tensor of shape [1, 2]""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers_2d, indices, operand, 8)

  @et.why_tpu_only("Custom op sparse_gather is TPU only.")
  def test_sparse_gather_invalid_indices_dim(self):
    device = et.device()
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device=device)
    indices_2d = torch.tensor([[0] * 8], dtype=torch.int32, device=device)
    operand = torch.ones(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather(): expected indices to be a 1D tensor, got a 2D tensor of shape [1, 8]""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices_2d, operand, 8)

  @et.why_tpu_only("Custom op sparse_gather is TPU only.")
  def test_sparse_gather_invalid_operand_dim(self):
    device = et.device()
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device=device)
    indices = torch.tensor([0] * 8, dtype=torch.int32, device=device)
    operand_1d = torch.ones(10, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather(): expected operand to be a 2D tensor, got a 1D tensor of shape [10]""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices, operand_1d, 8)

  @et.why_tpu_only("Custom op sparse_gather is TPU only.")
  def test_sparse_gather_invalid_indices_length(self):
    device = et.device()
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device=device)
    indices_wrong_len = torch.tensor([0] * 7, dtype=torch.int32, device=device)
    operand = torch.ones(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather(): expected indices length to match the maximum number of non-zeroes, i.e. row_pointers length * maximum number of non-zeroes per row (2 * 8 = 16), got 7""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices_wrong_len, operand, 8)

  @et.why_tpu_only("Custom op sparse_gather is TPU only.")
  def test_sparse_gather_unsupported_device_without_sparse_core(self):
    device = et.device()
    row_pointers = torch.tensor([0, 8], dtype=torch.int32, device=device)
    indices = torch.tensor([0] * 16, dtype=torch.int32, device=device)
    operand = torch.ones(10, 8, dtype=torch.float32, device=device)

    # Valid inputs executed on a TPU device without SparseCore support
    # (e.g., TPU v5 lite) should raise an error.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather(): sparse_gather requires a TPU device with SparseCore support""",
    ):
      torch.ops.tpu.sparse_gather(row_pointers, indices, operand, 8)

  @et.why_tpu_only("Custom op sparse_gather_backward is TPU only.")
  def test_sparse_gather_backward_invalid_grad_output_dim(self):
    device = et.device()
    grad_output_1d = torch.ones(8, dtype=torch.float32, device=device)
    indices = torch.tensor([0] * 8, dtype=torch.int32, device=device)
    grad_operand = torch.zeros(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather_backward(): expected grad_output to be a 2D tensor, got a 1D tensor of shape [8]""",
    ):
      torch.ops.tpu.sparse_gather_backward(
          grad_output_1d, indices, grad_operand
      )

  @et.why_tpu_only("Custom op sparse_gather_backward is TPU only.")
  def test_sparse_gather_backward_invalid_indices_dim(self):
    device = et.device()
    grad_output = torch.ones(8, 8, dtype=torch.float32, device=device)
    indices_2d = torch.tensor([[0] * 8], dtype=torch.int32, device=device)
    grad_operand = torch.zeros(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather_backward(): expected indices to be a 1D tensor, got a 2D tensor of shape [1, 8]""",
    ):
      torch.ops.tpu.sparse_gather_backward(
          grad_output, indices_2d, grad_operand
      )

  @et.why_tpu_only("Custom op sparse_gather_backward is TPU only.")
  def test_sparse_gather_backward_invalid_grad_operand_dim(self):
    device = et.device()
    grad_output = torch.ones(8, 8, dtype=torch.float32, device=device)
    indices = torch.tensor([0] * 8, dtype=torch.int32, device=device)
    grad_operand_1d = torch.zeros(10, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather_backward(): expected grad_operand to be a 2D tensor, got a 1D tensor of shape [10]""",
    ):
      torch.ops.tpu.sparse_gather_backward(
          grad_output, indices, grad_operand_1d
      )

  @et.why_tpu_only("Custom op sparse_gather_backward is TPU only.")
  def test_sparse_gather_backward_invalid_batch_size(self):
    device = et.device()
    grad_output = torch.ones(7, 8, dtype=torch.float32, device=device)
    indices = torch.tensor([0] * 8, dtype=torch.int32, device=device)
    grad_operand = torch.zeros(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather_backward(): expected indices length (8) to match grad_output batch size, got 7""",
    ):
      torch.ops.tpu.sparse_gather_backward(grad_output, indices, grad_operand)

  @et.why_tpu_only("Custom op sparse_gather_backward is TPU only.")
  def test_sparse_gather_backward_invalid_embedding_dim(self):
    device = et.device()
    grad_output = torch.ones(8, 4, dtype=torch.float32, device=device)
    indices = torch.tensor([0] * 8, dtype=torch.int32, device=device)
    grad_operand = torch.zeros(10, 8, dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""sparse_gather_backward(): expected embedding dimension to match (8), got 4""",
    ):
      torch.ops.tpu.sparse_gather_backward(grad_output, indices, grad_operand)

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

  @et.why_tpu_only("Sparse operations fallback to SparsePrivateUse1 on TPU.")
  def test_sparse_ops_not_supported(self):
    indices = torch.tensor([[0, 1], [1, 2]], device="tpu")
    values = torch.tensor([1.0, 2.0], device="tpu")
    # TODO(b/540532086): Update the error message once issue is resolved.
    with et.assert_raises_message(
        RuntimeError,
        tpu="""aten::_sparse_coo_tensor_with_dims_and_tensors(): sparse operators are not supported on TPU yet""",
        message_reviewed_by="wan",
    ):
      torch.sparse_coo_tensor(indices, values, (2, 3), device="tpu")

  @et.why_tpu_only("PReLU weight shape broadcastability check on TPU.")
  def test_prelu_kernel_weight_shape_not_broadcastable(self):
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(1, 2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel(): expected weight tensor shape to be broadcastable to self shape [2, 3], got [1, 2, 3]""",
    ):
      torch.ops.aten._prelu_kernel(self_tensor, weight)

  @et.why_tpu_only("PReLU backward weight shape broadcastability check on TPU.")
  def test_prelu_kernel_backward_weight_shape_not_broadcastable(self):
    self_tensor = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    grad_output = torch.ones(2, 3, device=et.device(), dtype=torch.float32)
    weight = torch.ones(1, 2, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""prelu_kernel_backward(): expected weight tensor shape to be broadcastable to self shape [2, 3], got [1, 2, 3]""",
    ):
      torch.ops.aten._prelu_kernel_backward(grad_output, self_tensor, weight)

  @et.why_tpu_only("torch._scaled_grouped_mm is not supported on GPU.")
  def test_scaled_grouped_mm_fast_accum_unsupported(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    scale_a = torch.tensor(1.0, device=et.device())
    scale_b = torch.tensor(1.0, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_grouped_mm(): use_fast_accum=true is not supported yet""",
    ):
      torch._scaled_grouped_mm(
          a, b, scale_a, scale_b, offs=offs, use_fast_accum=True
      )

  @et.why_tpu_only("torch._scaled_grouped_mm is not supported on GPU.")
  def test_scaled_grouped_mm_invalid_scale_a_dim(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    scale_a = torch.randn(2, 2, device=et.device())
    scale_b = torch.tensor(1.0, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_grouped_mm(): expected scale_a to be 1D or scalar, got 2D""",
    ):
      torch._scaled_grouped_mm(a, b, scale_a, scale_b, offs=offs)

  @et.why_tpu_only("torch._scaled_grouped_mm is not supported on GPU.")
  def test_scaled_grouped_mm_invalid_scale_b_dim(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    scale_a = torch.tensor(1.0, device=et.device())
    scale_b = torch.randn(2, 2, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_grouped_mm(): expected scale_b to be 1D or scalar, got 2D""",
    ):
      torch._scaled_grouped_mm(a, b, scale_a, scale_b, offs=offs)

  @et.why_tpu_only("torch._scaled_grouped_mm is not supported on GPU.")
  def test_scaled_grouped_mm_invalid_bias_dim(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    scale_a = torch.tensor(1.0, device=et.device())
    scale_b = torch.tensor(1.0, device=et.device())
    bias = torch.randn(2, 2, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_grouped_mm(): expected bias to be 1D or scalar, got 2D""",
    ):
      torch._scaled_grouped_mm(a, b, scale_a, scale_b, bias=bias, offs=offs)

  @et.why_tpu_only("torch._scaled_grouped_mm is not supported on GPU.")
  def test_scaled_grouped_mm_invalid_scale_result_dim(self):
    a = torch.randn(3, 4, device=et.device())
    b = torch.randn(3, 4, 8, device=et.device())
    scale_a = torch.tensor(1.0, device=et.device())
    scale_b = torch.tensor(1.0, device=et.device())
    scale_result = torch.randn(2, 2, device=et.device())
    offs = torch.tensor([1, 2, 3], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_grouped_mm(): expected scale_result to be 1D or scalar, got 2D""",
    ):
      torch._scaled_grouped_mm(
          a, b, scale_a, scale_b, scale_result=scale_result, offs=offs
      )

  @et.why_tpu_only("GPU linear allows broadcastable 2-D bias.")
  def test_native_multi_head_attention_invalid_proj_bias_dim(self):
    query = torch.ones(2, 4, 8, device=et.device())
    key = torch.ones(2, 4, 8, device=et.device())
    value = torch.ones(2, 4, 8, device=et.device())
    qkv_weight = torch.ones(24, 8, device=et.device())
    qkv_bias = torch.ones(24, device=et.device())
    proj_weight = torch.ones(8, 8, device=et.device())
    proj_bias = torch.ones(8, 1, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected 1-D proj_bias, got 2-D tensor""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 2, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  @et.why_tpu_only("GPU doesn't check if num_head >= 0.")
  def test_native_multi_head_attention_non_positive_num_head(self):
    query = torch.ones(2, 4, 8, device=et.device())
    key = torch.ones(2, 4, 8, device=et.device())
    value = torch.ones(2, 4, 8, device=et.device())
    qkv_weight = torch.ones(24, 8, device=et.device())
    qkv_bias = torch.ones(24, device=et.device())
    proj_weight = torch.ones(8, 8, device=et.device())
    proj_bias = torch.ones(8, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""native_multi_head_attention(): expected num_head to be positive, got 0""",
    ):
      torch.ops.aten._native_multi_head_attention(
          query, key, value, 8, 0, qkv_weight, qkv_bias, proj_weight, proj_bias
      )

  @et.why_tpu_only("Optimization Barrier requires non-empty list input")
  def test_optimization_barrier_empty_list_fails(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(
            r"""There were no tensor arguments to this function(.|\n)*"""
        ),
    ):
      torch.ops.tpu.optimization_barrier([])

  @et.why_tpu_only(
      "Native PyTorch accepts 2-D proj_bias without throwing an error."
  )
  def test_transformer_encoder_layer_fwd_invalid_proj_bias_dim(self):
    embed_dim = 16
    num_heads = 4
    d_ff = 32
    device = et.device()
    src = torch.randn(2, 8, 16, device=device)
    qkv_w = torch.randn(3 * embed_dim, embed_dim, device=device)
    qkv_b = torch.randn(3 * embed_dim, device=device)
    proj_w = torch.randn(embed_dim, embed_dim, device=device)
    proj_b = torch.randn(16, 1, device=device)
    norm_w1 = torch.ones(embed_dim, device=device)
    norm_b1 = torch.zeros(embed_dim, device=device)
    norm_w2 = torch.ones(embed_dim, device=device)
    norm_b2 = torch.zeros(embed_dim, device=device)
    ffn_w1 = torch.randn(d_ff, embed_dim, device=device)
    ffn_b1 = torch.randn(d_ff, device=device)
    ffn_w2 = torch.randn(embed_dim, d_ff, device=device)
    ffn_b2 = torch.randn(embed_dim, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""transformer_encoder_layer_fwd(): expected 1-D proj_bias, got 2-D tensor""",
    ):
      torch._transformer_encoder_layer_fwd(
          src,
          embed_dim,
          num_heads,
          qkv_w,
          qkv_b,
          proj_w,
          proj_b,
          True,
          True,
          1e-5,
          norm_w1,
          norm_b1,
          norm_w2,
          norm_b2,
          ffn_w1,
          ffn_b1,
          ffn_w2,
          ffn_b2,
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_jagged_offsets_2d(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([[0, 2, 5]], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected offsets to have only 1 dimension, got 2""",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_jagged_offsets_not_int64(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected offsets to be of dtype int64, got int32""",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_jagged_offsets_non_zero_start(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([1, 3, 5], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected the first offset to be 0, got 1""",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_jagged_offsets_decreasing(self):
    values = torch.randn(5, 4, device=et.device())
    offsets = torch.tensor([0, 3, 2], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected offsets to be non-decreasing, got offsets[1] (3) > offsets[2] (2)""",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_jagged_to_padded_values_0d(self):
    values = torch.tensor(1.0, device=et.device())
    offsets = torch.tensor([0], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected values to have >= 1 dimensions, got 0""",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_jagged_to_padded_offsets_out_of_bounds(self):
    values = torch.randn(3, 4, device=et.device())
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""jagged_to_padded_dense_forward(): expected the last offset to be <= the number of values (3), got 5""",
    ):
      torch.ops.aten._jagged_to_padded_dense_forward(
          values, [offsets], [3], 0.0
      )

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_padded_to_jagged_total_l_mismatch(self):
    dense = torch.randn(2, 5, 4, device=et.device())
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""padded_dense_to_jagged_forward(): expected the last offset to match total_L (6), got 5""",
    ):
      torch.ops.aten._padded_dense_to_jagged_forward(dense, [offsets], 6)

  @et.why_tpu_only("Custom eager input validation in TorchTPU jagged kernels")
  def test_padded_to_jagged_segment_exceeds_max_length(self):
    dense = torch.randn(2, 4, 4, device=et.device())
    offsets = torch.tensor([0, 5, 7], dtype=torch.int64, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""padded_dense_to_jagged_forward(): expected all batch items to have length <= 4 (max length specified by padded input), got 5""",
    ):
      torch.ops.aten._padded_dense_to_jagged_forward(dense, [offsets], 7)

  @et.why_tpu_only("WindowTPU DMA striping configuration is TPU-specific")
  def test_window_num_stripes_non_positive(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected window_num_stripes to be > 0, got 0""",
        message_reviewed_by="wan",
    ):
      torch.tpu.window_num_stripes = 0

  @et.why_tpu_only("WindowTPU DMA striping configuration is TPU-specific")
  def test_window_num_stripes_negative(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected window_num_stripes to be > 0, got -1""",
        message_reviewed_by="wan",
    ):
      torch.tpu.window_num_stripes = -1

  @et.why_tpu_only("WindowTPU DMA striping configuration is TPU-specific")
  def test_window_stripe_chunk_mb_non_positive(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected window_stripe_chunk_mb to be > 0, got 0""",
        message_reviewed_by="wan",
    ):
      torch.tpu.window_stripe_chunk_mb = 0

  @et.why_tpu_only("WindowTPU DMA striping configuration is TPU-specific")
  def test_window_stripe_chunk_mb_negative(self):
    with et.assert_raises_message(
        RuntimeError,
        tpu="""expected window_stripe_chunk_mb to be > 0, got -1""",
        message_reviewed_by="wan",
    ):
      torch.tpu.window_stripe_chunk_mb = -1

  @et.why_tpu_only("Only default driver ('gesvd') is supported on TPU")
  def test_linalg_svd_unsupported_driver(self):
    t = torch.randn(5, 3, device=et.device(), dtype=torch.float32)
    with et.assert_raises_message(
        NotImplementedError,
        tpu="""linalg_svd(): expected default driver ('gesvd'), got 'gesvdj'""",
    ):
      torch.linalg.svd(t, driver="gesvdj")

  @parameterized.named_parameters(
      # Tests that non-2D lhs tensors are rejected.
      dict(
          testcase_name="invalid_lhs_shape",
          lhs_arg=(2,),
          grad_output_arg=(2, 2),
          group_sizes_arg=[1, 1],
          expected_error="""ragged_dot_weight_grad(): expected lhs to be 2D, got dim: 1""",
      ),
      # Tests that non-2D grad_output tensors are rejected.
      dict(
          testcase_name="invalid_grad_output_shape",
          lhs_arg=(3, 2),
          grad_output_arg=(2,),
          group_sizes_arg=[1, 1],
          expected_error="""ragged_dot_weight_grad(): expected grad_output to be 2D, got dim: 1""",
      ),
      # Tests that non-1D group_sizes tensors are rejected.
      dict(
          testcase_name="invalid_group_sizes_shape",
          lhs_arg=(3, 2),
          grad_output_arg=(3, 2),
          group_sizes_arg=[[1, 1]],
          expected_error="""ragged_dot_weight_grad(): expected group_sizes to be 1D, got dim: 2""",
      ),
      # Tests that mismatched batch dimensions between lhs and grad_output are rejected.
      dict(
          testcase_name="mismatched_batch_dim",
          lhs_arg=(3, 2),
          grad_output_arg=(4, 2),
          group_sizes_arg=[1, 1],
          expected_error="""ragged_dot_weight_grad(): expected lhs and grad_output to have the same batch dimension, got 3 vs 4""",
      ),
  )
  @et.why_tpu_only("torch.ops.tpu.ragged_dot_weight_grad is a TPU-specific op")
  def test_ragged_dot_weight_grad(
      self, lhs_arg, grad_output_arg, group_sizes_arg, expected_error
  ):
    """Verifies that tpu.ragged_dot_weight_grad enforces shape validation.

    Ensures that invalid input dimensions (lhs not 2D, grad_output not 2D,
    group_sizes not 1D) or batch size mismatches (lhs.size(0) !=
    grad_output.size(0))
    raise a RuntimeError with the expected descriptive error message.
    """
    lhs = torch.ones(*lhs_arg, dtype=torch.float32, device=et.device())
    grad_output = torch.ones(
        *grad_output_arg, dtype=torch.float32, device=et.device()
    )
    group_sizes = torch.tensor(
        group_sizes_arg, dtype=torch.int32, device=et.device()
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu=expected_error,
    ):
      torch.ops.tpu.ragged_dot_weight_grad(lhs, grad_output, group_sizes)

  def _default_lstm_input_args(self):
    device = et.device()
    x = torch.randn(10, 3, 16, device=device)
    h0 = torch.randn(1, 3, 32, device=device)
    c0 = torch.randn(1, 3, 32, device=device)
    w_ih = torch.randn(128, 16, device=device)
    w_hh = torch.randn(128, 32, device=device)
    b_ih = torch.randn(128, device=device)
    b_hh = torch.randn(128, device=device)
    params = [w_ih, w_hh, b_ih, b_hh]
    return x, [h0, c0], params

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_invalid_input_dim(self):
    _, hx, params = self._default_lstm_input_args()
    x_2d = torch.randn(10, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected input to be a 3D tensor, got 2D""",
    ):
      torch.ops.aten.lstm.input(
          x_2d, hx, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_invalid_hx_size(self):
    x, hx, params = self._default_lstm_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected hx to contain exactly 2 tensors (h_0, c_0), got 1""",
    ):
      torch.ops.aten.lstm.input(
          x, [hx[0]], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_zero_seq_len(self):
    _, hx, params = self._default_lstm_input_args()
    x_zero_seq = torch.randn(0, 3, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected sequence length to be larger than 0 in RNN, got 0""",
    ):
      torch.ops.aten.lstm.input(
          x_zero_seq, hx, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_zero_batch_size(self):
    _, hx, params = self._default_lstm_input_args()
    x_zero_batch = torch.randn(10, 0, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected batch size > 0 in RNN, got 0""",
    ):
      torch.ops.aten.lstm.input(
          x_zero_batch, hx, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_zero_num_layers(self):
    x, hx, params = self._default_lstm_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected num_layers > 0 in RNN, got 0""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, params, True, 0, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_invalid_dropout(self):
    x, hx, params = self._default_lstm_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected dropout to be in range [0, 1], got -0.5""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, params, True, 1, -0.5, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_h0_not_3d(self):
    x, hx, params = self._default_lstm_input_args()
    h0_2d = torch.randn(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected h_0 to be a 3D tensor [num_layers * num_directions, batch, out_size], got 2D""",
    ):
      torch.ops.aten.lstm.input(
          x, [h0_2d, hx[1]], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_c0_not_3d(self):
    x, hx, params = self._default_lstm_input_args()
    c0_2d = torch.randn(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected c_0 to be a 3D tensor [num_layers * num_directions, batch, hidden_size], got 2D""",
    ):
      torch.ops.aten.lstm.input(
          x, [hx[0], c0_2d], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_h0_layer_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    h0_bad_layers = torch.randn(2, 3, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected h_0 size(0) to match num_layers * num_directions (1), got 2""",
    ):
      torch.ops.aten.lstm.input(
          x, [h0_bad_layers, hx[1]], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_h0_batch_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    h0_bad_batch = torch.randn(1, 4, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected h_0 size(1) to match batch (3), got 4""",
    ):
      torch.ops.aten.lstm.input(
          x, [h0_bad_batch, hx[1]], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_c0_layer_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    c0_bad_layers = torch.randn(2, 3, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected c_0 size(0) to match num_layers * num_directions (1), got 2""",
    ):
      torch.ops.aten.lstm.input(
          x, [hx[0], c0_bad_layers], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_c0_batch_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    c0_bad_batch = torch.randn(1, 4, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected c_0 size(1) to match batch (3), got 4""",
    ):
      torch.ops.aten.lstm.input(
          x, [hx[0], c0_bad_batch], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_h0_out_size_exceeds_c0(self):
    x, _, params = self._default_lstm_input_args()
    h0_large_out = torch.randn(1, 3, 64, device=et.device())
    c0 = torch.randn(1, 3, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): h_0 size(2) (64) cannot exceed c_0 size(2) (32)""",
    ):
      torch.ops.aten.lstm.input(
          x, [h0_large_out, c0], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_params_count_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): expected 4 parameters, got 3""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, params[:3], True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_w_ih_shape_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    w_ih_bad = torch.randn(127, 16, device=et.device())
    bad_params = [w_ih_bad, params[1], params[2], params[3]]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): w_ih layer 0 dir 0 expected shape [128, 16], got [127, 16]""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_w_hh_shape_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    w_hh_bad = torch.randn(128, 31, device=et.device())
    bad_params = [params[0], w_hh_bad, params[2], params[3]]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): w_hh layer 0 dir 0 expected shape [128, 32], got [128, 31]""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_b_ih_shape_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    b_ih_bad = torch.randn(127, device=et.device())
    bad_params = [params[0], params[1], b_ih_bad, params[3]]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): b_ih layer 0 dir 0 expected shape [128], got [127]""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_b_hh_shape_mismatch(self):
    x, hx, params = self._default_lstm_input_args()
    b_hh_bad = torch.randn(127, device=et.device())
    bad_params = [params[0], params[1], params[2], b_hh_bad]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): b_hh layer 0 dir 0 expected shape [128], got [127]""",
    ):
      torch.ops.aten.lstm.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_w_hr_shape_mismatch_with_bias(self):
    device = et.device()
    x = torch.randn(10, 3, 16, device=device)
    h0 = torch.randn(1, 3, 16, device=device)
    c0 = torch.randn(1, 3, 32, device=device)
    w_ih = torch.randn(128, 16, device=device)
    w_hh = torch.randn(128, 16, device=device)
    b_ih = torch.randn(128, device=device)
    b_hh = torch.randn(128, device=device)
    w_hr_bad = torch.randn(15, 32, device=device)
    params = [w_ih, w_hh, b_ih, b_hh, w_hr_bad]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): w_hr layer 0 dir 0 expected shape [16, 32], got [15, 32]""",
    ):
      torch.ops.aten.lstm.input(
          x, [h0, c0], params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.lstm.input error validations on TPU")
  def test_lstm_input_w_hr_shape_mismatch_without_bias(self):
    device = et.device()
    x = torch.randn(10, 3, 16, device=device)
    h0 = torch.randn(1, 3, 16, device=device)
    c0 = torch.randn(1, 3, 32, device=device)
    w_ih = torch.randn(128, 16, device=device)
    w_hh = torch.randn(128, 16, device=device)
    w_hr_bad = torch.randn(15, 32, device=device)
    params = [w_ih, w_hh, w_hr_bad]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""lstm(): w_hr layer 0 dir 0 expected shape [16, 32], got [15, 32]""",
    ):
      torch.ops.aten.lstm.input(
          x, [h0, c0], params, False, 1, 0.0, False, False, False
      )

  def _default_gru_input_args(self):
    device = et.device()
    x = torch.randn(10, 3, 16, device=device)
    h0 = torch.randn(1, 3, 32, device=device)
    w_ih = torch.randn(96, 16, device=device)
    w_hh = torch.randn(96, 32, device=device)
    b_ih = torch.randn(96, device=device)
    b_hh = torch.randn(96, device=device)
    params = [w_ih, w_hh, b_ih, b_hh]
    return x, h0, params

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_invalid_input_dim(self):
    """Verifies that non-3D input tensor raises RuntimeError.

    What is being tested:
      - Passing a 2D tensor of shape [10, 3] as input x instead of expected 3D
      tensor.

    Expected result:
      - RuntimeError: 'gru(): expected input to be a 3D tensor, got 2D'
    """
    _, hx, params = self._default_gru_input_args()
    x_2d = torch.randn(10, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected input to be a 3D tensor, got 2D""",
    ):
      torch.ops.aten.gru.input(
          x_2d, hx, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_zero_seq_len(self):
    """Verifies that sequence length of zero raises RuntimeError.

    What is being tested:
      - Passing an input tensor with seq_len=0 (shape [0, 3, 16]).

    Expected result:
      - RuntimeError: 'gru(): expected sequence length to be larger than 0 in
      RNN, got 0'
    """
    _, hx, params = self._default_gru_input_args()
    x_zero_seq = torch.randn(0, 3, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected sequence length to be larger than 0 in RNN, got 0""",
    ):
      torch.ops.aten.gru.input(
          x_zero_seq, hx, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_zero_batch_size(self):
    """Verifies that batch size of zero raises RuntimeError.

    What is being tested:
      - Passing an input tensor with batch_size=0 (shape [10, 0, 16]).

    Expected result:
      - RuntimeError: 'gru(): expected batch size > 0 in RNN, got 0'
    """
    _, hx, params = self._default_gru_input_args()
    x_zero_batch = torch.randn(10, 0, 16, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected batch size > 0 in RNN, got 0""",
    ):
      torch.ops.aten.gru.input(
          x_zero_batch, hx, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_zero_num_layers(self):
    """Verifies that num_layers=0 raises RuntimeError.

    What is being tested:
      - Passing num_layers=0 to aten.gru.input.

    Expected result:
      - RuntimeError: 'gru(): expected num_layers > 0 in RNN, got 0'
    """
    x, hx, params = self._default_gru_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected num_layers > 0 in RNN, got 0""",
    ):
      torch.ops.aten.gru.input(x, hx, params, True, 0, 0.0, False, False, False)

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_invalid_dropout(self):
    """Verifies that dropout probability out of range [0, 1] raises RuntimeError.

    What is being tested:
      - Passing dropout=2.0 (outside [0, 1]).

    Expected result:
      - RuntimeError: 'gru(): expected dropout to be in range [0, 1], got 2'
    """
    x, hx, params = self._default_gru_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected dropout to be in range [0, 1], got 2""",
    ):
      torch.ops.aten.gru.input(x, hx, params, True, 1, 2.0, False, False, False)

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_hx_not_3d(self):
    """Verifies that non-3D initial hidden state hx raises RuntimeError.

    What is being tested:
      - Passing a 2D tensor of shape [1, 3] as hx.

    Expected result:
      - RuntimeError: 'gru(): expected hx to be a 3D tensor [num_layers *
      num_directions, batch, hidden_size], got 2D'
    """
    x, _, params = self._default_gru_input_args()
    hx_2d = torch.randn(1, 3, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected hx to be a 3D tensor [num_layers * num_directions, batch, hidden_size], got 2D""",
    ):
      torch.ops.aten.gru.input(
          x, hx_2d, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_hx_layer_mismatch(self):
    """Verifies that layer count mismatch between hx and num_layers raises RuntimeError.

    What is being tested:
      - Passing hx with size(0)=2 when num_layers * num_directions = 1.

    Expected result:
      - RuntimeError: 'gru(): expected hx size(0) to match num_layers *
      num_directions (1), got 2'
    """
    x, _, params = self._default_gru_input_args()
    hx_bad_layers = torch.randn(2, 3, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected hx size(0) to match num_layers * num_directions (1), got 2""",
    ):
      torch.ops.aten.gru.input(
          x, hx_bad_layers, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_hx_batch_mismatch(self):
    """Verifies that batch mismatch between hx and input x raises RuntimeError.

    What is being tested:
      - Passing hx with batch=4 when input x has batch=3.

    Expected result:
      - RuntimeError: 'gru(): expected hx size(1) to match batch (3), got 4'
    """
    x, _, params = self._default_gru_input_args()
    hx_bad_batch = torch.randn(1, 4, 32, device=et.device())
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected hx size(1) to match batch (3), got 4""",
    ):
      torch.ops.aten.gru.input(
          x, hx_bad_batch, params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_params_count_mismatch(self):
    """Verifies that parameter count mismatch raises RuntimeError.

    What is being tested:
      - Providing 3 parameter tensors instead of 4 for a single-layer biased
      GRU.

    Expected result:
      - RuntimeError: 'gru(): expected 4 parameters, got 3'
    """
    x, hx, params = self._default_gru_input_args()
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): expected 4 parameters, got 3""",
    ):
      torch.ops.aten.gru.input(
          x, hx, params[:3], True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_w_ih_shape_mismatch(self):
    """Verifies that mismatched w_ih shape raises RuntimeError.

    What is being tested:
      - Providing w_ih with shape [95, 16] instead of expected [96, 16] (3 *
      hidden).

    Expected result:
      - RuntimeError: 'gru(): w_ih layer 0 dir 0 expected shape [96, 16], got
      [95, 16]'
    """
    x, hx, params = self._default_gru_input_args()
    w_ih_bad = torch.randn(95, 16, device=et.device())
    bad_params = [w_ih_bad, params[1], params[2], params[3]]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): w_ih layer 0 dir 0 expected shape [96, 16], got [95, 16]""",
    ):
      torch.ops.aten.gru.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_w_hh_shape_mismatch(self):
    """Verifies that mismatched w_hh shape raises RuntimeError.

    What is being tested:
      - Providing w_hh with shape [96, 31] instead of expected [96, 32].

    Expected result:
      - RuntimeError: 'gru(): w_hh layer 0 dir 0 expected shape [96, 32], got
      [96, 31]'
    """
    x, hx, params = self._default_gru_input_args()
    w_hh_bad = torch.randn(96, 31, device=et.device())
    bad_params = [params[0], w_hh_bad, params[2], params[3]]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): w_hh layer 0 dir 0 expected shape [96, 32], got [96, 31]""",
    ):
      torch.ops.aten.gru.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_b_ih_shape_mismatch(self):
    """Verifies that mismatched b_ih shape raises RuntimeError.

    What is being tested:
      - Providing b_ih with shape [95] instead of expected [96].

    Expected result:
      - RuntimeError: 'gru(): b_ih layer 0 dir 0 expected shape [96], got [95]'
    """
    x, hx, params = self._default_gru_input_args()
    b_ih_bad = torch.randn(95, device=et.device())
    bad_params = [params[0], params[1], b_ih_bad, params[3]]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): b_ih layer 0 dir 0 expected shape [96], got [95]""",
    ):
      torch.ops.aten.gru.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )

  @et.why_tpu_only("torch.ops.aten.gru.input error validations on TPU")
  def test_gru_input_b_hh_shape_mismatch(self):
    """Verifies that mismatched b_hh shape raises RuntimeError.

    What is being tested:
      - Providing b_hh with shape [95] instead of expected [96].

    Expected result:
      - RuntimeError: 'gru(): b_hh layer 0 dir 0 expected shape [96], got [95]'
    """
    x, hx, params = self._default_gru_input_args()
    b_hh_bad = torch.randn(95, device=et.device())
    bad_params = [params[0], params[1], params[2], b_hh_bad]
    with et.assert_raises_message(
        RuntimeError,
        tpu="""gru(): b_hh layer 0 dir 0 expected shape [96], got [95]""",
    ):
      torch.ops.aten.gru.input(
          x, hx, bad_params, True, 1, 0.0, False, False, False
      )


if __name__ == "__main__":
  absltest.main()
