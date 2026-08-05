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

"""Tests that view ops return layout information that is consistent with CPU."""

from collections.abc import Sequence
import dataclasses
import time
from typing import Callable, Tuple

from absl.testing import absltest
import torch
from torch_tpu._internal import compile as compile_lib
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def setUpModule():
  # Initialize seed with the current system time
  torch.manual_seed(time.time())
  # Uncomment to set a specific seed value.
  #  torch.manual_seed(1234)
  print(f"Torch initial seed: {torch.initial_seed()}")


@dataclasses.dataclass(frozen=True)
class TensorLayout:
  """Represents the view-specific layout of a strided tensor."""

  strides: tuple[int, ...]
  dtype: torch.dtype
  # Offset is returned in terms of elements, NOT bytes.
  storage_offset: int
  storage_nbytes: int

  @classmethod
  def from_tensor(cls, tensor: torch.Tensor) -> "TensorLayout":
    """Returns the layout of a strided tensor, with validity checks."""
    shape = tensor.shape
    print(f"tensor.shape: {shape}")
    strides = tensor.stride()
    print(f"tensor.stride(): {strides}")
    if len(shape) != len(strides):
      raise ValueError(
          f"Tensor has different rank in sizes and strides: {shape} vs"
          f" {strides}"
      )

    storage_offset = tensor.storage_offset()
    print(f"tensor.storage_offset(): {storage_offset}")
    if storage_offset < 0:
      raise ValueError(f"Tensor has negative storage offset: {storage_offset}")

    # NB: "storage()"" returns a TypedStorage, which is deprecated
    # https://docs.pytorch.org/docs/stable/storage.html#legacy-typed-storage
    # untyped_storage() returns a Storage (untyped) instead.
    storage_nbytes = tensor.untyped_storage().nbytes()
    if storage_nbytes < 0:
      raise ValueError(f"Tensor has negative storage nbytes: {storage_nbytes}")

    # Check that addressable elements are within the storage's bytes boundaries.
    elements_span = storage_offset
    if not strides:
      # Scalars have one element
      elements_span += 1
    else:
      for size, stride in zip(tensor.shape, strides):
        if size < 0:
          raise ValueError(f"Tensor has negative size: {size}")
        if size == 0:
          elements_span = storage_offset
          break
        elements_span += (size - 1) * stride
    bytes_span = elements_span * tensor.dtype.itemsize
    if bytes_span > storage_nbytes:
      raise ValueError(
          "Tensor spans wider than storage bytes allow: "
          f"{bytes_span} > {storage_nbytes}"
      )

    return cls(
        strides=strides,
        dtype=tensor.dtype,
        storage_offset=storage_offset,
        storage_nbytes=storage_nbytes,
    )


# Base class with functionality for testing.
class LayoutTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.tpu_device = torch.device("tpu")

  def _assert_same_layout(
      self, tensor_cpu: torch.Tensor, tensor_tpu: torch.Tensor
  ):
    layout_cpu = TensorLayout.from_tensor(tensor_cpu)
    print(f"layout_cpu: {layout_cpu}")
    layout_tpu = TensorLayout.from_tensor(tensor_tpu)
    print(f"layout_tpu: {layout_tpu}")
    self.assertEqual(layout_cpu.dtype, layout_tpu.dtype)
    self.assertEqual(layout_cpu.strides, layout_tpu.strides)
    self.assertEqual(layout_cpu.storage_offset, layout_tpu.storage_offset)
    self.assertEqual(layout_cpu.storage_nbytes, layout_tpu.storage_nbytes)

  def _assert_same_layout_tpu_vs_cpu(
      self, tensor_from_device: Callable[[torch.device], torch.Tensor]
  ):
    cpu_result = tensor_from_device(torch.device("cpu"))
    tpu_result = tensor_from_device(self.tpu_device)
    if isinstance(cpu_result, Tuple):
      for cpu_result_i, tpu_result_i in zip(cpu_result, tpu_result):
        self._assert_same_layout(cpu_result_i, tpu_result_i)

  def _assert_same_mutation_from_base(
      self,
      base_shape: tuple[int, ...],
      view_operation: Callable[
          [torch.Tensor], torch.Tensor | Sequence[torch.Tensor]
      ],
  ):
    # Ensure every element in the base tensor is unique using arange
    numel = 1
    for dim in base_shape:
      numel *= dim
    base_cpu = (
        torch.arange(numel, dtype=torch.int32, device=torch.device("cpu"))
        .view(*base_shape)
        .clone()
    )
    base_tpu = base_cpu.to(self.tpu_device)

    view_cpu = view_operation(base_cpu)
    view_tpu = view_operation(base_tpu)

    # Mutate the base tensors in a way that still preserves element uniqueness.
    base_cpu.mul_(-1)
    base_tpu.mul_(-1)

    # Check that the layouts and values match after the mutation.
    if isinstance(view_cpu, Sequence):
      self.assertLen(view_tpu, len(view_cpu))
      for view_cpu_i, view_tpu_i in zip(view_cpu, view_tpu):
        self._assert_same_layout(view_cpu_i, view_tpu_i)
        utils.assert_close(view_tpu_i.cpu(), view_cpu_i)
    else:
      self._assert_same_layout(view_cpu, view_tpu)
      utils.assert_close(view_tpu.cpu(), view_cpu)

  def _assert_no_mutation_from_base(
      self,
      base_shape: tuple[int, ...],
      non_view_operation: Callable[[torch.Tensor], torch.Tensor],
  ):
    # Use zeros to easily identify if any elements are mutated
    numel = 1
    for dim in base_shape:
      numel *= dim
    base_cpu = (
        torch.zeros(numel, dtype=torch.int32, device=torch.device("cpu"))
        .view(*base_shape)
        .clone()
    )
    base_tpu = base_cpu.to(self.tpu_device)

    non_view_cpu = non_view_operation(base_cpu)
    non_view_tpu = non_view_operation(base_tpu)

    # Mutate the base tensors and check that the non-views don't change.
    base_cpu.add_(1)
    base_tpu.add_(1)

    expected = torch.zeros_like(non_view_cpu)
    utils.assert_close(non_view_cpu, expected)
    utils.assert_close(non_view_tpu.cpu(), expected)

  def _assert_same_mutation_from_view(
      self,
      base_shape: tuple[int, ...],
      view_operation: Callable[
          [torch.Tensor], torch.Tensor | Sequence[torch.Tensor]
      ],
  ):
    # Ensure every element in the base tensor is unique using arange
    numel = 1
    for dim in base_shape:
      numel *= dim
    base_cpu = (
        torch.arange(numel, dtype=torch.int32, device=torch.device("cpu"))
        .view(*base_shape)
        .clone()
    )
    base_tpu = base_cpu.to(self.tpu_device)

    view_cpu = view_operation(base_cpu)
    view_tpu = view_operation(base_tpu)

    # Mutate the view tensors in a way that still preserves element uniqueness.
    if isinstance(view_cpu, Sequence):
      self.assertLen(view_tpu, len(view_cpu))
      for view_cpu_i, view_tpu_i in zip(view_cpu, view_tpu):
        view_cpu_i.mul_(-1)
        view_tpu_i.mul_(-1)

    # Check that the base layouts and values match after the mutation.
    self._assert_same_layout(base_cpu, base_tpu)
    utils.assert_close(base_tpu.cpu(), base_cpu)


# Tests aten view ops t, t_, transpose.int, transpose_, permute.
# These are all implemented by stablehlo.transpose.
class TransposeTest(LayoutTest):

  def test_t(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, device=device).t()
    )

  def test_t_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3),
        lambda tensor: tensor.t(),
    )

  def test_t_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3),
        lambda tensor: tensor.t(),
    )

  def test_t_inplace(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, device=device).t_()
    )

  def test_transpose(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, device=device).transpose(0, 2)
    )

  def test_transpose_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4),
        lambda tensor: tensor.transpose(0, 2),
    )

  def test_transpose_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4),
        lambda tensor: tensor.transpose(0, 2),
    )

  def test_transpose_inplace(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, device=device).transpose_(0, 2)
    )

  def test_permute(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).permute(
            3, 2, 0, 1
        )
    )

  def test_permute_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.permute(3, 2, 0, 1),
    )

  def test_permute_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        lambda tensor: tensor.permute(3, 2, 0, 1),
    )


class DetachTest(LayoutTest):

  def test_detached_tensor(self):
    """Detaching a tensor does not change their layout."""
    tensor = torch.randn(2, 3, device=self.tpu_device)
    detached = tensor.detach()
    self._assert_same_layout(tensor, detached)

  def test_detached_tensor_no_grad(self):
    """Detached tensors have requires_grad=False."""
    x = torch.randn(3, 2, requires_grad=True, device=self.tpu_device)
    y = x.detach()
    self.assertTrue(x.requires_grad)
    self.assertFalse(y.requires_grad)

  def test_inplace_detach_no_grad(self):
    x = torch.randn(3, 2, device=self.tpu_device, requires_grad=True)
    y = x * 2.0
    self.assertTrue(y.requires_grad)
    self.assertIsNotNone(y.grad_fn)
    y.detach_()
    self.assertFalse(y.requires_grad)
    self.assertIsNone(y.grad_fn)

  def test_inplace_detach_fails_with_view(self):
    x = torch.randn(3, 2, device=self.tpu_device, requires_grad=True)
    y = x.view(-1)
    with self.assertRaises(RuntimeError) as cm:
      y.detach_()
      y.to("cpu")

    self.assertIn("Can't detach views in-place", str(cm.exception))

  def test_detached_tensor_skips_grad(self):
    """A detached tensor is not included in the backwards pass."""
    x = torch.randn(3, 2, device=self.tpu_device, requires_grad=True)
    # First check the typical case -- the gradient of x + x is 2.
    (x + x).sum().backward()
    expected_grad = torch.full_like(x, fill_value=2.0, device="cpu")
    utils.assert_close(x.grad.to("cpu"), expected_grad)

    # The gradient of x + x.detach() is 1 because the detached tensor
    # does not contribute.
    x = torch.randn(3, 2, device=self.tpu_device, requires_grad=True)
    (x + x.detach()).sum().backward()
    expected_grad = torch.full_like(x, fill_value=1.0, device="cpu")
    utils.assert_close(x.grad.to("cpu"), expected_grad)


class FlattenTest(LayoutTest):

  def test_flatten_middle_dims(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).flatten(1, 2)
    )

  def test_flatten_suffix_dims(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).flatten(2, 3)
    )

  def test_flatten_prefix_dims(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).flatten(0, 1)
    )

  def test_flatten_all_dims(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).flatten()
    )

  def test_flatten_view_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.flatten(),
    )

  def test_flatten_view_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        lambda tensor: tensor.flatten(),
    )

  def test_flatten_with_copy(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device)
        .permute(1, 0, 3, 2)
        .flatten(0, 3)
    )

  def test_flatten_with_copy_no_mutation_from_base(self):
    self._assert_no_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.permute(1, 0, 3, 2).flatten(0, 3),
    )


class ReshapeTest(LayoutTest):

  def test_reshape_tensor_to_scalar(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(1, 1, 1, 1, device=device).reshape(shape=[])
    )

  def test_reshape_as_tensor_to_scalar(self):
    def _test_fn(device):
      a = torch.randn(1, 1, 1, 1, device=device)
      b = torch.scalar_tensor(1.0, dtype=torch.float32, device=device)
      return a.reshape_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_reshape_scalar_to_tensor(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.scalar_tensor(
            1.0, dtype=torch.float32, device=device
        ).reshape(1, 1, 1, 1)
    )

  def test_reshape_as_scalar_to_tensor(self):
    def _test_fn(device):
      a = torch.scalar_tensor(1.0, dtype=torch.float32, device=device)
      b = torch.randn(1, 1, 1, 1, device=device)
      return a.reshape_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_reshape_no_copy(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).reshape(6, 20)
    )

  def test_reshape_no_copy_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.reshape(6, 20),
    )

  def test_reshape_no_copy_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        lambda tensor: tensor.reshape(6, 20),
    )

  def test_reshape_as_no_copy(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device)
      b = torch.randn(6, 20, device=device)
      return a.reshape_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_reshape_with_copy(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device)
        .permute(1, 0, 3, 2)
        .reshape(3, 10, 4)
    )

  def test_reshape_with_copy_no_mutation_from_base(self):
    self._assert_no_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.permute(1, 0, 3, 2).reshape(3, 10, 4),
    )

  def test_reshape_as_with_copy(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device)
      b = torch.randn(3, 10, 4, device=device)
      return a.permute(1, 0, 3, 2).reshape_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)


# Specifically testing view() and view_as() here.
class ViewTest(LayoutTest):

  def test_view_tensor_to_scalar(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(
            1, 1, 1, 1, dtype=torch.float32, device=device
        ).view(size=tuple())
    )

  def test_view_as_tensor_to_scalar(self):
    def _test_fn(device):
      a = torch.randn(1, 1, 1, 1, device=device)
      b = torch.scalar_tensor(1.0, dtype=torch.float32, device=device)
      return a.view_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_view_scalar_to_tensor(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.scalar_tensor(
            1.0, dtype=torch.float32, device=device
        ).view(1, 1, 1, 1)
    )

  def test_view_as_scalar_to_tensor(self):
    def _test_fn(device):
      a = torch.scalar_tensor(1.0, dtype=torch.float32, device=device)
      b = torch.randn(1, 1, 1, 1, device=device)
      return a.view_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_view(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).view(6, 20)
    )

  def test_view_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.view(6, 20),
    )

  def test_view_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        lambda tensor: tensor.view(6, 20),
    )

  def test_view_as(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device)
      b = torch.randn(6, 20, device=device)
      return a.reshape_as(b)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)


class SliceTest(LayoutTest):

  def test_slice_no_offset_or_step(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[:5]
    )

  def test_slice_offset(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[1:6]
    )

  def test_slice_step(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[::2]
    )

  def test_slice_middle_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 4, 2, device=device)[:, 1:3, :]
    )

  def test_slice_no_op(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, device=device)[0:2:1, 0:3:1, 0:4:1]
    )

  def test_slice_start_equals_end(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[5:5]
    )

  def test_slice_start_greater_than_end(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[5:4]
    )

  def test_slice_negative_start(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[-5:]
    )

  def test_slice_negative_end(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[:-5]
    )

  def test_slice_negative_start_and_end(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, device=device)[-5:-3]
    )

  def test_slice_mutate_base(self):
    self._assert_same_mutation_from_base(
        (10, 10),
        lambda tensor: tensor[1:99, ::2],
    )

  def test_slice_mutate_view(self):
    self._assert_same_mutation_from_view(
        (10, 10),
        lambda tensor: tensor[1:99, ::2],
    )


class SelectTest(LayoutTest, seed_test_utils.RepeatableTest):
  # Slice performs a select in cases where it changes the number of dimensions.

  def test_select_as_slice(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn((10, 10), device=device)[:, 1]
    )

  def test_select_standard_case(self):
    self.do_test_select(0, 1)

  def test_select_negative_index(self):
    self.do_test_select(0, -1)

  def test_select_negative_dim(self):
    self.do_test_select(-1, 0)

  def test_select_negative_dim_index(self):
    self.do_test_select(-1, -1)

  def do_test_select(self, index1, index2):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn((10, 10), device=device).select(
            index1, index2
        )
    )

  def test_select_to_scalar(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(3, device=device).select(0, 0)
    )

  def test_select_mutate_base(self):
    self._assert_same_mutation_from_base((3, 3), lambda tensor: tensor[:, 1])

  def test_select_mutate_view(self):
    self._assert_same_mutation_from_view((3, 3), lambda tensor: tensor[:, 1])


class ResizeTest(LayoutTest):

  def test_resize_no_op(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, device=device)
      a.resize_(2, 3, 4)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_resize_as_view(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device)
      a.resize_(6, 20)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_resize_as_view_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.resize_(6, 20),
    )

  def test_resize_as_view_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        lambda tensor: tensor.resize_(6, 20),
    )

  def test_resize_shrink(self):
    def _test_fn(device):
      a = torch.randn(100, device=device)
      a.resize_(99)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_aminmax_out_resize(self):
    def _test_fn(device):
      x = torch.randn(4, 4, device=device)
      min_out = torch.empty(2, 2, device=device)
      max_out = torch.empty(2, 2, device=device)
      torch.aminmax(x, out=(min_out, max_out))
      return min_out, max_out

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_resize_shrink_mutate_base(self):
    self._assert_same_mutation_from_base(
        (100,),
        lambda tensor: tensor.resize_(99),
    )

  def test_resize_shrink_mutate_view(self):
    self._assert_same_mutation_from_view(
        (100,),
        lambda tensor: tensor.resize_(99),
    )

  def test_resize_non_contiguous_non_view(self):
    def _test_fn(device):
      a = torch.empty_strided((2, 3), (1, 2), device=device)
      a.resize_(6)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_resize_non_contiguous_view(self):
    # Test with transpose view (size 6, storage 6)
    # Resize to 7 (requires storage resize)
    a_cpu = torch.randn(2, 3, device="cpu")
    b_cpu = a_cpu.t()
    b_cpu.resize_(7)

    a_tpu = a_cpu.to(self.tpu_device)
    b_tpu = a_tpu.t()
    b_tpu.resize_(7)

    self._assert_same_layout(b_cpu, b_tpu)
    self._assert_same_layout(a_cpu, a_tpu)
    utils.assert_close(b_tpu.cpu()[:6], b_cpu[:6])

    # Test with slice view (size 5, storage 10)
    # Resize to 11 (requires storage resize)
    x_cpu = torch.randn(10, device="cpu")
    y_cpu = x_cpu[::2]
    y_cpu.resize_(11)

    x_tpu = x_cpu.to(self.tpu_device)
    y_tpu = x_tpu[::2]
    y_tpu.resize_(11)

    self._assert_same_layout(y_cpu, y_tpu)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(y_tpu.cpu()[:10], y_cpu[:10])

  def test_resize_grow(self):
    def _test_fn(device):
      a = torch.randn(100, device=device)
      a.resize_(101)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_resize_grow_preserves_values(self):
    tensor_cpu = torch.arange(100, dtype=torch.int32, device="cpu")
    tensor_tpu = tensor_cpu.to(self.tpu_device)

    tensor_cpu.resize_(101)
    tensor_tpu.resize_(101)

    expected = torch.arange(100, dtype=torch.int32, device="cpu")
    utils.assert_close(tensor_cpu[:100], expected)

    utils.assert_close(tensor_tpu[:100].cpu(), expected)

  def test_resize_grow_mutate_base(self):
    base_cpu = torch.arange(100, dtype=torch.int32, device="cpu")
    base_tpu = base_cpu.to(self.tpu_device)

    view_cpu = base_cpu[:50]
    view_tpu = base_tpu[:50]

    base_cpu.resize_(101)
    base_cpu[100] = 100
    base_cpu.mul_(-1)

    base_tpu.resize_(101)
    base_tpu[100] = 100
    base_tpu.mul_(-1)

    self._assert_same_layout(view_cpu, view_tpu)
    utils.assert_close(view_tpu.cpu(), view_cpu)

  def test_resize_grow_mutate_view(self):
    base_cpu = torch.arange(100, dtype=torch.int32, device="cpu")
    base_tpu = base_cpu.to(self.tpu_device)

    view_cpu = base_cpu[:50]
    view_tpu = base_tpu[:50]

    base_cpu.resize_(101)
    base_cpu[100] = 100

    base_tpu.resize_(101)
    base_tpu[100] = 100

    view_cpu.mul_(-1)
    view_tpu.mul_(-1)

    self._assert_same_layout(base_cpu, base_tpu)
    utils.assert_close(base_tpu.cpu(), base_cpu)

  def test_resize_view_different_dtype_false_grow(self):
    x_cpu = torch.randint(0, 100, (7,), dtype=torch.uint32, device="cpu")
    x_tpu = x_cpu.to(self.tpu_device)
    y_cpu = x_cpu.view(torch.bool)
    y_tpu = x_tpu.view(torch.bool)

    y_cpu.resize_(28)
    y_tpu.resize_(28)

    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

  def test_resize_view_different_dtype_true_grow(self):
    x_cpu = torch.randint(0, 100, (7,), dtype=torch.uint32, device="cpu")
    x_tpu = x_cpu.to(self.tpu_device)
    y_cpu = x_cpu.view(torch.bool)
    y_tpu = x_tpu.view(torch.bool)

    y_cpu.resize_(29)
    y_tpu.resize_(29)

    self.assertEqual(y_cpu.size(), y_tpu.size())
    self.assertEqual(y_cpu.dtype, y_tpu.dtype)
    utils.assert_close(y_cpu[:28], y_tpu[:28].cpu())


class SqueezeTest(LayoutTest):

  def test_squeeze(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(
            1, 2, 1, 3, 1, 4, 1, 5, 1, device=device
        ).squeeze()
    )

  def test_squeeze_inplace(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      a.squeeze_()
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_dims(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(
            1, 2, 1, 3, 1, 4, 1, 5, 1, device=device
        ).squeeze(0, 4, -1)
    )

  def test_squeeze_dims_inplace(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      a.squeeze_(0, 4, -1)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(
            1, 2, 1, 3, 1, 4, 1, 5, 1, device=device
        ).squeeze(4)
    )

  def test_squeeze_dim_inplace(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      a.squeeze_(4)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_mutate_base(self):
    self._assert_same_mutation_from_base(
        (1, 2, 1, 3, 1, 4, 1, 5, 1),
        lambda tensor: tensor.squeeze(),
    )

  def test_squeeze_mutate_view(self):
    self._assert_same_mutation_from_view(
        (1, 2, 1, 3, 1, 4, 1, 5, 1),
        lambda tensor: tensor.squeeze(),
    )

  def test_squeeze_copy(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      return torch.squeeze_copy(a)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_copy_out(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      out = torch.empty(2, 3, 4, 5, device=device)
      torch.squeeze_copy(a, out=out)
      return out

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_copy_dims(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      return torch.squeeze_copy(a, (0, 4, -1))

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_copy_dims_out(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      out = torch.empty(2, 1, 3, 4, 1, 5, device=device)
      torch.squeeze_copy(a, (0, 4, -1), out=out)
      return out

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_copy_dim(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      return torch.squeeze_copy(a, 4)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_copy_dim_out(self):
    def _test_fn(device):
      a = torch.randn(1, 2, 1, 3, 1, 4, 1, 5, 1, device=device)
      out = torch.empty(1, 2, 1, 3, 4, 1, 5, 1, device=device)
      torch.squeeze_copy(a, 4, out=out)
      return out

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_squeeze_copy_no_mutation_from_base(self):
    self._assert_no_mutation_from_base(
        (1, 2, 1, 3, 1, 4, 1, 5, 1),
        torch.squeeze_copy,
    )


class UnsqueezeTest(LayoutTest):

  def test_unsqueeze(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).unsqueeze(2)
    )

  def test_unsqueeze_negative_index(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(2, 3, 4, 5, device=device).unsqueeze(-2)
    )

  def test_unsqueeze_inplace(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device)
      a.unsqueeze_(2)
      return a

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_unsqueeze_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: tensor.unsqueeze(2),
    )

  def test_unsqueeze_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        lambda tensor: tensor.unsqueeze(2),
    )

  def test_unsqueeze_copy(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device).permute(3, 2, 1, 0)
      return torch.unsqueeze_copy(a, 2)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_unsqueeze_copy_out(self):
    def _test_fn(device):
      a = torch.randn(2, 3, 4, 5, device=device).permute(3, 2, 1, 0)
      out = torch.empty(5, 4, 1, 3, 2, device=device)
      return torch.unsqueeze_copy(a, 2, out=out)

    self._assert_same_layout_tpu_vs_cpu(_test_fn)

  def test_unsqueeze_copy_no_mutation_from_base(self):
    self._assert_no_mutation_from_base(
        (2, 3, 4, 5),
        lambda tensor: torch.unsqueeze_copy(tensor, 2),
    )


# There's no torch.alias() function, but we can get to it this way.
alias = torch.ops.aten.alias


class AliasTest(LayoutTest):

  def check_alias(self, a: torch.Tensor, b: torch.Tensor) -> None:
    """Checks that a and b are aliases of each other."""

    assert a is not b, "a and b must not be the same object"
    utils.assert_close(a, b)
    self._assert_same_layout(a, b)
    # TODO(bawilson): check that mutating a affects b and vice versa.

  def test_alias_scalar(self):
    a = torch.scalar_tensor(1.0, dtype=torch.float32, device="tpu")
    self.check_alias(a, alias(a))

  def test_alias_contiguous_tensor(self):
    a = torch.randn(2, 3, 4, 5, device="tpu")
    self.check_alias(a, alias(a))

  def test_alias_noncontiguous_tensor(self):
    a = torch.randn(2, 3, 4, 5, device="tpu").permute(3, 1, 2, 0)
    self.check_alias(a, alias(a))

  def test_alias_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4, 5),
        alias,
    )

  def test_alias_mutate_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4, 5),
        alias,
    )


class UnbindTest(LayoutTest):

  def test_unbind_default_dim(self):
    a_tpu = torch.randn((2, 3, 4), device="tpu").unbind()
    a_cpu = torch.randn((2, 3, 4), device="tpu").unbind()
    for t1, t2 in zip(a_cpu, a_tpu):
      self._assert_same_layout(t1, t2)

  def test_unbind_dim_1(self):
    a_tpu = torch.randn((2, 3, 4), device="tpu").unbind(1)
    a_cpu = torch.randn((2, 3, 4), device="tpu").unbind(1)
    for t1, t2 in zip(a_cpu, a_tpu):
      self._assert_same_layout(t1, t2)

  def test_unbind_dim_2(self):
    a_tpu = torch.randn((2, 3, 4), device="tpu").unbind(2)
    a_cpu = torch.randn((2, 3, 4), device="tpu").unbind(2)
    for t1, t2 in zip(a_cpu, a_tpu):
      self._assert_same_layout(t1, t2)

  def test_unbind_mutate_base(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4),
        lambda tensor: tensor.unbind(),
    )

  def test_unbind_mutate_from_view(self):
    self._assert_same_mutation_from_view(
        (2, 3, 4),
        lambda tensor: tensor.unbind()[0],
    )


class ExpandTest(LayoutTest):

  def test_expand_1d(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(1, device=device).expand(999)
    )

  def test_expand_2d_all_dims(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(1, 1, device=device).expand(999, 999)
    )

  def test_expand_2d_one_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(1, 999, device=device).expand(999, 999)
    )

  def test_expand_mutate_base(self):
    self._assert_same_mutation_from_base(
        (1, 1),
        lambda tensor: tensor.expand(999, 999),
    )

  def test_expand_mutate_view_fails(self):
    base = torch.zeros(1, device=self.tpu_device)
    view = base.expand(1000)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        # This error message is generated by PyTorch before our kernel is
        # called.
        "unsupported operation: more than one element of the written-to tensor"
        " refers to a single memory location. Please clone() the tensor before"
        " performing the operation.",
    ):
      view.add_(1.0)


class SplitTest(LayoutTest):

  def test_split_default_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, 20, device=device).split(2)
    )

  def test_split_with_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, 20, device=device).split(2, dim=1)
    )

  def test_split_with_sizes_default_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, 20, device=device).split_with_sizes(
            [1, 2, 3, 4]
        )
    )

  def test_split_with_sizes_and_dim(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(10, 20, device=device).split_with_sizes(
            [2, 4, 6, 8], dim=1
        )
    )

  def test_split_mutate_base(self):
    self._assert_same_mutation_from_base(
        (10, 20),
        lambda tensor: tensor.split(2),
    )

  def test_split_mutate_view(self):
    self._assert_same_mutation_from_view(
        (10, 20),
        lambda tensor: tensor.split(2)[0],
    )

  def test_split_with_sizes_mutate_base(self):
    self._assert_same_mutation_from_base(
        (10, 20),
        lambda tensor: tensor.split_with_sizes([1, 2, 3, 4]),
    )

  def test_split_with_sizes_mutate_view(self):
    self._assert_same_mutation_from_view(
        (10, 20),
        lambda tensor: tensor.split_with_sizes([1, 2, 3, 4])[1],
    )


class ViewDtypeTest(LayoutTest):

  def _assert_bitcast_layout_and_values(
      self,
      base_shape: torch.Size,
      base_int_type: torch.dtype,
      view_int_dtype: torch.dtype,
  ):
    # Generate the base as random integers.
    base_iinfo = torch.iinfo(base_int_type)
    x_cpu = torch.randint(
        low=base_iinfo.min,
        high=base_iinfo.max,
        size=base_shape,
        dtype=base_int_type,
    )
    x_tpu = x_cpu.to(self.tpu_device)

    # Cast to the view type and check that they match.
    y_cpu = x_cpu.view(view_int_dtype)
    y_tpu = x_tpu.view(view_int_dtype)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    # Mutate the base and check that the view is updated.
    x_cpu.bitwise_xor_(base_iinfo.max)
    x_tpu.bitwise_xor_(base_iinfo.max)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    # Mutate the view and check that the base is updated.
    y_cpu.bitwise_and_(0)
    y_tpu.bitwise_and_(0)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())

  def test_real_to_real_same_size(self):
    self._assert_bitcast_layout_and_values(
        base_shape=(17,),
        base_int_type=torch.uint32,
        view_int_dtype=torch.int32,
    )

  def test_real_to_real_larger_size(self):
    self._assert_bitcast_layout_and_values(
        base_shape=(34,),
        base_int_type=torch.uint32,
        view_int_dtype=torch.uint64,
    )

  def test_real_to_real_smaller_size(self):
    self._assert_bitcast_layout_and_values(
        base_shape=(17,),
        base_int_type=torch.uint32,
        view_int_dtype=torch.uint16,
    )

  def test_bool_to_real(self):
    # Generating bool values is a bit trickier than ordinary uints.
    # Need to use bernoulli rather than randint because torch.iinfo does not
    # support bool.
    x_cpu = torch.bernoulli(
        torch.full(size=(28,), fill_value=0.5, dtype=torch.float32),
        out=torch.empty(28, dtype=torch.bool),
    )
    x_tpu = x_cpu.to(self.tpu_device)

    y_cpu = x_cpu.view(torch.uint32)
    y_tpu = x_tpu.view(torch.uint32)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    # Mutate the base and check that the view is updated.
    x_cpu.logical_not_()
    x_tpu.logical_not_()
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    # Mutate the view and check that the base is updated.
    y_cpu.mul_(2)
    y_tpu.mul_(2)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())

  def test_real_to_bool(self):
    base_shape = (7,)
    base_int_type = torch.uint32
    # Generate the base as random integers.
    base_iinfo = torch.iinfo(base_int_type)
    x_cpu = torch.randint(
        low=base_iinfo.min,
        high=base_iinfo.max,
        size=base_shape,
        dtype=base_int_type,
    )
    x_tpu = x_cpu.to(self.tpu_device)

    # Cast to the view type and check that they match.
    y_cpu = x_cpu.view(torch.bool)
    y_tpu = x_tpu.view(torch.bool)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    # Mutate the base and check that the view is updated.
    x_cpu.bitwise_xor_(base_iinfo.max)
    x_tpu.bitwise_xor_(base_iinfo.max)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    # Mutate the view and check that the base is updated.
    y_cpu.logical_not_()
    y_tpu.logical_not_()
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())


class ComplexViewsTest(LayoutTest):

  def test_view_as_real(self):
    x_cpu = torch.view_as_complex(
        torch.randn(17, 2, device="cpu", dtype=torch.float32)
    ).clone()
    x_tpu = x_cpu.to(self.tpu_device)

    y_cpu = torch.view_as_real(x_cpu)
    y_tpu = torch.view_as_real(x_tpu)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    x_cpu.mul_(2.0)
    x_tpu.mul_(2.0)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    y_cpu.mul_(2.0)
    y_tpu.mul_(2.0)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())

  def test_real_half(self):
    x_cpu = torch.view_as_complex(
        torch.randn(17, 2, device="cpu", dtype=torch.float32)
    ).clone()
    x_tpu = x_cpu.to(self.tpu_device)

    y_cpu = torch.real(x_cpu)
    y_tpu = torch.real(x_tpu)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    x_cpu.mul_(2.0)
    x_tpu.mul_(2.0)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    y_cpu.mul_(2.0)
    y_tpu.mul_(2.0)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())

  def test_imag_half(self):
    x_cpu = torch.view_as_complex(
        torch.randn(17, 2, device="cpu", dtype=torch.float32)
    ).clone()
    x_tpu = x_cpu.to(self.tpu_device)

    y_cpu = torch.imag(x_cpu)
    y_tpu = torch.imag(x_tpu)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    x_cpu.mul_(2.0)
    x_tpu.mul_(2.0)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    y_cpu.mul_(2.0)
    y_tpu.mul_(2.0)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())

  def test_view_as_complex(self):
    x_cpu = torch.randn(17, 2, device="cpu", dtype=torch.float32)
    x_tpu = x_cpu.to(self.tpu_device)

    y_cpu = torch.view_as_complex(x_cpu)
    y_tpu = torch.view_as_complex(x_tpu)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    x_cpu.mul_(2.0)
    x_tpu.mul_(2.0)
    self._assert_same_layout(y_cpu, y_tpu)
    utils.assert_close(y_cpu, y_tpu.cpu())

    y_cpu.mul_(2.0)
    y_tpu.mul_(2.0)
    self._assert_same_layout(x_cpu, x_tpu)
    utils.assert_close(x_cpu, x_tpu.cpu())


class UnfoldTest(LayoutTest):

  def test_unfold(self):
    self._assert_same_layout_tpu_vs_cpu(
        lambda device: torch.randn(1024, 128, device=device).unfold(
            dimension=1, size=127, step=1
        )
    )

  def test_unfold_mutate_base(self):
    self._assert_same_mutation_from_base(
        (1024, 128),
        lambda tensor: tensor.unfold(dimension=1, size=127, step=1),
    )

  def test_unfold_mutate_view_fails(self):
    base = torch.zeros(1024, 128, device=self.tpu_device)
    view = base.unfold(dimension=1, size=127, step=1)
    with self.assertRaisesRegex(
        RuntimeError,
        "inplace writes to overlapping views are undefined behavior and"
        " are not supported",
    ):
      view.add_(1.0)


class SymbolicViewsTest(LayoutTest):
  """Tests to ensure no false cache hits for symbolic views."""

  def test_transpose(self):
    self._assert_same_mutation_from_base(
        (2, 3),
        lambda tensor: tensor.transpose(1, 0),
    )
    self._assert_same_mutation_from_base(
        (2, 4),
        lambda tensor: tensor.transpose(1, 0),
    )

  def test_reshape(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4),
        lambda tensor: tensor.reshape(6, 4),
    )
    self._assert_same_mutation_from_base(
        (2, 4, 4),
        lambda tensor: tensor.reshape(8, 4),
    )

  def test_multiple_view_ops(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4),
        lambda tensor: tensor.transpose(1, 0).reshape(6, 4),
    )
    self._assert_same_mutation_from_base(
        (2, 4, 4),
        lambda tensor: tensor.transpose(1, 0).reshape(8, 4),
    )

  def test_multiple_view_ops_different_chain(self):
    self._assert_same_mutation_from_base(
        (2, 3, 4),
        lambda tensor: tensor.transpose(2, 0).reshape(12, 2).transpose(1, 0),
    )
    self._assert_same_mutation_from_base(
        (2, 3, 4),
        lambda tensor: tensor.transpose(1, 0).reshape(6, 4).transpose(1, 0),
    )


class StridedSliceViewTest(LayoutTest):

  def test_view_assignment(self):
    # Simulates the mrope interleaved assignment from vLLM.
    # Mrope_section = [11, 11, 10], rotary_dim // 2 = 32.
    x = torch.arange(
        3 * 10 * 32, dtype=torch.float32, device=self.tpu_device
    ).reshape(3, 10, 32)
    x_t = x[0].clone()
    x_t[..., 1:33:3] = x[1, ..., 1:33:3]
    x_t[..., 2:30:3] = x[2, ..., 2:30:3]

    x_cpu = x.cpu()
    x_t_cpu = x_cpu[0].clone()
    x_t_cpu[..., 1:33:3] = x_cpu[1, ..., 1:33:3]
    x_t_cpu[..., 2:30:3] = x_cpu[2, ..., 2:30:3]

    utils.assert_close(x_t.cpu(), x_t_cpu)


class CompileTest(LayoutTest):

  def test_cast_non_contiguous_tensor_view(self):
    class CastNonContiguousTensorViewModule(torch.nn.Module):

      def forward(self, x):
        x_fp32 = x.float()
        x_fp32 = x_fp32 + 1.0
        x_bf16 = x_fp32.type_as(x)
        out = (
            x_bf16.transpose(1, 2)
            .contiguous()
            .reshape(1, 512, 1024)
            .contiguous()
        )
        return out

    model = CastNonContiguousTensorViewModule().to(self.tpu_device)
    # The input to the model should be a non-contiguous tensor.
    x = torch.randn(
        1, 512, 4, 256, dtype=torch.bfloat16, device=self.tpu_device
    ).transpose(1, 2)

    compiled_model = torch.compile(model, backend=compile_lib.TpuBackend())

    y_compiled = compiled_model(x)
    self.assertEqual(y_compiled.shape, torch.Size([1, 512, 1024]))


if __name__ == "__main__":
  absltest.main()
