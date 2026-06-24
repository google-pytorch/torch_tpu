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

"""Unit tests for index_put op."""


from absl.testing import absltest
import torch
from torch_tpu._internal.utils import utils
from tests import op_testing


OpInput = op_testing.OpInput
TorchTpuVsCpuTestBase = op_testing.TorchTpuVsCpuTestBase
to = op_testing.to


class IndexPutTest(TorchTpuVsCpuTestBase):
  """Valid tests for index_put op with integer indices."""

  def _run_index_put(
      self, device, self_tensor, indices_tuple, values_tensor, accumulate=False
  ):
    t = to(self_tensor, device)
    # Ensure indices are on the correct device
    indices = tuple(to(i, device) for i in indices_tuple if i is not None)
    values = to(values_tensor, device)

    t_clone = t.clone()
    t_clone.index_put_(indices, values, accumulate=accumulate)
    return t_clone

  # pointwise update
  def test_pointwise_2D(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.arange(12, dtype=torch.float32).view(3, 4),
            (torch.tensor([0, 1, 0]), torch.tensor([1, 2, 3])),  # B = (3,)
            torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32),
        )
    )

  # basic slice update
  def test_basic_slice_3d_index_2d(self):
    # Indices broadcast shape B = (2)
    # Values broadcasts to (2, 5)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, 5, dtype=torch.float32),
            (torch.tensor([0, 1]), torch.tensor([0, 2])),
            torch.arange(1, 11, dtype=torch.float32).view(2, 5),
        )
    )

  # index broadcasting
  def test_index_broadcast_2D(self):
    idx0 = torch.tensor([[0], [1]])  # Shape (2, 1)
    idx1 = torch.tensor([[0, 2, 3]])  # Shape (1, 3)
    # Broadcast shape B = (2, 3)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, dtype=torch.float32),
            (idx0, idx1),
            torch.arange(1, 7, dtype=torch.float32).view(2, 3),
        )
    )

  # slice update with fewer indices than rank
  def test_slice_1D_index_in_3D(self):
    # Indices broadcast shape B = (2,)
    # Values broadcasts to (2, 4, 5)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            (torch.tensor([0, 2]),),  # Indices for dim 0, B = (2,)
            # values shape must be B + self.shape[1:] = (2, 4, 5)
            torch.arange(1, 41, dtype=torch.float32).view(2, 4, 5),
        )
    )

  # values broadcasts over the slice dimensions
  def test_values_slice_broadcast_3D(self):
    idx0 = torch.tensor([0, 2]).view(-1, 1)  # (2, 1)
    idx1 = torch.tensor([1, 3, 0]).view(1, -1)  # (1, 3)
    # Indices broadcast shape B = (2, 3)
    # Values broadcasts to (2, 3, 5)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            (idx0, idx1),
            # values (2, 3, 1) will broadcast to (2, 3, 5)
            torch.arange(1, 7, dtype=torch.float32).view(2, 3, 1),
        )
    )

  # accumulate=True
  def test_accumulate_2D_broadcast(self):
    # Broadcast shape is (2, 3)
    idx0 = torch.tensor([[0], [1]])  # (2, 1)
    idx1 = torch.tensor([[0, 2, 3]])  # (1, 3)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(3, 4, dtype=torch.float32),
            (idx0, idx1),
            torch.arange(1, 7, dtype=torch.float32).view(2, 3),
            accumulate=True,
        )
    )

  # different integer dtypes for indices
  def test_index_dtype_int32(self):
    # Broadcast shape B = (3,)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.arange(12, dtype=torch.float32).view(3, 4),
            (
                torch.tensor([0, 1, 0], dtype=torch.int32),
                torch.tensor([1, 2, 3], dtype=torch.int32),
            ),
            torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32),
        )
    )

  # all dimensions indexed with broadcasting
  def test_all_dims_indexed_3D_broadcast(self):
    idx0 = torch.tensor([0, 1]).view(-1, 1, 1)  # (2, 1, 1)
    idx1 = torch.tensor([1, 2]).view(1, -1, 1)  # (1, 2, 1)
    idx2 = torch.tensor([3, 4]).view(1, 1, -1)  # (1, 1, 2)
    # Broadcast shape B = (2, 2, 2)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            (idx0, idx1, idx2),
            torch.arange(1, 9, dtype=torch.float32).view(2, 2, 2),
        )
    )

  # accumulate=True with scalar values
  def test_accumulate_scalar(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(2, 3, dtype=torch.float32),
            (torch.tensor([0, 1]), torch.tensor([1, 2])),
            torch.tensor(5.0, dtype=torch.float32),
            accumulate=True,
        )
    )

  # all indices are scalar
  def test_all_scalar_2d(self):
    # Expected: self[1, 2] = 99.0
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, dtype=torch.float32),
            (torch.tensor(1), torch.tensor(2)),
            torch.tensor(99.0, dtype=torch.float32),
        )
    )

  # mixed scalar and 1D tensor
  def test_scalar_and_1d_2d(self):
    # Expected: self[1, 0]=77, self[1, 2]=88, self[1, 3]=99
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, dtype=torch.float32),
            # indices for dim 1, B=(3,)
            (torch.tensor(1), torch.tensor([0, 2, 3])),
            torch.tensor([77.0, 88.0, 99.0], dtype=torch.float32),
        )
    )

  # mixed 1D tensor and scalar
  def test_1d_and_scalar_2d(self):
    # Expected: self[0, 1]=55, self[2, 1]=66
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, dtype=torch.float32),
            (torch.tensor([0, 2]), torch.tensor(1)),  # B=(2,)
            torch.tensor([55.0, 66.0], dtype=torch.float32),
        )
    )

  # scalar index for slice update
  def test_scalar_index_slice_3d(self):
    # Expected: self[1, :, :] is updated with values
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            (torch.tensor(1),),  # Index for dim 0, B=()
            # values shape must be B + self.shape[1:] = (4, 5)
            torch.arange(20, dtype=torch.float32).view(4, 5),
        )
    )

  # mixed scalar and 1D for slice update
  def test_scalar_and_1d_slice_3d(self):
    # Expected: self[2, 0, :] and self[2, 3, :] are updated
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            # indices for dim 0, 1. B=(2,)
            (torch.tensor(2), torch.tensor([0, 3])),
            # Unindexed dim 2, size 5. values shape (2, 5)
            torch.arange(1, 11, dtype=torch.float32).view(2, 5),
        )
    )

  # accumulate with scalar index
  def test_accumulate_scalar_2d(self):
    # Expected: self[1, 2] = 1.0 + 5.0 = 6.0
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(3, 4, dtype=torch.float32),
            (torch.tensor(1), torch.tensor(2)),
            torch.tensor(5.0, dtype=torch.float32),
            accumulate=True,
        )
    )

  # scalar index with scalar value broadcast
  def test_scalar_index_scalar_value_slice(self):
    # Expected: self[1, :, :] is updated with -1.0
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            (torch.tensor(1),),  # Index for dim 0, B=()
            torch.tensor(-1.0, dtype=torch.float32),  # Broadcasts to (4, 5)
        )
    )

  # all scalar indices on 3D tensor
  def test_all_scalar_3d(self):
    # Expected: self[0, 1, 2] = 123.0
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, 4, dtype=torch.float32),
            (torch.tensor(0), torch.tensor(1), torch.tensor(2)),
            torch.tensor(123.0, dtype=torch.float32),
        )
    )

  # boolean mask with the same shape as self.
  def test_boolean_full_shape(self):
    # Indices broadcast shape (3)
    # Values broadcast shape (3)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, dtype=torch.float32),
            (torch.tensor([[True, False, True], [False, True, False]]),),
            torch.tensor([10, 20, 30], dtype=torch.float32),
        )
    )

  def test_boolean_full_shape_multiple_indices(self):
    # Indices broadcast shape (2)
    # Values broadcast shape (2)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, dtype=torch.float32),
            (torch.tensor([True, False]), torch.tensor([True, True, False])),
            torch.tensor([10, 20], dtype=torch.float32),
        )
    )

  # values tensor requires broadcasting.
  def test_boolean_broadcast_values(self):
    # Indices broadcast shape (1)
    # Values broadcast shape (1, 3)
    # Expected: Row 0 is [10, 20, 30], Row 1 is zeros.
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, dtype=torch.float32),
            (torch.tensor([True, False]),),
            torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
        )
    )

  # accumulate=True with boolean mask.
  def test_boolean_accumulate(self):
    # Indices broadcast shape (2)
    # Values broadcast shape (2, 2)
    # Expected: Rows 0 and 1 are t + vals = 6, row 2 remains 1.
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(3, 2, dtype=torch.float32),
            (torch.tensor([True, True, False]),),
            torch.ones(2, 2, dtype=torch.float32) * 5,
            accumulate=True,
        )
    )

  # scalar value with boolean mask.
  def test_boolean_scalar_value(self):
    # Indices broadcast shape (2)
    # Values broadcast shape (2)
    # Expected: [[77, 0], [0, 77]]
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 2, dtype=torch.float32),
            (torch.tensor([[True, False], [False, True]]),),
            torch.tensor(77.0, dtype=torch.float32),
        )
    )

  # scalar value with full mask. (Engages the select path)
  def test_boolean_scalar_value_with_full_mask(self):
    # Expected: [[77, 0, 77], [0, 77, 0]]
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, dtype=torch.float32),
            (torch.tensor([[True, False, True], [False, True, False]]),),
            torch.tensor(77.0, dtype=torch.float32),
        )
    )

  # scalar value with partial mask. (Engages the select path)
  def test_boolean_scalar_value_with_partial_mask(self):
    # Expected: [[77, 77, 77], [0, 0, 0]]
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, dtype=torch.float32),
            (torch.tensor([True, False]),),
            torch.tensor(77.0, dtype=torch.float32),
        )
    )

  # scalar value with partial mask + accumulate. (Engages the select path)
  def test_boolean_scalar_value_with_partial_mask_accumulate(self):
    # Expected: [[78, 78, 78], [1, 1, 1]]
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(2, 3, dtype=torch.float32),
            (torch.tensor([True, False]),),
            torch.tensor(77.0, dtype=torch.float32),
            accumulate=True,
        )
    )

  # empty boolean mask
  def test_boolean_empty_mask(self):
    # Indices broadcast shape (0)
    # Values broadcast shape (0)
    # Expected: No changes to t, should remain all ones.
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(2, 3, dtype=torch.float32),
            (torch.zeros(2, 3, dtype=torch.bool),),
            torch.empty(0, dtype=torch.float32),
        )
    )

  # all True boolean mask
  def test_boolean_all_true_mask(self):
    # Indices broadcast shape (6)
    # Values broadcast shape (6)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(2, 3, dtype=torch.float32),
            (torch.ones(2, 3, dtype=torch.bool),),
            torch.zeros(6, dtype=torch.float32),
        )
    )

  def test_mixed_int_bool(self):
    # Indices broadcast shape (2)
    # Values broadcast shape (2)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, dtype=torch.float32),
            (torch.tensor([0, 2]), torch.tensor([True, False, True, False])),
            torch.ones(2, dtype=torch.float32),
        )
    )

  def test_mixed_bool_int(self):
    # Indices broadcast shape (2)
    # Values broadcast shape (2)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(4, 3, dtype=torch.float32),
            (torch.tensor([True, False, True, False]), torch.tensor([0, 2])),
            torch.ones(2, dtype=torch.float32),
        )
    )

  def test_mixed_multiple_indices(self):
    # Indices broadcast shape (2, 1, 3)
    # Values broadcast shape (2, 1, 3, 4)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(2, 3, 4, dtype=torch.float32),
            (
                torch.arange(2).reshape(
                    2, 1, 1
                ),  # Integer index for dim 0, shape (2, 1, 1)
                torch.tensor(True),  # Boolean SCALAR index for dim 1
                torch.arange(3).reshape(
                    1, 1, 3
                ),  # Integer index for dim 2, shape (1, 1, 3)
            ),
            torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4),
        )
    )

  def test_decompose_with_mask(self):
    def test_fn(device):
      tensor = torch.arange(10, device=device).view(2, 5)
      # This mask will select elements at odd indices (1, 3, 5, 7, 9)
      boolean_mask = tensor % 2 != 0
      values_to_assign = torch.tensor(
          [100.0, 200.0, 300.0, 400.0, 500.0],
          dtype=tensor.dtype,
          device=device,
      )
      tensor[boolean_mask] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  # Engages the select path for boolean mask
  def test_decompose_with_mask_scalar_value(self):
    def test_fn(device):
      tensor = torch.arange(10, device=device).view(2, 5)
      boolean_mask = tensor % 2 != 0
      tensor[boolean_mask] = 100
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  # Engages the select path for boolean mask
  def test_decompose_with_mask_scalar_value_slice(self):
    def test_fn(device):
      tensor = torch.arange(10, device=device).view(2, 5)
      boolean_mask = tensor[1] % 2 != 0
      tensor[:, boolean_mask] = 100
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  # Engages the select path for boolean mask
  def test_decompose_with_mask_scalar_value_multiple_slices(self):
    def test_fn(device):
      tensor = torch.arange(210, device=device).view(2, 3, 5, 7)
      boolean_mask_dim1_dim2 = tensor[0, :, :, 0] % 2 != 0
      tensor[:, boolean_mask_dim1_dim2, :] = 100
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  # multiple boolean masks, does not enable the select path
  def test_decompose_with_multiple_mask_scalar_value_multiple_slices(self):
    def test_fn(device):
      tensor = torch.arange(270, device=device).view(2, 3, 5, 9)
      boolean_mask_dim1 = tensor[0, :, 0, 0] % 2 != 0
      boolean_mask_dim3 = tensor[0, 0, 0, :] % 2 != 0
      tensor[:, boolean_mask_dim1, :, boolean_mask_dim3] = 100
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_with_integer_indices(self):
    def test_fn(device):
      tensor = torch.arange(10, device=device).view(2, 5)
      indices = (
          torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device),
          torch.tensor([1, 3, 0, 2, 4], dtype=torch.long, device=device),
      )
      values_to_assign = torch.tensor(
          [100.0, 200.0, 300.0, 400.0, 500.0],
          dtype=tensor.dtype,
          device=device,
      )
      tensor[indices] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5, device=device).view(3, 4, 5)
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device)
      values_to_assign = torch.arange(
          3 * 4 * 2, dtype=tensor.dtype, device=device
      ).view(3, 4, 2)
      # Single index tensor means contiguous, so will not move to the front,
      # so broadcast shape (B) will be (2,) and slice shape (S_before)
      # will be (dim0=3, dim1=4)
      # Hence, values shape should broadcast to (S_before + B) = (3, 4, 2)
      tensor[:, :, indices_dim2] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_middle(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5, device=device).view(3, 4, 5)
      indices_dim0 = torch.tensor([0, 2], dtype=torch.long, device=device).view(
          2, 1
      )
      indices_dim2 = torch.tensor(
          [1, 2, 3], dtype=torch.long, device=device
      ).view(1, 3)
      values_to_assign = (
          torch.arange(2 * 3 * 4, dtype=tensor.dtype, device=device).view(
              2, 3, 4
          )
      )
      # Here indices will move to the front as non-contiguous index tensors
      # are present, so broadcast shape (B) will be (2, 3) and slice shape
      # (S_after) will be (dim1=4)
      # Hence, values shape should broadcast to (B + S_after) = (2, 3, 4)
      tensor[indices_dim0, :, indices_dim2] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_end(self):
    def test_fn(device):
      tensor = torch.zeros(12, device=device).view(3, 4)
      indices = torch.tensor([0, 2], dtype=torch.long, device=device)
      values_to_assign = torch.tensor(
          [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]],
          dtype=tensor.dtype,
          device=device,
      )
      # Single index tensor means contiguous, so will not move to the front,
      # so broadcast shape (B) will be (2,) and slice shape (S_after)
      # will be (dim1=4)
      # Hence, values shape should broadcast to (B + S_after) = (2, 4)
      tensor[indices, :] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start_end_variation_1(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5 * 6 * 7, device=device).view(3, 4, 5, 6, 7)
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device)

      values_to_assign = torch.arange(
          3 * 4 * 2 * 7, dtype=tensor.dtype, device=device
      ).view(3, 4, 2, 7)
      # Here indices are contiguous, so will not move to the front,
      # so broadcast shape (B) will be (2,) and first slice shape (S_before)
      # will be (dim0=3, dim1=4), second slice shape (S_after) will be (dim4=7)
      # Hence, values shape should broadcast to
      # (S_before + B + S_after) = (3, 4, 2, 7)
      tensor[:, :, indices_dim2, indices_dim2, :] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start_end_variation_2(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5 * 6 * 7, device=device).view(3, 4, 5, 6, 7)
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device).view(
          2, 1
      )
      indices_dim4 = torch.tensor(
          [1, 2, 3, 4], dtype=torch.long, device=device
      ).view(1, 4)

      values_to_assign = torch.arange(
          3 * 4 * 2 * 4 * 7, dtype=tensor.dtype, device=device
      ).view(3, 4, 2, 4, 7)

      # Here indices are contiguous, so will not move to the front,
      # so broadcast shape (B) will be (2, 4) and first slice shape (S_before)
      # will be (dim0=3, dim1=4), second slice shape (S_after) will be (dim4=7)
      # Hence, values shape should broadcast to
      # (S_before + B + S_after) = (3, 4, 2, 4, 7)
      tensor[:, :, indices_dim2, indices_dim4, :] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start_middle_variation_1(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5 * 6 * 7, device=device).view(3, 4, 5, 6, 7)
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device)

      values_to_assign = torch.arange(
          3 * 4 * 2 * 6, dtype=tensor.dtype, device=device
      ).view(2, 3, 4, 6)

      # Here indices will move to the front as non-contiguous index tensors
      # are present, so broadcast shape (B) will be (2,) and slice shape
      # (S_after) will be (dim0=3, dim1=4, dim3=6)
      # Hence, values shape should broadcast to (B + S_after) = (2, 3, 4, 6)
      tensor[:, :, indices_dim2, :, indices_dim2] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start_middle_variation_2(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5 * 6 * 7, device=device).view(3, 4, 5, 6, 7)
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device).view(
          2, 1
      )
      indices_dim4 = torch.tensor(
          [1, 2, 3, 4], dtype=torch.long, device=device
      ).view(1, 4)

      values_to_assign = torch.arange(
          3 * 4 * 2 * 6 * 4, dtype=tensor.dtype, device=device
      ).view(2, 4, 3, 4, 6)

      # Here indices will move to the front as non-contiguous index tensors
      # are present, so broadcast shape (B) will be (2, 4) and slice shape
      # (S_after) will be (dim0=3, dim1=4, dim3=6)
      # Hence, values shape should broadcast to (B + S_after) = (2, 4, 3, 4, 6)
      tensor[:, :, indices_dim2, :, indices_dim4] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start_middle_end_variation_1(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5 * 6 * 7 * 8, device=device).view(
          3, 4, 5, 6, 7, 8
      )
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device)

      values_to_assign = torch.arange(
          3 * 4 * 2 * 6 * 8, dtype=tensor.dtype, device=device
      ).view(2, 3, 4, 6, 8)

      # Here indices will move to the front as non-contiguous index tensors
      # are present, so broadcast shape (B) will be (2,) and slice shape
      # (S_after) will be (dim0=3, dim1=4, dim3=6, dim5=8)
      # Hence, values shape should broadcast to (B + S_after) = (2, 3, 4, 6, 8)
      tensor[:, :, indices_dim2, :, indices_dim2, :] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_slice_at_start_middle_end_variation_2(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5 * 6 * 7 * 8, device=device).view(
          3, 4, 5, 6, 7, 8
      )
      indices_dim2 = torch.tensor([1, 3], dtype=torch.long, device=device).view(
          2, 1
      )
      indices_dim4 = torch.tensor(
          [1, 2, 3, 4], dtype=torch.long, device=device
      ).view(1, 4)

      values_to_assign = torch.arange(
          3 * 4 * 2 * 6 * 4 * 8, dtype=tensor.dtype, device=device
      ).view(2, 4, 3, 4, 6, 8)

      # Here indices will move to the front as non-contiguous index tensors
      # are present, so broadcast shape (B) will be (2,4) and slice shape
      # (S_after) will be (dim0=3, dim1=4, dim3=6, dim5=8)
      # Hence, values shape should broadcast to (B + S_after)=(2, 4, 3, 4, 6, 8)
      tensor[:, :, indices_dim2, :, indices_dim4, :] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_decompose_only_slices(self):
    def test_fn(device):
      tensor = torch.zeros(3 * 4 * 5, device=device).view(3, 4, 5)
      values_to_assign = (
          torch.arange(3 * 4 * 5, dtype=tensor.dtype, device=device).view(
              3, 4, 5
          )
      )
      # this does not decompose to index_put
      tensor[:, :, :] = values_to_assign
      return tensor

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_index_put_leading_none_with_optracer(self):
    def test_fn(device):
      data = torch.arange(60, dtype=torch.int32, device=device).reshape(2, 5, 6)
      indices = torch.tensor([1, 3], dtype=torch.int64, device=device)
      values = torch.ones(2, 2, 6, dtype=torch.int32, device=device)
      with utils.OpTracer():
        data[:, indices] = values
      return data

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_index_put_leading_none_with_optracer_boolean_mask_scalar_value(self):
    def test_fn(device):
      data = torch.arange(60, dtype=torch.int32, device=device).reshape(2, 5, 6)
      indices = torch.tensor(
          [False, True, False, True, False], dtype=torch.bool, device=device
      )
      values = 100
      with utils.OpTracer():
        data[:, indices] = values
      return data

    self.assert_close_tpu_vs_cpu(test_fn)

  def test_negative_indices_1d(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(5, dtype=torch.float32),
            (torch.tensor([-1, -3, 1]),),
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
        )
    )

  def test_negative_indices_2d_pointwise(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.arange(12, dtype=torch.float32).view(3, 4),
            (torch.tensor([-3, 1, -3]), torch.tensor([1, -2, -1])),
            torch.tensor([100.0, 101.0, 102.0], dtype=torch.float32),
        )
    )

  def test_negative_indices_broadcast_3d(self):
    idx0 = torch.tensor([[-3], [-1]])  # shape (2, 1)
    idx1 = torch.tensor([[-3, 2, -1]])  # shape (1, 3)
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.zeros(3, 4, 5, dtype=torch.float32),
            (idx0, idx1),
            torch.arange(1, 7, dtype=torch.float32).view(2, 3, 1),
        )
    )

  def test_negative_indices_accumulate(self):
    self.assert_close_tpu_vs_cpu(
        lambda device: self._run_index_put(
            device,
            torch.ones(3, 4, dtype=torch.float32),
            (torch.tensor([-1, -2, 1]), torch.tensor([-1, -3, 0])),
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
            accumulate=True,
        )
    )


if __name__ == "__main__":
  absltest.main()
