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

"""Tests PyTorch distributed collective all_to_all on multiple TPUs.

Uses `TpuMultiProcContinuousTest` to spawn a persistent pool of TPU worker
processes per test class and run distributed collective checks across ranks.
"""

from collections.abc import Sequence
import math
from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils

# 1D ragged-all-to-all factors operands onto 1024-element (`8x128`) TPU tiles.
_1D_FACTOR_UNIT = 1024


class AllToAllSingleTestData:
  """Input and output data for all_to_all tests where data is dependent on rank and world_size."""

  def __init__(
      self,
      input_split_sizes: list[int] = None,
      output_split_sizes: list[int] = None,
      dtype: torch.dtype = torch.int64,
      md_shapes: bool = False,
  ):
    self._input_tensor = None
    self._output_tensor = None
    self._expected_output_tensor = None
    self._input_split_sizes = input_split_sizes
    self._output_split_sizes = output_split_sizes
    self._dtype = dtype
    self._md_shapes = md_shapes

  def get_input(self, rank: int, world_size: int) -> torch.Tensor:
    """Returns the input tensor for a given rank."""
    if self._input_tensor is not None:
      return self._input_tensor

    if self._dtype.is_complex:
      if self._dtype == torch.cfloat:
        real_dtype = torch.float32
      else:
        real_dtype = torch.float64
      real_val = (
          torch.arange(world_size, dtype=real_dtype) + 1 + rank * world_size
      )
      return torch.complex(real_val, real_val)

    # Input tensors on different ranks:
    # tensor([0, 1, 2, 3, 4, 5, 6, 7])           # Rank 0
    # tensor([8, 9, 10, 11, 12, 13, 14, 15])     # Rank 1
    # tensor([16, 17, 18, 19, 20, 21, 22, 23])   # Rank 2
    # ...
    base = torch.arange(world_size, dtype=self._dtype) + rank * world_size
    if self._md_shapes:
      # this creates a 2D tensor with shape [world_size, 2]
      # each row i will be [base[i], base[i] + 1]
      # Input tensor becomes:
      # tensor([[0, 1], [1, 2], [2, 3], ... [7, 8]])      # Rank 0
      # tensor([[8, 9], [9, 10], [10, 11], ... [15, 16]]) # Rank 1
      # ...
      return base.unsqueeze(-1) + torch.arange(2, dtype=self._dtype)
    return base

  def get_output(self, rank: int, world_size: int) -> torch.Tensor:
    """Returns the expected output tensor for a given rank."""
    if self._output_tensor is not None:
      return self._output_tensor

    if self._dtype.is_complex:
      if self._dtype == torch.cfloat:
        real_dtype = torch.float32
      else:
        real_dtype = torch.float64
      real_val = (
          rank + 1 + torch.arange(world_size, dtype=real_dtype) * world_size
      )
      return torch.complex(real_val, real_val)

    # Expected output tensors on different ranks:
    # tensor([0, 8, 16, 24, 32, 40, 48, 56])    # Rank 0
    # tensor([1, 9, 17, 25, 33, 41, 49, 57])    # Rank 1
    # tensor([2, 10, 18, 26, 34, 42, 50, 58])   # Rank 2
    # ...
    base = torch.arange(world_size, dtype=self._dtype) * world_size + rank
    if self._md_shapes:
      # this creates a 2D tensor with shape [world_size, 2]
      # each row i will be [base[i], base[i] + 1]
      # Output tensor becomes:
      # tensor([[0, 1], [8, 9], [16, 17], ... [56, 57]])  # Rank 0
      # tensor([[1, 2], [9, 10], [17, 18], ... [57, 58]]) # Rank 1
      # ...
      return base.unsqueeze(-1) + torch.arange(2, dtype=self._dtype)
    return base

  def get_input_split_sizes(self, rank: int, world_size: int) -> list[int]:
    del rank, world_size  # Unused in base implementation
    return self._input_split_sizes

  def get_output_split_sizes(self, rank: int, world_size: int) -> list[int]:
    del rank, world_size  # Unused in base implementation
    return self._output_split_sizes

  def get_dtype(self) -> torch.dtype:
    return self._dtype

  def get_output_tensor(self, rank: int, world_size: int) -> torch.Tensor:
    """Returns an empty output tensor with the correct shape."""
    return torch.empty(
        self.get_output(rank, world_size).shape,
        dtype=self._dtype,
    )


def run_all_to_all_single(
    test_data: AllToAllSingleTestData,
) -> None:
  """Tests all-to-all functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensor = test_data.get_input(rank, world_size).to(device)
  output_tensor = test_data.get_output_tensor(rank, world_size).to(device)

  logging.info(
      "Before all_to_all_single (rank=%d, dtype=%s): input = %s, out_splits=%s,"
      " in_splits=%s",
      rank,
      test_data.get_dtype(),
      input_tensor.cpu(),
      test_data.get_output_split_sizes(rank, world_size),
      test_data.get_input_split_sizes(rank, world_size),
  )

  work = torch.distributed.all_to_all_single(
      output_tensor,
      input_tensor,
      test_data.get_output_split_sizes(rank, world_size),
      test_data.get_input_split_sizes(rank, world_size),
      async_op=True,
  )
  logging.info("Rank %d waiting on work...", rank)
  if work is not None:
    work.wait()
  logging.info("Rank %d work wait completed.", rank)

  logging.info(
      "After all_to_all_single (rank=%d, dtype=%s): output = %s",
      rank,
      test_data.get_dtype(),
      output_tensor.cpu(),
  )

  expected_tensor = test_data.get_output(rank, world_size)
  logging.info("Rank %d expected = %s", rank, expected_tensor.cpu())
  utils.assert_close(
      actual=output_tensor.cpu(),
      expected=expected_tensor.cpu(),
      preamble=f"Rank {rank} failed",
  )
  logging.info("Rank %d assert_close passed.", rank)


class AllToAllSingleUnevenTestData:
  """Input and output data generator for uneven all_to_all_single collective tests.

  The communication pattern is defined by a 2D split matrix of dimensions
  [world_size, world_size], where split_matrix[src][dst] specifies the number of
  elements (rows along dimension 0) sent from rank `src` to rank `dst`.
  """

  def __init__(
      self,
      split_matrix: Sequence[Sequence[int]],
      dtype: torch.dtype = torch.float32,
      extra_dims: tuple[int, ...] = (),
      md_shapes: bool = False,
  ):
    """Initializes uneven test data with a split matrix and tensor properties.

    Args:
      split_matrix: 2D matrix where entry [i][j] is the count sent from rank i
        to rank j.
      dtype: Data type of generated tensors.
      extra_dims: Trailing shape dimensions (e.g. (3, 4) for a [rows, 3, 4]
        tensor).
      md_shapes: Convenience flag to add a default trailing dimension of (2,).
    """
    self._split_matrix = split_matrix
    self._world_size = len(split_matrix)
    self._dtype = dtype
    if md_shapes and not extra_dims:
      self._extra_dims = (2,)
    else:
      self._extra_dims = extra_dims

  def get_input_split_sizes(self, rank: int, world_size: int) -> list[int]:
    """Returns the list of slice sizes that `rank` sends to all destination ranks."""
    del world_size
    # Row `rank` contains outgoing counts to destination ranks 0..world_size-1.
    return self._split_matrix[rank]

  def get_output_split_sizes(self, rank: int, world_size: int) -> list[int]:
    """Returns the list of slice sizes that `rank` receives from all source ranks."""
    del world_size
    # Column `rank` contains incoming counts from source ranks 0..world_size-1.
    return [self._split_matrix[src][rank] for src in range(self._world_size)]

  def get_input(self, rank: int, world_size: int) -> torch.Tensor:
    """Constructs the local input tensor for `rank` with identifiable values."""
    del world_size
    in_splits = self.get_input_split_sizes(rank, self._world_size)
    total_rows = sum(in_splits)
    if total_rows == 0:
      # Return empty tensor with proper trailing dimensions if rank sends 0 elements.
      shape = (0, *self._extra_dims)
      return torch.empty(shape, dtype=self._dtype)

    slices = []
    for dst, count in enumerate(in_splits):
      if count > 0:
        # Generate recognizable numbers keyed by (src, dst): e.g., rank 0 -> dst 1 is 1020..
        base_val = (rank + 1) * 1000 + (dst + 1) * 10
        val = torch.arange(base_val, base_val + count, dtype=torch.float32).to(
            self._dtype
        )
        if self._extra_dims:
          val = val.view(-1, *([1] * len(self._extra_dims))).expand(
              -1, *self._extra_dims
          )
        slices.append(val)

    return (
        torch.cat(slices, dim=0)
        if slices
        else torch.empty((0, *self._extra_dims), dtype=self._dtype)
    )

  def get_output(self, rank: int, world_size: int) -> torch.Tensor:
    """Constructs the expected received tensor on `rank` after collective exchange."""
    del world_size
    out_splits = self.get_output_split_sizes(rank, self._world_size)
    total_rows = sum(out_splits)
    if total_rows == 0:
      # Return empty tensor with proper trailing dimensions if rank receives 0 elements.
      shape = (0, *self._extra_dims)
      return torch.empty(shape, dtype=self._dtype)

    # In all_to_all_single, received slices arrive sequentially in source rank order (0..N-1).
    slices = []
    for src in range(self._world_size):
      count = self._split_matrix[src][rank]
      if count > 0:
        base_val = (src + 1) * 1000 + (rank + 1) * 10
        val = torch.arange(base_val, base_val + count, dtype=torch.float32).to(
            self._dtype
        )
        if self._extra_dims:
          val = val.view(-1, *([1] * len(self._extra_dims))).expand(
              -1, *self._extra_dims
          )
        slices.append(val)

    return (
        torch.cat(slices, dim=0)
        if slices
        else torch.empty((0, *self._extra_dims), dtype=self._dtype)
    )

  def get_dtype(self) -> torch.dtype:
    """Returns the tensor element data type."""
    return self._dtype

  def get_output_tensor(self, rank: int, world_size: int) -> torch.Tensor:
    """Allocates a pre-sized destination output buffer for the collective call."""
    return torch.zeros(
        self.get_output(rank, world_size).shape,
        dtype=self._dtype,
    )


class AllToAllSingleCollectiveTest(
    continuous_test_utils.TpuMultiProcContinuousTest
):

  world_size = distributed_utils.get_tpu_device_count()

  def test_no_split_sizes_1d(self):
    rank_data = AllToAllSingleTestData()
    run_all_to_all_single(rank_data)

  def test_no_split_sizes_multi_dim(self):
    rank_data = AllToAllSingleTestData(md_shapes=True)
    run_all_to_all_single(rank_data)

  def test_with_equal_splits(self):
    splits = [1] * self.world_size
    rank_data = AllToAllSingleTestData(
        input_split_sizes=splits, output_split_sizes=splits
    )
    run_all_to_all_single(rank_data)

  def test_uneven_splits_1d(self):
    # matrix of split sizes: rank i sends 2 to (i + 1) % world_size, 1 to others
    # (total world_size + 1)
    matrix = [
        [
            (2 if j == (i + 1) % self.world_size else 1)
            for j in range(self.world_size)
        ]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.int32)
    run_all_to_all_single(rank_data)

  def test_uneven_splits_multi_dim(self):
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(
        matrix, dtype=torch.bfloat16, md_shapes=True
    )
    run_all_to_all_single(rank_data)

  def test_uneven_splits_zero_sized_slices(self):
    matrix = [
        [(1 if (i + j) % 2 == 0 else 0) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.float32)
    run_all_to_all_single(rank_data)

  def test_uneven_splits_cyclic_pattern(self):
    matrix = [
        [
            (3 if j == (i + 1) % self.world_size else 0)
            for j in range(self.world_size)
        ]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.int32)
    run_all_to_all_single(rank_data)

  def test_uneven_splits_float32(self):
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.float32)
    run_all_to_all_single(rank_data)

  def test_uneven_splits_asymmetric(self):
    # Asymmetric split pattern with a mixture of sizes (3, 2, 0, 1) per peer.
    matrix = [
        [
            (
                3
                if j == (i + 1) % self.world_size
                else (
                    2
                    if j == (i + 2) % self.world_size
                    else (0 if j == (i + 3) % self.world_size else 1)
                )
            )
            for j in range(self.world_size)
        ]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.int32)
    run_all_to_all_single(rank_data)

  def test_uneven_splits_3d_float32(self):
    # 3D tensor with trailing shape (2, 4) and float32
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(
        matrix, dtype=torch.float32, extra_dims=(2, 4)
    )
    run_all_to_all_single(rank_data)

  def test_uneven_splits_1d_factoring_and_asymmetric(self):
    """Tests 1024-aligned 1D splits, asymmetric buffers, and subgroups."""
    run_dist_all_to_all_single_1d_factoring_and_asymmetric_test()

  def test_uneven_splits_large_asymmetric_zero_row_and_stream(self):
    """Tests multi-MiB asymmetric buffers and a 0-row rank on a side stream."""
    run_dist_all_to_all_single_large_asymmetric_zero_row_and_stream_test()


def _fill_1d_step_input_and_expected(
    matrix: list[list[int]],
    rank: int,
    step: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Builds 1D CPU input and expected output tensors for one uneven step."""
  group_size = len(matrix)
  send_sizes = matrix[rank]
  total_send = sum(send_sizes)
  in_cpu = torch.empty(total_send, dtype=dtype)
  offset = 0
  for dst, count in enumerate(send_sizes):
    if count > 0:
      value = float((step + 1) * 100 + (rank + 1) * 10 + (dst + 1))
      in_cpu[offset : offset + count] = value
      offset += count

  expected_parts = []
  for src in range(group_size):
    count = matrix[src][rank]
    if count > 0:
      value = float((step + 1) * 100 + (src + 1) * 10 + (rank + 1))
      expected_parts.append(torch.full((count,), value, dtype=dtype))
  if expected_parts:
    expected = torch.cat(expected_parts)
  else:
    expected = torch.empty(0, dtype=dtype)
  return in_cpu, expected


def _run_uneven_1d_matrix_and_verify(
    matrix: list[list[int]],
    dtype: torch.dtype,
    *,
    extra_pad: int = 0,
) -> None:
  """Runs 1D uneven all_to_all_single (optionally on prefix-slice views)."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  send_sizes = matrix[rank]
  recv_sizes = [matrix[src][rank] for src in range(world_size)]
  total_send = sum(send_sizes)
  total_recv = sum(recv_sizes)

  in_backing = torch.full(
      (total_send + extra_pad,), -3.0, dtype=dtype, device="tpu"
  )
  out_backing = torch.full(
      (total_recv + extra_pad,), -7.0, dtype=dtype, device="tpu"
  )
  in_cpu, expected = _fill_1d_step_input_and_expected(
      matrix, rank, step=0, dtype=dtype
  )
  if total_send > 0:
    in_backing[:total_send].copy_(in_cpu.to("tpu"))

  dist.all_to_all_single(
      out_backing[:total_recv],
      in_backing[:total_send],
      output_split_sizes=recv_sizes,
      input_split_sizes=send_sizes,
  )

  if total_recv > 0:
    utils.assert_close(out_backing[:total_recv].cpu(), expected)

  if extra_pad > 0:
    utils.assert_close(
        out_backing[total_recv:].cpu(),
        torch.full((extra_pad,), -7.0, dtype=dtype),
    )


def run_dist_all_to_all_single_1d_factoring_and_asymmetric_test() -> None:
  """Tests 1024-aligned 1D uneven splits, asymmetric buffers, and subgroups."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # Test 1024-aligned 1D uneven splits (bfloat16 and float32) on prefix-slice
  # views of larger backing buffers.
  for dtype, scale in [
      (torch.bfloat16, 1),
      (torch.bfloat16, 2),
      (torch.float32, 1),
  ]:
    chunk = _1D_FACTOR_UNIT * scale
    matrix = [
        [
            (2 * chunk if j == (i + 1) % world_size else chunk)
            for j in range(world_size)
        ]
        for i in range(world_size)
    ]
    _run_uneven_1d_matrix_and_verify(matrix, dtype, extra_pad=512)

  # Test asymmetric cross-rank buffer sizes with varying capacities and
  # 0-element ranks.
  asymmetric_matrix = [
      [_1D_FACTOR_UNIT * (src + 1) for _ in range(world_size)]
      for src in range(world_size)
  ]
  _run_uneven_1d_matrix_and_verify(asymmetric_matrix, torch.bfloat16)

  # Test gather (rank 0 receives `_1D_FACTOR_UNIT` from all ranks; peers
  # receive 0) and scatter (rank 0 sends `_1D_FACTOR_UNIT` to all ranks; ranks
  # 1..N-1 send 0).
  owned_gather = [
      [_1D_FACTOR_UNIT if dst == 0 else 0 for dst in range(world_size)]
      for _ in range(world_size)
  ]
  owned_scatter = [
      [_1D_FACTOR_UNIT if src == 0 else 0 for _ in range(world_size)]
      for src in range(world_size)
  ]
  _run_uneven_1d_matrix_and_verify(owned_gather, torch.bfloat16, extra_pad=512)
  _run_uneven_1d_matrix_and_verify(owned_scatter, torch.bfloat16, extra_pad=512)

  # Exercise single-host `/dev/shm` metadata exchange on concurrent subgroups
  # with distinct uneven split matrices so store prefixes stay isolated.
  subgroup_size = world_size // 2
  group_lo = dist.new_group(list(range(subgroup_size)))
  group_hi = dist.new_group(list(range(subgroup_size, world_size)))
  try:
    if rank < subgroup_size:
      my_group = group_lo
      local_sub_rank = rank
      sub_matrix = [
          [
              _1D_FACTOR_UNIT * (3 if dst == (src + 1) % subgroup_size else 1)
              for dst in range(subgroup_size)
          ]
          for src in range(subgroup_size)
      ]
    else:
      my_group = group_hi
      local_sub_rank = rank - subgroup_size
      sub_matrix = [
          [
              _1D_FACTOR_UNIT
              * (
                  2
                  if dst in (
                      (src + 1) % subgroup_size,
                      (src + 2) % subgroup_size,
                  )
                  else 1
              )
              for dst in range(subgroup_size)
          ]
          for src in range(subgroup_size)
      ]

    sub_send = sub_matrix[local_sub_rank]
    sub_recv = [sub_matrix[src][local_sub_rank] for src in range(subgroup_size)]
    sub_in_cpu, expected_sub = _fill_1d_step_input_and_expected(
        sub_matrix,
        local_sub_rank,
        10 if rank >= subgroup_size else 0,
        torch.bfloat16,
    )
    sub_in = sub_in_cpu.to("tpu")
    sub_out = torch.zeros(sum(sub_recv), dtype=torch.bfloat16, device="tpu")
    dist.all_to_all_single(sub_out, sub_in, sub_recv, sub_send, group=my_group)
    utils.assert_close(sub_out.cpu(), expected_sub)
  finally:
    dist.destroy_process_group(group_hi)
    dist.destroy_process_group(group_lo)


def run_dist_all_to_all_single_large_asymmetric_zero_row_and_stream_test() -> (
    None
):
  """Tests multi-MiB asymmetric buffers with a 0-row rank on a side stream.

  Allocates rank-dependent prior HBM buffers so stack bases differ across ranks,
  dispatches on a non-default TPU stream, and sends 2 MiB chunks into a 16 MiB
  rank-0 output buffer while the last rank holds a 0-element output buffer.
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  non_owner_rank = world_size - 1
  owner_chunk_elems = 512 * _1D_FACTOR_UNIT

  prior_hbm_padding = torch.ones(
      (rank + 1) * 256 * _1D_FACTOR_UNIT,
      dtype=torch.float32,
      device="tpu",
  )

  matrix = []
  for src in range(world_size):
    row = []
    for dst in range(world_size):
      if dst == 0:
        count = owner_chunk_elems
      elif dst == non_owner_rank:
        count = 0
      else:
        count = ((src + dst) % 2 + 1) * 128 * _1D_FACTOR_UNIT
      row.append(count)
    matrix.append(row)

  send_sizes = matrix[rank]
  recv_sizes = [matrix[src][rank] for src in range(world_size)]
  total_send = sum(send_sizes)
  total_recv = sum(recv_sizes)

  in_buf = torch.zeros(total_send, dtype=torch.float32, device="tpu")
  if total_recv > 0:
    out_buf = torch.zeros(total_recv, dtype=torch.float32, device="tpu")
  else:
    out_buf = torch.empty((0,), dtype=torch.float32, device="tpu")

  transfer_stream = torch.tpu.Stream()
  for step in range(2):
    in_cpu, expected_cpu = _fill_1d_step_input_and_expected(
        matrix, rank, step, torch.float32
    )
    in_buf.copy_(in_cpu)
    default_stream = torch.accelerator.current_stream()
    transfer_stream.wait_stream(default_stream)
    with torch.tpu.stream(transfer_stream):
      dist.all_to_all_single(out_buf, in_buf, recv_sizes, send_sizes)
    default_stream.wait_stream(transfer_stream)
    if total_recv > 0:
      utils.assert_close(out_buf.cpu(), expected_cpu)
    else:
      assert out_buf.numel() == 0, f"Expected empty output, got {out_buf.shape}"

  # Keep the rank-dependent padding live until both collective steps finish so
  # HBM stack bases remain asymmetric across ranks during execution.
  del prior_hbm_padding  # Unused.


class AllToAllTestData:
  """Input and output data for all_to_all tests where data is dependent on rank and world_size."""

  def __init__(
      self,
      dtype: torch.dtype = torch.int64,
      md_shapes: bool = False,
  ):
    self._dtype = dtype
    self._md_shapes = md_shapes

  def _tensor(self, data: Sequence[int]):
    if self._dtype.is_complex:
      real_dtype = (
          torch.float32 if self._dtype == torch.cfloat else torch.float64
      )
      return torch.complex(
          torch.tensor(data, dtype=real_dtype),
          torch.tensor(data, dtype=real_dtype),
      )
    return torch.tensor(data, dtype=self._dtype)

  def get_input_tensors(
      self, rank: int, world_size: int
  ) -> Sequence[torch.Tensor]:
    """Returns the input tensor list for a given rank."""
    if self._md_shapes:
      return [
          self._tensor(
              [rank * world_size + i, rank * world_size + i + 1],
          ).reshape(2, 1)
          for i in range(world_size)
      ]
    return [self._tensor([rank * world_size + i]) for i in range(world_size)]

  def get_expected_output_tensors(
      self, rank: int, world_size: int
  ) -> Sequence[torch.Tensor]:
    """Returns the expected output tensor list for a given rank."""
    if self._md_shapes:
      return [
          self._tensor(
              [i * world_size + rank, i * world_size + rank + 1],
          ).reshape(2, 1)
          for i in range(world_size)
      ]
    return [self._tensor([i * world_size + rank]) for i in range(world_size)]

  def get_output_tensors(
      self, rank: int, world_size: int
  ) -> Sequence[torch.Tensor]:
    """Returns an empty output tensor list with the correct shape."""
    return [
        torch.empty_like(t)
        for t in self.get_expected_output_tensors(rank, world_size)
    ]

  def get_dtype(self) -> torch.dtype:
    return self._dtype


class AllToAllNonUniformTestData:
  """Input and output data for non-uniform tensor list all_to_all tests."""

  def __init__(
      self,
      matrix: Sequence[Sequence[int]],
      dtype: torch.dtype = torch.int32,
      extra_dims: tuple[int, ...] = (),
  ):
    self._matrix = matrix
    self._dtype = dtype
    self._extra_dims = extra_dims

  def _make_tensor(self, count: int, offset: int) -> torch.Tensor:
    shape = (count,) + self._extra_dims
    num_elems = count * math.prod(self._extra_dims)
    return torch.arange(num_elems, dtype=self._dtype).reshape(shape) + offset

  def get_input_tensors(
      self, rank: int, world_size: int
  ) -> Sequence[torch.Tensor]:
    return [
        self._make_tensor(self._matrix[rank][dst], rank * 100 + dst * 10)
        for dst in range(world_size)
    ]

  def get_expected_output_tensors(
      self, rank: int, world_size: int
  ) -> Sequence[torch.Tensor]:
    return [
        self._make_tensor(self._matrix[src][rank], src * 100 + rank * 10)
        for src in range(world_size)
    ]

  def get_output_tensors(
      self, rank: int, world_size: int
  ) -> Sequence[torch.Tensor]:
    return [
        torch.empty_like(t)
        for t in self.get_expected_output_tensors(rank, world_size)
    ]

  def get_dtype(self) -> torch.dtype:
    return self._dtype


def run_all_to_all(
    test_data: AllToAllTestData | AllToAllNonUniformTestData,
) -> None:
  """Tests all-to-all functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensors = [
      t.to(device) for t in test_data.get_input_tensors(rank, world_size)
  ]
  output_tensors = [
      t.to(device) for t in test_data.get_output_tensors(rank, world_size)
  ]

  logging.info(
      "Before all_to_all (rank=%d, dtype=%s): input = %s",
      rank,
      test_data.get_dtype(),
      [t.cpu() for t in input_tensors],
  )

  torch.distributed.all_to_all(
      output_tensors,
      input_tensors,
      async_op=True,
  ).wait()

  logging.info(
      "After all_to_all (rank=%d, dtype=%s): output = %s",
      rank,
      test_data.get_dtype(),
      [t.cpu() for t in output_tensors],
  )

  expected_output_tensors = test_data.get_expected_output_tensors(
      rank, world_size
  )
  for i, output_tensor in enumerate(output_tensors):
    utils.assert_close(
        actual=output_tensor.cpu(),
        expected=expected_output_tensors[i].cpu(),
        preamble=f"Rank {rank} failed for output tensor {i}",
    )


class AllToAllCollectiveTest(continuous_test_utils.TpuMultiProcContinuousTest):

  world_size = distributed_utils.get_tpu_device_count()

  def test_uniform_tensors_1d(self):
    rank_data = AllToAllTestData()
    run_all_to_all(rank_data)

  def test_uniform_tensors_multi_dim(self):
    rank_data = AllToAllTestData(md_shapes=True)
    run_all_to_all(rank_data)

  def test_non_uniform_tensors_1d(self):
    # Alternating 1D tensor sizes (1, 2, 1, 2...) across ranks.
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllNonUniformTestData(matrix, dtype=torch.int32)
    run_all_to_all(rank_data)

  def test_non_uniform_tensors_multi_dim(self):
    # Multi-dimensional tensors with trailing hidden dimension (e.g. MoE
    # [tokens, hidden_dim]) in bfloat16.
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllNonUniformTestData(
        matrix, dtype=torch.bfloat16, extra_dims=(2,)
    )
    run_all_to_all(rank_data)

  def test_non_uniform_tensors_zero_sized(self):
    # Zero-sized slices to test empty communication channels between ranks.
    matrix = [
        [(1 if (i + j) % 2 == 0 else 0) for j in range(self.world_size)]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllNonUniformTestData(matrix, dtype=torch.float32)
    run_all_to_all(rank_data)

  def test_non_uniform_tensors_asymmetric(self):
    # Asymmetric split pattern with a mixture of sizes (3, 2, 0, 1) per peer.
    matrix = [
        [
            (
                3
                if j == (i + 1) % self.world_size
                else (
                    2
                    if j == (i + 2) % self.world_size
                    else (0 if j == (i + 3) % self.world_size else 1)
                )
            )
            for j in range(self.world_size)
        ]
        for i in range(self.world_size)
    ]
    rank_data = AllToAllNonUniformTestData(matrix, dtype=torch.int32)
    run_all_to_all(rank_data)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
