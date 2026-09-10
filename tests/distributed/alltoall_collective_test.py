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

- For each test case, the main process creates a spec of the world (common
  information all workers need to agree on) and N worker processes (one
  per TPU) based on the same world spec.
- Each worker process is started with a different rank. It initializes
  its environment and runs the test logic for its rank.
- The test logic is expressed as a (rank, world_spec) -> None function. It runs
  PyTorch distributed collectives and checks that they behave as expected.

This isn't quite how this will be used in open source, so we should revisit this
before the open source release.
"""

import math
import os
from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


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
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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

  dist.barrier()
  logging.info("Rank %d barrier passed.", rank)
  dist.destroy_process_group()


class AllToAllSingleUnevenTestData:
  """Input and output data generator for uneven all_to_all_single collective tests.

  The communication pattern is defined by a 2D split matrix of dimensions
  [world_size, world_size], where split_matrix[src][dst] specifies the number of
  elements (rows along dimension 0) sent from rank `src` to rank `dst`.
  """

  def __init__(
      self,
      split_matrix: list[list[int]],
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


class AllToAllSingleCollectiveTest(seed_test_utils.MultiProcessRepeatableTest):

  _world_size = 8

  def test_no_split_sizes_1d(self):
    rank_data = AllToAllSingleTestData()
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_no_split_sizes_multi_dim(self):
    rank_data = AllToAllSingleTestData(md_shapes=True)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_with_equal_splits(self):
    splits = [1, 1, 1, 1, 1, 1, 1, 1]
    rank_data = AllToAllSingleTestData(
        input_split_sizes=splits, output_split_sizes=splits
    )
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_1d(self):
    # matrix of split sizes: rank i sends 2 to (i+1)%8, 1 to others (total 9)
    matrix = [
        [
            (2 if j == (i + 1) % self._world_size else 1)
            for j in range(self._world_size)
        ]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.int32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_multi_dim(self):
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(
        matrix, dtype=torch.bfloat16, md_shapes=True
    )
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_zero_sized_slices(self):
    matrix = [
        [(1 if (i + j) % 2 == 0 else 0) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.float32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_cyclic_pattern(self):
    matrix = [
        [
            (3 if j == (i + 1) % self._world_size else 0)
            for j in range(self._world_size)
        ]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.int32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_float32(self):
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.float32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_asymmetric(self):
    # Asymmetric split pattern with a mixture of sizes (3, 2, 0, 1) per peer.
    matrix = [
        [
            (
                3
                if j == (i + 1) % self._world_size
                else (
                    2
                    if j == (i + 2) % self._world_size
                    else (0 if j == (i + 3) % self._world_size else 1)
                )
            )
            for j in range(self._world_size)
        ]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(matrix, dtype=torch.int32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uneven_splits_3d_float32(self):
    # 3D tensor with trailing shape (2, 4) and float32
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllSingleUnevenTestData(
        matrix, dtype=torch.float32, extra_dims=(2, 4)
    )
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single, world_size=self._world_size
        ),
        test_data=rank_data,
    )


class AllToAllTestData:
  """Input and output data for all_to_all tests where data is dependent on rank and world_size."""

  def __init__(
      self,
      dtype: torch.dtype = torch.int64,
      md_shapes: bool = False,
  ):
    self._dtype = dtype
    self._md_shapes = md_shapes

  def _tensor(self, data: list[int]):
    if self._dtype.is_complex:
      real_dtype = (
          torch.float32 if self._dtype == torch.cfloat else torch.float64
      )
      return torch.complex(
          torch.tensor(data, dtype=real_dtype),
          torch.tensor(data, dtype=real_dtype),
      )
    return torch.tensor(data, dtype=self._dtype)

  def get_input_tensors(self, rank: int, world_size: int) -> list[torch.Tensor]:
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
  ) -> list[torch.Tensor]:
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
  ) -> list[torch.Tensor]:
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
      matrix: list[list[int]],
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

  def get_input_tensors(self, rank: int, world_size: int) -> list[torch.Tensor]:
    return [
        self._make_tensor(self._matrix[rank][dst], rank * 100 + dst * 10)
        for dst in range(world_size)
    ]

  def get_expected_output_tensors(
      self, rank: int, world_size: int
  ) -> list[torch.Tensor]:
    return [
        self._make_tensor(self._matrix[src][rank], src * 100 + rank * 10)
        for src in range(world_size)
    ]

  def get_output_tensors(
      self, rank: int, world_size: int
  ) -> list[torch.Tensor]:
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
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


class AllToAllCollectiveTest(seed_test_utils.MultiProcessRepeatableTest):

  _world_size = 8

  def test_uniform_tensors_1d(self):
    rank_data = AllToAllTestData()
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_uniform_tensors_multi_dim(self):
    rank_data = AllToAllTestData(md_shapes=True)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_non_uniform_tensors_1d(self):
    # Alternating 1D tensor sizes (1, 2, 1, 2...) across ranks.
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllNonUniformTestData(matrix, dtype=torch.int32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_non_uniform_tensors_multi_dim(self):
    # Multi-dimensional tensors with trailing hidden dimension (e.g. MoE
    # [tokens, hidden_dim]) in bfloat16.
    matrix = [
        [(1 if (i + j) % 2 == 0 else 2) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllNonUniformTestData(
        matrix, dtype=torch.bfloat16, extra_dims=(2,)
    )
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_non_uniform_tensors_zero_sized(self):
    # Zero-sized slices to test empty communication channels between ranks.
    matrix = [
        [(1 if (i + j) % 2 == 0 else 0) for j in range(self._world_size)]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllNonUniformTestData(matrix, dtype=torch.float32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all, world_size=self._world_size
        ),
        test_data=rank_data,
    )

  def test_non_uniform_tensors_asymmetric(self):
    # Asymmetric split pattern with a mixture of sizes (3, 2, 0, 1) per peer.
    matrix = [
        [
            (
                3
                if j == (i + 1) % self._world_size
                else (
                    2
                    if j == (i + 2) % self._world_size
                    else (0 if j == (i + 3) % self._world_size else 1)
                )
            )
            for j in range(self._world_size)
        ]
        for i in range(self._world_size)
    ]
    rank_data = AllToAllNonUniformTestData(matrix, dtype=torch.int32)
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all, world_size=self._world_size
        ),
        test_data=rank_data,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
