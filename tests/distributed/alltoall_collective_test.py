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

import os
from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


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
      "Before all_to_all_single (rank=%d, dtype=%s): input = %s",
      rank,
      test_data.get_dtype(),
      input_tensor.cpu(),
  )

  torch.distributed.all_to_all_single(
      output_tensor,
      input_tensor,
      test_data.get_output_split_sizes(rank, world_size),
      test_data.get_input_split_sizes(rank, world_size),
      async_op=True,
  ).wait()

  logging.info(
      "After all_to_all_single (rank=%d, dtype=%s): output = %s",
      rank,
      test_data.get_dtype(),
      output_tensor.cpu(),
  )

  expected_tensor = test_data.get_output(rank, world_size)
  utils.assert_close(
      actual=output_tensor.cpu(),
      expected=expected_tensor.cpu(),
      preamble=f"Rank {rank} failed",
  )


class AllToAllSingleCollectiveTest(absltest.TestCase):
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


def run_all_to_all(
    test_data: AllToAllTestData,
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


class AllToAllCollectiveTest(absltest.TestCase):
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


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
