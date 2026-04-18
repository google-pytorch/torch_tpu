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

"""Tests PyTorch distributed collective all_to_all errors on multiple TPUs."""

import os
import re

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from tests import error_testing as et
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_all_to_all_single_dtype_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  input_tensor = torch.ones((1, 2), dtype=torch.float64, device="tpu")
  output_tensor = torch.empty((1, 2), dtype=torch.int64, device="tpu")
  # This message comes from native PyTorch.
  expected_msg = re.compile("Invalid usage of tensors with different dtypes.*")
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.all_to_all_single(output_tensor, input_tensor)


def run_all_to_all_single_invalid_tensor_size_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  input_tensor = torch.ones((1, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((1, 2), dtype=torch.int64, device="tpu")
  expected_msg = (
      "distributed.all_to_all_single(): tensor first dimension must be "
      "divisible by process group size, got 8 for process group size "
      "and 1 for tensor shape [1, 2] dim 0"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(output_tensor, input_tensor)


def run_all_to_all_single_invalid_split_sizes_size_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  input_tensor = torch.ones((8, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((8, 2), dtype=torch.int64, device="tpu")
  input_split_sizes = [2] * 9
  output_split_sizes = [1] * 8
  expected_msg = (
      "distributed.all_to_all_single(): split sizes must have the same "
      "size as process group size, got 8 for process group size and "
      "9 for split sizes [2 2 2 2 2 2 2 2 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_invalid_split_sizes_sum_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  input_tensor = torch.ones((8, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((8, 2), dtype=torch.int64, device="tpu")
  input_split_sizes = [1, 3, 1, 1, 1, 1, 1, 1]
  output_split_sizes = [1] * 8
  expected_msg = (
      "distributed.all_to_all_single(): split sizes sum must be equal to tensor"
      " first dimension, got 10 for split sizes [1 3 1 1 1 1 1 1] and 8 for"
      " tensor shape [8, 2] dim 0"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


class AllToAllSingleCollectiveErrorsTest(absltest.TestCase):
  _world_size = 8

  def test_invalid_dtype(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single_dtype_error, world_size=self._world_size
        ),
    )

  def test_invalid_tensor_size(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single_invalid_tensor_size_error,
            world_size=self._world_size,
        ),
    )

  def test_invalid_split_sizes_size(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single_invalid_split_sizes_size_error,
            world_size=self._world_size,
        ),
    )

  def test_invalid_split_sizes_sum(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single_invalid_split_sizes_sum_error,
            world_size=self._world_size,
        ),
    )


def run_all_to_all_unequal_tensor_list_size_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  expected_msg = (
      "distributed.all_to_all(): output and input tensors must have the same"
      " number of tensors, got 9 for output and 8 for input"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_wrong_num_input_tensors_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  expected_msg = (
      "distributed.all_to_all(): input tensors must have the same number of"
      " tensors as the process group size, got 9 for input and 8 for process"
      " group size"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_unequal_input_tensor_shape_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones(2, device=device, dtype=torch.int64)
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  output_tensors[1] = torch.ones(2, device=device, dtype=torch.int64)
  expected_msg = (
      "distributed.all_to_all(): all input tensors must be of same shape, got"
      " [2] at index 1 and [1] at index 0"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_unequal_input_output_tensor_shape_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  output_tensors[0] = torch.ones(2, device=device, dtype=torch.int64)
  expected_msg = (
      "distributed.all_to_all(): output and input tensors must have the same"
      " shape, got [2] for output and [1] for input at index 0"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


class AllToAllCollectiveErrorsTest(absltest.TestCase):
  _world_size = 8

  def test_unequal_tensor_list_size(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_unequal_tensor_list_size_error,
            world_size=self._world_size,
        ),
    )

  def test_wrong_num_input_tensors(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_wrong_num_input_tensors_error,
            world_size=self._world_size,
        ),
    )

  def test_unequal_input_tensor_shape(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_unequal_input_tensor_shape_error,
            world_size=self._world_size,
        ),
    )

  def test_unequal_input_output_tensor_shape(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_unequal_input_output_tensor_shape_error,
            world_size=self._world_size,
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
