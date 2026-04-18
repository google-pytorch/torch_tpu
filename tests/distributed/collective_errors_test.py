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

"""Tests PyTorch distributed collectives on multiple TPUs.

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

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from tests import error_testing as et
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_all_reduce_error() -> None:
  """Tests all-reduce error."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device="tpu")
  # BAND is only supported for integer tensors. This should raise RuntimeError.
  handle = torch.distributed.all_reduce(
      x, op=torch.distributed.ReduceOp.BAND, async_op=True
  )
  handle.wait()


def run_all_gather() -> None:
  """Tests all-gather functionality."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.tensor([1.0, float(rank)], device="tpu", dtype=torch.float32)
  outputs = [
      torch.zeros(2, device="tpu", dtype=torch.float32)
      for _ in range(world_size)
  ]
  handle = torch.distributed.all_gather(outputs, x, async_op=True)
  handle.wait()
  outputs_cpu = [o.to("cpu") for o in outputs]

  for i, output in enumerate(outputs_cpu):
    assert (
        output[0] == 1.0
    ), f"Got {output[0]} for the first element of tensor {i}, expected 1.0"
    assert output[1] == i, (
        f"Got {output[1]} for the second element of tensor {i}, expected {i}."
        f" Full outputs: {outputs_cpu}."
    )


def run_all_gather_tensor_wrong_number_output_dimensions() -> None:
  """Tests incorrect output dimensionality."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty((world_size, 2, 2, 1), device="tpu", dtype=torch.float32)
  expected_msg = (
      "distributed.all_gather_into_tensor(): invalid output tensor shape."
      " Number of output dimensions must equal number of input dimensions"
      " (concatenation) or input dimensions + 1 (stacking). Got output shape"
      f" [{world_size}, 2, 2, 1], input shape [2, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_wrong_concat_dimension() -> None:
  """Tests incorrect size along concatenation dimension."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty(
      (2 * world_size - 1, 2), device="tpu", dtype=torch.float32
  )
  expected_msg = (
      "distributed.all_gather_into_tensor(): for concatenation, output tensor"
      " size at dimension 0 must be world size * input tensor size at"
      f" dimension 0. Got output shape [{2 * world_size - 1}, 2], input shape"
      f" [2, 2], world size {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_wrong_stack_dimension() -> None:
  """Tests incorrect size along stacking dimension."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty(
      (world_size - 1, 2, 2), device="tpu", dtype=torch.float32
  )
  expected_msg = (
      "distributed.all_gather_into_tensor(): for stacking, output tensor size"
      f" at dimension 0 must be world size. Got output shape [{world_size - 1},"
      f" 2, 2], input shape [2, 2], world size {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_stack_mismatched_dim() -> None:
  """Tests incorrect size along a non-stacking dimension."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty((world_size, 3, 2), device="tpu", dtype=torch.float32)
  expected_msg = (
      "distributed.all_gather_into_tensor(): for stacking, output tensor shape"
      " must match input tensor shape along all other dimensions. Got output"
      f" shape [{world_size}, 3, 2], input shape [2, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_concat_mismatched_dim() -> None:
  """Tests incorrect size along a non-concatenation dimension."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty((world_size * 2, 3), device="tpu", dtype=torch.float32)
  expected_msg = (
      "distributed.all_gather_into_tensor(): for concatenation, output tensor"
      " shape must match input tensor shape along all other dimensions. Got"
      f" output shape [{world_size * 2}, 3], input shape [2, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_wrong_scalar_dimension() -> None:
  """Tests incorrect size when gathering scalars."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.tensor(1.0, device="tpu", dtype=torch.float32)
  output = torch.empty(
      (world_size, world_size), device="tpu", dtype=torch.float32
  )
  expected_msg = (
      "distributed.all_gather_into_tensor(): for scalar input, output tensor"
      " must be 1-dimensional with size equal to world size. Got output shape"
      f" [{world_size}, {world_size}], world size {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_uneven_output_sizes() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.full(
      (rank + 1,), fill_value=rank, device="tpu", dtype=torch.float32
  )
  outputs = [
      torch.empty(i + 1, device="tpu", dtype=torch.float32)
      for i in range(world_size)
  ]
  expected_msg = (
      "distributed.all_gather(): tensors in the list must have the same"
      " shape, got [1] at index 0 and [2] at index 1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather(outputs, x)
    [o.to("cpu") for o in outputs]  # pylint: disable=expression-not-assigned


def run_all_gather_mismatch_input_size() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  outputs = [
      torch.empty((3, 2), device="tpu", dtype=torch.float32)
      for _ in range(world_size)
  ]
  expected_msg = (
      "distributed.all_gather(): input tensor shape [2, 2] must match output"
      " tensor shape [3, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather(outputs, x)
    [o.to("cpu") for o in outputs]  # pylint: disable=expression-not-assigned


def run_all_gather_mismatch_dtype() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.ones((3, 2), device="tpu", dtype=torch.int32)
  outputs = [
      torch.empty((3, 2), device="tpu", dtype=torch.float32)
      for _ in range(world_size)
  ]
  # This message comes from native PyTorch.
  expected_msg = (
      "Invalid usage of tensors with different dtypesFound torch.float32 and "
      " torch.int32"
  )
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.all_gather(outputs, x)
    [o.to("cpu") for o in outputs]  # pylint: disable=expression-not-assigned


def run_reduce_scatter_errors() -> None:
  """Tests reduce-scatter functionality with invalid inputs."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])

  # Helps the test look nicer.
  def new_inputs():
    return [
        torch.zeros((2, 3), device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]

  # Length of input tensors list must match world size.
  inputs = new_inputs()[0 : (world_size - 1)]  # Short by one.
  output = torch.zeros((2, 3), device="tpu")
  expected_msg = (
      "distributed.reduce_scatter(): length of input tensors list must match"
      f" world size, got {world_size-1} input tensors and"
      f" {world_size} processes."
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.reduce_scatter(output, inputs)

  # Dtypes must match. This is validated in PyTorch layer.
  inputs = new_inputs()
  output = torch.zeros((2, 3), device="tpu", dtype=torch.bfloat16)
  expected_msg = (
      "Invalid usage of tensors with different dtypesFound torch.bfloat16 and "
      " torch.float32"
  )
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.reduce_scatter(output, inputs)

  # Output shape must match the input shape:
  inputs = new_inputs()
  output = torch.zeros((3, 3), device="tpu")  # Wrong shape.
  expected_msg = (
      "distributed.reduce_scatter(): output tensor shape [3, 3] must match"
      " input tensor shape [2, 3]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.reduce_scatter(output, inputs)

  # Input tensors must have the same shape (torch_tpu-specific limitation):
  inputs = new_inputs()
  inputs[3] = torch.zeros((3, 3), device="tpu")  # One shape is different.
  output = torch.zeros((2, 3), device="tpu")
  expected_msg = (
      "distributed.reduce_scatter(): tensors in the list must have the same"
      " shape, got [2, 3] at index 0 and [3, 3] at index 3"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.reduce_scatter(output, inputs)


def run_reduce_scatter_tensor_errors() -> None:
  """Tests reduce_scatter_tensor with invalid inputs."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])

  output = torch.zeros((3, 4, 5), device="tpu")
  input_invalid = torch.zeros((10, 4, 5), device="tpu")

  expected_msg = (
      "distributed.reduce_scatter_tensor(): input tensor shape must be either "
      f"[{world_size*3}, 4, 5] or [{world_size}, 3, 4, 5], but got [10, 4, 5]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.reduce_scatter_tensor(output, input_invalid)


def run_gather_wrong_input_size() -> None:
  """Tests gather failure when input_tensors.size() != 1."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  dst = 0
  pg = dist.group.WORLD
  opts = dist.GatherOptions()
  opts.rootRank = dst
  tensor = torch.zeros(2, device="tpu")
  expected_msg = (
      "distributed.gather(): a single input tensor must be provided, got 2"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    pg.gather([[]], [tensor, tensor], opts)


def run_gather_wrong_output_list_size() -> None:
  """Tests gather failure when output_tensors.size() != 1 on root."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  dst = 0
  pg = dist.group.WORLD
  opts = dist.GatherOptions()
  opts.rootRank = dst
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    expected_msg = (
        "distributed.gather(): there must be a single list of output tensors on"
        " the root rank, got 0"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      pg.gather([], [tensor], opts)
  else:
    pg.gather([], [tensor], opts)


def run_gather_wrong_output_tensor_count() -> None:
  """Tests gather failure when output_tensor_list.size() != world_size."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  dst = 0
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    gather_list = [torch.zeros(2, device="tpu") for _ in range(world_size - 1)]
    expected_msg = (
        "distributed.gather(): the number of output tensors on the root rank"
        f" must be equal to the group size, got {world_size - 1} tensors and"
        f" {world_size} processes"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)
  else:
    torch.distributed.gather(tensor, gather_list=None, dst=dst)


def run_gather_mismatch_input_size() -> None:
  """Tests gather failure when input shape != output shape."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  dst = 0
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    gather_list = [torch.zeros(3, device="tpu") for _ in range(world_size)]
    expected_msg = (
        "distributed.gather(): input tensor shape must match output tensor"
        " shape"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)
  else:
    torch.distributed.gather(tensor, gather_list=None, dst=dst)


def run_gather_non_uniform_output_shapes() -> None:
  """Tests gather failure when output tensors have different shapes."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  dst = 0
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    gather_list = [torch.zeros(2, device="tpu") for _ in range(world_size)]
    gather_list[1] = torch.zeros(3, device="tpu")
    expected_msg = (
        "distributed.gather(): output tensors on the root rank: tensors in the"
        " list must have the same shape, got [2] at index 0 and [3] at index 1"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)
  else:
    torch.distributed.gather(tensor, gather_list=None, dst=dst)


def run_gather_output_on_non_root() -> None:
  """Tests gather failure when output list is provided on non-root."""
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  dst = 0
  pg = dist.group.WORLD
  opts = dist.GatherOptions()
  opts.rootRank = dst
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    # Multiple output lists to fail on root as well.
    expected_msg = (
        "distributed.gather(): there must be a single list of output tensors on"
        " the root rank, got 2"
    )
    gather_list = [torch.zeros(2, device="tpu") for _ in range(world_size)]
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      pg.gather([gather_list, gather_list], [tensor], opts)
  else:
    expected_msg = (
        f"distributed.gather(): on non-root rank {rank} the list of output"
        " tensors must be empty"
    )
    gather_list = [torch.zeros(2, device="tpu") for _ in range(world_size)]
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      pg.gather([gather_list], [tensor], opts)


class CollectiveErrorsTest(et.TpuOnlyErrorTestBase):

  def test_all_gather(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_all_gather, world_size=8),
    )

  def test_all_gather_tensor_wrong_number_output_dimensions(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_tensor_wrong_number_output_dimensions,
            world_size=8,
        ),
    )

  def test_all_gather_tensor_wrong_concat_dimension(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_tensor_wrong_concat_dimension, world_size=8
        ),
    )

  def test_all_gather_tensor_wrong_stack_dimension(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_tensor_wrong_stack_dimension, world_size=8
        ),
    )

  def test_all_gather_tensor_stack_mismatched_dim(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_tensor_stack_mismatched_dim, world_size=8
        ),
    )

  def test_all_gather_tensor_concat_mismatched_dim(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_tensor_concat_mismatched_dim, world_size=8
        ),
    )

  def test_all_gather_tensor_wrong_scalar_dimension(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_tensor_wrong_scalar_dimension, world_size=8
        ),
    )

  def test_all_gather_uneven_output_sizes(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_uneven_output_sizes, world_size=8
        ),
    )

  def test_all_gather_mismatch_input_size(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_mismatch_input_size, world_size=8
        ),
    )

  def test_all_gather_mismatch_dtype(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_mismatch_dtype, world_size=8
        ),
    )

  def test_reduce_scatter_errors(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_reduce_scatter_errors, world_size=8
        ),
    )

  def test_reduce_scatter_tensor_errors(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_reduce_scatter_tensor_errors, world_size=8
        ),
    )

  def test_gather_wrong_input_size(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_wrong_input_size, world_size=8
        ),
    )

  def test_gather_wrong_output_list_size(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_wrong_output_list_size, world_size=8
        ),
    )

  def test_gather_wrong_output_tensor_count(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_wrong_output_tensor_count, world_size=8
        ),
    )

  def test_gather_mismatch_input_size(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_mismatch_input_size, world_size=8
        ),
    )

  def test_gather_non_uniform_output_shapes(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_non_uniform_output_shapes, world_size=8
        ),
    )

  def test_gather_output_on_non_root(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_output_on_non_root, world_size=8
        ),
    )

  def test_all_reduce_error(self):
    expected_msg = (
        "distributed.all_reduce(): bitwise reduction ops "
        "(BAND, BOR, BXOR) are only supported for integer tensors, got"
        " Float"
    )
    with et.assert_subprocess_raises_message(RuntimeError, expected_msg):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=singlehost_wrapper.tpu_env_wrapper(
              run_all_reduce_error, world_size=8
          ),
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
