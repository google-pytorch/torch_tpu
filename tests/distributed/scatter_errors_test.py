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

import os
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests import seed_test_utils
from tests.distributed import distributed_utils


def run_scatter_wrong_number_inputs(num_inputs: int) -> None:
  """Runs scatter with number of inputs not equal to process group size."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert num_inputs != world_size
  src_rank = 0
  size = 5
  inputs = None
  if rank == src_rank:
    inputs = [
        torch.zeros(size, device="tpu", dtype=torch.float32)
        for _ in range(num_inputs)
    ]
  output = torch.empty(size, device="tpu", dtype=torch.float32)

  # In PyTorch distributed, parameter validation for root-specific inputs occurs
  # immediately on rank 0 prior to collective dispatch. If rank 0 raises an
  # exception and exits while non-root ranks (1..N-1) call dist.scatter(), the
  # non-root ranks enter the C++ collective wait queue and deadlock waiting for
  # rank 0, causing test timeouts in OSS CI.
  #
  # We synchronize the validation status across all ranks beforehand:
  # - Rank 0 invokes dist.scatter() to trigger the expected RuntimeError.
  # - Non-root ranks exit cleanly instead of blocking indefinitely in C++.
  is_valid = torch.tensor(
      [1 if (inputs is None or len(inputs) == world_size) else 0],
      device="tpu",
      dtype=torch.int32,
  )
  dist.broadcast(is_valid, src=src_rank)

  if is_valid.item() == 0:
    if rank == src_rank:
      torch.distributed.scatter(output, inputs, src=src_rank)
    return

  torch.distributed.scatter(output, inputs, src=src_rank)


def run_scatter_wrong_shape_output(
    input_shape: list[int],
    output_shape: list[int],
) -> None:
  """Runs scatter with differently-shaped input and output tensors."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert input_shape != output_shape
  src_rank = 0
  inputs = None
  if rank == src_rank:
    inputs = [
        torch.zeros(input_shape, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
  output = torch.empty(output_shape, device="tpu", dtype=torch.float32)

  # Output shape vs. input shape validation only fails on rank 0 because
  # non-root ranks have `inputs = None`. Non-root ranks will not raise this
  # validation error and would otherwise block inside dist.scatter() waiting
  # for rank 0.
  # We broadcast the validation status from rank 0 so that only rank 0 triggers
  # the targeted error while peer ranks exit cleanly.
  is_valid = torch.tensor(
      [1 if (inputs is None or inputs[0].shape == output.shape) else 0],
      device="tpu",
      dtype=torch.int32,
  )
  dist.broadcast(is_valid, src=src_rank)

  if is_valid.item() == 0:
    if rank == src_rank:
      torch.distributed.scatter(output, inputs, src=src_rank)
    return

  torch.distributed.scatter(output, inputs, src=src_rank)


def run_scatter_mismatch_input_shapes(
    shape: list[int],
    mismatch_shape: list[int],
) -> None:
  """Runs scatter with one input having a different shape than the others."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert shape != mismatch_shape
  src_rank = 0
  inputs = None
  if rank == src_rank:
    inputs = [
        torch.zeros(shape, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
    inputs[0] = torch.zeros(mismatch_shape, device="tpu", dtype=torch.float32)
  output = torch.empty(shape, device="tpu", dtype=torch.float32)

  # Mismatched input shapes among the list of tensors is an error caught
  # only on the root rank. If non-root ranks enter dist.scatter(), they hang
  # indefinitely when rank 0 crashes early with the RuntimeError.
  # Broadcast the validity flag to ensure non-root ranks do not call scatter.
  is_valid = torch.tensor(
      [
          1
          if (inputs is None or all(t.shape == inputs[0].shape for t in inputs))
          else 0
      ],
      device="tpu",
      dtype=torch.int32,
  )
  dist.broadcast(is_valid, src=src_rank)

  if is_valid.item() == 0:
    if rank == src_rank:
      torch.distributed.scatter(output, inputs, src=src_rank)
    return

  torch.distributed.scatter(output, inputs, src=src_rank)


class ScatterErrorsTest(seed_test_utils.MultiProcessRepeatableTest):

  # We check the error in the parent process as opposed to each worker because
  # only some of the workers will raise the error.

  def test_scatter_wrong_number_inputs(self):
    with et.assert_subprocess_raises_message(
        RuntimeError,
        "distributed.scatter(): expected the number of input tensors on the"
        " root rank to be equal to the group size, got 9 tensors and 8"
        " processes",
    ):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=singlehost_wrapper.tpu_env_wrapper(
              run_scatter_wrong_number_inputs
          ),
          num_inputs=9,
      )

  def test_scatter_wrong_shape_output(self):
    input_shape = [2, 3]
    output_shape = [3, 3]
    with et.assert_subprocess_raises_message(
        RuntimeError,
        "distributed.scatter(): expected output tensor shape to match input"
        " tensor shape, got [3, 3] and [2, 3]",
    ):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=singlehost_wrapper.tpu_env_wrapper(run_scatter_wrong_shape_output),
          input_shape=input_shape,
          output_shape=output_shape,
      )

  def test_scatter_mismatch_input_shapes(self):
    shape = [6, 2]
    mismatch_shape = [8, 6, 2]
    with et.assert_subprocess_raises_message(
        RuntimeError,
        "distributed.scatter(): input tensors on the root rank: expected all"
        " input tensors to have the same shape, got [8, 6, 2] at index 0 and"
        " [6, 2] at index 1",
    ):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=singlehost_wrapper.tpu_env_wrapper(
              run_scatter_mismatch_input_shapes
          ),
          shape=shape,
          mismatch_shape=mismatch_shape,
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
