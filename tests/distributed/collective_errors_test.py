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

import datetime
import operator
import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.sharding import PartitionSpec as P
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import process_group_utils
from torch_tpu._internal.distributed import torchcomm_tpu
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils

try:
  import torchcomms
except ImportError:
  torchcomms = None

ProcessGroupId = process_group_utils.ProcessGroupId

_UNSUPPORTED_64BIT_REDUCE_OPS = (
    (
        torch.int64,
        (
            dist.ReduceOp.PRODUCT,
            dist.ReduceOp.MIN,
            dist.ReduceOp.MAX,
            dist.ReduceOp.BAND,
            dist.ReduceOp.BOR,
            dist.ReduceOp.BXOR,
        ),
    ),
    (
        torch.float64,
        (
            dist.ReduceOp.PRODUCT,
            dist.ReduceOp.MIN,
            dist.ReduceOp.MAX,
        ),
    ),
)


def run_all_reduce_error() -> None:
  """Tests all-reduce error."""
  x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device="tpu")
  expected_msg = (
      "distributed.all_reduce(): expected the tensors dtype to be integer on"
      " bitwise reduction ops (BAND, BOR, BXOR), got float32"
  )
  # BAND is only supported for integer tensors. This should raise RuntimeError.
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    handle = torch.distributed.all_reduce(
        x, op=torch.distributed.ReduceOp.BAND, async_op=True
    )
    handle.wait()

  for dtype, ops in _UNSUPPORTED_64BIT_REDUCE_OPS:
    t = torch.ones(4, dtype=dtype, device="tpu")
    dtype_name = str(dtype).removeprefix("torch.")
    for reduce_op in ops:
      op_name = reduce_op.name.lower()
      expected_64bit_msg = (
          f"distributed.all_reduce(): reduce op {op_name} is not supported for"
          f" {dtype_name} tensors; only sum and avg are supported for 64-bit"
          " dtypes"
      )
      with et.assert_raises_message(
          NotImplementedError, tpu=expected_64bit_msg
      ):
        dist.all_reduce(t, op=reduce_op)


def run_all_gather_tensor_wrong_number_output_dimensions() -> None:
  """Tests incorrect output dimensionality."""
  world_size = dist.get_world_size()
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty((world_size, 2, 2, 1), device="tpu", dtype=torch.float32)
  expected_msg = (
      "distributed.all_gather_into_tensor(): expected number of output"
      " dimensions to be equal the number of input dimensions (concatenation)"
      " or input dimensions + 1 (stacking), got output shape"
      f" [{world_size}, 2, 2, 1] and input shape [2, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_wrong_concat_dimension() -> None:
  """Tests incorrect size along concatenation dimension."""
  world_size = dist.get_world_size()
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty(
      (2 * world_size - 1, 2), device="tpu", dtype=torch.float32
  )
  expected_msg = (
      "distributed.all_gather_into_tensor(): expected output for concatenation"
      " to be a tensor with size equal to world size * input size at"
      " dimension 0, got output shape"
      f" [{2 * world_size - 1}, 2], input shape [2, 2], and world size"
      f" {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_wrong_stack_dimension() -> None:
  """Tests incorrect size along stacking dimension."""
  world_size = dist.get_world_size()
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty(
      (world_size - 1, 2, 2), device="tpu", dtype=torch.float32
  )
  expected_msg = (
      "distributed.all_gather_into_tensor(): expected output for stacking to be"
      " a tensor with size equal to the world size at dimension 0, got output"
      f" shape [{world_size - 1}, 2, 2], input shape [2, 2], and world size"
      f" {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_stack_mismatched_dim() -> None:
  """Tests incorrect size along a non-stacking dimension."""
  world_size = dist.get_world_size()
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty((world_size, 3, 2), device="tpu", dtype=torch.float32)
  expected_msg = (
      "distributed.all_gather_into_tensor(): expected output for stacking to"
      " match the input tensor shape along all other dimensions, got output"
      f" shape [{world_size}, 3, 2] and input shape [2, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_concat_mismatched_dim() -> None:
  """Tests incorrect size along a non-concatenation dimension."""
  world_size = dist.get_world_size()
  x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
  output = torch.empty((world_size * 2, 3), device="tpu", dtype=torch.float32)
  expected_msg = (
      "distributed.all_gather_into_tensor(): expected output for concatenation"
      " to match the input tensor shape along all other dimensions, got output"
      f" shape [{world_size * 2}, 3] and input shape [2, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_tensor_wrong_scalar_dimension() -> None:
  """Tests incorrect size when gathering scalars."""
  world_size = dist.get_world_size()
  x = torch.tensor(1.0, device="tpu", dtype=torch.float32)
  output = torch.empty(
      (world_size, world_size), device="tpu", dtype=torch.float32
  )
  expected_msg = (
      "distributed.all_gather_into_tensor(): expected output for scalar inputs"
      " to be a 1D tensor with size equal to the world size, got output shape"
      f" [{world_size}, {world_size}] and world size {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_gather_into_tensor(output, x)


def run_all_gather_uneven_output_sizes() -> None:
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  x = torch.full(
      (rank + 1,), fill_value=rank, device="tpu", dtype=torch.float32
  )
  outputs = [
      torch.empty(i + 1, device="tpu", dtype=torch.float32)
      for i in range(world_size)
  ]
  with et.assert_raises_message(
      RuntimeError,
      tpu="""distributed.all_gather(): failed to validate output tensors: expected all tensors to have the same shape, got [1] at index 0 and [2] at index 1""",
  ):
    torch.distributed.all_gather(outputs, x)
    [o.to("cpu") for o in outputs]  # pylint: disable=expression-not-assigned


def run_all_gather_mismatch_input_size() -> None:
  world_size = dist.get_world_size()
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
  world_size = dist.get_world_size()
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
  world_size = dist.get_world_size()

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
      "distributed.reduce_scatter(): expected 1 input tensor for each process,"
      f" got {world_size-1} input tensors and {world_size} processes"
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
  diff_idx = world_size - 1
  inputs[diff_idx] = torch.zeros(
      (3, 3), device="tpu"
  )  # One shape is different.
  output = torch.zeros((2, 3), device="tpu")
  with et.assert_raises_message(
      RuntimeError,
      tpu=f"""distributed.reduce_scatter(): failed to validate input tensors: expected all tensors to have the same shape, got [2, 3] at index 0 and [3, 3] at index {diff_idx}""",
  ):
    torch.distributed.reduce_scatter(output, inputs)

  # 64-bit dtypes only support sum and avg reduction ops:
  for dtype, ops in _UNSUPPORTED_64BIT_REDUCE_OPS:
    inputs = [
        torch.zeros((2, 3), device="tpu", dtype=dtype)
        for _ in range(world_size)
    ]
    output = torch.zeros((2, 3), device="tpu", dtype=dtype)
    dtype_name = str(dtype).removeprefix("torch.")
    for reduce_op in ops:
      op_name = reduce_op.name.lower()
      expected_64bit_msg = (
          f"distributed.reduce_scatter(): reduce op {op_name} is not supported"
          f" for {dtype_name} tensors; only sum and avg are supported for"
          " 64-bit dtypes"
      )
      with et.assert_raises_message(
          NotImplementedError, tpu=expected_64bit_msg
      ):
        torch.distributed.reduce_scatter(output, inputs, op=reduce_op)


def run_reduce_scatter_tensor_errors() -> None:
  """Tests reduce_scatter_tensor with invalid inputs."""
  world_size = dist.get_world_size()

  output = torch.zeros((3, 4, 5), device="tpu")
  input_invalid = torch.zeros((10, 4, 5), device="tpu")

  expected_msg = (
      "distributed.reduce_scatter_tensor(): expected the input tensor to be"
      f" either of shape [{world_size*3}, 4, 5] or of shape [{world_size}, 3,"
      " 4, 5], got [10, 4, 5]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.reduce_scatter_tensor(output, input_invalid)

  for dtype, ops in _UNSUPPORTED_64BIT_REDUCE_OPS:
    output = torch.zeros((3, 4), device="tpu", dtype=dtype)
    input_tensor = torch.zeros((world_size * 3, 4), device="tpu", dtype=dtype)
    dtype_name = str(dtype).removeprefix("torch.")
    for reduce_op in ops:
      op_name = reduce_op.name.lower()
      expected_64bit_msg = (
          f"distributed.reduce_scatter_tensor(): reduce op {op_name} is not"
          f" supported for {dtype_name} tensors; only sum and avg are supported"
          " for 64-bit dtypes"
      )
      with et.assert_raises_message(
          NotImplementedError, tpu=expected_64bit_msg
      ):
        torch.distributed.reduce_scatter_tensor(
            output, input_tensor, op=reduce_op
        )


def run_gather_wrong_input_size() -> None:
  """Tests gather failure when input_tensors.size() != 1."""
  dst = 0
  pg = dist.group.WORLD
  opts = dist.GatherOptions()
  opts.rootRank = dst
  tensor = torch.zeros(2, device="tpu")
  expected_msg = "distributed.gather(): expected a single input tensor, got 2"
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    pg.gather([[]], [tensor, tensor], opts)


def run_gather_wrong_output_list_size() -> None:
  """Tests gather failure when output_tensors.size() != 1 on root."""
  rank = dist.get_rank()
  dst = 0
  pg = dist.group.WORLD
  opts = dist.GatherOptions()
  opts.rootRank = dst
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    expected_msg = (
        "distributed.gather(): expected a single list of output tensors on the"
        " root rank, got 0"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      pg.gather([], [tensor], opts)
  # Intentionally do NOT call gather on non-dst ranks.
  # The rank-specific error will prevent the gather() from actually starting;
  # trying to gather() on non-dst ranks would deadlock.


def run_gather_wrong_output_tensor_count() -> None:
  """Tests gather failure when output_tensor_list.size() != world_size."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dst = 0
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    gather_list = [torch.zeros(2, device="tpu") for _ in range(world_size - 1)]
    expected_msg = (
        "distributed.gather(): expected the number of output tensors on the"
        f" root rank to be equal to the group size, got {world_size - 1}"
        f" tensors and {world_size} processes"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)
  # Intentionally do NOT call gather on non-dst ranks.
  # The rank-specific error will prevent the gather() from actually starting;
  # trying to gather() on non-dst ranks would deadlock.


def run_gather_mismatch_input_size() -> None:
  """Tests gather failure when input shape != output shape."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
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
  # Intentionally do NOT call gather on non-dst ranks.
  # The rank-specific error will prevent the gather() from actually starting;
  # trying to gather() on non-dst ranks would deadlock.


def run_gather_non_uniform_output_shapes() -> None:
  """Tests gather failure when output tensors have different shapes."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dst = 0
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    gather_list = [torch.zeros(2, device="tpu") for _ in range(world_size)]
    gather_list[1] = torch.zeros(3, device="tpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""distributed.gather(): failed to validate output tensors on root rank: expected all tensors to have the same shape, got [2] at index 0 and [3] at index 1""",
    ):
      torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)
  # Intentionally do NOT call gather on non-dst ranks.
  # The rank-specific error will prevent the gather() from actually starting;
  # trying to gather() on non-dst ranks would deadlock.


def run_gather_output_on_non_root() -> None:
  """Tests gather failure when output list is provided on non-root."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dst = 0
  pg = dist.group.WORLD
  opts = dist.GatherOptions()
  opts.rootRank = dst
  tensor = torch.zeros(2, device="tpu")
  if rank == dst:
    # Multiple output lists to fail on root as well.
    expected_msg = (
        "distributed.gather(): expected a single list of output tensors on the"
        " root rank, got 2"
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


def run_all_gather_wrong_num_tensors() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones((1, 2), dtype=torch.float32, device="tpu")
  output_tensors = [torch.empty((1, 2), dtype=torch.float32, device="tpu")]
  # This message comes from native PyTorch.
  expected_msg = re.compile(
      rf"distributed\.all_gather\(\): expected output tensor list to have one"
      rf" tensor per process, got 1 tensor and {world_size} processes.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    dist.all_gather(output_tensors, input_tensor)


def run_all_gather_dispatch_failure() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones((1, 2), dtype=torch.float32, device="tpu")
  output_tensors = [
      torch.empty((1, 2), dtype=torch.float32, device="tpu")
      for _ in range(world_size)
  ]
  tt_testing.set_op_dispatch_failure(
      "distributed.all_gather", "forced dispatch failure"
  )
  try:
    expected_msg = re.compile(
        r"distributed\.all_gather\(\): forced dispatch failure.*"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.all_gather(output_tensors, input_tensor)
  finally:
    tt_testing.set_op_dispatch_failure("", "")


def run_scatter_wrong_number_inputs() -> None:
  """Runs scatter with number of inputs not equal to process group size."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  num_inputs = world_size + 1
  assert num_inputs != world_size
  src_rank = 0
  size = 5
  output = torch.empty(size, device="tpu", dtype=torch.float32)
  if rank == src_rank:
    inputs = [
        torch.zeros(size, device="tpu", dtype=torch.float32)
        for _ in range(num_inputs)
    ]
    expected_msg = (
        "distributed.scatter(): expected the number of input tensors on the"
        f" root rank to be equal to the group size, got {num_inputs} tensors"
        f" and {world_size} processes"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.scatter(output, inputs, src=src_rank)
  # Intentionally do NOT call scatter on non-root ranks.
  # The root-rank validation error prevents scatter() from starting; calling
  # scatter() on non-root ranks would block waiting for the root rank.


def run_scatter_wrong_shape_output() -> None:
  """Runs scatter with differently-shaped input and output tensors."""
  input_shape = [2, 3]
  output_shape = [3, 3]
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  assert input_shape != output_shape
  src_rank = 0
  output = torch.empty(output_shape, device="tpu", dtype=torch.float32)
  if rank == src_rank:
    inputs = [
        torch.zeros(input_shape, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
    expected_msg = (
        "distributed.scatter(): expected output tensor shape to match input"
        f" tensor shape, got {output_shape} and {input_shape}"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.scatter(output, inputs, src=src_rank)
  # Intentionally do NOT call scatter on non-root ranks.


def run_scatter_mismatch_input_shapes() -> None:
  """Runs scatter with one input having a different shape than the others."""
  shape = [6, 2]
  mismatch_shape = [8, 6, 2]
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  assert shape != mismatch_shape
  src_rank = 0
  output = torch.empty(shape, device="tpu", dtype=torch.float32)
  if rank == src_rank:
    inputs = [
        torch.zeros(shape, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
    inputs[0] = torch.zeros(mismatch_shape, device="tpu", dtype=torch.float32)
    expected_msg = (
        "distributed.scatter(): failed to validate input tensors on root rank:"
        f" expected all tensors to have the same shape, got {mismatch_shape}"
        f" at index 0 and {shape} at index 1"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.scatter(output, inputs, src=src_rank)
  # Intentionally do NOT call scatter on non-root ranks.


def run_barrier_rejects_foreign_device_ids() -> None:
  """Tests barrier rejects device ids this process does not own."""
  # A process addresses a single TPU, so only [0] is meaningful. When the first
  # device id is out of bounds (e.g. [1]), c10d::ProcessGroup::barrier fails
  # while allocating the dispatch tensor on tpu:1. When the first id is 0 but
  # multiple ids are given, ProcessGroupTpu::barrier rejects device_ids.
  for device_ids, exc_type, msg_regex in (
      (
          [1],
          RuntimeError,
          r"device index 1 is out of bounds of the number of addressable"
          r" devices 1",
      ),
      (
          [0, 0],
          NotImplementedError,
          r"device_ids in barrier options is not supported, except "
          r"device_ids=\[0\] which names this process's own device",
      ),
      (
          [0, 1],
          NotImplementedError,
          r"device_ids in barrier options is not supported, except "
          r"device_ids=\[0\] which names this process's own device",
      ),
  ):
    with et.assert_raises_message(
        exc_type,
        tpu=re.compile(f".*{msg_regex}.*", re.DOTALL),
    ):
      torch.distributed.barrier(device_ids=device_ids)


def run_collective_validation_errors() -> None:
  """Runs synchronous collective input validation error checks."""
  run_barrier_rejects_foreign_device_ids()
  run_all_reduce_error()
  run_all_gather_tensor_wrong_number_output_dimensions()
  run_all_gather_tensor_wrong_concat_dimension()
  run_all_gather_tensor_wrong_stack_dimension()
  run_all_gather_tensor_stack_mismatched_dim()
  run_all_gather_tensor_concat_mismatched_dim()
  run_all_gather_tensor_wrong_scalar_dimension()
  run_all_gather_uneven_output_sizes()
  run_all_gather_mismatch_input_size()
  run_all_gather_mismatch_dtype()
  run_all_gather_wrong_num_tensors()
  run_all_gather_dispatch_failure()
  run_reduce_scatter_errors()
  run_reduce_scatter_tensor_errors()
  run_gather_wrong_input_size()
  run_gather_wrong_output_list_size()
  run_gather_wrong_output_tensor_count()
  run_gather_mismatch_input_size()
  run_gather_non_uniform_output_shapes()
  run_gather_output_on_non_root()
  run_scatter_wrong_number_inputs()
  run_scatter_wrong_shape_output()
  run_scatter_mismatch_input_shapes()


def run_conflicting_device_assignments_error() -> None:
  """Tests that compatible tiled subgroups fuse cleanly while conflicting device assignments fail."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  half = world_size // 2

  low = dist.new_group(ranks=list(range(half)))
  high = dist.new_group(ranks=list(range(half, world_size)))
  subgroup = low if rank < half else high

  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def compatible_tiled_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a)
    dist.all_reduce(b, group=subgroup)
    return a + b

  x = torch.ones(4, device="tpu")
  y = torch.ones(4, device="tpu")
  res = compatible_tiled_fn(x, y).cpu()
  expected = torch.full((4,), float(world_size + half), dtype=torch.float32)
  utils.assert_close(res, expected)

  mesh1 = torch.tpu.pallas.get_pallas_mesh(axis_names=("x",))
  mesh2 = jax.sharding.Mesh(mesh1.devices[::-1], axis_names=mesh1.axis_names)

  def add_one(x: jax.Array) -> jax.Array:
    return x + 1.0

  kernel1 = pallas.jax_op(
      "custom::k1",
      add_one,
      mesh=mesh1,
      input_partition_specs=[P("x")],
      output_partition_specs=[P("x")],
  )
  kernel2 = pallas.jax_op(
      "custom::k2",
      add_one,
      mesh=mesh2,
      input_partition_specs=[P("x")],
      output_partition_specs=[P("x")],
  )

  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def conflicting_fn(a: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a)
    return kernel2(kernel1(a))

  expected_msg = re.compile(
      r".*expected Pallas kernels and scans in the same program to have the"
      r" same device assignment, got .*",
      re.DOTALL,
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    _ = conflicting_fn(x).cpu()


def run_all_to_all_single_dtype_error() -> None:
  input_tensor = torch.ones((1, 2), dtype=torch.float64, device="tpu")
  output_tensor = torch.empty((1, 2), dtype=torch.int64, device="tpu")
  # This message comes from native PyTorch.
  expected_msg = re.compile("Invalid usage of tensors with different dtypes.*")
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.all_to_all_single(output_tensor, input_tensor)


def run_all_to_all_single_invalid_tensor_size_error() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones((1, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((1, 2), dtype=torch.int64, device="tpu")
  expected_msg = (
      "distributed.all_to_all_single(): expected tensor first dimension to be "
      f"divisible by process group size, got 1 is not divisible by {world_size}"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(output_tensor, input_tensor)


def run_all_to_all_single_invalid_split_sizes_size_error() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones((world_size, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((world_size, 2), dtype=torch.int64, device="tpu")
  input_split_sizes = [2] * (world_size + 1)
  output_split_sizes = [1] * world_size
  expected_msg = (
      "distributed.all_to_all_single(): expected split sizes to have the same "
      f"size as process group size, got {world_size + 1} for split sizes and"
      f" {world_size} for process group size"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_invalid_split_sizes_sum_error() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones((world_size, 2), dtype=torch.int64, device="tpu")
  output_tensor = torch.empty((world_size, 2), dtype=torch.int64, device="tpu")
  input_split_sizes = [1, 3] + [1] * (world_size - 2)
  output_split_sizes = [1] * world_size
  expected_msg = (
      "distributed.all_to_all_single(): expected split sizes sum to be equal to"
      f" tensor first dimension, got {world_size + 2} for split sizes sum and"
      f" {world_size} for tensor shape [{world_size}, 2] dim 0"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_uneven_trailing_dims_mismatch_error() -> None:
  world_size = dist.get_world_size()
  total_rows = world_size + 2
  input_tensor = torch.ones((total_rows, 4), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty(
      (total_rows, 8), dtype=torch.float32, device="tpu"
  )
  input_split_sizes = [2] + [1] * (world_size - 2) + [2]
  output_split_sizes = [1, 2] + [1] * (world_size - 4) + [2, 1]
  expected_msg = (
      "distributed.all_to_all_single(): expected trailing dimensions of input"
      f" and output to match, got [{total_rows}, 4] and [{total_rows}, 8]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_uneven_dim_count_mismatch_error() -> None:
  world_size = dist.get_world_size()
  total_rows = world_size + 2
  input_tensor = torch.ones((total_rows, 4), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty(total_rows, dtype=torch.float32, device="tpu")
  input_split_sizes = [2] + [1] * (world_size - 2) + [2]
  output_split_sizes = [1, 2] + [1] * (world_size - 4) + [2, 1]
  expected_msg = (
      "distributed.all_to_all_single(): expected input and output tensors to"
      " have the same number of dimensions, got 2 and 1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_negative_split_size_error() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones((world_size, 2), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty(
      (world_size, 2), dtype=torch.float32, device="tpu"
  )
  input_split_sizes = [-1, 3] + [1] * (world_size - 2)
  output_split_sizes = [1] * world_size
  expected_msg = (
      "distributed.all_to_all_single(): expected split sizes to be"
      " non-negative, got -1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_send_recv_mismatch_error() -> None:
  world_size = dist.get_world_size()
  input_tensor = torch.ones(
      (world_size + 1, 2), dtype=torch.float32, device="tpu"
  )
  output_tensor = torch.empty(
      (world_size + 2, 2), dtype=torch.float32, device="tpu"
  )
  input_split_sizes = [2] + [1] * (world_size - 1)
  output_split_sizes = [2, 2] + [1] * (world_size - 2)
  expected_msg = re.compile(
      ".*send and recv slice size mismatch between rank.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_asymmetric_cross_rank_buffer_error() -> None:
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if rank == 0:
    local_size = world_size + 1
    input_split_sizes = [2] + [1] * (world_size - 1)
    output_split_sizes = [2] + [1] * (world_size - 1)
  else:
    local_size = world_size + 2
    input_split_sizes = [2, 2] + [1] * (world_size - 2)
    output_split_sizes = [2, 2] + [1] * (world_size - 2)
  input_tensor = torch.ones((local_size, 2), dtype=torch.float32, device="tpu")
  output_tensor = torch.empty(
      (local_size, 2), dtype=torch.float32, device="tpu"
  )
  expected_msg = re.compile(
      ".*asymmetric .*buffer sizes across ranks are not supported on TPU.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_asymmetric_cross_rank_output_buffer_error() -> None:
  """Keeps input buffers symmetric so the output buffer check is reached."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  # Every rank allocates the same input capacity (world_size + 1), so the input
  # buffer check passes and validation proceeds to the output buffer check.
  input_split_sizes = [2] + [1] * (world_size - 1)
  if rank == 0:
    output_split_sizes = [2] + [1] * (world_size - 1)
  else:
    output_split_sizes = [3] + [1] * (world_size - 1)
  input_tensor = torch.ones(
      (sum(input_split_sizes), 2), dtype=torch.float32, device="tpu"
  )
  output_tensor = torch.empty(
      (sum(output_split_sizes), 2), dtype=torch.float32, device="tpu"
  )
  expected_msg = re.compile(
      ".*asymmetric output buffer sizes across ranks are not supported on TPU.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all_single(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes
    )


def run_all_to_all_single_errors() -> None:
  run_all_to_all_single_send_recv_mismatch_error()
  run_all_to_all_single_asymmetric_cross_rank_buffer_error()
  run_all_to_all_single_asymmetric_cross_rank_output_buffer_error()
  run_all_to_all_single_negative_split_size_error()
  run_all_to_all_single_dtype_error()
  run_all_to_all_single_invalid_tensor_size_error()
  run_all_to_all_single_invalid_split_sizes_size_error()
  run_all_to_all_single_invalid_split_sizes_sum_error()
  run_all_to_all_single_uneven_trailing_dims_mismatch_error()
  run_all_to_all_single_uneven_dim_count_mismatch_error()


def run_all_to_all_unequal_tensor_list_size_error() -> None:
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64)
      for _ in range(world_size + 1)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected output and input tensors to have the"
      f" same number of tensors, got {world_size + 1} for output and"
      f" {world_size} for input"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_wrong_num_input_tensors_error() -> None:
  world_size = dist.get_world_size()
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
      "distributed.all_to_all(): expected input tensors to have the same"
      f" number of tensors as the process group size, got {world_size + 1} for"
      f" input and {world_size} for process group size"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_unequal_input_tensor_trailing_dims_error() -> None:
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones((1, 4), device=device, dtype=torch.int64)
  output_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected trailing dimensions of input tensors"
      " to match, got [1, 4] and [1, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_unequal_output_tensor_trailing_dims_error() -> None:
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  output_tensors = [
      torch.ones((1, 2), device=device, dtype=torch.int64)
      for _ in range(world_size)
  ]
  output_tensors[0] = torch.ones((1, 4), device=device, dtype=torch.int64)
  expected_msg = (
      "distributed.all_to_all(): expected trailing dimensions of output"
      " tensors to match, got [1, 4] and [1, 2]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_dim_count_mismatch_error() -> None:
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones((1, 2), device=device, dtype=torch.int64)
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  expected_msg = (
      "distributed.all_to_all(): expected all input tensors to have the same"
      " number of dimensions, got 2 and 1"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_dtype_mismatch_error() -> None:
  world_size = dist.get_world_size()
  device = "tpu"
  input_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  input_tensors[1] = torch.ones(1, device=device, dtype=torch.float32)
  output_tensors = [
      torch.ones(1, device=device, dtype=torch.int64) for _ in range(world_size)
  ]
  expected_msg = re.compile("Invalid usage of tensors with different dtypes.*")
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    torch.distributed.all_to_all(output_tensors, input_tensors)


def run_all_to_all_errors() -> None:
  run_all_to_all_unequal_tensor_list_size_error()
  run_all_to_all_wrong_num_input_tensors_error()
  run_all_to_all_unequal_input_tensor_trailing_dims_error()
  run_all_to_all_unequal_output_tensor_trailing_dims_error()
  run_all_to_all_dim_count_mismatch_error()
  run_all_to_all_dtype_mismatch_error()


def _make_ragged_all_to_all_metadata(
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Builds valid uniform offsets and slice sizes for ragged_all_to_all."""
  offsets = torch.arange(world_size, dtype=torch.int32).tpu()
  sizes = torch.ones(world_size, dtype=torch.int32).tpu()
  return offsets, sizes, offsets, sizes


def run_ragged_all_to_all_dim_count_mismatch_error() -> None:
  world_size = dist.get_world_size()
  input_offsets, send_sizes, output_offsets, recv_sizes = (
      _make_ragged_all_to_all_metadata(world_size)
  )
  operand = torch.ones(world_size, dtype=torch.float32, device="tpu")
  output = torch.zeros((world_size, 2), dtype=torch.float32, device="tpu")
  expected_msg = re.compile(
      ".*expected operand and output to have the same number of dimensions.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.tpu.ops.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )


def run_ragged_all_to_all_trailing_dims_mismatch_error() -> None:
  world_size = dist.get_world_size()
  input_offsets, send_sizes, output_offsets, recv_sizes = (
      _make_ragged_all_to_all_metadata(world_size)
  )
  operand = torch.ones((world_size, 2), dtype=torch.float32, device="tpu")
  output = torch.zeros((world_size, 3), dtype=torch.float32, device="tpu")
  expected_msg = re.compile(
      ".*expected trailing dimensions of operand and output to match.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.tpu.ops.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )


def run_ragged_all_to_all_dtype_mismatch_error() -> None:
  world_size = dist.get_world_size()
  input_offsets, send_sizes, output_offsets, recv_sizes = (
      _make_ragged_all_to_all_metadata(world_size)
  )
  operand = torch.ones((world_size, 2), dtype=torch.float32, device="tpu")
  output = torch.zeros((world_size, 2), dtype=torch.bfloat16, device="tpu")
  expected_msg = re.compile(
      ".*expected operand and output to have the same dtype.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.tpu.ops.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        dist.group.WORLD.group_name,
    )


def run_ragged_all_to_all_errors() -> None:
  run_ragged_all_to_all_dim_count_mismatch_error()
  run_ragged_all_to_all_trailing_dims_mismatch_error()
  run_ragged_all_to_all_dtype_mismatch_error()


def run_all_to_all_collective_errors() -> None:
  run_all_to_all_single_errors()
  run_all_to_all_errors()
  run_ragged_all_to_all_errors()


# Constants and helpers for ProcessGroupTpuErrorsTest.
_PG_WORLD_SIZE = 8

# In the missing-rank test, this rank never creates the process group.
_ABSENT_RANK = _PG_WORLD_SIZE - 1
# In the missing-rank test, this rank times out waiting for the gathered table
# before the collector (rank 0) gives up and publishes its error.
_EARLY_TIMEOUT_RANK = _PG_WORLD_SIZE - 2

# Store timeouts. The collector must time out well after
# `_EARLY_TIMEOUT_RANK` and well before the other ranks.
_EARLY_TIMEOUT = datetime.timedelta(seconds=1)
_COLLECTOR_TIMEOUT = datetime.timedelta(seconds=3)
_LONG_TIMEOUT = datetime.timedelta(seconds=180)

# Root op name that may prefix errors raised from the backend factory.
_API_PREFIX = r"(?:create_process_group\(\): )?"

# Error that the collector (rank 0) reports when `_ABSENT_RANK` never joins.
_COLLECTOR_ERROR = (
    r"collector \(node holding world_topo:0\): failed after \d+ms with 7/8"
    r" peer keys filled from 7 arrived nodes; missing keys \[world_topo:7\]:"
    r" world_store wait nlag:world_topo:0:8:node:7: wait timeout after \d+ms,"
    r" keys: \S+"
)


def _init_tpu() -> None:
  """Brings up this rank's TPU before any process group is created."""
  torch.zeros(1, device="tpu").cpu()


def _barrier(store: dist.Store, name: str) -> None:
  """Blocks until all `_PG_WORLD_SIZE` ranks reach the barrier `name`."""
  if store.add(f"{name}:count", 1) == _PG_WORLD_SIZE:
    store.set(f"{name}:done", "1")
  store.wait([f"{name}:done"])


def run_missing_rank_and_local_rank_out_of_range() -> None:
  """Tests missing-rank store timeouts on ranks 0..6 and out-of-range LOCAL_RANK on `_ABSENT_RANK`."""
  rank = int(os.environ["RANK"])
  store = dist.TCPStore(
      os.environ["MASTER_ADDR"],
      int(os.environ["MASTER_PORT"]),
      _PG_WORLD_SIZE,
      is_master=rank == 0,
      timeout=_LONG_TIMEOUT,
  )
  _init_tpu()
  _barrier(store, "ready")

  if rank == _ABSENT_RANK:
    # TorchTPU caches LOCAL_RANK (0-7 here) before this function runs, but on a
    # single host first reads TORCH_TPU_DEVICES_PER_HOST when the process group
    # is created, so shrink the latter on `_ABSENT_RANK` to put it out of range.
    # It fails before touching the store, leaving `world_topo:7` missing for the
    # remaining ranks.
    local_world_size = _PG_WORLD_SIZE // 2
    os.environ["TORCH_TPU_DEVICES_PER_HOST"] = str(local_world_size)
    expected_msg = re.compile(
        _API_PREFIX
        + r"expected environment variable LOCAL_RANK to be in \[0, 4\)"
        rf" \(TORCH_TPU_DEVICES_PER_HOST\), got {rank} \(local_rank={rank}"
        r" devices_per_host=4 world_size=8\).*",
        re.DOTALL,
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.init_process_group(
          backend="tpu_dist",
          store=store,
          rank=rank,
          world_size=_PG_WORLD_SIZE,
          timeout=_LONG_TIMEOUT,
      )
  else:
    # With one rank per host, every rank publishes to the world store directly,
    # so rank 0, which holds the first key, is the collector.
    os.environ["TORCH_TPU_DEVICES_PER_HOST"] = "1"
    if rank == 0:
      timeout = _COLLECTOR_TIMEOUT
      expected_pattern = _COLLECTOR_ERROR
    elif rank == _EARLY_TIMEOUT_RANK:
      timeout = _EARLY_TIMEOUT
      expected_pattern = (
          r"world_store get nlag:world_topo:0:8:table: wait timeout after"
          r" \d+ms, keys: \S+"
      )
    else:
      timeout = _LONG_TIMEOUT
      expected_pattern = (
          r"failed to all-gather via the node-local store: node leader"
          r" \(my_key=world_topo:0\): " + _COLLECTOR_ERROR
      )
    store.set_timeout(timeout)
    expected_msg = re.compile(_API_PREFIX + expected_pattern + ".*", re.DOTALL)
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      dist.init_process_group(
          backend="tpu_dist",
          store=store,
          rank=rank,
          world_size=_PG_WORLD_SIZE,
          timeout=timeout,
      )
    store.set_timeout(_LONG_TIMEOUT)

  # Keep every rank (and rank 0's store server) alive until all are done.
  _barrier(store, "finished")


def _remove_node_local_store_file() -> None:
  """Removes the node-local store file of workers spawned by this process."""
  # Workers name the file after their parent's pid, which is this process.
  try:
    os.remove(f"/dev/shm/torch_tpu_pg_store_{os.getpid()}")
  except FileNotFoundError:
    pass


class CollectiveErrorsTest(
    continuous_test_utils.TpuMultiProcContinuousTest,
    et.TpuOnlyDistributedErrorTestBase,
):
  world_size = distributed_utils.get_tpu_device_count()

  def test_collective_validation_errors(self):
    run_collective_validation_errors()

  def test_conflicting_device_assignments_error(self):
    run_conflicting_device_assignments_error()

  def test_all_to_all_collective_errors(self):
    run_all_to_all_collective_errors()

  def test_uneven_all_gather_into_tensor_dim0_raises(self):
    world_size = dist.get_world_size()
    x = torch.ones((2, 2), device="tpu", dtype=torch.float32)
    output = torch.empty(
        (2 * world_size - 1, 2), device="tpu", dtype=torch.float32
    )
    expected_msg = (
        "distributed.all_gather_into_tensor(): expected output for"
        " concatenation to be a tensor with size equal to world size * input"
        " size at dimension 0, got output shape"
        f" [{2 * world_size - 1}, 2], input shape [2, 2], and world size"
        f" {world_size}"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.all_gather_into_tensor(output, x)

  def test_uneven_reduce_scatter_tensor_dim0_raises(self):
    world_size = dist.get_world_size()
    output = torch.zeros((3, 4, 5), device="tpu")
    input_invalid = torch.zeros((10, 4, 5), device="tpu")
    expected_msg = (
        "distributed.reduce_scatter_tensor(): expected the input tensor to be"
        f" either of shape [{world_size*3}, 4, 5] or of shape [{world_size},"
        " 3, 4, 5], got [10, 4, 5]"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.reduce_scatter_tensor(output, input_invalid)


class ProcessGroupTpuErrorsTest(et.TpuOnlyDistributedErrorTestBase):

  def setUp(self):
    super().setUp()
    # Keys left over from a previous case would be reused, since scope keys
    # repeat across cases.
    _remove_node_local_store_file()

  def tearDown(self):
    _remove_node_local_store_file()
    super().tearDown()

  def test_missing_rank_and_local_rank_out_of_range(self):
    distributed_utils.dist_run(
        nproc_per_node=_PG_WORLD_SIZE,
        fn=run_missing_rank_and_local_rank_out_of_range,
    )

  def test_collective_after_finalize_raises_runtime_error(self):
    if torchcomms is None:
      self.skipTest("torchcomms is not available")
    torchcomm_tpu.register_torchcomms_tpu()
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_finalize")
    comm.finalize()
    dummy_tensor = torch.tensor([1.0])
    with et.assert_raises_message(
        RuntimeError, tpu=re.compile(r".*finalized.*", re.DOTALL)
    ):
      comm.all_reduce(dummy_tensor, op=torchcomms.ReduceOp.SUM, async_op=False)

  def test_to_dist_reduce_op_unsupported_raises_value_error(self):
    if torchcomms is None:
      self.skipTest("torchcomms is not available")
    torchcomm_tpu.register_torchcomms_tpu()
    invalid_op = mock.MagicMock(spec=torchcomms.ReduceOp)
    invalid_op.type = "UNSUPPORTED_TYPE"
    with et.assert_raises_message(ValueError, tpu=re.compile(r".*", re.DOTALL)):
      torchcomm_tpu._to_dist_reduce_op(invalid_op)

  @parameterized.named_parameters(
      ("create_pair_comm", lambda b, t: b.create_pair_comm(1)),
      ("split_group", lambda b, t: b.split_group([0, 1])),
      ("reconfigure", lambda b, t: b.reconfigure(ranks=[0, 1], size=2)),
      ("create_window", lambda b, t: b.create_window(t)),
      ("map_remote_tensor", lambda b, t: b.map_remote_tensor(t, peer_rank=1)),
      ("create_batch", lambda b, t: b.create_batch()),
      ("all_reduce_coalesced", lambda b, t: b.all_reduce_coalesced([t])),
      ("all_gather_coalesced", lambda b, t: b.all_gather_coalesced([[t]], [t])),
      (
          "reduce_scatter_coalesced",
          lambda b, t: b.reduce_scatter_coalesced([t], [[t]]),
      ),
      (
          "all_to_all_coalesced",
          lambda b, t: b.all_to_all_coalesced([[t]], [[t]]),
      ),
      ("reduce", lambda b, t: b.reduce(t, dst=0)),
      ("gather", lambda b, t: b.gather(input_tensor=t, dst=0)),
      ("scatter", lambda b, t: b.scatter(output_tensor=t, src=0)),
      ("monitored_barrier", lambda b, t: b.monitored_barrier()),
      (
          "register_flight_recorder_hook",
          lambda b, t: b.register_flight_recorder_hook(mock.MagicMock()),
      ),
      (
          "get_flight_recorder_traces",
          lambda b, t: b.get_flight_recorder_traces(),
      ),
      (
          "broadcast_object_list",
          lambda b, t: b.broadcast_object_list([1, 2, 3]),
      ),
      ("all_gather_object", lambda b, t: b.all_gather_object([], {"a": 1})),
      ("gather_object", lambda b, t: b.gather_object({"a": 1})),
      ("scatter_object_list", lambda b, t: b.scatter_object_list([])),
  )
  def test_unsupported_stubs_raise_not_implemented(self, stub_fn):
    if torchcomms is None:
      self.skipTest("torchcomms is not available")
    torchcomm_tpu.register_torchcomms_tpu()
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_stubs")
    backend = comm.get_backend_impl()
    t = torch.zeros(4, device="cpu")
    try:
      with et.assert_raises_message(
          NotImplementedError, tpu=re.compile(r".*", re.DOTALL)
      ):
        stub_fn(backend, t)
    finally:
      comm.finalize()

  def test_get_device_SetsDefault(self):
    comm = torchcomm_tpu.TorchCommTPU(backend_name="tpu", device=None)
    self.assertIsNotNone(comm.get_device())

  def test_all_gather_single_mismatched_size_raises(self):
    comm = torchcomm_tpu.TorchCommTPU(backend_name="tpu", device="cpu")
    with et.assert_raises_message(
        RuntimeError, tpu=re.compile(r".*not initialized.*", re.DOTALL)
    ):
      comm.all_gather_single(torch.zeros(4), torch.zeros(2))

  def test_extract_process_group_id_non_collective_op_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(operator.add, args=(x, x))
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(
            r".*expected node target to be in .* torch_tpu bug.*", re.DOTALL
        ),
    ):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_empty_ranks_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", []),
    )
    with et.assert_raises_message(
        ValueError, tpu=re.compile(r".*ranks cannot be empty.*", re.DOTALL)
    ):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_missing_group_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum"),
    )
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(
            r".*Could not infer process group argument.*", re.DOTALL
        ),
    ):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_unsupported_group_type_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", 3.14),
    )
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(r".*Unsupported process group type.*", re.DOTALL),
    ):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_uninitialized_dist_str_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node_str = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", "0"),
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(r".*Could not resolve the process group.*", re.DOTALL),
    ):
      process_group_utils._extract_process_group_id_from_node(node_str)

  def test_extract_process_group_id_uninitialized_dist_int_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node_int = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", 0),
    )
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(r".*Could not resolve the process group.*", re.DOTALL),
    ):
      process_group_utils._extract_process_group_id_from_node(node_int)

  def test_empty_ranks_raises_value_error(self) -> None:
    with et.assert_raises_message(
        ValueError, tpu=re.compile(r".*ranks cannot be empty.*", re.DOTALL)
    ):
      ProcessGroupId([])

  def test_negative_rank_raises_value_error(self) -> None:
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(r".*All ranks must be non-negative.*", re.DOTALL),
    ):
      ProcessGroupId([-1, 0])

  @parameterized.named_parameters(
      ("zero", 0),
      ("negative", -2),
  )
  def test_invalid_world_size_raises_value_error(self, world_size: int) -> None:
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(r".*world_size must be strictly positive.*", re.DOTALL),
    ):
      ProcessGroupId([0, 1], world_size=world_size)

  @parameterized.named_parameters(
      ("rank_4_world_size_4", [0, 4], 4, r".*All ranks must be in \[0, 4\).*"),
      ("rank_2_world_size_2", [0, 2], 2, r".*All ranks must be in \[0, 2\).*"),
  )
  def test_rank_out_of_world_size_raises_value_error(
      self, ranks: list[int], world_size: int, pattern: str
  ) -> None:
    with et.assert_raises_message(
        ValueError, tpu=re.compile(pattern, re.DOTALL)
    ):
      ProcessGroupId(ranks, world_size=world_size)

  def test_duplicate_ranks_raises_value_error(self) -> None:
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(r".*ranks must not contain duplicates.*", re.DOTALL),
    ):
      ProcessGroupId([0, 1, 1])

  @parameterized.named_parameters(
      ("descending", [1, 0]),
      ("unsorted_tail", [0, 3, 2]),
  )
  def test_unsorted_ranks_raises_value_error(self, ranks: list[int]) -> None:
    with et.assert_raises_message(
        ValueError,
        tpu=re.compile(
            r".*ranks must be sorted in ascending order.*", re.DOTALL
        ),
    ):
      ProcessGroupId(ranks)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
