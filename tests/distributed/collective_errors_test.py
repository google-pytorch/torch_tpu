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
import os
import re

from absl.testing import absltest
import jax
from jax.sharding import PartitionSpec as P
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal import pallas
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests.distributed import distributed_utils


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


def run_all_gather_tensor_wrong_number_output_dimensions() -> None:
  """Tests incorrect output dimensionality."""
  world_size = int(os.environ["WORLD_SIZE"])
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
  world_size = int(os.environ["WORLD_SIZE"])
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
  world_size = int(os.environ["WORLD_SIZE"])
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
  world_size = int(os.environ["WORLD_SIZE"])
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
  world_size = int(os.environ["WORLD_SIZE"])
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
  world_size = int(os.environ["WORLD_SIZE"])
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
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


def run_reduce_scatter_tensor_errors() -> None:
  """Tests reduce_scatter_tensor with invalid inputs."""
  world_size = int(os.environ["WORLD_SIZE"])

  output = torch.zeros((3, 4, 5), device="tpu")
  input_invalid = torch.zeros((10, 4, 5), device="tpu")

  expected_msg = (
      "distributed.reduce_scatter_tensor(): expected the input tensor to be"
      f" either of shape [{world_size*3}, 4, 5] or of shape [{world_size}, 3,"
      " 4, 5], got [10, 4, 5]"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    torch.distributed.reduce_scatter_tensor(output, input_invalid)


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
  rank = int(os.environ["RANK"])
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
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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
  # Intentionally do NOT call gather on non-dst ranks.
  # The rank-specific error will prevent the gather() from actually starting;
  # trying to gather() on non-dst ranks would deadlock.


def run_gather_non_uniform_output_shapes() -> None:
  """Tests gather failure when output tensors have different shapes."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


def run_collective_validation_errors() -> None:
  """Runs synchronous collective input validation error checks."""
  dist.init_process_group(backend="tpu_dist")
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
  run_reduce_scatter_errors()
  run_reduce_scatter_tensor_errors()
  run_gather_wrong_input_size()
  run_gather_wrong_output_list_size()
  run_gather_wrong_output_tensor_count()
  run_gather_mismatch_input_size()
  run_gather_non_uniform_output_shapes()
  run_gather_output_on_non_root()


def run_conflicting_device_assignments_error() -> None:
  """Tests that groups fuse over the world while conflicting Pallas meshes fail."""
  dist.init_process_group("tpu")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  half = world_size // 2

  low = dist.new_group(ranks=list(range(half)))
  high = dist.new_group(ranks=list(range(half, world_size)))
  subgroup = low if rank < half else high

  # The world and subgroup collectives run on different device sets; the
  # program runs on the world, with the subgroup collective tiled over both
  # halves.
  @torch.compile(backend="tpu", fullgraph=True)
  @torch.tpu.distributed.spmd_safe
  def world_and_subgroup_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a)
    dist.all_reduce(b, group=subgroup)
    return a + b

  x = torch.ones(4, device="tpu")
  y = torch.ones(4, device="tpu")
  res = world_and_subgroup_fn(x, y).cpu()
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


# Constants and helpers for ProcessGroupTpuErrorsTest.
_PG_WORLD_SIZE = 8

# In the missing-rank test, this rank never creates the process group.
_ABSENT_RANK = _PG_WORLD_SIZE - 1
# In the missing-rank test, this rank times out waiting for the gathered table
# before the collector (rank 0) gives up and publishes its error.
_EARLY_TIMEOUT_RANK = _PG_WORLD_SIZE - 2

# Store timeouts. The collector must time out well after
# `_EARLY_TIMEOUT_RANK` and well before the other ranks.
_EARLY_TIMEOUT = datetime.timedelta(seconds=2)
_COLLECTOR_TIMEOUT = datetime.timedelta(seconds=6)
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


def run_missing_rank() -> None:
  """Creates the process group on all ranks but `_ABSENT_RANK`."""
  # With one rank per host, every rank publishes to the world store directly,
  # so rank 0, which holds the first key, is the collector. On a single host,
  # TORCH_TPU_DEVICES_PER_HOST is first read when the process group is created.
  os.environ["TORCH_TPU_DEVICES_PER_HOST"] = "1"
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

  if rank != _ABSENT_RANK:
    if rank == 0:
      timeout = _COLLECTOR_TIMEOUT
      expected_msg = _COLLECTOR_ERROR
    elif rank == _EARLY_TIMEOUT_RANK:
      timeout = _EARLY_TIMEOUT
      expected_msg = (
          r"world_store get nlag:world_topo:0:8:table: wait timeout after"
          r" \d+ms, keys: \S+"
      )
    else:
      timeout = _LONG_TIMEOUT
      expected_msg = (
          r"failed to all-gather via the node-local store: node leader"
          r" \(my_key=world_topo:0\): " + _COLLECTOR_ERROR
      )
    store.set_timeout(timeout)
    with et.assert_raises_message(
        RuntimeError,
        tpu=re.compile(_API_PREFIX + expected_msg + ".*", re.DOTALL),
    ):
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


def run_local_rank_out_of_range() -> None:
  """Creates the process group on ranks with LOCAL_RANK >= devices per host."""
  # TorchTPU caches LOCAL_RANK (0-7 here) before this function runs, but on a
  # single host first reads TORCH_TPU_DEVICES_PER_HOST when the process group is
  # created, so shrink the latter to put the upper ranks out of range. These
  # ranks fail before touching the store.
  local_world_size = _PG_WORLD_SIZE // 2
  os.environ["TORCH_TPU_DEVICES_PER_HOST"] = str(local_world_size)
  rank = int(os.environ["RANK"])
  store = dist.TCPStore(
      os.environ["MASTER_ADDR"],
      int(os.environ["MASTER_PORT"]),
      _PG_WORLD_SIZE,
      is_master=rank == 0,
      timeout=_LONG_TIMEOUT,
  )

  if rank >= local_world_size:
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

  # Keep every rank (and rank 0's store server) alive until all are done.
  _barrier(store, "finished")


def _remove_node_local_store_file() -> None:
  """Removes the node-local store file of workers spawned by this process."""
  # Workers name the file after their parent's pid, which is this process.
  try:
    os.remove(f"/dev/shm/torch_tpu_pg_store_{os.getpid()}")
  except FileNotFoundError:
    pass


class CollectiveErrorsTest(et.TpuOnlyDistributedErrorTestBase):
  _world_size = 8

  def test_collective_validation_errors(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=run_collective_validation_errors,
    )

  def test_conflicting_device_assignments_error(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=run_conflicting_device_assignments_error,
    )


class ProcessGroupTpuErrorsTest(et.TpuOnlyDistributedErrorTestBase):

  def setUp(self):
    super().setUp()
    # Keys left over from a previous case would be reused, since scope keys
    # repeat across cases.
    _remove_node_local_store_file()

  def tearDown(self):
    _remove_node_local_store_file()
    super().tearDown()

  def test_missing_rank(self):
    distributed_utils.dist_run(
        nproc_per_node=_PG_WORLD_SIZE,
        fn=run_missing_rank,
    )

  def test_local_rank_out_of_range(self):
    distributed_utils.dist_run(
        nproc_per_node=_PG_WORLD_SIZE,
        fn=run_local_rank_out_of_range,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
