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

- `CollectiveOpsTest` uses `TpuMultiProcContinuousTest`: N worker processes (one
  per TPU) are spawned once and share a process group across tests. Each test
  method runs on every rank.
- Tests that need fresh worker processes (e.g. to exercise process exit) use
  `distributed_utils.dist_run()`, which spawns N workers per test.
"""

from collections.abc import Callable, Sequence
import datetime
import os
import threading
import time
from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

if __name__ == "__main__":  # We are in the parent process.
  # Pick a likely unique cache root for this run. This makes it less likely for
  # multiple runs of this test to interfere with each other via the compilation
  # cache. This must be done once in the parent process, before importing torch.
  #
  # Note that this test is sharded, each shard running on a different host.
  # We pick a cache path that is the same for all shards in a single run, but
  # likely different from one run to the next.
  date = time.strftime("%Y-%m-%d", time.localtime())
  random_seed = os.environ.get("TEST_RANDOM_SEED", "0")
  os.environ[
      "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
  ] += f"/{date}.{random_seed}"
# pylint: disable=g-code-after-main

import torch  # pylint: disable=g-import-not-at-top
from torch import distributed as dist
from torch.distributed import device_mesh
import torch.multiprocessing as mp

from torch_tpu._internal import execution_mode
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils

_SKIP_P2P_ON_TPU_V7_REASON = (
    "P2P send/recv aborts in libtpu on TPU v7 (b/540414751)."
)
try:
  _IS_TPU_V7 = torch.tpu.get_device_properties().name == "TPU v7"
except RuntimeError:  # No TPU generation that TorchTPU supports is attached.
  _IS_TPU_V7 = False

# torch < 2.13 cannot return Work.exception() to Python.
_WORK_EXCEPTION_IS_BOUND = utils.parse_version(torch.__version__) >= (2, 13, 0)


def _test_wrapper(
    test_fn: Callable[..., None], *args: Any, **kwargs: Any
) -> None:
  """Wrapper to initialize and cleanup the distributed environment for tests.

  This helper ensures that every test rank initializes the TPU device and
  process group. After the test function completes, it performs a final
  synchronization barrier to ensure all workers have finished before
  destroying the process group, preventing premature exit crashes.

  Args:
    test_fn: The test function to execute.
    *args: Positional arguments for the test function.
    **kwargs: Keyword arguments for the test function.
  """
  dist.init_process_group(backend="tpu_dist")
  try:
    test_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def _all_reduce_input_fn_rank(r, w):
  """Generates rank-based input for all-reduce tests.

  Args:
    r: Rank of the current process (0 to w-1).
    w: World size.

  Returns:
    List of 4 floats for all-reduce tests.
  """
  del w  # unused
  return [0.0, 1.0, float(r), float(r**2)]


def _all_reduce_input_fn_const(r, w):
  """Generates constant input for all-reduce tests.

  Args:
    r: Rank of the current process (0 to w-1).
    w: World size.

  Returns:
    List of 4 floats for all-reduce tests.
  """
  del r, w  # unused
  return [1.0, 2.0, 3.0, 4.0]


def _all_reduce_bitwise_input_fn(r, w):
  """Generates integer inputs for a given rank for bitwise operations.

  Args:
    r: Rank of the current process (0 to w-1).
    w: World size.

  Returns:
    List of 3 integers for bitwise operation tests.
  """

  # Example values:
  # val1: Sets the r-th bit and the (w-1-r)-th bit.
  # val2: Simple arithmetic series based on rank.
  # val3: A constant value with all bits set in a byte.
  val1 = (1 << r) | (1 << (w - 1 - r))
  val2 = r * 3
  val3 = 0xFF
  return [val1, val2, val3]


def _run_all_reduce(
    input_fn: Callable[[int, int], Sequence[int | float]],
    reduce_op: torch.distributed.ReduceOp,
    expected: Sequence[int | float],
    dtype: torch.dtype,
) -> None:
  """Tests all-reduce functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = "tpu"
  input_list = input_fn(rank, world_size)
  x = torch.tensor(input_list, dtype=dtype, device=device)
  logging.info(
      "Before all_reduce (rank=%d, op=%s, dtype=%s): x = %s",
      rank,
      reduce_op,
      dtype,
      x.cpu(),
  )

  handle = torch.distributed.all_reduce(x, op=reduce_op, async_op=True)
  handle.wait()

  logging.info(
      "After all_reduce (rank=%d, op=%s, dtype=%s): x = %s",
      rank,
      reduce_op,
      dtype,
      x.cpu(),
  )

  expected_tensor = torch.tensor(expected, dtype=dtype)
  utils.assert_close(x.cpu(), expected_tensor)


def _run_all_gather() -> None:
  """Tests all-gather functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
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


def _run_all_gather_scalar() -> None:
  """Tests all_gather on scalar input."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  outputs = [
      torch.tensor(0, device="tpu", dtype=torch.float32)
      for _ in range(world_size)
  ]
  x = torch.tensor(rank, device="tpu", dtype=torch.float32)
  torch.distributed.all_gather(outputs, x, async_op=False)
  outputs_cpu = [o.to("cpu") for o in outputs]
  for idx, scalar in enumerate(outputs_cpu):
    assert scalar.item() == idx


def _run_all_gather_tensor_concat() -> None:
  """Tests all_gather_into_tensor with concatenation."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  x = torch.tensor([rank, rank], device="tpu", dtype=torch.float32)
  output = torch.zeros(world_size * 2, device="tpu", dtype=torch.float32)
  handle = torch.distributed.all_gather_into_tensor(output, x, async_op=True)
  handle.wait()
  output_cpu = output.cpu()

  # expected = [0, 0, 1, 1, ...]
  iota = torch.arange(world_size, dtype=torch.float32)
  expected = torch.stack([iota, iota]).T.flatten()
  utils.assert_close(output_cpu, expected)


def _run_all_gather_tensor_scalar() -> None:
  """Tests all_gather_into_tensor on scalar input."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  x = torch.tensor(rank, device="tpu", dtype=torch.float32)
  output = torch.zeros(world_size, device="tpu", dtype=torch.float32)
  torch.distributed.all_gather_into_tensor(output, x, async_op=False)
  output_cpu = output.cpu()
  expected = torch.arange(world_size, dtype=torch.float32)
  utils.assert_close(output_cpu, expected)


def _run_all_gather_tensor_stack() -> None:
  """Tests all_gather_into_tensor with stacking."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  x = torch.tensor([rank, rank], device="tpu", dtype=torch.float32)
  output = torch.zeros((world_size, 2), device="tpu", dtype=torch.float32)
  torch.distributed.all_gather_into_tensor(output, x, async_op=False)
  output_cpu = output.cpu()

  # expected = [[0, 0], [1, 1], ...]
  iota = torch.arange(world_size, dtype=torch.float32)
  expected = torch.stack([iota, iota]).T
  utils.assert_close(output_cpu, expected)


def _run_all_gather_rank_order() -> None:
  """Tests all_gather_into_tensor and all_to_all rank ordering across TPU cores.

  Verifies that collectives preserve logical rank ordering [0, 1, ...,
  world_size-1]
  and are not permuted by physical PJRT/ICI chip topology [0, 1, 4, 5, 6, 7, 2,
  3]
  on multi-core TPU hosts (e.g. TPU v5e 2x4 cores), covering both default
  process group (group=None) and explicit full-world subgroup (dist.new_group).
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = torch.device("tpu", torch.accelerator.current_device_index())
  expected = torch.arange(world_size, dtype=torch.float32)

  # 1. all_reduce (permutation invariant check matching reproducer)
  summed = torch.full((8,), float(rank), dtype=torch.float32, device=device)
  torch.distributed.all_reduce(summed, group=None)
  expected_sum = world_size * (world_size - 1) / 2.0
  utils.assert_close(
      summed.cpu(), torch.full((8,), expected_sum, dtype=torch.float32)
  )

  # 2. all_gather_into_tensor on default process group (group=None)
  gathered_default = torch.empty(world_size, dtype=torch.float32, device=device)
  inp = torch.full((1,), float(rank), dtype=torch.float32, device=device)
  torch.distributed.all_gather_into_tensor(gathered_default, inp, group=None)
  utils.assert_close(gathered_default.cpu(), expected)

  # 3. all_gather_into_tensor on explicit world group (dist.new_group)
  all_ranks = list(range(world_size))
  explicit_world_pg = torch.distributed.new_group(ranks=all_ranks)
  gathered_new_group = torch.empty(
      world_size, dtype=torch.float32, device=device
  )
  torch.distributed.all_gather_into_tensor(
      gathered_new_group, inp, group=explicit_world_pg
  )
  utils.assert_close(gathered_new_group.cpu(), expected)

  # 4. all_to_all_single on default process group (group=None)
  send = torch.full(
      (world_size,), float(rank), dtype=torch.float32, device=device
  )
  recv = torch.empty(world_size, dtype=torch.float32, device=device)
  torch.distributed.all_to_all_single(recv, send, group=None)
  utils.assert_close(recv.cpu(), expected)

  # 5. all_to_all_single on explicit world group
  recv_new_group = torch.empty(world_size, dtype=torch.float32, device=device)
  torch.distributed.all_to_all_single(
      recv_new_group, send, group=explicit_world_pg
  )
  utils.assert_close(recv_new_group.cpu(), expected)


def _check_dim_collectives(fn: Callable[..., Any], group: Any) -> None:
  # Encode inputs by world rank so that a wrong group membership changes the
  # result, not just a wrong order within the group.
  ranks = dist.get_process_group_ranks(
      dist.group.WORLD if group is None else group
  )
  rank = dist.get_rank(group)
  size = len(ranks)
  x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
  y = torch.arange(6 * size, dtype=torch.float32).reshape(2, 3 * size)
  gathered, scattered_sum, scattered_avg = fn(
      (x + 100 * dist.get_rank()).to("tpu"),
      (y + 1000 * dist.get_rank()).to("tpu"),
      group,
  )
  utils.assert_close(
      gathered.cpu(), torch.cat([x + 100 * r for r in ranks], dim=1)
  )
  utils.assert_close(
      scattered_sum.cpu(),
      (size * y + 1000 * sum(ranks)).chunk(size, dim=1)[rank],
  )
  utils.assert_close(
      scattered_avg.cpu(),
      (y + 1000 * sum(ranks) / size).chunk(size, dim=1)[rank],
  )


def _run_experimental_collectives_on_dim() -> None:
  """Tests the experimental collectives on a non-leading dim of a 2D mesh."""
  # Mesh dim 0 groups are strided ({0, 4}, ...), dim 1 groups are contiguous.
  mesh = device_mesh.init_device_mesh("tpu", (2, dist.get_world_size() // 2))
  mesh_groups = (mesh.get_group(0), mesh.get_group(1))

  def _dim_collectives(x, y, group):
    return (
        torch.tpu.ops.all_gather_dim(x, 1, group),
        torch.tpu.ops.reduce_scatter_dim(y, -1, group),
        torch.tpu.ops.reduce_scatter_dim(y, -1, group, reduce_op="avg"),
    )

  debugs = []
  compiled = torch.compile(
      _dim_collectives,
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={"serializable": False, "debug_callback": debugs.append},
  )
  # Compile first so the first call of each public op happens under Dynamo.
  for group in mesh_groups:
    _check_dim_collectives(compiled, group)
  for group in (None, *mesh_groups):
    _check_dim_collectives(_dim_collectives, group)

  stablehlo = "\n".join(
      text for debug in debugs for text in debug.stablehlo_forward_text
  )
  assert "all_gather_dim = 1" in stablehlo, stablehlo
  assert "scatter_dimension = 1" in stablehlo, stablehlo


def _run_gather_scalar() -> None:
  """Tests gather on scalar input."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dst = 0
  tensor = torch.tensor(rank, device="tpu", dtype=torch.float32)
  if rank == dst:
    gather_list = [
        torch.tensor(0, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
  else:
    gather_list = None

  torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)

  if rank == dst:
    for i in range(world_size):
      utils.assert_close(
          gather_list[i].cpu(), torch.tensor(i, dtype=torch.float32)
      )


def _run_gather_tensor() -> None:
  """Tests gather on tensor input."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dst = 0
  tensor = torch.tensor([rank, rank], device="tpu", dtype=torch.float32)
  if rank == dst:
    gather_list = [
        torch.zeros(2, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
  else:
    gather_list = None

  torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)

  if rank == dst:
    for i in range(world_size):
      utils.assert_close(
          gather_list[i].cpu(), torch.tensor([i, i], dtype=torch.float32)
      )


def _run_gather_with_root_subset_use() -> None:
  """Tests gather when only a subset of outputs is used on the root rank.

  This test verifies that gather does not hang when only a subset of output
  tensors are used on the root rank, matching eager PyTorch behavior.
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dst = 0
  tensor = torch.tensor([rank, rank], device="tpu", dtype=torch.float32)
  if rank == dst:
    gather_list = [
        torch.zeros(2, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
  else:
    gather_list = None

  torch.distributed.gather(tensor, gather_list=gather_list, dst=dst)

  if rank == dst:
    res = gather_list[0] + 1.0
    res.cpu()


def _run_reduce_scatter() -> None:
  """Tests reduce-scatter functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  inputs = []
  for i in range(world_size):
    inputs.append(
        torch.tensor(
            [
                [1.0, float(i), float(i**2)],
                [float(rank), float(rank**2), float(i * rank)],
            ],
            device="tpu",
        )
    )

  output = torch.zeros(2, 3, device="tpu")
  handle = torch.distributed.reduce_scatter(output, inputs, async_op=True)
  handle.wait()

  output_cpu = output.cpu()
  logging.info("reduce_scatter output (rank %d): %s", rank, output_cpu)

  sum_r = float(sum(range(world_size)))
  sum_r2 = float(sum(i**2 for i in range(world_size)))
  expected = torch.tensor([
      [world_size, rank * world_size, rank**2 * world_size],
      [sum_r, sum_r2, rank * sum_r],
  ])

  logging.info("reduce_scatter expected (rank %d): %s", rank, expected)
  utils.assert_close(output_cpu, expected)


def _run_reduce_scatter_tensor_stack() -> None:
  """Tests reduce_scatter_tensor (in stack input mode) functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # We run a [world_size, world_size, 3, 4, 5] --> [3, 4, 5] reduce-scatter.
  # Construct input tensors for all ranks (each unreduced, full-size) on CPU:
  torch.manual_seed(42)
  x_full_cpu = torch.randn((world_size, world_size, 3, 4, 5))

  # Every rank gets a different unreduced full-size tensor:
  x = x_full_cpu[rank].to(device="tpu")
  output = torch.zeros((3, 4, 5), device="tpu")
  handle = torch.distributed.reduce_scatter_tensor(output, x, async_op=True)
  handle.wait()

  output_cpu = output.cpu()
  logging.info("reduce_scatter_tensor output (rank %d): %s", rank, output_cpu)

  expected = x_full_cpu.sum(dim=0)[rank]
  logging.info("reduce_scatter_tensor expected (rank %d): %s", rank, expected)

  assert (
      output_cpu.shape == expected.shape
  ), f"Got {output_cpu.shape}, expected {expected.shape}"

  utils.assert_close(output_cpu, expected, rtol=1e-4, atol=1e-6)


def _run_reduce_scatter_tensor_concat() -> None:
  """Tests reduce_scatter_tensor (in concat input mode) functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # We run a [world_size*3, 4, 5] --> [3, 4, 5] reduce-scatter operation.
  # Construct input tensors for all ranks (each unreduced, full-size) on CPU:
  torch.manual_seed(42)
  x_full_cpu = torch.randn((world_size, world_size * 3, 4, 5))

  # Every rank gets a different unreduced full-size tensor:
  x = x_full_cpu[rank].to(device="tpu")
  output = torch.zeros((3, 4, 5), device="tpu")
  handle = torch.distributed.reduce_scatter_tensor(output, x, async_op=True)
  handle.wait()

  output_cpu = output.cpu()
  logging.info("reduce_scatter_tensor output (rank %d): %s", rank, output_cpu)

  expected = x_full_cpu.sum(dim=0).view(-1, 3, 4, 5)[rank]
  logging.info("reduce_scatter_tensor expected (rank %d): %s", rank, expected)

  assert (
      output_cpu.shape == expected.shape
  ), f"Got {output_cpu.shape}, expected {expected.shape}"

  utils.assert_close(output_cpu, expected, rtol=1e-4, atol=1e-6)


def _run_reduce_scatter_tensor_avg() -> None:
  """Tests reduce_scatter_tensor with an AVG reduction op."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # We run a [world_size*3, 4] --> [3, 4] reduce-scatter operation.
  # Construct input tensors for all ranks (each unreduced, full-size) on CPU:
  torch.manual_seed(42)
  x_full_cpu = torch.randn((world_size, world_size * 3, 4))

  # Every rank gets a different unreduced full-size tensor:
  x = x_full_cpu[rank].to(device="tpu")
  output = torch.zeros((3, 4), device="tpu")
  torch.distributed.reduce_scatter_tensor(
      output, x, op=torch.distributed.ReduceOp.AVG
  )

  output_cpu = output.cpu()
  expected = x_full_cpu.mean(dim=0).view(-1, 3, 4)[rank]

  utils.assert_close(output_cpu, expected)


def _run_broadcast() -> None:
  """Tests collective broadcast functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # Try broadcasting from all possible src ranks:
  for src_rank in range(world_size):
    # Start with different tensors on each rank:
    x = torch.tensor([0.0, 1.0, float(rank), float(rank**2)], device="tpu")

    # Should end up with the same tensor on all ranks:
    expected = torch.tensor([0.0, 1.0, float(src_rank), float(src_rank**2)])

    handle = torch.distributed.broadcast(x, src=src_rank, async_op=True)
    handle.wait()
    utils.assert_close(x.cpu(), expected)


def _run_broadcast_objects() -> None:
  """Tests broadcast_object_list collective."""
  rank = dist.get_rank()
  source_objects = [17, "foo", {"key": False}]
  num_objects = len(source_objects)
  objects: list[Any | None]
  if rank == 0:
    objects = source_objects
  else:
    objects = [None] * num_objects
  torch.distributed.broadcast_object_list(objects, src=0)
  assert (
      objects == source_objects
  ), f"Rank {rank}: expected object list {source_objects}, got {objects}"


def _run_send_recv() -> None:
  """Tests point-to-point blocking send and recv functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  if world_size < 2:
    return

  x = torch.tensor([[float(rank), float(rank**2)]], device="tpu")
  recv_buffer = torch.zeros_like(x, device="tpu")

  if rank < world_size - 1:
    torch.tpu.ops.send([x], dst=rank + 1, tag=rank).wait()

  if rank > 0:
    src_rank = rank - 1
    torch.tpu.ops.recv([recv_buffer], src=src_rank, tag=src_rank).wait()
    expected = torch.tensor([[float(src_rank), float(src_rank**2)]])
    utils.assert_close(recv_buffer.cpu(), expected)


def _run_isend_irecv() -> None:
  """Tests point-to-point asynchronous isend and irecv functionality."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  if world_size < 2:
    return

  x = torch.tensor([[float(rank), float(rank**2)]], device="tpu")
  recv_buffer = torch.zeros_like(x, device="tpu")

  send_work = None
  if rank < world_size - 1:
    send_work = torch.tpu.ops.send([x], dst=rank + 1, tag=rank)

  if rank > 0:
    src_rank = rank - 1
    recv_work = torch.tpu.ops.recv([recv_buffer], src=src_rank, tag=src_rank)

    # For irecv, we must wait to ensure the buffer is filled before assertion
    assert recv_work is not None
    recv_work.wait()

    expected = torch.tensor([[float(src_rank), float(src_rank**2)]])
    utils.assert_close(recv_buffer.cpu(), expected)

  # Ensure the send operation is also completed before exiting
  if send_work is not None:
    send_work.wait()


def _synchronized_store_num_keys(store: dist.Store) -> int:
  """Returns `store.num_keys()` read while no rank can mutate the store.

  The store is shared by all ranks, so an unsynchronized read can race with
  another rank posting or deleting P2P descriptor keys. The leading barrier
  ensures every rank has finished its preceding store operations; the trailing
  barrier ensures no rank starts new ones until every rank has read the count.

  Args:
    store: The c10d store shared by all ranks.

  Returns:
    The number of keys in the store.
  """
  dist.barrier()
  num_keys = store.num_keys()
  dist.barrier()
  return num_keys


def _run_send_recv_same_tag() -> None:
  """Tests send and recv with the same tag across pairs and repeated calls."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size < 2:
    return

  store = dist.distributed_c10d._get_default_store()
  num_keys_before = _synchronized_store_num_keys(store)

  # Reuse the same tag across different (src, dst) pairs, repeated transfers
  # between the same pair, reverse-direction transfers, and one-to-many sends.
  p2p_tag = 0
  tensor_shape = (32, 32)

  ranks = list(range(world_size))
  for step in range(3):
    for [r0, r1] in zip(ranks[0::2], ranks[1::2]):
      for src_rank, dst_rank in ((r0, r1), (r1, r0)):
        tensor_value = (
            float(src_rank) ** 2 + float(dst_rank) + float(step * 100)
        )
        if rank == src_rank:
          x = torch.full(tensor_shape, tensor_value, device="tpu")
          torch.tpu.ops.send([x], dst=dst_rank, tag=p2p_tag).wait()
        elif rank == dst_rank:
          recv_buffer = torch.zeros(tensor_shape, device="tpu")
          torch.tpu.ops.recv([recv_buffer], src=src_rank, tag=p2p_tag).wait()
          expected = torch.full(tensor_shape, tensor_value)
          utils.assert_close(recv_buffer.cpu(), expected)

  # Verify per-(peer_rank, tag) counter isolation when rank 0 sends to each peer
  # using the same tag.
  for dst_rank in range(1, world_size):
    tensor_value = float(1000 + dst_rank)
    if rank == 0:
      x = torch.full(tensor_shape, tensor_value, device="tpu")
      torch.tpu.ops.send([x], dst=dst_rank, tag=p2p_tag).wait()
    elif rank == dst_rank:
      recv_buffer = torch.zeros(tensor_shape, device="tpu")
      torch.tpu.ops.recv([recv_buffer], src=0, tag=p2p_tag).wait()
      expected = torch.full(tensor_shape, tensor_value)
      utils.assert_close(recv_buffer.cpu(), expected)

  num_keys_after = _synchronized_store_num_keys(store)
  assert (
      num_keys_after == num_keys_before
  ), f"P2P transfers leaked {num_keys_after - num_keys_before} store keys"


def _run_isend_irecv_does_not_leak_store_keys() -> None:
  """Tests that repeated ring isend/irecv hand-offs do not grow the store."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size < 2:
    return

  store = dist.distributed_c10d._get_default_store()
  dst_rank = (rank + 1) % world_size
  src_rank = (rank - 1) % world_size
  tensor_shape = (16, 16)
  num_steps = 20

  num_keys_before = _synchronized_store_num_keys(store)

  for step in range(num_steps):
    x = torch.full(tensor_shape, float(rank * 100 + step), device="tpu")
    recv_buffer = torch.zeros(tensor_shape, device="tpu")
    # Post the recv before the send so the upstream rank's send never waits on
    # a receiver that hasn't published its descriptor yet (avoids deadlock).
    recv_work = torch.tpu.ops.recv([recv_buffer], src=src_rank, tag=0)
    send_work = torch.tpu.ops.send([x], dst=dst_rank, tag=0)
    recv_work.wait()
    send_work.wait()
    expected = torch.full(tensor_shape, float(src_rank * 100 + step))
    utils.assert_close(recv_buffer.cpu(), expected)

  num_keys_after = _synchronized_store_num_keys(store)
  assert num_keys_after == num_keys_before, (
      f"{num_steps} ring isend/irecv steps leaked "
      f"{num_keys_after - num_keys_before} store keys"
  )


def _run_async_work_tracks_completion() -> None:
  """Tests that the Work of an async collective reports its completion."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  store = dist.distributed_c10d._get_default_store()
  gate_key = "async_work_completion_rank0_checked"
  x = torch.full((4,), float(rank), device="tpu")

  # Ranks > 0 hold off issuing, so the collective cannot complete on rank 0.
  if rank != 0:
    store.wait([gate_key])
  work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
  completed_at_issue = rank == 0 and work.is_completed()
  success_at_issue = rank == 0 and work.is_success()
  exc_at_issue = (
      work.exception() if rank == 0 and _WORK_EXCEPTION_IS_BOUND else None
  )
  if rank == 0:
    store.set(gate_key, "1")
  assert work.wait(datetime.timedelta(seconds=60))
  assert work.wait(datetime.timedelta(seconds=5))
  dist.barrier()
  if rank == 0:
    store.delete_key(gate_key)

  assert not completed_at_issue, "is_completed() was True before peers issued"
  assert not success_at_issue, "is_success() was True before peers issued"
  assert exc_at_issue is None
  assert work.is_completed() and work.is_success()
  if _WORK_EXCEPTION_IS_BOUND:
    assert work.exception() is None
  utils.assert_close(x.cpu(), torch.full((4,), float(sum(range(world_size)))))


def _run_async_work_wait_holds_the_current_stream() -> None:
  """Tests cross-stream wait/synchronize/block_current_stream and prefetch."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  store = dist.distributed_c10d._get_default_store()
  gate_key = "async_work_prefetch_rank0_waited"
  side = torch.tpu.Stream()
  x1 = torch.full((4,), float(rank), device="tpu")
  x2 = torch.ones((4,), device="tpu")
  dist.all_reduce(x1, op=dist.ReduceOp.SUM, async_op=True).wait()
  x1.fill_(float(rank))
  torch.accelerator.synchronize()
  dist.barrier()

  ones = torch.ones(1024, device="tpu")
  held = torch.zeros(1024, pin_memory=True)

  # Prefetch `x1` then `x2` on `side`, each flushed on its own; ranks > 0 hold
  # off `x2`, so `work1.wait()` on the default stream must not wait on `x2`.
  if rank == 0:
    with torch.tpu.stream(side):
      work1 = dist.all_reduce(x1, op=dist.ReduceOp.SUM, async_op=True)
      side.record_event()
      work2 = dist.all_reduce(x2, op=dist.ReduceOp.SUM, async_op=True)
      side.record_event()
    work1.wait()
    held.copy_(ones, non_blocking=True)
    torch.tpu.default_stream().record_event().synchronize()
    work2_still_pending = not work2.is_completed()
    # Peers have not issued `x2` yet, so the default stream must stay held.
    work2.block_current_stream()
    blocked = torch.tpu.default_stream().record_event()
    deadline = time.monotonic() + 0.5
    while not blocked.query() and time.monotonic() < deadline:
      time.sleep(0.001)
    stream_held = not blocked.query()
    store.set(gate_key, "1")
  else:
    with torch.tpu.stream(side):
      work1 = dist.all_reduce(x1, op=dist.ReduceOp.SUM, async_op=True)
    work1.synchronize()
    held.copy_(ones, non_blocking=True)
    torch.tpu.default_stream().record_event().synchronize()
    store.wait([gate_key])
    with torch.tpu.stream(side):
      work2 = dist.all_reduce(x2, op=dist.ReduceOp.SUM, async_op=True)
    work2_still_pending = True
    work2.block_current_stream()
    blocked = torch.tpu.default_stream().record_event()
    stream_held = True

  blocked.synchronize()
  completed_with_event = work2.is_completed()
  dist.barrier()
  if rank == 0:
    store.delete_key(gate_key)

  assert work1.is_completed() and work2.is_completed()
  assert work2_still_pending, "work1.wait() waited on later collective work2"
  assert stream_held, "block_current_stream() did not hold the current stream"
  assert completed_with_event, "the held stream resumed before work2 completed"
  utils.assert_close(held, torch.ones(1024))
  utils.assert_close(x1.cpu(), torch.full((4,), float(sum(range(world_size)))))
  utils.assert_close(x2.cpu(), torch.full((4,), float(world_size)))


def _run_async_work_future_resolves_to_results() -> None:
  """Tests that get_future() resolves to the results of the collective."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  x = torch.full((4,), float(rank), device="tpu")

  work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
  future = work.get_future()
  results = future.wait()
  assert future.done() and len(results) == 1
  utils.assert_close(
      results[0].cpu(), torch.full((4,), float(sum(range(world_size))))
  )

  # Two outputs, so the Work tracks more than one buffer.
  c_in = [torch.full((2,), float(rank + i), device="tpu") for i in range(2)]
  c_out = [torch.empty((2 * world_size,), device="tpu") for _ in range(2)]
  coalesced_opts = dist.distributed_c10d.AllgatherOptions()
  coalesced_opts.asyncOp = True
  pg = dist.distributed_c10d._get_default_group()
  c_work = pg.allgather_into_tensor_coalesced(c_out, c_in, coalesced_opts)
  c_results = c_work.get_future().wait()
  assert c_work.wait(datetime.timedelta(seconds=60))
  assert c_work.is_completed() and c_work.is_success()
  assert len(c_results) == 2
  ranks = torch.arange(world_size, dtype=torch.float32).repeat_interleave(2)
  for i, c_result in enumerate(c_results):
    utils.assert_close(c_result.cpu(), ranks + i)


def _run_async_work_future_is_complete_at_issue() -> None:
  """Tests that get_future() is complete once the collective is issued."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  store = dist.distributed_c10d._get_default_store()
  gate_key = "async_work_future_rank0_checked"
  x = torch.full((4,), float(rank), device="tpu")

  # Ranks > 0 hold off issuing, so the collective cannot complete on rank 0.
  if rank != 0:
    store.wait([gate_key])
  work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
  future = work.get_future()
  done_at_issue = future.done()
  callback_threads: list[int] = []
  future.then(lambda _: callback_threads.append(threading.get_ident()))
  ran_inline = callback_threads == [threading.get_ident()]
  completed_at_issue = rank == 0 and work.is_completed()
  if rank == 0:
    store.set(gate_key, "1")

  assert work.wait(datetime.timedelta(seconds=60))
  dist.barrier()
  if rank == 0:
    store.delete_key(gate_key)

  assert done_at_issue, "get_future() was not done right after issue"
  assert ran_inline, "then() did not run the callback on the calling thread"
  assert not completed_at_issue, "is_completed() was True before peers issued"
  utils.assert_close(x.cpu(), torch.full((4,), float(sum(range(world_size)))))


def _run_async_work_future_callback_updates_the_result() -> None:
  """Tests a callback chained on the future, as DDP comm hooks do."""
  world_size = dist.get_world_size()
  x = torch.full((4,), float(dist.get_rank()), device="tpu")

  mean = (
      dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
      .get_future()
      .then(lambda future: future.value()[0].div_(world_size))
      .wait()
  )

  utils.assert_close(
      mean.cpu(), torch.full((4,), sum(range(world_size)) / world_size)
  )


def _run_async_work_of_gather_has_results_on_root_only() -> None:
  """Tests that result() of a gather Work is empty off the root rank."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  gather_in = torch.full((2,), float(rank), device="tpu")
  gather_out = (
      [torch.empty((2,), device="tpu") for _ in range(world_size)]
      if rank == 0
      else None
  )

  gather_work = dist.gather(gather_in, gather_out, dst=0, async_op=True)
  gather_work.wait()

  if rank == 0:
    assert len(gather_work.result()) == world_size
  else:
    assert not gather_work.result()


def _run_async_work_of_barrier_blocks_the_host() -> None:
  """Tests that wait() on an async barrier blocks until all ranks enter it."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  store = dist.distributed_c10d._get_default_store()
  barrier_key = f"async_barrier_entered:{rank}"

  dist.barrier()
  if rank == 0:
    b_work = dist.barrier(async_op=True)
    b_work.wait()
    peers_entered = all(
        store.check([f"async_barrier_entered:{r}"])
        for r in range(1, world_size)
    )
  else:
    time.sleep(0.02)
    store.set(barrier_key, "1")
    b_work = dist.barrier(async_op=True)
    b_work.wait()
    peers_entered = True
  dist.barrier()
  if rank != 0:
    store.delete_key(barrier_key)
  assert (
      peers_entered
  ), "async barrier wait() returned before peer ranks arrived"


def _run_async_work_ignores_later_stream_work() -> None:
  """Tests that is_completed() tracks the collective, not later stream work."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  store = dist.distributed_c10d._get_default_store()
  gate_key = "async_work_isolation_rank0_checked"
  x = torch.full((4,), float(rank), device="tpu")
  y = torch.ones((4,), device="tpu")
  dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True).wait()
  x.fill_(float(rank))
  torch.accelerator.synchronize()
  dist.barrier()

  work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
  # Starts the collective in every eager mode.
  torch.tpu.default_stream().record_event()
  if rank == 0:
    # Ranks > 0 hold off issuing `y`, so `stream_done` cannot complete yet.
    dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
    stream_done = torch.tpu.default_stream().record_event()
    deadline = time.monotonic() + 60
    while not work.is_completed() and time.monotonic() < deadline:
      time.sleep(0.001)
    completed_before_gate = work.is_completed()
    later_still_pending = not stream_done.query()
    store.set(gate_key, "1")
  else:
    store.wait([gate_key])
    dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
    stream_done = torch.tpu.default_stream().record_event()
    completed_before_gate = True
    later_still_pending = True

  stream_done.synchronize()
  dist.barrier()
  if rank == 0:
    store.delete_key(gate_key)

  assert completed_before_gate, "is_completed() waited on later stream work"
  assert later_still_pending, "the later collective completed without peers"
  assert work.is_completed()
  utils.assert_close(x.cpu(), torch.full((4,), float(sum(range(world_size)))))
  utils.assert_close(y.cpu(), torch.full((4,), float(world_size)))


def _run_async_work_succeeds_after_output_updated_in_place() -> None:
  """Tests that the Work succeeds after its output was updated in place."""
  world_size = dist.get_world_size()
  a = torch.full((4,), float(dist.get_rank()), device="tpu")

  work = dist.all_reduce(a, op=dist.ReduceOp.SUM, async_op=True)
  a.add_(1.0)
  # Lets the update run: it takes over the buffer that the collective wrote.
  torch.accelerator.synchronize()

  assert work.wait(datetime.timedelta(seconds=60))
  assert work.is_completed() and work.is_success()
  utils.assert_close(
      a.cpu(), torch.full((4,), float(sum(range(world_size))) + 1.0)
  )


def _run_async_work_polling_starts_fused_collective() -> None:
  """Tests that polling starts a collective that `spmd_safe` left fused."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()

  @torch.tpu.distributed.spmd_safe
  def _issue_spmd_safe_async(t: torch.Tensor) -> dist.Work:
    return dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

  with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
    s = torch.full((4,), float(rank), device="tpu")
    work = _issue_spmd_safe_async(s)
    assert work.wait()
    deadline = time.monotonic() + 60
    while not work.is_completed() and time.monotonic() < deadline:
      time.sleep(0.001)
    assert work.is_completed() and work.is_success()
  utils.assert_close(s.cpu(), torch.full((4,), float(sum(range(world_size)))))


def _run_async_work_reports_success_without_being_polled() -> None:
  """Tests that is_success() observes a fused collective nothing polled."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()

  @torch.tpu.distributed.spmd_safe
  def _issue_spmd_safe_async(t: torch.Tensor) -> dist.Work:
    return dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)

  with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
    s = torch.full((4,), float(rank), device="tpu")
    work = _issue_spmd_safe_async(s)
    assert work.wait()
    torch.accelerator.synchronize()
    assert work.is_success(), "is_success() was False after the stream synced"
    if _WORK_EXCEPTION_IS_BOUND:
      assert work.exception() is None
  utils.assert_close(s.cpu(), torch.full((4,), float(sum(range(world_size)))))


def _run_async_work_tracks_collective_fused_with_update(
    blocking_flush: bool,
) -> None:
  """Tests the Work of a collective fused with a later in-place update."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()

  @torch.tpu.distributed.spmd_safe
  def _reduce_and_average(t: torch.Tensor) -> dist.Work:
    work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    t.div_(world_size)
    return work

  with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
    t = torch.full((4,), float(rank), device="tpu")
    work = _reduce_and_average(t)
    # Flushes the collective fused with the update.
    if blocking_flush:
      t.cpu()
    else:
      torch.tpu.default_stream().record_event()
    assert work.wait(datetime.timedelta(seconds=60))
    assert work.is_completed() and work.is_success()
    if _WORK_EXCEPTION_IS_BOUND:
      assert work.exception() is None
    torch.accelerator.synchronize()
  utils.assert_close(
      t.cpu(), torch.full((4,), sum(range(world_size)) / world_size)
  )


def _run_async_work_tracks_collective_while_fused_update_is_queued() -> None:
  """Tests the Work of a collective whose fused block is cut but not run."""
  world_size = dist.get_world_size()
  rank = dist.get_rank()

  @torch.tpu.distributed.spmd_safe
  def _reduce_average_and_reduce(
      t: torch.Tensor, u: torch.Tensor
  ) -> tuple[dist.Work, dist.Work]:
    work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    t.div_(world_size)
    other = dist.all_reduce(u, op=dist.ReduceOp.SUM, async_op=True)
    return work, other

  with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
    t = torch.full((4,), float(rank), device="tpu")
    u = torch.full((4,), float(rank) + 1.0, device="tpu")
    work, other = _reduce_average_and_reduce(t, u)
    # Cuts the fused block without waiting for it to start.
    other.is_completed()
    assert work.wait(datetime.timedelta(seconds=60))
    assert work.is_completed() and work.is_success()
    assert other.wait(datetime.timedelta(seconds=60))
    assert other.is_completed() and other.is_success()
    torch.accelerator.synchronize()
  utils.assert_close(
      t.cpu(), torch.full((4,), sum(range(world_size)) / world_size)
  )
  utils.assert_close(
      u.cpu(), torch.full((4,), float(sum(range(world_size)) + world_size))
  )


def _run_collectives_with_non_uniform_deferred_ops() -> None:
  """Tests collective functionality with non-uniform (per-rank) deferred ops."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # Create input tensors with deferred ops on some (but not all) of the ranks.
  def make_input():
    x = torch.tensor([0.0, 1.0, float(rank), float(rank**2)], device="tpu")
    if rank % 3 == 1:
      x = (x @ torch.eye(x.size(0)).to(device="tpu")).relu()
    return x

  # Make sure all these collectives finish (don't hang):
  x = make_input()
  torch.distributed.all_reduce(x, torch.distributed.ReduceOp.SUM)
  logging.info("deferred + all_reduce (rank=%d): out = %s", rank, x.cpu())

  x = make_input()
  torch.distributed.broadcast(x, src=1)
  logging.info("deferred + broadcast (rank=%d): out = %s", rank, x.cpu())

  x = make_input()
  outs = [torch.zeros(4, device="tpu") for _ in range(world_size)]
  torch.distributed.all_gather(outs, x)
  outs = [o.to(device="cpu") for o in outs]
  logging.info("deferred + all_gather (rank=%d): out = %s", rank, outs)

  xs = [make_input() for _ in range(world_size)]
  out = torch.zeros(4, device="tpu")
  torch.distributed.reduce_scatter(out, xs)
  logging.info("deferred + reduce_scatter (rank=%d): out = %s", rank, out.cpu())

  x = torch.cat([make_input() for _ in range(world_size)])
  out = torch.zeros(4, device="tpu")
  torch.distributed.reduce_scatter_tensor(out, x)
  logging.info(
      "deferred + reduce_scatter_tensor (rank=%d): out = %s", rank, out.cpu()
  )


def _run_barrier_blocking(async_op: bool) -> None:
  """Tests barrier functionality blocks."""
  rank = dist.get_rank()

  # Ensure all ranks start reasonably close to each other.
  torch.distributed.barrier()

  sleep_seconds = 0.5
  epsilon = 0.05

  start = time.time()
  if rank == 0:
    time.sleep(sleep_seconds)

  torch.distributed.barrier(async_op=async_op)

  elapsed = time.time() - start

  print(f"Rank {rank} finished barrier in {elapsed}s")

  if async_op:
    # In async mode, only rank 0 should be blocked. Other ranks should return
    # immediately.
    if rank == 0:
      if not elapsed >= sleep_seconds - epsilon:
        raise RuntimeError(
            f"Rank 0 took {elapsed}s, expected >= {sleep_seconds}"
        )
    else:
      if not elapsed < epsilon:
        raise RuntimeError(
            f"Rank {rank} took {elapsed}s, expected < {sleep_seconds}"
        )

  else:
    # In sync mode, all ranks should be blocked until rank 0 wakes up and enters
    # the barrier. Therefore, the elapsed time for everyone should be at least
    # sleep_seconds.
    if not elapsed >= sleep_seconds - epsilon:
      raise RuntimeError(
          f"Rank {rank} took {elapsed}s, expected >= {sleep_seconds}"
      )


def _run_rank_variable_dead_collective_without_hang(world_size: int) -> None:
  """Tests that process exits on some ranks will not cause hangs on others."""
  rank = int(os.environ["RANK"])
  input_tensor = torch.ones(1, dtype=torch.float32, device="cpu").to(
      device="tpu"
  )
  dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
  if rank != 0:
    # Ranks other than 0 will try to exit without waiting for the collective to
    # finish.
    return
  # Rank 0 will wait for the result of the all_reduce.
  # This should not hang.
  result = input_tensor.cpu()
  expected = torch.tensor([world_size], dtype=torch.float32, device="cpu")
  utils.assert_close(result, expected)


class CollectiveOpsTest(continuous_test_utils.TpuMultiProcContinuousTest):
  world_size = distributed_utils.get_tpu_device_count()

  def test_global_device_count(self) -> None:
    # Verify that rank-related environment variables are set by launcher.
    self.assertIn("RANK", os.environ)
    self.assertIn("LOCAL_RANK", os.environ)
    self.assertIn("GROUP_RANK", os.environ)
    self.assertIn("WORLD_SIZE", os.environ)
    self.assertEqual(int(os.environ["WORLD_SIZE"]), self.world_size)
    self.assertEqual(int(os.environ["RANK"]), self.rank)

    self.assertEqual(dist.get_world_size(), self.world_size)
    self.assertEqual(dist.get_rank(), self.rank)

    global_device_ids = tpu_distributed.all_global_device_ids()
    self.assertLen(global_device_ids, self.world_size)

    this_device = torch.accelerator.current_device_index()
    # torch.accelerator.current_device_index() returns the id among all
    # addressable devices, not the global device id.
    self.assertGreaterEqual(this_device, 0)
    self.assertLess(
        this_device,
        torch.accelerator.device_count(),
        msg=(
            "Expected device id to be in range [0,"
            f" {torch.accelerator.device_count()}), got {this_device}"
        ),
    )
    # A worker uses only its own device, and PyTorch picks device
    # `rank % device_count()`, so the count must be 1.
    self.assertEqual(torch.accelerator.device_count(), 1)

  def test_all_reduce_sum(self) -> None:
    w = self.world_size
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.SUM,
        expected=[
            0.0,
            float(w),
            float(sum(range(w))),
            float(sum(r**2 for r in range(w))),
        ],
        dtype=torch.float32,
    )

  def test_all_reduce_avg(self) -> None:
    w = self.world_size
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.AVG,
        expected=[
            0.0,
            1.0,
            sum(range(w)) / w,
            sum(r**2 for r in range(w)) / w,
        ],
        dtype=torch.float32,
    )

  def test_all_reduce_product(self) -> None:
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.PRODUCT,
        expected=[0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

  def test_all_reduce_min(self) -> None:
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.MIN,
        expected=[0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

  def test_all_reduce_max(self) -> None:
    w = self.world_size
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.MAX,
        expected=[0.0, 1.0, float(w - 1), float((w - 1) ** 2)],
        dtype=torch.float32,
    )

  def test_all_reduce_band(self) -> None:
    _run_all_reduce(
        input_fn=_all_reduce_bitwise_input_fn,
        reduce_op=torch.distributed.ReduceOp.BAND,
        expected=[0, 0, 255],
        dtype=torch.int32,
    )

  def test_all_reduce_bor(self) -> None:
    w = self.world_size
    bor_val2 = 0
    for r in range(w):
      bor_val2 |= r * 3
    _run_all_reduce(
        input_fn=_all_reduce_bitwise_input_fn,
        reduce_op=torch.distributed.ReduceOp.BOR,
        expected=[(1 << w) - 1, bor_val2, 255],
        dtype=torch.int32,
    )

  def test_all_reduce_bxor(self) -> None:
    w = self.world_size
    bxor_val2 = 0
    for r in range(w):
      bxor_val2 ^= r * 3
    _run_all_reduce(
        input_fn=_all_reduce_bitwise_input_fn,
        reduce_op=torch.distributed.ReduceOp.BXOR,
        expected=[0, bxor_val2, 0 if w % 2 == 0 else 255],
        dtype=torch.int32,
    )

  def test_all_reduce_sum_fixed_values(self) -> None:
    w = self.world_size
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_const,
        reduce_op=torch.distributed.ReduceOp.SUM,
        expected=[1.0 * w, 2.0 * w, 3.0 * w, 4.0 * w],
        dtype=torch.float32,
    )

  def test_all_reduce_product_fixed_values(self) -> None:
    w = self.world_size
    _run_all_reduce(
        input_fn=_all_reduce_input_fn_const,
        reduce_op=torch.distributed.ReduceOp.PRODUCT,
        expected=[1.0, float(2**w), float(3**w), float(4**w)],
        dtype=torch.float32,
    )

  def test_all_gather(self) -> None:
    _run_all_gather()

  def test_all_gather_scalar(self) -> None:
    _run_all_gather_scalar()

  def test_all_gather_tensor_concat(self) -> None:
    _run_all_gather_tensor_concat()

  def test_all_gather_tensor_scalar(self) -> None:
    _run_all_gather_tensor_scalar()

  def test_all_gather_tensor_stack(self) -> None:
    _run_all_gather_tensor_stack()

  def test_all_gather_rank_order(self) -> None:
    _run_all_gather_rank_order()

  def test_experimental_collectives_on_dim(self) -> None:
    _run_experimental_collectives_on_dim()

  def test_gather_scalar(self) -> None:
    _run_gather_scalar()

  def test_gather_tensor(self) -> None:
    _run_gather_tensor()

  def test_gather_with_root_subset_use(self) -> None:
    _run_gather_with_root_subset_use()

  def test_scatter_tensor(self) -> None:
    src_rank = 0
    size = 5
    inputs = None
    if self.rank == src_rank:
      inputs = [
          torch.full((size,), i, device="tpu", dtype=torch.float32)
          for i in range(self.world_size)
      ]
    output = torch.empty(size, device="tpu", dtype=torch.float32)

    torch.distributed.scatter(output, inputs, src=src_rank)
    expected = torch.full((size,), fill_value=self.rank, dtype=torch.float32)
    utils.assert_close(output.cpu(), expected)

  def test_scatter_scalar(self) -> None:
    src_rank = 0
    inputs = None
    if self.rank == src_rank:
      inputs = [
          torch.tensor(i, device="tpu", dtype=torch.float32)
          for i in range(self.world_size)
      ]
    output = torch.tensor(-1, device="tpu", dtype=torch.float32)

    torch.distributed.scatter(output, inputs, src=src_rank)
    expected = torch.tensor(self.rank, dtype=torch.float32)
    utils.assert_close(output.cpu(), expected)

  def test_reduce_scatter(self) -> None:
    _run_reduce_scatter()

  def test_reduce_scatter_tensor_stack(self) -> None:
    _run_reduce_scatter_tensor_stack()

  def test_reduce_scatter_tensor_concat(self) -> None:
    _run_reduce_scatter_tensor_concat()

  def test_reduce_scatter_tensor_avg(self) -> None:
    _run_reduce_scatter_tensor_avg()

  def test_broadcast(self) -> None:
    _run_broadcast()

  def test_broadcast_objects(self) -> None:
    _run_broadcast_objects()

  def test_async_work_tracks_completion(self) -> None:
    _run_async_work_tracks_completion()

  def test_async_work_wait_holds_the_current_stream(self) -> None:
    _run_async_work_wait_holds_the_current_stream()

  def test_async_work_future_resolves_to_results(self) -> None:
    _run_async_work_future_resolves_to_results()

  def test_async_work_future_is_complete_at_issue(self) -> None:
    _run_async_work_future_is_complete_at_issue()

  def test_async_work_future_callback_updates_the_result(self) -> None:
    _run_async_work_future_callback_updates_the_result()

  def test_async_work_of_gather_has_results_on_root_only(self) -> None:
    _run_async_work_of_gather_has_results_on_root_only()

  def test_async_work_of_barrier_blocks_the_host(self) -> None:
    _run_async_work_of_barrier_blocks_the_host()

  def test_async_work_ignores_later_stream_work(self) -> None:
    _run_async_work_ignores_later_stream_work()

  def test_async_work_ignores_later_stream_work_when_deferred(self) -> None:
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      _run_async_work_ignores_later_stream_work()

  def test_async_work_succeeds_after_output_updated_in_place(self) -> None:
    _run_async_work_succeeds_after_output_updated_in_place()

  def test_async_work_polling_starts_fused_collective(self) -> None:
    _run_async_work_polling_starts_fused_collective()

  def test_async_work_reports_success_without_being_polled(self) -> None:
    _run_async_work_reports_success_without_being_polled()

  def test_async_work_tracks_collective_fused_with_update(self) -> None:
    _run_async_work_tracks_collective_fused_with_update(blocking_flush=True)

  def test_async_work_tracks_collective_while_fused_update_starts(self) -> None:
    _run_async_work_tracks_collective_fused_with_update(blocking_flush=False)

  def test_async_work_tracks_collective_while_fused_update_is_queued(
      self,
  ) -> None:
    _run_async_work_tracks_collective_while_fused_update_is_queued()

  def test_collectives_with_non_uniform_deferred_ops(self) -> None:
    _run_collectives_with_non_uniform_deferred_ops()

  def test_barrier(self) -> None:
    torch.distributed.barrier(async_op=False)
    torch.distributed.barrier(async_op=True)

  def test_barrier_with_device_ids(self) -> None:
    torch.distributed.barrier(
        device_ids=[torch.accelerator.current_device_index()], async_op=False
    )
    torch.distributed.barrier(
        device_ids=[torch.accelerator.current_device_index()], async_op=True
    )

  def test_barrier_rejects_foreign_device_ids(self) -> None:
    # A process addresses a single TPU, so only [0] is meaningful. When the
    # first device id is out of bounds (e.g. [1]), c10d::ProcessGroup::barrier
    # fails while allocating the dispatch tensor on tpu:1. When the first id is
    # 0 but multiple ids are given, ProcessGroupTpu::barrier rejects device_ids.
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
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=just using helper
          exc_type, msg_regex
      ):
        torch.distributed.barrier(device_ids=device_ids)

  @parameterized.named_parameters(
      ("sync", False),
      ("async", True),
  )
  def test_barrier_blocking(self, async_op: bool) -> None:
    _run_barrier_blocking(async_op=async_op)

  @absltest.skipIf(_IS_TPU_V7, _SKIP_P2P_ON_TPU_V7_REASON)
  def test_send_recv(self) -> None:
    _run_send_recv()

  @absltest.skipIf(_IS_TPU_V7, _SKIP_P2P_ON_TPU_V7_REASON)
  def test_isend_irecv(self) -> None:
    _run_isend_irecv()

  @absltest.skipIf(_IS_TPU_V7, _SKIP_P2P_ON_TPU_V7_REASON)
  def test_send_recv_same_tag(self) -> None:
    _run_send_recv_same_tag()

  @absltest.skipIf(_IS_TPU_V7, _SKIP_P2P_ON_TPU_V7_REASON)
  def test_isend_irecv_does_not_leak_store_keys(self) -> None:
    _run_isend_irecv_does_not_leak_store_keys()


class ProcessExitTest(seed_test_utils.MultiProcessRepeatableTest):
  """Tests that need each worker process to exit at the end of the test."""

  _world_size = distributed_utils.get_tpu_device_count()

  def test_rank_variable_dead_collective_without_hang(self) -> None:
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=_test_wrapper,
        test_fn=_run_rank_variable_dead_collective_without_hang,
        world_size=self._world_size,
    )


if __name__ == "__main__":  # We are in the parent process.
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
