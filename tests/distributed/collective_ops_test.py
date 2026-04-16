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
import time
from typing import Any, Callable, List, Union

from absl import logging
from absl.testing import absltest

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
      "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
  ] += f"/{date}.{random_seed}"


import torch  # pylint: disable=g-import-not-at-top
from torch import distributed as dist
import torch.multiprocessing as mp

from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


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
    # Ensure all ranks synchronize at the end of each test so that every worker
    # in the slice completes. This prevents one rank from destroying the process
    # group while others are still performing collectives or verification.
    # TODO: remove this wrapper when we figure out a principled way to prevent
    # the premature exit.
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

  # Assuming world size w = 8
  # Example values:
  # val1: Sets the r-th bit and the (w-1-r)-th bit.
  # val2: Simple arithmetic series based on rank.
  # val3: A constant value with all bits set in a byte.
  val1 = (1 << r) | (1 << (w - 1 - r))
  val2 = r * 3
  val3 = 0xFF
  return [val1, val2, val3]


def run_all_reduce(
    input_fn: Callable[[int, int], List[Union[int, float]]],
    reduce_op: torch.distributed.ReduceOp,
    expected: List[Union[int, float]],
    dtype: torch.dtype,
) -> None:
  """Tests all-reduce functionality."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


def run_all_gather_scalar() -> None:
  """Tests all_gather on scalar input."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  outputs = [
      torch.tensor(0, device="tpu", dtype=torch.float32)
      for _ in range(world_size)
  ]
  x = torch.tensor(rank, device="tpu", dtype=torch.float32)
  torch.distributed.all_gather(outputs, x, async_op=False)
  outputs_cpu = [o.to("cpu") for o in outputs]
  for idx, scalar in enumerate(outputs_cpu):
    assert scalar.item() == idx


def run_all_gather_tensor_concat() -> None:
  """Tests all_gather_into_tensor with concatenation."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.tensor([rank, rank], device="tpu", dtype=torch.float32)
  output = torch.zeros(world_size * 2, device="tpu", dtype=torch.float32)
  handle = torch.distributed.all_gather_into_tensor(output, x, async_op=True)
  handle.wait()
  output_cpu = output.cpu()

  # expected = [0, 0, 1, 1, ...]
  iota = torch.arange(world_size, dtype=torch.float32)
  expected = torch.stack([iota, iota]).T.flatten()
  utils.assert_close(output_cpu, expected)


def run_all_gather_tensor_scalar() -> None:
  """Tests all_gather_into_tensor on scalar input."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.tensor(rank, device="tpu", dtype=torch.float32)
  output = torch.zeros(world_size, device="tpu", dtype=torch.float32)
  torch.distributed.all_gather_into_tensor(output, x, async_op=False)
  output_cpu = output.cpu()
  expected = torch.arange(world_size, dtype=torch.float32)
  utils.assert_close(output_cpu, expected)


def run_all_gather_tensor_stack() -> None:
  """Tests all_gather_into_tensor with stacking."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  x = torch.tensor([rank, rank], device="tpu", dtype=torch.float32)
  output = torch.zeros((world_size, 2), device="tpu", dtype=torch.float32)
  torch.distributed.all_gather_into_tensor(output, x, async_op=False)
  output_cpu = output.cpu()

  # expected = [[0, 0], [1, 1], ...]
  iota = torch.arange(world_size, dtype=torch.float32)
  expected = torch.stack([iota, iota]).T
  utils.assert_close(output_cpu, expected)


def run_gather_scalar() -> None:
  """Tests gather on scalar input."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


def run_gather_tensor() -> None:
  """Tests gather on tensor input."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


def run_gather_with_root_subset_use() -> None:
  """Tests gather when only a subset of outputs is used on the root rank.

  This test verifies that gather does not hang when only a subset of output
  tensors are used on the root rank, matching eager PyTorch behavior.
  """
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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


def run_reduce_scatter() -> None:
  """Tests reduce-scatter functionality."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
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

  expected = torch.tensor([
      [world_size, rank * world_size, rank**2 * world_size],
      [28.0, 140.0, rank * 28.0],
  ])

  logging.info("reduce_scatter expected (rank %d): %s", rank, expected)
  utils.assert_close(output_cpu, expected)


def run_reduce_scatter_tensor_stack() -> None:
  """Tests reduce_scatter_tensor (in stack input mode) functionality."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

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


def run_reduce_scatter_tensor_concat() -> None:
  """Tests reduce_scatter_tensor (in concat input mode) functionality."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

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


def run_reduce_scatter_tensor_avg() -> None:
  """Tests reduce_scatter_tensor with an AVG reduction op."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

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


def run_broadcast() -> None:
  """Tests collective broadcast functionality."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Try broadcasting from all possible src ranks:
  for src_rank in range(world_size):
    # Start with different tensors on each rank:
    x = torch.tensor([0.0, 1.0, float(rank), float(rank**2)], device="tpu")

    # Should end up with the same tensor on all ranks:
    expected = torch.tensor([0.0, 1.0, float(src_rank), float(src_rank**2)])

    handle = torch.distributed.broadcast(x, src=src_rank, async_op=True)
    handle.wait()
    utils.assert_close(x.cpu(), expected)


def run_broadcast_objects() -> None:
  """Tests broadcast_object_list collective."""
  rank = int(os.environ["RANK"])
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


def run_send_recv() -> None:
  """Tests point-to-point blocking send and recv functionality."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  if world_size < 2:
    return

  x = torch.tensor([[float(rank), float(rank**2)]], device="tpu")
  recv_buffer = torch.zeros_like(x, device="tpu")

  if rank < world_size - 1:
    torch.ops.torch_tpu.experimental_send([x], dst=rank + 1, tag=rank).wait()

  if rank > 0:
    src_rank = rank - 1
    torch.ops.torch_tpu.experimental_recv(
        [recv_buffer], src=src_rank, tag=src_rank
    ).wait()
    expected = torch.tensor([[float(src_rank), float(src_rank**2)]])
    utils.assert_close(recv_buffer.cpu(), expected)


def run_isend_irecv() -> None:
  """Tests point-to-point asynchronous isend and irecv functionality."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  if world_size < 2:
    return

  x = torch.tensor([[float(rank), float(rank**2)]], device="tpu")
  recv_buffer = torch.zeros_like(x, device="tpu")

  send_work = None
  if rank < world_size - 1:
    send_work = torch.ops.torch_tpu.experimental_send(
        [x], dst=rank + 1, tag=rank
    )

  if rank > 0:
    src_rank = rank - 1
    recv_work = torch.ops.torch_tpu.experimental_recv(
        [recv_buffer], src=src_rank, tag=src_rank
    )

    # For irecv, we must wait to ensure the buffer is filled before assertion
    assert recv_work is not None
    recv_work.wait()

    expected = torch.tensor([[float(src_rank), float(src_rank**2)]])
    utils.assert_close(recv_buffer.cpu(), expected)

  # Ensure the send operation is also completed before exiting
  if send_work is not None:
    send_work.wait()


def run_send_recv_same_tag():
  """Tests send and recv with the same tag for different src and dst ranks."""
  dist.init_process_group(backend="tpu_dist")

  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size < 2:
    return

  # Use the same tag for all send/recv pairs to check if tag uniqueness is not
  # required when src and dst ranks are different.
  p2p_tag = 0

  ranks = list(range(world_size))
  for [src_rank, dst_rank] in zip(ranks[0::2], ranks[1::2]):
    tensor_shape = (32, 32)
    tensor_value = float(src_rank) ** 2 + float(dst_rank)

    if rank == src_rank:
      x = torch.full(tensor_shape, tensor_value, device="tpu")
      torch.ops.torch_tpu.experimental_send(
          [x], dst=dst_rank, tag=p2p_tag
      ).wait()
    elif rank == dst_rank:
      recv_buffer = torch.zeros(tensor_shape, device="tpu")
      torch.ops.torch_tpu.experimental_recv(
          [recv_buffer], src=src_rank, tag=p2p_tag
      ).wait()

      expected = torch.full(tensor_shape, tensor_value)
      utils.assert_close(recv_buffer.cpu(), expected)


def run_collectives_with_non_uniform_deferred_ops() -> None:
  """Tests collective functionality with non-uniform (per-rank) deferred ops."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

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


def run_barrier(async_op: bool) -> None:
  """Tests barrier functionality."""
  torch.distributed.barrier(async_op=async_op)


def run_barrier_blocking(async_op: bool) -> None:
  """Tests barrier functionality blocks."""
  rank = int(os.environ["RANK"])

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


class CollectiveOpsTest(absltest.TestCase):
  _world_size = 8

  def test_all_reduce_sum(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.SUM,
        expected=[0.0, 8.0, 28.0, 140.0],
        dtype=torch.float32,
    )

  def test_all_reduce_avg(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.AVG,
        expected=[0.0, 1.0, 3.5, 17.5],
        dtype=torch.float32,
    )

  def test_all_reduce_product(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.PRODUCT,
        expected=[0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

  def test_all_reduce_min(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.MIN,
        expected=[0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

  def test_all_reduce_max(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_rank,
        reduce_op=torch.distributed.ReduceOp.MAX,
        expected=[0.0, 1.0, 7.0, 49.0],
        dtype=torch.float32,
    )

  def test_all_reduce_band(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_bitwise_input_fn,
        reduce_op=torch.distributed.ReduceOp.BAND,
        expected=[0, 0, 255],
        dtype=torch.int32,
    )

  def test_all_reduce_bor(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_bitwise_input_fn,
        reduce_op=torch.distributed.ReduceOp.BOR,
        expected=[255, 31, 255],
        dtype=torch.int32,
    )

  def test_all_reduce_bxor(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_bitwise_input_fn,
        reduce_op=torch.distributed.ReduceOp.BXOR,
        expected=[0, 8, 0],
        dtype=torch.int32,
    )

  def test_all_reduce_sum_fixed_values(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_const,
        reduce_op=torch.distributed.ReduceOp.SUM,
        expected=[8, 16, 24, 32],
        dtype=torch.float32,
    )

  def test_all_reduce_product_fixed_values(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_reduce,
        input_fn=_all_reduce_input_fn_const,
        reduce_op=torch.distributed.ReduceOp.PRODUCT,
        expected=[1.0, 256.0, 6561.0, 65536.0],
        dtype=torch.float32,
    )

  def test_all_gather_scalar(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_gather_scalar,
    )

  def test_all_gather_tensor_concat(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_gather_tensor_concat,
    )

  def test_all_gather_tensor_scalar(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_gather_tensor_scalar,
    )

  def test_all_gather_tensor_stack(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_gather_tensor_stack,
    )

  def test_gather_scalar(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_gather_scalar,
    )

  def test_gather_tensor(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_gather_tensor,
    )

  def test_gather_with_root_subset_use(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_gather_with_root_subset_use,
    )

  def test_reduce_scatter(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_reduce_scatter,
    )

  def test_reduce_scatter_tensor_stack(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_reduce_scatter_tensor_stack,
    )

  def test_reduce_scatter_tensor_concat(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_reduce_scatter_tensor_concat,
    )

  def test_reduce_scatter_tensor_avg(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_reduce_scatter_tensor_avg,
    )

  def test_broadcast(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_broadcast,
    )

  def test_broadcast_objects(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_broadcast_objects,
    )

  def test_collectives_with_non_uniform_deferred_ops(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_collectives_with_non_uniform_deferred_ops,
    )

  def test_barrier(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_barrier,
        async_op=False,
    )
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_barrier,
        async_op=True,
    )

  def test_send_recv(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_send_recv, world_size=self._world_size
        ),
    )

  def test_isend_irecv(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_isend_irecv, world_size=self._world_size
        ),
    )

  def test_send_recv_same_tag(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_send_recv_same_tag, world_size=self._world_size
        ),
    )


if __name__ == "__main__":  # We are in the parent process.
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
