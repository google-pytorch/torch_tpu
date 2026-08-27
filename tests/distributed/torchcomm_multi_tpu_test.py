# Copyright 2026 Google LLC
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

"""Multi-TPU tests for TorchCommsTPU communicator and collectives."""

from typing import Callable
from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import torchcomm_tpu
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import distributed_utils
import torchcomms


def _run_with_torchcomms(
    fn: Callable[[torchcomms.TorchComm, int, int], None],
) -> None:
  """Worker wrapper that initializes TPU distributed PG and creates communicator via torchcomms.new_comm."""
  torchcomm_tpu.register_torchcomms_tpu()
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  dev = torch.device("tpu", torch.tpu.current_device())

  comm = torchcomms.new_comm(
      "tpu",
      device=dev,
      name=f"tpu_comm_world_{world_size}",
  )

  try:
    fn(comm, rank, world_size)
  finally:
    if not comm.get_backend_impl().is_finalized():
      comm.finalize()
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def _worker_test_comm_attributes(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  assert comm.get_rank() == rank
  assert comm.get_size() == world_size
  assert comm.get_backend() == "tpu"
  assert not comm.get_backend_impl().is_finalized()
  logging.info("Rank %d/%d verified communicator attributes.", rank, world_size)


def _worker_test_all_reduce(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  tensor = torch.tensor([float(rank + 1)], dtype=torch.float32, device="tpu")
  work = comm.all_reduce(tensor, op=torchcomms.ReduceOp.SUM, async_op=False)
  work.wait()
  assert work.is_completed()

  expected_sum = float(world_size * (world_size + 1) // 2)
  actual_val = tensor.item()
  assert (
      abs(actual_val - expected_sum) < 1e-4
  ), f"Rank {rank}: expected all_reduce sum {expected_sum}, got {actual_val}"
  logging.info("Rank %d: all_reduce sum verified %f.", rank, actual_val)


def _worker_test_all_gather(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  in_tensor = torch.tensor(
      [float(rank * 10 + 1)], dtype=torch.float32, device="tpu"
  )
  tensor_list = [
      torch.zeros(1, dtype=torch.float32, device="tpu")
      for _ in range(world_size)
  ]
  work = comm.all_gather(tensor_list, in_tensor, async_op=False)
  work.wait()
  assert work.is_completed()

  for i in range(world_size):
    expected_val = float(i * 10 + 1)
    actual_val = tensor_list[i].item()
    assert abs(actual_val - expected_val) < 1e-4, (
        f"Rank {rank}: all_gather slot {i} expected {expected_val}, got"
        f" {actual_val}"
    )
  logging.info(
      "Rank %d: all_gather verified across all %d slots.", rank, world_size
  )


def _worker_test_all_gather_single(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  in_tensor = torch.tensor([float(rank)], dtype=torch.float32, device="tpu")
  out_tensor = torch.zeros(world_size, dtype=torch.float32, device="tpu")
  work = comm.all_gather_single(out_tensor, in_tensor, async_op=False)
  work.wait()
  assert work.is_completed()

  for i in range(world_size):
    actual_val = out_tensor[i].item()
    assert abs(actual_val - float(i)) < 1e-4, (
        f"Rank {rank}: all_gather_single slot {i} expected {float(i)}, got"
        f" {actual_val}"
    )
  logging.info("Rank %d: all_gather_single verified.", rank)


def _worker_test_broadcast(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  if rank == 0:
    tensor = torch.tensor([123.456, 789.0], dtype=torch.float32, device="tpu")
  else:
    tensor = torch.zeros(2, dtype=torch.float32, device="tpu")

  work = comm.broadcast(tensor, root=0, async_op=False)
  work.wait()
  assert work.is_completed()

  assert abs(tensor[0].item() - 123.456) < 1e-3
  assert abs(tensor[1].item() - 789.0) < 1e-3
  logging.info("Rank %d: broadcast from rank 0 verified.", rank)


def _worker_test_barrier(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  work = comm.barrier(async_op=False)
  work.wait()
  assert work.is_completed()
  logging.info("Rank %d: barrier completed successfully.", rank)


def _worker_test_finalize(
    comm: torchcomms.TorchComm, rank: int, world_size: int
) -> None:
  backend_impl = comm.get_backend_impl()
  assert not backend_impl.is_finalized()
  comm.finalize()
  assert backend_impl.is_finalized()
  assert backend_impl.get_backend() is None
  logging.info("Rank %d: finalize completed successfully.", rank)


def _run_test_attributes():
  _run_with_torchcomms(_worker_test_comm_attributes)


def _run_test_all_reduce():
  _run_with_torchcomms(_worker_test_all_reduce)


def _run_test_all_gather():
  _run_with_torchcomms(_worker_test_all_gather)


def _run_test_all_gather_single():
  _run_with_torchcomms(_worker_test_all_gather_single)


def _run_test_broadcast():
  _run_with_torchcomms(_worker_test_broadcast)


def _run_test_barrier():
  _run_with_torchcomms(_worker_test_barrier)


def _run_test_finalize():
  _run_with_torchcomms(_worker_test_finalize)


class TorchCommsMultiTpuTest(
    absltest.TestCase,  # ABSLTEST_OK=Distributed multi-process test.
):

  WORLD_SIZE = 8

  def test_multi_tpu_attributes(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_attributes,
            world_size=self.WORLD_SIZE,
        ),
    )

  def test_multi_tpu_all_reduce(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_all_reduce,
            world_size=self.WORLD_SIZE,
        ),
    )

  def test_multi_tpu_all_gather(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_all_gather,
            world_size=self.WORLD_SIZE,
        ),
    )

  def test_multi_tpu_all_gather_single(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_all_gather_single,
            world_size=self.WORLD_SIZE,
        ),
    )

  def test_multi_tpu_broadcast(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_broadcast,
            world_size=self.WORLD_SIZE,
        ),
    )

  def test_multi_tpu_barrier(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_barrier,
            world_size=self.WORLD_SIZE,
        ),
    )

  def test_multi_tpu_finalize(self):
    distributed_utils.dist_run(
        nproc_per_node=self.WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _run_test_finalize,
            world_size=self.WORLD_SIZE,
        ),
    )


# Alias for backward compatibility
TorchCommMultiTpuTest = TorchCommsMultiTpuTest

if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
