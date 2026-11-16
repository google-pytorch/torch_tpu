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

from typing import Any
from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal.distributed import torchcomm_tpu
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils
import torchcomms


class TorchCommsMultiTpuTest(continuous_test_utils.TpuMultiProcContinuousTest):

  @property
  def world_size(self) -> int:
    return 8

  @classmethod
  def _init_pg(cls, rank: int, world_size: int, rdvz_file: str | None) -> None:
    torchcomm_tpu.register_torchcomms_tpu()
    super()._init_pg(rank, world_size, rdvz_file)

  def _create_comm(self) -> torchcomms.TorchComm:
    dev = torch.device("tpu", torch.accelerator.current_device_index())
    return torchcomms.new_comm(
        "tpu",
        device=dev,
        name=f"tpu_comm_world_{self.world_size}",
    )

  def _run_with_comm(self, fn: Any) -> None:
    comm = self._create_comm()
    try:
      fn(comm)
    finally:
      if not comm.get_backend_impl().is_finalized():
        comm.finalize()

  def test_multi_tpu_attributes(self):
    def _body(comm: torchcomms.TorchComm) -> None:
      self.assertEqual(comm.get_rank(), self.rank)
      self.assertEqual(comm.get_size(), self.world_size)
      self.assertEqual(comm.get_backend(), "tpu")
      self.assertFalse(comm.get_backend_impl().is_finalized())
      logging.info(
          "Rank %d/%d verified communicator attributes.",
          self.rank,
          self.world_size,
      )

    self._run_with_comm(_body)

  def test_multi_tpu_all_reduce(self):
    def _body(comm: torchcomms.TorchComm) -> None:
      tensor = torch.tensor(
          [float(self.rank + 1)], dtype=torch.float32, device="tpu"
      )
      work = comm.all_reduce(tensor, op=torchcomms.ReduceOp.SUM, async_op=False)
      work.wait()
      self.assertTrue(work.is_completed())

      expected_sum = float(self.world_size * (self.world_size + 1) // 2)
      actual_val = tensor.item()
      self.assertAlmostEqual(
          actual_val,
          expected_sum,
          places=4,
          msg=(
              f"Rank {self.rank}: expected all_reduce sum {expected_sum}, got"
              f" {actual_val}"
          ),
      )
      logging.info(
          "Rank %d: all_reduce sum verified %f.", self.rank, actual_val
      )

    self._run_with_comm(_body)

  def test_multi_tpu_all_gather(self):
    def _body(comm: torchcomms.TorchComm) -> None:
      in_tensor = torch.tensor(
          [float(self.rank * 10 + 1)], dtype=torch.float32, device="tpu"
      )
      tensor_list = [
          torch.zeros(1, dtype=torch.float32, device="tpu")
          for _ in range(self.world_size)
      ]
      work = comm.all_gather(tensor_list, in_tensor, async_op=False)
      work.wait()
      self.assertTrue(work.is_completed())

      for i in range(self.world_size):
        expected_val = float(i * 10 + 1)
        actual_val = tensor_list[i].item()
        self.assertAlmostEqual(
            actual_val,
            expected_val,
            places=4,
            msg=(
                f"Rank {self.rank}: all_gather slot {i} expected"
                f" {expected_val}, got {actual_val}"
            ),
        )
      logging.info(
          "Rank %d: all_gather verified across all %d slots.",
          self.rank,
          self.world_size,
      )

    self._run_with_comm(_body)

  def test_multi_tpu_all_gather_single(self):
    def _body(comm: torchcomms.TorchComm) -> None:
      in_tensor = torch.tensor(
          [float(self.rank)], dtype=torch.float32, device="tpu"
      )
      out_tensor = torch.zeros(
          self.world_size, dtype=torch.float32, device="tpu"
      )
      work = comm.all_gather_single(out_tensor, in_tensor, async_op=False)
      work.wait()
      self.assertTrue(work.is_completed())

      for i in range(self.world_size):
        actual_val = out_tensor[i].item()
        self.assertAlmostEqual(
            actual_val,
            float(i),
            places=4,
            msg=(
                f"Rank {self.rank}: all_gather_single slot {i} expected"
                f" {float(i)}, got {actual_val}"
            ),
        )
      logging.info("Rank %d: all_gather_single verified.", self.rank)

    self._run_with_comm(_body)

  def test_multi_tpu_broadcast(self):
    def _body(comm: torchcomms.TorchComm) -> None:
      if self.rank == 0:
        tensor = torch.tensor(
            [123.456, 789.0], dtype=torch.float32, device="tpu"
        )
      else:
        tensor = torch.zeros(2, dtype=torch.float32, device="tpu")

      work = comm.broadcast(tensor, root=0, async_op=False)
      work.wait()
      self.assertTrue(work.is_completed())

      self.assertAlmostEqual(tensor[0].item(), 123.456, places=3)
      self.assertAlmostEqual(tensor[1].item(), 789.0, places=3)
      logging.info("Rank %d: broadcast from rank 0 verified.", self.rank)

    self._run_with_comm(_body)

  def test_multi_tpu_barrier(self):

    def _body(comm: torchcomms.TorchComm) -> None:
      work = comm.barrier(async_op=False)
      work.wait()
      self.assertTrue(work.is_completed())
      logging.info("Rank %d: barrier completed successfully.", self.rank)

    self._run_with_comm(_body)

  def test_multi_tpu_finalize(self):

    def _body(comm: torchcomms.TorchComm) -> None:
      backend_impl = comm.get_backend_impl()
      self.assertFalse(backend_impl.is_finalized())
      comm.finalize()
      self.assertTrue(backend_impl.is_finalized())
      self.assertIsNone(backend_impl.get_backend())
      logging.info("Rank %d: finalize completed successfully.", self.rank)

    self._run_with_comm(_body)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
