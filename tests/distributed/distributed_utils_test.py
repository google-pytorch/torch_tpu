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

"""Tests for distributed_utils module."""

import os
import random
import socket
from unittest import mock

from absl.testing import absltest
import portpicker
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


def _worker_seed_check(
    expected_base_seed: int,
    output_dir: str,
) -> None:
  rank = int(os.environ["RANK"])
  expected_seed = expected_base_seed + rank

  # Check python random
  py_random_val = random.random()
  expected_py_val = random.Random(expected_seed).random()

  # Check torch random
  torch_val = torch.rand(4)
  torch.manual_seed(expected_seed)
  expected_torch_val = torch.rand(4)

  assert (
      py_random_val == expected_py_val
  ), f"Rank {rank}: expected py rand {expected_py_val}, got {py_random_val}"
  assert torch.equal(
      torch_val, expected_torch_val
  ), f"Rank {rank}: expected torch rand {expected_torch_val}, got {torch_val}"

  rank_file = os.path.join(output_dir, f"rank_{rank}.txt")
  with open(rank_file, "w") as f:
    f.write(f"{py_random_val},{torch_val.tolist()}\n")


def _worker_xla_flags_check(output_dir: str) -> None:
  rank = os.environ["RANK"]
  xla_flags = os.environ.get("XLA_FLAGS", "")
  rank_file = os.path.join(output_dir, f"rank_{rank}_xla_flags.txt")
  with open(rank_file, "w") as f:
    f.write(xla_flags)


def _worker_env_check() -> None:
  local_rank = int(os.environ["LOCAL_RANK"])
  node_rank = int(os.environ["GROUP_RANK"])
  nproc = int(os.environ["LOCAL_WORLD_SIZE"])
  world_size = int(os.environ["WORLD_SIZE"])
  rank = int(os.environ["RANK"])

  assert rank == node_rank * nproc + local_rank
  assert world_size == 2
  assert nproc == 2


def _worker_init_process_group() -> None:
  torch.distributed.init_process_group(
      backend="gloo",
      init_method="env://",
  )
  torch.distributed.destroy_process_group()


def _worker_gloo_all_reduce(world_size: int) -> None:
  assert torch.distributed.is_initialized()
  assert torch.distributed.get_backend() == "gloo"
  assert torch.distributed.get_world_size() == world_size
  rank = torch.distributed.get_rank()
  tensor = torch.tensor([float(rank + 1)])
  torch.distributed.all_reduce(tensor)
  expected = float(world_size * (world_size + 1) // 2)
  assert tensor.item() == expected, f"rank {rank}: got {tensor.item()}"


def _worker_fail_on_rank_one() -> None:
  if torch.distributed.get_rank() == 1:
    raise AssertionError("rank 1 failed on purpose")


class DistributedUtilsTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_automatic_per_rank_seeding(self) -> None:
    """Verifies default per-rank RNG seeding based on RepeatableTest seed."""
    output_dir = self.create_tempdir().full_path
    base_seed = seed_test_utils.RepeatableTest.choose_seed()

    distributed_utils.dist_run(
        2,
        _worker_seed_check,
        base_seed,
        output_dir,
    )

    rank0_file = os.path.join(output_dir, "rank_0.txt")
    rank1_file = os.path.join(output_dir, "rank_1.txt")
    self.assertTrue(os.path.exists(rank0_file))
    self.assertTrue(os.path.exists(rank1_file))

    with open(rank0_file) as f:
      rank0_content = f.read().strip()
    with open(rank1_file) as f:
      rank1_content = f.read().strip()

    # Rank 0 and Rank 1 must have different random streams (rank diversity)
    self.assertNotEqual(rank0_content, rank1_content)

  def test_explicit_base_seed(self) -> None:
    """Verifies that an explicitly passed base_seed is used."""
    output_dir = self.create_tempdir().full_path
    custom_seed = 9876

    distributed_utils.dist_run(
        2,
        _worker_seed_check,
        custom_seed,
        output_dir,
        base_seed=custom_seed,
    )

    rank0_file = os.path.join(output_dir, "rank_0.txt")
    rank1_file = os.path.join(output_dir, "rank_1.txt")
    self.assertTrue(os.path.exists(rank0_file))
    self.assertTrue(os.path.exists(rank1_file))

    with open(rank0_file) as f:
      rank0_content = f.read().strip()
    with open(rank1_file) as f:
      rank1_content = f.read().strip()

    self.assertNotEqual(rank0_content, rank1_content)

  def test_env_variables_set(self) -> None:
    """Verifies distributed environment variables set in worker."""
    distributed_utils.dist_run(
        2,
        _worker_env_check,
    )

  def test_xla_flags_expansion(self) -> None:
    """Verifies that $RANK / ${RANK} in XLA_FLAGS is expanded per worker."""
    output_dir = self.create_tempdir().full_path
    template_flag = "--xla_dump_to=/tmp/dumps/rank_${RANK} --xla_tpu_rank=$RANK"
    with mock.patch.dict(os.environ, {"XLA_FLAGS": template_flag}):
      distributed_utils.dist_run(
          2,
          _worker_xla_flags_check,
          output_dir,
      )

    rank0_file = os.path.join(output_dir, "rank_0_xla_flags.txt")
    rank1_file = os.path.join(output_dir, "rank_1_xla_flags.txt")
    self.assertTrue(os.path.exists(rank0_file))
    self.assertTrue(os.path.exists(rank1_file))

    with open(rank0_file) as f:
      self.assertEqual(
          f.read().strip(),
          "--xla_dump_to=/tmp/dumps/rank_0 --xla_tpu_rank=0",
      )
    with open(rank1_file) as f:
      self.assertEqual(
          f.read().strip(),
          "--xla_dump_to=/tmp/dumps/rank_1 --xla_tpu_rank=1",
      )

  def test_gloo_run_provides_working_process_group(self) -> None:
    """Verifies that gloo_run ranks share an initialized gloo group."""
    distributed_utils.gloo_run(2, _worker_gloo_all_reduce, 2)

  def test_gloo_run_propagates_rank_failure(self) -> None:
    """Verifies that an assertion failing on one rank fails the caller."""
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests the test fixture
        mp.ProcessRaisedException, "rank 1 failed on purpose"
    ):
      distributed_utils.gloo_run(2, _worker_fail_on_rank_one)

  def test_dist_run_retries_and_succeeds_with_real_socket_collision(
      self,
  ) -> None:
    """Verifies that dist_run handles a real OS port collision (EADDRINUSE).

    We simulate a race condition by holding open a listening TCP socket
    on the port returned by portpicker on the first attempt.
    dist_run will fail on attempt 1 with EADDRINUSE from
    gloo/init_process_group,
    catch the ProcessRaisedException, request a new port from portpicker, and
    succeed on attempt 2.
    """
    free_port = portpicker.pick_unused_port()

    # Open and hold a real listening socket to guarantee EADDRINUSE on attempt 1.
    busy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy_sock.bind(("localhost", 0))
    busy_sock.listen(1)
    busy_port = busy_sock.getsockname()[1]

    try:
      with mock.patch.object(
          portpicker,
          "pick_unused_port",
          side_effect=[busy_port, free_port],
      ):
        distributed_utils.dist_run(
            2,
            _worker_init_process_group,
            max_retries=3,
        )
    finally:
      busy_sock.close()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
