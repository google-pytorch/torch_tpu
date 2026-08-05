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

"""Tests for distributed handshake protocol using PyZMQ and asyncio."""

import os
import time
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import handshake
from torch_tpu._internal.distributed.handshake import _HandshakeClient
from torch_tpu._internal.distributed.handshake import _HandshakeServer
from torch_tpu._internal.distributed.handshake import Handshake
from torch_tpu._internal.distributed.handshake import HandshakeMessage
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def test_wrapper(
    target_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
  """Wrapper for distributed test worker functions.

  Initializes the PyTorch process group before invoking target_fn and tears it
  down afterwards.
  """
  dist.init_process_group(
      backend="gloo",
      init_method="env://",
      rank=int(os.environ.get("RANK", "0")),
      world_size=int(os.environ.get("WORLD_SIZE", "1")),
  )
  try:
    return target_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


def _run_handshake_message_matching(coordinator_rank: int) -> None:
  """Worker function to test matching HandshakeMessage across ranks."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)
  msg = HandshakeMessage(
      executable_fingerprint="matching_fingerprint_123",
      is_last_frame_in_compiled_region=True,
      participating_ranks=list(range(world_size)),
      rank=rank,
  )
  res = hs.submit(msg)
  assert res is True, f"Rank {rank} expected True, got {res}"


def _run_handshake_message_mismatching(coordinator_rank: int) -> None:
  """Worker function to test mismatching HandshakeMessage fingerprints across ranks."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)
  # Rank 0 has a different fingerprint than other ranks
  fingerprint = "fp_coordinator" if rank == 0 else f"fp_worker_{rank}"
  msg = HandshakeMessage(
      executable_fingerprint=fingerprint,
      is_last_frame_in_compiled_region=True,
      participating_ranks=list(range(world_size)),
      rank=rank,
  )
  res = hs.submit(msg)
  assert res is False, f"Rank {rank} expected False for mismatch, got {res}"


def _run_multiple_consecutive_submissions(coordinator_rank: int) -> None:
  """Worker function to test multiple consecutive step submissions."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)

  for step in range(5):
    # Step where all ranks match
    msg_match = HandshakeMessage(
        executable_fingerprint=f"step_{step}_match",
        is_last_frame_in_compiled_region=(step == 4),
        participating_ranks=list(range(world_size)),
        rank=rank,
    )
    assert hs.submit(msg_match) is True

    # Step where ranks mismatch
    msg_mismatch = HandshakeMessage(
        executable_fingerprint=f"step_{step}_rank_{rank}",
        is_last_frame_in_compiled_region=(step == 4),
        participating_ranks=list(range(world_size)),
        rank=rank,
    )
    assert hs.submit(msg_mismatch) is False


def _run_select_handshake_port_pg_broadcast(coordinator_rank: int) -> None:
  """Verifies that non-coordinator ranks receive the broadcasted port."""
  rank = int(os.environ["RANK"])

  port = handshake._select_handshake_port(
      current_rank=rank, coordinator_rank=coordinator_rank
  )

  world_size = dist.get_world_size()
  gathered_ports = [None for _ in range(world_size)] if rank == 0 else None
  dist.gather_object(port, gathered_ports, dst=0)

  if rank == 0:
    assert gathered_ports is not None
    unique_ports = set(gathered_ports)
    assert (
        len(unique_ports) == 1
    ), f"All ranks must see identical port, got {gathered_ports}"
    assert (
        gathered_ports[0] > 0
    ), f"Port must be valid positive int, got {gathered_ports[0]}"


def _run_handshake_with_initialized_process_group(
    coordinator_rank: int,
) -> None:
  """Worker function to test Handshake port exchange over process group."""
  Handshake._reset_instances()
  hs = Handshake(coordinator_rank=coordinator_rank)
  msg = HandshakeMessage(
      executable_fingerprint="pg_fingerprint_test",
      is_last_frame_in_compiled_region=True,
      participating_ranks=list(range(int(os.environ["WORLD_SIZE"]))),
      rank=int(os.environ["RANK"]),
  )
  res = hs.submit(msg)
  assert res is True, f"Rank {os.environ['RANK']} expected True, got {res}"


def _run_subset_handshake_without_coordinator(coordinator_rank: int) -> None:
  """Worker function to test subset handshake when coordinator is not participating."""
  rank = int(os.environ["RANK"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=coordinator_rank)
  # Ranks 1, 2, 3 participate, coordinator rank 0 does not call submit()
  if rank != coordinator_rank:
    msg = HandshakeMessage(
        executable_fingerprint="subset_fingerprint_456",
        is_last_frame_in_compiled_region=True,
        participating_ranks=[1, 2, 3],
        rank=rank,
    )
    res = hs.submit(msg)
    assert (
        res is True
    ), f"Rank {rank} expected True for subset handshake, got {res}"
  else:
    time.sleep(2)


def _run_mismatched_frame_count_deadlock() -> None:
  """Worker function to test mismatched frame count between rank 0 and ranks 1-7."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  Handshake._reset_instances()

  hs = Handshake(coordinator_rank=0)
  participating_ranks = list(range(world_size))

  if rank == 0:
    # Rank 0 sends 2 handshakes:
    # 1st handshake: fingerprint_A, is_last_frame_in_compiled_region=False
    msg1 = HandshakeMessage(
        executable_fingerprint="fingerprint_A",
        is_last_frame_in_compiled_region=False,
        participating_ranks=participating_ranks,
        rank=0,
    )
    res1 = hs.submit(msg1)
    assert res1 is False, f"Rank 0 expected False for 1st submit, got {res1}"
    # 2nd handshake: fingerprint_B, is_last_frame_in_compiled_region=True
    msg2 = HandshakeMessage(
        executable_fingerprint="fingerprint_B",
        is_last_frame_in_compiled_region=True,
        participating_ranks=participating_ranks,
        rank=0,
    )
    res2 = hs.submit(msg2)
    assert res2 is False, f"Rank 0 expected False for 2nd submit, got {res2}"
  else:
    # Ranks 1-7 send 1 handshake: fingerprint_C, is_last_frame_in_compiled_region=True
    msg = HandshakeMessage(
        executable_fingerprint="fingerprint_C",
        is_last_frame_in_compiled_region=True,
        participating_ranks=participating_ranks,
        rank=rank,
    )
    res = hs.submit(msg)
    assert res is False, f"Rank {rank} expected False, got {res}"


class HandshakeTest(parameterized.TestCase):
  """Unit tests for distributed Handshake protocol on CPU."""

  def setUp(self) -> None:
    super().setUp()
    Handshake._reset_instances()

  def tearDown(self) -> None:
    super().tearDown()
    Handshake._reset_instances()

  def test_handshake_server_singleton(self) -> None:
    """Tests that _HandshakeServer acts as a singleton per process."""
    server1 = _HandshakeServer(current_rank=0, port=31234, world_size=1)
    server2 = _HandshakeServer(current_rank=0, port=31234, world_size=1)
    self.assertIs(server1, server2)

  def test_handshake_client_per_port(self) -> None:
    """Tests that _HandshakeClient maintains singletons per port."""
    client1 = _HandshakeClient(port=31234)
    client2 = _HandshakeClient(port=31234)
    client3 = _HandshakeClient(port=31235)
    self.assertIs(client1, client2)
    self.assertIsNot(client1, client3)

  def test_handshake_port_cache(self) -> None:
    """Tests that Handshake caches port selection per coordinator rank."""
    with mock.patch.object(
        handshake, "_select_handshake_port", return_value=31234
    ):
      _ = Handshake(current_rank=0, coordinator_rank=0)
      self.assertEqual(Handshake._port_cache[0], 31234)

  def test_select_handshake_port_exception(self) -> None:
    """Tests that RuntimeError is raised when process group is not initialized."""
    with self.assertRaisesRegex(RuntimeError, "cannot select Handshake port"):
      handshake._select_handshake_port(current_rank=0, coordinator_rank=0)

  def test_submit_invalid_message_type(self) -> None:
    """Tests that submitting non-HandshakeMessage raises TypeError."""
    with mock.patch.object(
        handshake, "_select_handshake_port", return_value=31234
    ):
      hs = Handshake(current_rank=0, coordinator_rank=0)
      with self.assertRaises(TypeError):
        hs.submit("invalid_string")  # type: ignore

  def test_select_handshake_port_pg_broadcast(self) -> None:
    """Tests that worker ranks receive the broadcasted port from coordinator."""
    distributed_utils.dist_run(
        3, test_wrapper, _run_select_handshake_port_pg_broadcast, 0
    )

  @parameterized.parameters(0, 1)
  def test_handshake_message_matching(self, coordinator_rank: int) -> None:
    distributed_utils.dist_run(
        4, test_wrapper, _run_handshake_message_matching, coordinator_rank
    )

  @parameterized.parameters(0, 1)
  def test_handshake_message_mismatching(self, coordinator_rank: int) -> None:
    distributed_utils.dist_run(
        4, test_wrapper, _run_handshake_message_mismatching, coordinator_rank
    )

  def test_multiple_consecutive_submissions(self) -> None:
    distributed_utils.dist_run(
        3, test_wrapper, _run_multiple_consecutive_submissions, 0
    )

  def test_handshake_with_initialized_process_group(self) -> None:
    distributed_utils.dist_run(
        3, test_wrapper, _run_handshake_with_initialized_process_group, 0
    )

  def test_subset_handshake_without_coordinator(self) -> None:
    distributed_utils.dist_run(
        4, test_wrapper, _run_subset_handshake_without_coordinator, 0
    )

  def test_mismatched_frame_count_deadlock(self) -> None:
    """Tests that mismatched frame count across ranks is exhausted cleanly without deadlocking."""
    distributed_utils.dist_run(
        8, test_wrapper, _run_mismatched_frame_count_deadlock
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
