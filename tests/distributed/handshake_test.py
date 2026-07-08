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

"""Tests for distributed handshake protocol."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import handshake
from torch_tpu._internal.distributed.handshake import Handshake
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def _run_handshake_value_consensus(coordinator_rank: int) -> None:
  """Worker function to test value consensus for various data types and match scenarios."""
  os.environ.setdefault(
      "TORCH_TPU_INTERNAL_HANDSHAKE_PORT",
      os.environ.get("MASTER_PORT", "29500"),
  )
  rank = int(os.environ["RANK"])
  hs = Handshake(coordinator_rank=coordinator_rank)

  # 1. All ranks submit True -> True
  assert hs.submit("all_true", True).wait()

  # 2. Ranks submit different booleans -> False
  assert not hs.submit("some_false", rank != 0).wait()

  # 3. All ranks submit False -> True (checking identical non-True match)
  assert hs.submit("all_false", False).wait()

  # 4. Ranks submit different non-boolean integers -> False
  assert not hs.submit("diff_vals", 42 if rank == 0 else 100).wait()

  # 5. All ranks submit identical complex dictionary -> True
  complex_val = {"layers": [1, 2, 3], "name": "transformer", "active": True}
  assert hs.submit("same_complex", complex_val).wait()


def _run_handshake_multiple_keys_and_instances(coordinator_rank: int) -> None:
  """Worker function to test multiple keys and multiple Handshake instances."""
  os.environ.setdefault(
      "TORCH_TPU_INTERNAL_HANDSHAKE_PORT",
      os.environ.get("MASTER_PORT", "29500"),
  )
  rank = int(os.environ["RANK"])
  hs1 = Handshake(coordinator_rank=coordinator_rank)
  hs2 = Handshake(coordinator_rank=coordinator_rank)

  # Multiple keys on a single instance
  assert hs1.submit("key_1", True).wait()
  assert not hs1.submit("key_2", rank % 2 == 0).wait()

  # Multiple instances sharing the same TCPStore and coordinator
  assert hs1.submit("instance_key_a", "val_a").wait()
  assert not hs2.submit("instance_key_b", rank % 2 == 0).wait()


def _run_handshake_cleanup(coordinator_rank: int) -> None:
  """Worker function to test that handshake keys are cleaned up after resolution."""
  os.environ.setdefault(
      "TORCH_TPU_INTERNAL_HANDSHAKE_PORT",
      os.environ.get("MASTER_PORT", "29500"),
  )
  hs = Handshake(coordinator_rank=coordinator_rank)
  fut = hs.submit("test_cleanup", True)
  res = fut.wait()
  rank = hs._current_rank
  assert res, f"Rank {rank} expected True, got {res}"

  if rank == coordinator_rank:
    for r in hs._participating_ranks:
      assert not hs._backend.check(
          [handshake._format_rank_topic_key("test_cleanup", r)]
      ), f"Rank {rank}: key 'test_cleanup/rank_{r}' was not cleaned up."
      assert not hs._backend.check(
          [handshake._format_ack_topic_key("test_cleanup", r)]
      ), f"Rank {rank}: key 'test_cleanup/ack_{r}' was not cleaned up."
    assert not hs._backend.check(
        [handshake._format_result_topic_key("test_cleanup")]
    ), f"Rank {rank}: key 'test_cleanup/result' was not cleaned up."


def _run_select_handshake_port_pg_broadcast(coordinator_rank: int) -> None:
  """Verifies that non-coordinator ranks receive the broadcasted port."""
  dist.init_process_group(
      backend="gloo",
      init_method="env://",
      rank=int(os.environ["RANK"]),
      world_size=int(os.environ["WORLD_SIZE"]),
  )
  try:
    rank = int(os.environ["RANK"])
    # Do not set TORCH_TPU_INTERNAL_HANDSHAKE_PORT so it falls back to process
    # group exchange
    port = handshake._select_handshake_port(rank, coordinator_rank)

    # Gather the port seen by each rank to explicitly verify that all worker
    # ranks received the exact same port number chosen by the coordinator:
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
      master_port = int(os.environ.get("MASTER_PORT", "29500"))
      assert (
          gathered_ports[0] != master_port
      ), f"Handshake port {gathered_ports[0]} must not collide with MASTER_PORT"
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


def _run_handshake_with_initialized_process_group(
    coordinator_rank: int,
) -> None:
  """Worker function to test Handshake port exchange over process group."""
  dist.init_process_group(
      backend="gloo",
      init_method="env://",
      rank=int(os.environ["RANK"]),
      world_size=int(os.environ["WORLD_SIZE"]),
  )
  try:
    # Do not set TORCH_TPU_INTERNAL_HANDSHAKE_PORT so it falls back to process
    # group exchange
    hs = Handshake(coordinator_rank=coordinator_rank)
    fut = hs.submit("test_pg_port_exchange", True)
    res = fut.wait()
    rank = hs._current_rank
    assert res, f"Rank {rank} expected True, got {res}"
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


def _run_handshake_multiple_instances(coordinator_rank: int) -> None:
  """Worker function to test multiple Handshake instances sharing class store."""
  os.environ.setdefault(
      "TORCH_TPU_INTERNAL_HANDSHAKE_PORT",
      os.environ.get("MASTER_PORT", "29500"),
  )
  hs1 = Handshake(coordinator_rank=coordinator_rank)
  hs2 = Handshake(coordinator_rank=coordinator_rank)
  rank = hs1._current_rank
  # Because they share a single class-level store, submitting sequentially on
  # hs1 and hs2 works seamlessly over the shared TCPStore:
  fut1 = hs1.submit("instance_key_1", True)
  assert fut1.wait()
  fut2 = hs2.submit("instance_key_2", rank % 2 == 0)
  assert not fut2.wait()


def _run_handshake_subset_ranks(coordinator_rank: int) -> None:
  """Worker function to test Handshake running on a subset of ranks."""
  os.environ.setdefault(
      "TORCH_TPU_INTERNAL_HANDSHAKE_PORT",
      os.environ.get("MASTER_PORT", "29500"),
  )
  rank = int(os.environ["RANK"])
  participating_ranks = [0, 2]

  if rank in participating_ranks:
    hs = Handshake(
        coordinator_rank=coordinator_rank,
        participating_ranks=participating_ranks,
    )
    fut = hs.submit("test_subset", "same_val")
    res = fut.wait()
    assert res, f"Rank {rank} expected True, got {res}"
  else:
    hs = Handshake(
        coordinator_rank=coordinator_rank,
        participating_ranks=participating_ranks,
    )
    with absltest.TestCase().assertRaisesRegex(
        ValueError, "not in participating ranks"
    ):
      hs.submit("test_subset", "should_fail")


def _run_handshake_multiple_coordinators(
    coordinator_rank: int, use_pg: bool
) -> None:
  """Worker function to test multiple Handshake instances with different coordinators."""
  del coordinator_rank  # Unused.
  if use_pg:
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
  try:
    rank = int(os.environ["RANK"])
    # Note: In production across multiple machines, every coordinator resides
    # on a distinct host, allowing all groups to share the exact same handshake
    # port. In single-host `mp.spawn` tests on localhost, PyTorch `TCPStore`
    # binds `0.0.0.0`, requiring distinct ephemeral ports for local coordinators
    # to prevent `EADDRINUSE`.
    if not use_pg:
      os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "31234"
    hs0 = Handshake(coordinator_rank=0)
    assert hs0.submit(
        "coord0_key", "val0"
    ).wait(), f"Rank {rank} failed coord 0"

    if not use_pg:
      os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "31235"
    hs1 = Handshake(coordinator_rank=1)
    assert hs1.submit(
        "coord1_key", "val1"
    ).wait(), f"Rank {rank} failed coord 1"

    if not use_pg:
      os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "31236"
    hs2 = Handshake(coordinator_rank=2)
    assert hs2.submit(
        "coord2_key", "val2"
    ).wait(), f"Rank {rank} failed coord 2"
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


def _run_handshake_disjoint_groups(coordinator_rank: int) -> None:
  """Worker function to test handshakes between disjoint participant groups without process group."""
  del coordinator_rank  # Unused.
  rank = int(os.environ["RANK"])

  # In single-host tests, Group A and Group B run concurrently on localhost.
  # We use distinct local ports so their `TCPStore` servers do not collide.
  if rank in [0, 1]:
    os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "31240"
    hs_a = Handshake(coordinator_rank=0, participating_ranks=[0, 1])
    fut_a = hs_a.submit("group_a_key", "val_a")
    assert fut_a.wait(), f"Rank {rank} failed Group A handshake"
  elif rank in [2, 3]:
    os.environ["TORCH_TPU_INTERNAL_HANDSHAKE_PORT"] = "31241"
    hs_b = Handshake(coordinator_rank=2, participating_ranks=[2, 3])
    fut_b = hs_b.submit("group_b_key", "val_b")
    assert fut_b.wait(), f"Rank {rank} failed Group B handshake"


class HandshakeTest(parameterized.TestCase):
  """Unit tests for distributed Handshake protocol on CPU."""

  def test_select_handshake_port_env_var_priority(self) -> None:
    """Tests that TORCH_TPU_INTERNAL_HANDSHAKE_PORT is the first choice for port selection."""
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_HANDSHAKE_PORT": "31234"}, clear=True
    ):
      port = handshake._select_handshake_port(rank=0, coordinator_rank=0)
      self.assertEqual(port, 31234)

  def test_select_handshake_port_exception(self) -> None:
    """Tests that RuntimeError is raised when no port can be selected."""
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertRaisesRegex(RuntimeError, "cannot select Handshake port"):
        handshake._select_handshake_port(rank=0, coordinator_rank=0)

  def test_select_handshake_port_pg_broadcast(self) -> None:
    """Tests that worker ranks receive the broadcasted port from coordinator."""
    distributed_utils.dist_run(3, _run_select_handshake_port_pg_broadcast, 0)

  def test_custom_backend(self) -> None:
    """Tests Handshake protocol with a custom mock backend."""
    storage = {}
    backend = mock.create_autospec(handshake.HandshakeBackend, instance=True)
    backend.check.side_effect = lambda keys: all(k in storage for k in keys)

    def _set(k, v):
      storage[k] = v.encode("utf-8") if isinstance(v, str) else v

    def _delete(k):
      if k in storage:
        del storage[k]
        return True
      return False

    backend.set.side_effect = _set
    backend.get.side_effect = lambda k: storage.get(k, b"0")
    backend.wait.return_value = None
    backend.delete.side_effect = _delete

    with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
      hs = Handshake(backend=backend)
      fut = hs.submit("custom_key", True)
      self.assertTrue(fut.wait())
      self.assertEqual(storage, {})

  @parameterized.parameters(0, 1)
  def test_handshake_value_consensus(self, coordinator_rank: int) -> None:
    distributed_utils.dist_run(
        4, _run_handshake_value_consensus, coordinator_rank
    )

  def test_handshake_multiple_keys_and_instances(self) -> None:
    distributed_utils.dist_run(3, _run_handshake_multiple_keys_and_instances, 0)

  def test_handshake_cleanup(self) -> None:
    distributed_utils.dist_run(3, _run_handshake_cleanup, 0)

  def test_duplicate_submission(self) -> None:
    """Tests that submitting twice for the same key raises ValueError."""
    backend = mock.create_autospec(handshake.HandshakeBackend, instance=True)
    submitted = set()

    def _check(keys):
      return all(k in submitted for k in keys)

    def _set(k, v):
      del v  # Unused.
      submitted.add(k)

    backend.check.side_effect = _check
    backend.set.side_effect = _set
    backend.wait.return_value = None
    backend.delete.return_value = True

    with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
      hs = Handshake(backend=backend)
      hs.submit("dup_test_key", True)
      with self.assertRaisesRegex(
          ValueError, "already submitted value for key"
      ):
        hs.submit("dup_test_key", True)

  def test_exception_propagation_through_future(self) -> None:
    """Tests that exceptions in the background thread propagate to the future."""
    backend = mock.create_autospec(handshake.HandshakeBackend, instance=True)
    backend.check.return_value = False
    backend.set.return_value = None
    backend.wait.side_effect = RuntimeError("backend connection lost")

    with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
      hs = Handshake(backend=backend)
      fut = hs.submit("error_key", True)
      with self.assertRaisesRegex(RuntimeError, "backend connection lost"):
        fut.wait()

  def test_handshake_with_initialized_process_group(self) -> None:
    distributed_utils.dist_run(
        3, _run_handshake_with_initialized_process_group, 0
    )

  def test_handshake_subset_ranks(self) -> None:
    distributed_utils.dist_run(4, _run_handshake_subset_ranks, 0)

  @parameterized.parameters(True, False)
  def test_handshake_multiple_coordinators(self, use_pg: bool) -> None:
    distributed_utils.dist_run(
        3, _run_handshake_multiple_coordinators, 0, use_pg=use_pg
    )

  def test_handshake_disjoint_groups(self) -> None:
    distributed_utils.dist_run(4, _run_handshake_disjoint_groups, 0)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
