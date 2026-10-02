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

"""Unit tests for TorchTPU integration using TorchComm directly."""

import datetime
import os
import time
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import torchcomm_tpu
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils
import torchcomms

_WORLD_SIZE = 2


def _mock_int_env(values: dict[str, int]):
  """Mocks env.get_int_env_once() so that it reports `values`.

  The real reader memoizes in C++, so it cannot be steered by patching the
  process environment more than once per process.

  Args:
    values: The environment variables to report as set.

  Returns:
    A context manager that installs the mock.
  """
  return mock.patch.object(
      torchcomm_tpu.env,
      "get_int_env_once",
      side_effect=lambda name, default_value=None: values.get(
          name, default_value
      ),
  )


def _new_gloo_comm(name: str) -> torchcomms.TorchComm:
  """Creates a TPU TorchComm on top of the current rank's gloo default group.

  Must run inside `distributed_utils.gloo_run`. With no store, TorchCommsTPU
  adopts the initialized default process group, and reads its rank and size
  from the RANK / WORLD_SIZE variables that the launcher sets.

  Args:
    name: The communicator name.

  Returns:
    The new communicator.
  """
  torchcomm_tpu.register_torchcomms_tpu()
  comm = torchcomms.new_comm("tpu", torch.device("cpu"), name=name)
  assert comm.get_rank() == dist.get_rank()
  assert comm.get_size() == dist.get_world_size()
  return comm


def _wait(work: torchcomms.TorchWork) -> None:
  assert isinstance(work, torchcomms.TorchWork), type(work)
  work.wait()
  assert work.is_completed()


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
  test_utils.assert_close(
      actual, expected, rtol=0, atol=0, preamble=f"rank {dist.get_rank()}"
  )


def _worker_all_reduce() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_all_reduce")

  tensor = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
  _wait(comm.all_reduce(tensor, op=torchcomms.ReduceOp.SUM, async_op=False))
  total = world_size * (world_size + 1) / 2
  _assert_equal(tensor, torch.tensor([total, 10.0 * total]))

  # A non-default op shows that the torchcomms op is translated, not dropped.
  tensor = torch.tensor([rank + 1.0])
  _wait(comm.all_reduce(tensor, op=torchcomms.ReduceOp.MAX, async_op=False))
  _assert_equal(tensor, torch.tensor([float(world_size)]))
  comm.finalize()


def _worker_broadcast() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_broadcast")

  # Broadcast from the last rank so that a dropped root (defaulting to 0) shows.
  root = world_size - 1
  tensor = torch.tensor([rank + 1.0, rank + 2.0])
  _wait(comm.broadcast(tensor, root=root, async_op=False))
  _assert_equal(tensor, torch.tensor([root + 1.0, root + 2.0]))
  comm.finalize()


def _worker_all_gather() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_all_gather")
  tensor = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
  expected = [
      torch.tensor([i + 1.0, 10.0 * (i + 1)]) for i in range(world_size)
  ]

  out_list = [torch.zeros(2) for _ in range(world_size)]
  _wait(comm.all_gather(out_list, tensor, async_op=False))
  for actual, want in zip(out_list, expected, strict=True):
    _assert_equal(actual, want)

  out_list_v = [torch.zeros(2) for _ in range(world_size)]
  _wait(comm.all_gather_v(out_list_v, tensor, async_op=False))
  for actual, want in zip(out_list_v, expected, strict=True):
    _assert_equal(actual, want)
  comm.finalize()


def _worker_all_gather_single() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_all_gather_single")

  tensor = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
  out = torch.zeros(2 * world_size)
  _wait(comm.all_gather_single(out, tensor, async_op=False))
  _assert_equal(
      out,
      torch.tensor(
          [x for i in range(world_size) for x in (i + 1.0, 10.0 * (i + 1))]
      ),
  )
  comm.finalize()


def _worker_reduce_scatter() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_reduce_scatter")
  # Rank k contributes 10 * k + i to chunk i, so chunk `rank` sums to
  # 10 * sum(k) + world_size * rank.
  in_list = [torch.full((2,), 10.0 * rank + i) for i in range(world_size)]
  expected = torch.full((2,), 10.0 * sum(range(world_size)) + world_size * rank)

  out = torch.zeros(2)
  _wait(
      comm.reduce_scatter(
          out, in_list, op=torchcomms.ReduceOp.SUM, async_op=False
      )
  )
  _assert_equal(out, expected)

  out_v = torch.zeros(2)
  _wait(
      comm.reduce_scatter_v(
          out_v, in_list, op=torchcomms.ReduceOp.SUM, async_op=False
      )
  )
  _assert_equal(out_v, expected)
  comm.finalize()


def _worker_reduce_scatter_single() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_reduce_scatter_single")

  in_t = torch.arange(2.0 * world_size) + 100.0 * rank
  out = torch.zeros(2)
  _wait(
      comm.reduce_scatter_single(
          out, in_t, op=torchcomms.ReduceOp.SUM, async_op=False
      )
  )
  chunk = torch.arange(2.0 * rank, 2.0 * rank + 2)
  _assert_equal(out, world_size * chunk + 100.0 * sum(range(world_size)))
  comm.finalize()


def _worker_barrier(marker_dir: str) -> None:
  rank = dist.get_rank()
  comm = _new_gloo_comm("test_barrier")
  marker = os.path.join(marker_dir, "rank0_done")

  if rank == 0:
    # Give the other ranks time to reach the check if the barrier is a no-op.
    time.sleep(0.5)
    with open(marker, "w") as f:
      f.write("done")
  _wait(comm.barrier(async_op=False))
  assert os.path.exists(marker), f"rank {rank} passed the barrier before rank 0"
  comm.finalize()


class _P2PProcessGroup:
  """Exposes a c10d group's send/recv under ProcessGroupTpu's P2P method names.

  TorchCommsTPU drives point-to-point traffic through `experimental_send` /
  `experimental_recv`, which only ProcessGroupTpu provides. This forwards them
  to a real gloo group, so the transfer itself still happens.
  """

  def __init__(self, pg: dist.ProcessGroup):
    self._pg = pg

  def experimental_send(
      self, tensors: list[torch.Tensor], dst: int, tag: int
  ) -> dist.Work:
    return self._pg.send(tensors, dst, tag)

  def experimental_recv(
      self, tensors: list[torch.Tensor], src: int, tag: int
  ) -> dist.Work:
    return self._pg.recv(tensors, src, tag)


def _worker_send_recv() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  # Without the P2P methods, send/recv must refuse rather than silently no-op.
  gloo_comm = _new_gloo_comm("test_sr_unsupported")
  try:
    gloo_comm.get_backend_impl().send(
        torch.zeros(1), dst=(rank + 1) % world_size
    )
  except NotImplementedError:
    pass
  else:
    raise AssertionError("send on a group without P2P support did not raise")
  gloo_comm.finalize()

  comm = torchcomm_tpu.TorchCommsTPU(
      device=torch.device("cpu"),
      name="test_sr",
      group_rank=rank,
      group_size=world_size,
      pg=_P2PProcessGroup(dist.group.WORLD),
  )
  # Pass a value around the ring: send to the next rank, receive from the
  # previous one. Even ranks send first so that the blocking calls pair up.
  dst, src = (rank + 1) % world_size, (rank - 1) % world_size
  payload = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
  received = torch.zeros(2)
  if rank % 2 == 0:
    comm.send(payload, dst=dst).wait()
    comm.recv(received, src=src).wait()
  else:
    comm.recv(received, src=src).wait()
    comm.send(payload, dst=dst).wait()
  _assert_equal(received, torch.tensor([src + 1.0, 10.0 * (src + 1)]))
  comm.finalize()


def _worker_all_to_all() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_all_to_all")

  # Rank r sends 10 * r + i to rank i, so rank r receives 10 * i + r from i.
  in_list = [torch.tensor([10.0 * rank + i]) for i in range(world_size)]
  out_list = [torch.zeros(1) for _ in range(world_size)]
  _wait(comm.all_to_all(out_list, in_list, async_op=False))
  for i, actual in enumerate(out_list):
    _assert_equal(actual, torch.tensor([10.0 * i + rank]))
  comm.finalize()


def _worker_all_to_all_single() -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  comm = _new_gloo_comm("test_all_to_all_single")

  # Even splits: row i of rank r's input goes to rank i.
  in_t = torch.tensor([[10.0 * rank + i] for i in range(world_size)])
  out = torch.zeros(world_size, 1)
  _wait(comm.all_to_all_single(out, in_t, async_op=False))
  _assert_equal(
      out, torch.tensor([[10.0 * i + rank] for i in range(world_size)])
  )

  # Uneven splits: every rank sends i + 1 elements to rank i, so rank r
  # receives r + 1 elements from each peer.
  in_splits = [i + 1 for i in range(world_size)]
  in_v = torch.cat(
      [torch.full((n,), 10.0 * rank + i) for i, n in enumerate(in_splits)]
  )
  out_splits = [rank + 1] * world_size
  out_v = torch.zeros(sum(out_splits))
  _wait(
      comm.all_to_all_v_single(
          out_v,
          in_v,
          output_split_sizes=out_splits,
          input_split_sizes=in_splits,
          async_op=False,
      )
  )
  _assert_equal(
      out_v,
      torch.cat(
          [torch.full((rank + 1,), 10.0 * i + rank) for i in range(world_size)]
      ),
  )
  comm.finalize()


class TorchCommsTPUTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torchcomm_tpu.register_torchcomms_tpu()

  def test_communicator_properties(self):
    """Verifies communicator attributes, rank, size, device, and backend name."""
    comm = torchcomms.new_comm(
        "tpu",
        torch.device("cpu"),
        name="test_comm_0",
    )
    self.assertIsInstance(comm, torchcomms.TorchComm)
    self.assertEqual(comm.get_rank(), 0)
    self.assertEqual(comm.get_size(), 1)
    self.assertEqual(comm.get_device(), torch.device("cpu"))
    self.assertEqual(comm.get_name(), "test_comm_0")
    self.assertEqual(comm.get_backend(), "tpu")

    # Verify that get_backend_impl() returns the underlying TorchCommsTPU backend
    backend_impl = comm.get_backend_impl()
    self.assertIsInstance(backend_impl, torchcomm_tpu.TorchCommsTPU)
    self.assertFalse(backend_impl.is_finalized())

    comm.finalize()
    self.assertTrue(backend_impl.is_finalized())

  def test_env_var_configuration(self):
    """Verifies rank and size initialization from TORCHCOMM_* environment variables."""
    with _mock_int_env({"TORCHCOMM_RANK": 3, "TORCHCOMM_SIZE": 16}):
      comm = torchcomms.new_comm(
          "tpu",
          torch.device("cpu"),
          name="test_env_comm",
      )
      self.assertEqual(comm.get_rank(), 3)
      self.assertEqual(comm.get_size(), 16)
      comm.finalize()

  def test_rank_size_fallback_to_rank_world_size(self):
    """Verifies fallback to standard RANK and WORLD_SIZE environment variables."""
    with _mock_int_env({"RANK": 1, "WORLD_SIZE": 4}):
      comm = torchcomms.new_comm(
          "tpu",
          torch.device("cpu"),
          name="test_fallback_comm",
      )
      self.assertEqual(comm.get_rank(), 1)
      self.assertEqual(comm.get_size(), 4)
      comm.finalize()

  def test_finalize_lifecycle(self):
    """Verifies communicator finalization lifecycle, idempotency, and post-finalize guard."""
    comm = torchcomms.new_comm(
        "tpu",
        torch.device("cpu"),
        name="test_fin_comm",
    )
    backend_impl = comm.get_backend_impl()
    self.assertFalse(backend_impl.is_finalized())

    comm.finalize()
    self.assertTrue(backend_impl.is_finalized())
    self.assertIsNone(backend_impl.get_backend())

    # Finalize should be idempotent
    comm.finalize()
    self.assertTrue(backend_impl.is_finalized())

    # Collectives should raise error after communicator is finalized
    dummy_tensor = torch.tensor([1.0])
    with self.assertRaisesRegex(RuntimeError, "finalized"):
      comm.all_reduce(dummy_tensor, op=torchcomms.ReduceOp.SUM, async_op=False)

  def test_split_subgroup(self):
    """Verifies splitting a communicator into a subgroup using TorchComm.split."""
    with _mock_int_env({"TORCHCOMM_RANK": 0, "TORCHCOMM_SIZE": 4}):
      comm = torchcomms.new_comm(
          "tpu",
          torch.device("cpu"),
          name="parent_mesh",
      )
      # Split communicator with subset of ranks
      sub_comm = comm.split([0, 2], name="sub_mesh")
      self.assertIsInstance(sub_comm, torchcomms.TorchComm)
      self.assertEqual(sub_comm.get_rank(), 0)
      self.assertEqual(sub_comm.get_size(), 2)
      self.assertEqual(sub_comm.get_name(), "sub_mesh")
      self.assertEqual(sub_comm.get_backend(), "tpu")

      # A rank not present in the split list receives None
      excluded_comm = comm.split([1, 3], name="other_mesh")
      self.assertIsNone(excluded_comm)

      comm.finalize()
      sub_comm.finalize()

  def test_registration_with_real_torchcomms(self):
    """Verifies that TorchTPU backend registration is correctly reflected in torchcomms."""
    self.assertTrue(torchcomm_tpu.is_torchcomms_registered("tpu"))
    self.assertTrue(torchcomm_tpu.is_torchcomms_registered("tpu_dist"))
    self.assertTrue(torchcomms._comms._is_backend_registered("tpu"))
    self.assertTrue(torchcomms._comms._is_backend_registered("tpu_dist"))

  def test_torchcomms_reduce_op_conversion(self):
    """Verifies bidirectional conversion between torchcomms.ReduceOp and dist.ReduceOp."""
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.SUM),
        dist.ReduceOp.SUM,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.PRODUCT),
        dist.ReduceOp.PRODUCT,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.MIN),
        dist.ReduceOp.MIN,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.MAX),
        dist.ReduceOp.MAX,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.BAND),
        dist.ReduceOp.BAND,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.BOR),
        dist.ReduceOp.BOR,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.BXOR),
        dist.ReduceOp.BXOR,
    )
    self.assertEqual(
        torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.AVG),
        dist.ReduceOp.AVG,
    )

    # Unsupported / unmapped reduction type raises ValueError
    with self.assertRaises(ValueError):
      torchcomm_tpu._to_dist_reduce_op(torchcomms.ReduceOp.PREMUL_SUM(2.0))

  def test_torchcomms_set_timeout(self):
    """Verifies setting operation timeout on the communicator."""
    comm = torchcomms.new_comm(
        "tpu", torch.device("cpu"), name="test_set_timeout"
    )
    comm.set_timeout(datetime.timedelta(milliseconds=4500))

    backend_impl = comm.get_backend_impl()
    self.assertEqual(
        backend_impl._timeout, datetime.timedelta(milliseconds=4500)
    )
    comm.finalize()

  def test_c10d_backend_availability(self):
    """Verifies backend availability registration with PyTorch c10d."""
    torchcomm_tpu.register_torchcomms_tpu()
    self.assertTrue(dist.is_backend_available("tpu_dist"))
    self.assertTrue(dist.is_backend_available("tpu"))
    self.assertEqual(dist.Backend.default_device_backend_map.get("tpu"), "tpu")

  def test_unsupported_stubs_raise_not_implemented(self):
    """Verifies that unsupported collective or communicator features raise NotImplementedError."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_stubs")
    backend = comm.get_backend_impl()
    t = torch.zeros(4, device="cpu")

    self.assertEqual(backend.backend_name, "tpu")
    self.assertEqual(backend.get_backend_name(), "tpu")
    self.assertIn(backend.device_type, ("cpu", "tpu"))
    self.assertEqual(backend.hints, {})
    self.assertIsNone(backend.store)
    self.assertIsNotNone(backend.timeout)

    with self.assertRaises(NotImplementedError):
      backend.create_pair_comm(1)

    with self.assertRaises(NotImplementedError):
      backend.split_group([0, 1])

    with self.assertRaises(NotImplementedError):
      backend.reconfigure(ranks=[0, 1], size=2)

    with self.assertRaises(NotImplementedError):
      backend.create_window(t)

    with self.assertRaises(NotImplementedError):
      backend.map_remote_tensor(t, peer_rank=1)

    with self.assertRaises(NotImplementedError):
      backend.create_batch()

    with self.assertRaises(NotImplementedError):
      backend.all_reduce_coalesced([t])

    with self.assertRaises(NotImplementedError):
      backend.all_gather_coalesced([[t]], [t])

    with self.assertRaises(NotImplementedError):
      backend.reduce_scatter_coalesced([t], [[t]])

    with self.assertRaises(NotImplementedError):
      backend.all_to_all_coalesced([[t]], [[t]])

    with self.assertRaises(NotImplementedError):
      backend.reduce(t, dst=0)

    with self.assertRaises(NotImplementedError):
      backend.gather(input_tensor=t, dst=0)

    with self.assertRaises(NotImplementedError):
      backend.scatter(output_tensor=t, src=0)

    with self.assertRaises(NotImplementedError):
      backend.monitored_barrier()

    with self.assertRaises(NotImplementedError):
      backend.register_flight_recorder_hook(lambda *_: None)

    with self.assertRaises(NotImplementedError):
      backend.get_flight_recorder_traces()

    with self.assertRaises(NotImplementedError):
      backend.broadcast_object_list([1, 2, 3])

    with self.assertRaises(NotImplementedError):
      backend.all_gather_object([], {"a": 1})

    with self.assertRaises(NotImplementedError):
      backend.gather_object({"a": 1})

    with self.assertRaises(NotImplementedError):
      backend.scatter_object_list([])

    comm.finalize()


class TorchCommsTPUCollectivesTest(seed_test_utils.MultiProcessRepeatableTest):
  """Runs each collective across real gloo ranks and checks the results."""

  def test_all_reduce(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_all_reduce)

  def test_broadcast(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_broadcast)

  def test_all_gather_and_all_gather_v(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_all_gather)

  def test_all_gather_single(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_all_gather_single)

  def test_reduce_scatter_and_reduce_scatter_v(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_reduce_scatter)

  def test_reduce_scatter_single(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_reduce_scatter_single)

  def test_barrier(self):
    distributed_utils.gloo_run(
        _WORLD_SIZE, _worker_barrier, self.create_tempdir().full_path
    )

  def test_send_recv(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_send_recv)

  def test_all_to_all(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_all_to_all)

  def test_all_to_all_single_and_v(self):
    distributed_utils.gloo_run(_WORLD_SIZE, _worker_all_to_all_single)


# Alias for backward compatibility
TorchCommTPUTest = TorchCommsTPUTest

if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
