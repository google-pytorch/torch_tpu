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
from unittest import mock
from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch_tpu import _loader
from torch_tpu._internal.distributed import torchcomm_tpu
from tests import seed_test_utils
import torchcomms

_loader._init_device("tpu")


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
    env_patch = {
        "TORCHCOMM_RANK": "3",
        "TORCHCOMM_SIZE": "16",
    }
    with mock.patch.dict(os.environ, env_patch):
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
    env_patch = {
        "RANK": "1",
        "WORLD_SIZE": "4",
    }
    with mock.patch.dict(os.environ, env_patch, clear=False), mock.patch.dict(
        os.environ, {"TORCHCOMM_RANK": "", "TORCHCOMM_SIZE": ""}
    ):
      os.environ.pop("TORCHCOMM_RANK", None)
      os.environ.pop("TORCHCOMM_SIZE", None)
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
    env_patch = {
        "TORCHCOMM_RANK": "0",
        "TORCHCOMM_SIZE": "4",
    }
    with mock.patch.dict(os.environ, env_patch):
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
    invalid_op = mock.MagicMock(spec=torchcomms.ReduceOp)
    invalid_op.type = "UNSUPPORTED_TYPE"
    with self.assertRaises(ValueError):
      torchcomm_tpu._to_dist_reduce_op(invalid_op)

  def test_torchcomms_all_reduce(self):
    """Verifies all_reduce collective delegation and TorchWork handle completion."""
    comm = torchcomms.new_comm(
        "tpu", torch.device("cpu"), name="test_all_reduce"
    )
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    tensor = torch.tensor([1.0, 2.0])
    work = comm.all_reduce(tensor, op=torchcomms.ReduceOp.SUM, async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.allreduce.called)
    self.assertEqual(
        mock_pg.allreduce.call_args[0][1].reduceOp, dist.ReduceOp.SUM
    )
    comm.finalize()

  def test_torchcomms_broadcast(self):
    """Verifies broadcast collective delegation and TorchWork handle completion."""
    comm = torchcomms.new_comm(
        "tpu", torch.device("cpu"), name="test_broadcast"
    )
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    tensor = torch.tensor([1.0, 2.0])
    work = comm.broadcast(tensor, root=0, async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.broadcast.called)
    comm.finalize()

  def test_torchcomms_all_gather_and_all_gather_v(self):
    """Verifies standard and vectorized all_gather collective operations."""
    comm = torchcomms.new_comm(
        "tpu", torch.device("cpu"), name="test_all_gather"
    )
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    tensor = torch.tensor([1.0, 2.0])
    out_list = [torch.empty_like(tensor), torch.empty_like(tensor)]

    # Standard all_gather
    work = comm.all_gather(out_list, tensor, async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.allgather.called)

    # Vectorized all_gather_v
    work_v = comm.all_gather_v(out_list, tensor, async_op=False)
    self.assertIsInstance(work_v, torchcomms.TorchWork)
    work_v.wait()
    self.assertTrue(work_v.is_completed())
    comm.finalize()

  def test_torchcomms_all_gather_single(self):
    """Verifies flattened all_gather_single contiguous tensor collective."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_ags")
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    tensor = torch.tensor([1.0, 2.0])
    out = torch.empty([2, 2])
    work = comm.all_gather_single(out, tensor, async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg._allgather_base.called)
    comm.finalize()

  def test_torchcomms_reduce_scatter_and_reduce_scatter_v(self):
    """Verifies standard and vectorized reduce_scatter collective operations."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_rs")
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    out = torch.empty([2])
    in_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

    # Standard reduce_scatter
    work = comm.reduce_scatter(
        out, in_list, op=torchcomms.ReduceOp.SUM, async_op=False
    )
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.reduce_scatter.called)

    # Vectorized reduce_scatter_v
    work_v = comm.reduce_scatter_v(
        out, in_list, op=torchcomms.ReduceOp.SUM, async_op=False
    )
    self.assertIsInstance(work_v, torchcomms.TorchWork)
    work_v.wait()
    self.assertTrue(work_v.is_completed())
    comm.finalize()

  def test_torchcomms_reduce_scatter_single(self):
    """Verifies contiguous single-tensor reduce_scatter_single collective."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_rss")
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    out = torch.empty([2])
    in_t = torch.empty([4])
    work = comm.reduce_scatter_single(
        out, in_t, op=torchcomms.ReduceOp.SUM, async_op=False
    )
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg._reduce_scatter_base.called)
    comm.finalize()

  def test_torchcomms_barrier(self):
    """Verifies barrier collective synchronization."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_barrier")
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    work = comm.barrier(async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.barrier.called)
    comm.finalize()

  def test_torchcomms_send_recv(self):
    """Verifies point-to-point send and recv operations."""
    env_patch = {
        "TORCHCOMM_RANK": "0",
        "TORCHCOMM_SIZE": "2",
    }
    with mock.patch.dict(os.environ, env_patch):
      comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_sr")
      mock_pg = mock.MagicMock()
      mock_pg.experimental_send = mock.MagicMock()
      mock_pg.experimental_recv = mock.MagicMock()
      comm.get_backend_impl()._pg = mock_pg

      tensor = torch.tensor([1.0, 2.0])
      work_send = comm.send(tensor, dst=1, async_op=False)
      self.assertIsInstance(work_send, torchcomms.TorchWork)
      work_send.wait()
      self.assertTrue(work_send.is_completed())
      self.assertTrue(mock_pg.experimental_send.called)

      work_recv = comm.recv(tensor, src=1, async_op=False)
      self.assertIsInstance(work_recv, torchcomms.TorchWork)
      work_recv.wait()
      self.assertTrue(work_recv.is_completed())
      self.assertTrue(mock_pg.experimental_recv.called)
      comm.finalize()

  def test_torchcomms_all_to_all(self):
    """Verifies all_to_all collective operation with tensor lists."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_a2a")
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    t_list = [torch.tensor([1.0]), torch.tensor([2.0])]
    work = comm.all_to_all(t_list, t_list, async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.alltoall.called)
    comm.finalize()

  def test_torchcomms_all_to_all_single_and_v(self):
    """Verifies all_to_all_single and split-based all_to_all_v_single."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_a2as")
    mock_pg = mock.MagicMock()
    comm.get_backend_impl()._pg = mock_pg

    out = torch.empty([2, 2])
    work = comm.all_to_all_single(out, out, async_op=False)
    self.assertIsInstance(work, torchcomms.TorchWork)
    work.wait()
    self.assertTrue(work.is_completed())
    self.assertTrue(mock_pg.alltoall_base.called)

    work_v = comm.all_to_all_v_single(
        out,
        out,
        output_split_sizes=[1, 1],
        input_split_sizes=[1, 1],
        async_op=False,
    )
    self.assertIsInstance(work_v, torchcomms.TorchWork)
    work_v.wait()
    self.assertTrue(work_v.is_completed())
    comm.finalize()

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

  def test_torchcomms_handles_backend_routing(self):
    """Verifies c10d._torchcomms_handles_backend recognizes tpu and tpu_dist."""
    with mock.patch.multiple(
        c10d,
        _TORCHCOMM_AVAILABLE=True,
        _torchcomms_is_backend_registered=lambda name: name
        in ("tpu", "tpu_dist"),
        _torchcomms_is_backend_built=lambda name: False,
        create=True,
    ):
      self.assertTrue(c10d._torchcomms_handles_backend("tpu"))
      self.assertTrue(c10d._torchcomms_handles_backend("tpu_dist"))
      self.assertTrue(c10d._torchcomms_handles_backend("TPU"))
      self.assertFalse(c10d._torchcomms_handles_backend("unknown_backend"))

  def test_c10d_backend_availability(self):
    """Verifies backend availability registration with PyTorch c10d."""
    mock_distributed = mock.MagicMock()
    with mock.patch(
        "torch_tpu._internal.distributed.torchcomm_tpu.tpu_distributed",
        mock_distributed,
    ):
      torchcomm_tpu.register_torchcomms_tpu()
      self.assertTrue(dist.is_backend_available("tpu_dist"))
      self.assertTrue(dist.is_backend_available("tpu"))
      self.assertEqual(
          dist.Backend.default_device_backend_map.get("tpu"), "tpu"
      )

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
      backend.register_flight_recorder_hook(mock.MagicMock())

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


# Alias for backward compatibility
TorchCommTPUTest = TorchCommsTPUTest

if __name__ == "__main__":
  absltest.main()
