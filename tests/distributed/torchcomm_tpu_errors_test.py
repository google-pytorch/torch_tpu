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

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.distributed import torchcomm_tpu
from tests import seed_test_utils
import torchcomms


class TorchCommsTPUTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torchcomm_tpu.register_torchcomms_tpu()

  def test_collective_after_finalize_raises_runtime_error(self):
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_finalize")
    comm.finalize()
    dummy_tensor = torch.tensor([1.0])
    with self.assertRaisesRegex(RuntimeError, "finalized"):
      comm.all_reduce(dummy_tensor, op=torchcomms.ReduceOp.SUM, async_op=False)

  def test_to_dist_reduce_op_unsupported_raises_value_error(self):
    invalid_op = mock.MagicMock(spec=torchcomms.ReduceOp)
    invalid_op.type = "UNSUPPORTED_TYPE"
    with self.assertRaises(ValueError):
      torchcomm_tpu._to_dist_reduce_op(invalid_op)

  def test_unsupported_stubs_raise_not_implemented(self):
    """Verifies that unsupported collective or communicator features raise NotImplementedError."""
    comm = torchcomms.new_comm("tpu", torch.device("cpu"), name="test_stubs")
    backend = comm.get_backend_impl()
    t = torch.zeros(4, device="cpu")

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


if __name__ == "__main__":
  absltest.main()
