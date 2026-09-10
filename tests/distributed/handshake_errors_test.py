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

"""Tests for distributed handshake protocol error handling."""

import asyncio
import json
import os
from unittest import mock

from absl.testing import absltest
import portpicker
import torch.distributed as dist
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.distributed import handshake
from tests import seed_test_utils
import zmq

_CollectiveHandshakeConsensus = handshake._CollectiveHandshakeConsensus
_CollectiveHandshakeRequestZMQEnvelope = (
    handshake._CollectiveHandshakeRequestZMQEnvelope
)
_ZMQClient = handshake._ZMQClient
_ZMQServer = handshake._ZMQServer
Handshake = handshake.Handshake
CollectiveHandshakeRequest = handshake.CollectiveHandshakeRequest
CollectiveHandshakeResponse = handshake.CollectiveHandshakeResponse
ProcessGroupId = handshake.ProcessGroupId
ProcessGroupCollectiveCount = handshake.ProcessGroupCollectiveCount
_get_handshake_timeout_s = handshake._get_handshake_timeout_s
RankCollectiveCounts = handshake.RankCollectiveCounts


def _make_request(
    rank: int,
    participating_ranks: list[int] | None = None,
    fingerprint: str = "fp",
    num_collectives_in_graph: int = 2,
    collective_count_before: int = 0,
) -> CollectiveHandshakeRequest:
  if participating_ranks is None:
    participating_ranks = [rank]
  pg_counts = RankCollectiveCounts({
      ProcessGroupId(participating_ranks): ProcessGroupCollectiveCount(
          collective_count_before=collective_count_before,
          num_collectives_in_graph=num_collectives_in_graph,
      )
  })
  return CollectiveHandshakeRequest(
      pg_collective_counts=pg_counts,
      executable_fingerprint=fingerprint,
      rank=rank,
  )


class ProcessGroupCollectiveCountTest(seed_test_utils.RepeatableTest):
  """Unit tests for ProcessGroupCollectiveCount error handling."""

  def test_invalid_collective_count_before(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "collective_count_before must be non-negative"
    ):
      ProcessGroupCollectiveCount(
          collective_count_before=-1,
          num_collectives_in_graph=1,
      )

  def test_invalid_num_collectives_in_graph(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "num_collectives_in_graph must be strictly positive"
    ):
      ProcessGroupCollectiveCount(
          collective_count_before=0,
          num_collectives_in_graph=0,
      )
    with self.assertRaisesRegex(
        ValueError, "num_collectives_in_graph must be strictly positive"
    ):
      ProcessGroupCollectiveCount(
          collective_count_before=0,
          num_collectives_in_graph=-2,
      )

  def test_increment_num_collectives_in_graph_invalid(self) -> None:
    count = ProcessGroupCollectiveCount(
        collective_count_before=0,
        num_collectives_in_graph=2,
    )
    with self.assertRaisesRegex(
        ValueError, "number_of_collectives must be greater or equal to 0"
    ):
      count.increment_num_collectives_in_graph(-1)


class RankCollectiveCountsTest(seed_test_utils.RepeatableTest):
  """Unit tests for RankCollectiveCounts error handling."""

  def test_missing_key_raises_key_error(self) -> None:
    counts = RankCollectiveCounts()
    pg = ProcessGroupId([0, 1])
    with self.assertRaises(KeyError):
      _ = counts[pg]

  def test_collective_count_empty_raises(self) -> None:
    counts = RankCollectiveCounts()
    with self.assertRaises(KeyError):
      counts.collective_count(ProcessGroupId([0, 1]))

  def test_collective_count_missing_key_raises(self) -> None:
    counts = RankCollectiveCounts({
        ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
            collective_count_before=0, num_collectives_in_graph=1
        ),
    })
    with self.assertRaises(KeyError):
      counts.collective_count(ProcessGroupId([0, 1, 2, 3]))


class CollectiveHandshakeRequestTest(seed_test_utils.RepeatableTest):
  """Unit tests for CollectiveHandshakeRequest error handling."""

  def test_invalid_participating_ranks_empty(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "participating_ranks cannot be empty"
    ):
      CollectiveHandshakeRequest(
          pg_collective_counts=RankCollectiveCounts(),
          executable_fingerprint="fp",
          rank=0,
      )

  def test_invalid_rank(self) -> None:
    pg_counts = RankCollectiveCounts()
    pg_counts.increment_pg_collective_counts({
        ProcessGroupId([0, 1, 2, 3]): 1,
    })
    with self.assertRaisesRegex(
        ValueError, r"rank \(4\) must be present in participating_ranks"
    ):
      CollectiveHandshakeRequest(
          pg_collective_counts=pg_counts,
          executable_fingerprint="fp",
          rank=4,
      )
    with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
      CollectiveHandshakeRequest(
          pg_collective_counts=pg_counts,
          executable_fingerprint="fp",
          rank=-1,
      )

  def test_from_bytes_invalid_payloads(self) -> None:
    with self.assertRaises(json.JSONDecodeError):
      CollectiveHandshakeRequest.from_bytes(b"invalid json")
    with self.assertRaises(UnicodeDecodeError):
      CollectiveHandshakeRequest.from_bytes(b"\xff\xfe")
    with self.assertRaises(KeyError):
      CollectiveHandshakeRequest.from_bytes(b"{}")


class CollectiveHandshakeResponseTest(seed_test_utils.RepeatableTest):
  """Unit tests for CollectiveHandshakeResponse error handling."""

  def test_from_bytes_invalid_payloads(self) -> None:
    with self.assertRaises(ValueError):
      CollectiveHandshakeResponse.from_bytes(b"invalid")


class GetHandshakeTimeoutMsTest(seed_test_utils.RepeatableTest):
  """Unit tests for _get_handshake_timeout_s error handling."""

  def test_invalid_string_timeout(self) -> None:
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S": "invalid"}
    ):
      with self.assertRaisesRegex(
          ValueError,
          "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S must be an integer",
      ):
        _get_handshake_timeout_s()

  def test_negative_timeout(self) -> None:
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S": "-100"}
    ):
      with self.assertRaisesRegex(
          ValueError,
          "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S must be non-negative",
      ):
        _get_handshake_timeout_s()


class CollectiveHandshakeConsensusTest(seed_test_utils.RepeatableTest):
  """Unit tests for _CollectiveHandshakeConsensus error handling."""

  def test_invalid_num_queues(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "num_queues must be strictly positive"
    ):
      _CollectiveHandshakeConsensus(num_queues=0)
    with self.assertRaisesRegex(
        ValueError, "num_queues must be strictly positive"
    ):
      _CollectiveHandshakeConsensus(num_queues=-1)

  def test_invalid_put_request_rank(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      msg = _make_request(rank=0)
      env = _CollectiveHandshakeRequestZMQEnvelope(client_id=b"c", request=msg)
      with self.assertRaisesRegex(ValueError, r"rank \(-1\) must be in range"):
        await consensus.put_request(-1, env)
      with self.assertRaisesRegex(ValueError, r"rank \(4\) must be in range"):
        await consensus.put_request(4, env)

    asyncio.run(_test())

  def test_invalid_get_from_ranks(self) -> None:
    async def _test() -> None:
      consensus = _CollectiveHandshakeConsensus(num_queues=4)
      with self.assertRaisesRegex(
          ValueError, "ranks must not contain duplicates"
      ):
        await consensus.get_from_ranks([0, 1, 0])
      with self.assertRaisesRegex(
          ValueError, r"rank \(-1\) in ranks must be in range"
      ):
        await consensus.get_from_ranks([-1, 0])
      with self.assertRaisesRegex(
          ValueError, r"rank \(4\) in ranks must be in range"
      ):
        await consensus.get_from_ranks([0, 4])

    asyncio.run(_test())


class GetValidatedHandshakePortTest(seed_test_utils.RepeatableTest):
  """Unit tests for _get_validated_port error handling."""

  def test_get_validated_port_in_use_raises_error_for_coordinator(self) -> None:
    """Tests that an exception is raised when the handshake port is in use."""
    with mock.patch.object(portpicker, "is_port_free", return_value=False):
      with self.assertRaisesRegex(
          RuntimeError, "TORCH_TPU_INTERNAL_HANDSHAKE_PORT"
      ):
        handshake._get_validated_port(is_coordinator=True)


class ZMQServerClientTest(seed_test_utils.RepeatableTest):
  """Unit tests for _ZMQServer and _ZMQClient error handling."""

  def test_zmq_server_invalid_port_number(self) -> None:
    with self.assertRaises(ValueError):
      _ZMQServer(port=-1)

    with self.assertRaises(ValueError):
      _ZMQServer(port=65536)

  def test_zmq_client_invalid_port_number(self) -> None:
    with self.assertRaises(ValueError):
      _ZMQClient(port=-1)
    with self.assertRaises(ValueError):
      _ZMQClient(port=65536)

  def test_zmq_server_invalid_timeout(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "send_timeout_s must be non-negative"
    ):
      _ZMQServer(port=12345, send_timeout_s=-1)
    with self.assertRaisesRegex(
        ValueError, "recv_timeout_s must be non-negative"
    ):
      _ZMQServer(port=12345, recv_timeout_s=-1)

  def test_zmq_client_invalid_timeout(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "send_timeout_s must be non-negative"
    ):
      _ZMQClient(port=12345, send_timeout_s=-1)
    with self.assertRaisesRegex(
        ValueError, "recv_timeout_s must be non-negative"
    ):
      _ZMQClient(port=12345, recv_timeout_s=-1)

  def test_zmq_server_close_idempotent_and_closed_operations(self) -> None:
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port)
    server.close()
    # Multiple close calls must be safe and idempotent
    server.close()

    async def _test() -> None:
      with self.assertRaises(zmq.ZMQError):
        await server.recv_request()
      with self.assertRaises(zmq.ZMQError):
        await server.send_reply(
            b"c0", CollectiveHandshakeResponse(success=True)
        )

    asyncio.run(_test())

  def test_zmq_client_close_idempotent_and_closed_operations(self) -> None:
    port = portpicker.pick_unused_port()
    client = _ZMQClient(port=port)
    client.close()
    # Multiple close calls must be safe and idempotent
    client.close()

    msg = _make_request(rank=0)
    with self.assertRaises(zmq.ZMQError):
      client.send(msg)
    with self.assertRaises(zmq.ZMQError):
      client.recv()


class HandshakeTest(seed_test_utils.RepeatableTest):
  """Unit tests for distributed Handshake protocol error handling on CPU."""

  def setUp(self) -> None:
    super().setUp()
    Handshake._reset_instances()

  def tearDown(self) -> None:
    super().tearDown()
    Handshake._reset_instances()

  def test_submit_invalid_message_type(self) -> None:
    """Tests that submitting non-CollectiveHandshakeRequest raises TypeError."""
    port = portpicker.pick_unused_port()
    with mock.patch.object(dist, "get_world_size", return_value=1):
      with mock.patch.object(
          tpu_torch_compile,
          "get_handshake_port_env_var_once",
          return_value=port,
      ):
        hs = Handshake(current_rank=0, coordinator_rank=0)
        with self.assertRaises(TypeError):
          hs.submit("invalid_string")  # pyrefly: ignore[bad-argument-type]

  def test_submit_subgroup_not_supported(self) -> None:
    """Tests that submitting participating_ranks != world_size raises RuntimeError."""
    port = portpicker.pick_unused_port()
    with mock.patch.object(dist, "get_world_size", return_value=4):
      with mock.patch.object(
          tpu_torch_compile,
          "get_handshake_port_env_var_once",
          return_value=port,
      ):
        hs = Handshake(current_rank=0, coordinator_rank=0)
        msg = _make_request(
            rank=0,
            participating_ranks=[0, 1],  # len 2 != world_size 4
            fingerprint="fp",
            num_collectives_in_graph=2,
            collective_count_before=0,
        )
        with self.assertRaisesRegex(
            RuntimeError, "do not support subgroup communication"
        ):
          hs.submit(msg)


if __name__ == "__main__":
  absltest.main()
