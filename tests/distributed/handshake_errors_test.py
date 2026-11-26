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
import socket
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import torch.distributed as dist
from torch_tpu._internal import env
from torch_tpu._internal.distributed import handshake
from torch_tpu._internal.distributed import handshake_consensus
from torch_tpu._internal.distributed import handshake_messages
from tests import seed_test_utils
import zmq

_ExpectedHandshakeClaim = handshake_consensus._ExpectedHandshakeClaim
_ProcessGroupState = handshake_consensus._ProcessGroupState
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
        ValueError, "num_collectives_in_graph must be non-negative"
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

  def setUp(self) -> None:
    super().setUp()
    # Participating-rank validation only runs in debug mode.
    self.enter_context(
        mock.patch.object(
            handshake_messages, "is_torch_tpu_debug_mode", return_value=True
        )
    )

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

  def test_negative_timeout(self) -> None:
    with mock.patch.object(env, "get_int_env_once", return_value=-100):
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


class HandshakeConsensusSubmitTest(seed_test_utils.RepeatableTest):
  """Unit tests for requests that the consensus for arbitrary groups refuses.

  The consensus is `handshake_consensus.CollectiveHandshakeConsensus`.
  """

  def setUp(self) -> None:
    super().setUp()
    self.consensus = handshake_consensus.CollectiveHandshakeConsensus()

  def submit(self, rank: int, start: int, end: int) -> dict[int, bool]:
    """Submits a request of `rank` for collectives `[start, end)` of {0, 1}.

    Args:
      rank: The rank sending the request.
      start: The number of collectives of the process group before the graph.
      end: `start` plus the number of collectives of the process group in the
        graph.

    Returns:
      Whether each response is a success, by rank.
    """
    responses = self.consensus.submit(
        _make_request(
            rank,
            participating_ranks=[0, 1],
            num_collectives_in_graph=end - start,
            collective_count_before=start,
        )
    )
    return {
        responder: response.success for responder, response in responses.items()
    }

  def test_second_request_of_a_rank_before_the_response(self) -> None:
    self.submit(0, 0, 1)

    with self.assertRaisesRegex(ValueError, "already has a pending request"):
      self.submit(0, 1, 2)

  def test_debug_mode_rejects_requests_that_restart_or_skip(self) -> None:
    self.enter_context(
        mock.patch.object(
            handshake_messages,
            "is_torch_tpu_debug_mode",
            autospec=True,
            return_value=True,
        )
    )
    self.submit(0, 0, 2)
    self.submit(1, 0, 2)

    with self.assertRaisesRegex(ValueError, "earlier requests end at 2"):
      self.submit(0, 1, 3)
    with self.assertRaisesRegex(ValueError, "earlier requests end at 2"):
      self.submit(0, 3, 4)
    # The refused requests are ignored.
    self.assertEmpty(self.submit(0, 2, 3))
    self.assertEqual(self.submit(1, 2, 3), {0: True, 1: True})

  def test_debug_mode_ignores_request_for_a_process_group_without_its_rank(
      self,
  ) -> None:
    self.enter_context(
        mock.patch.object(
            handshake_messages,
            "is_torch_tpu_debug_mode",
            autospec=True,
            return_value=True,
        )
    )
    one_collective = ProcessGroupCollectiveCount(
        collective_count_before=0, num_collectives_in_graph=1
    )
    # Rank 0 is a member of the first process group, but not of the second.
    request = CollectiveHandshakeRequest(
        pg_collective_counts=RankCollectiveCounts({
            ProcessGroupId([0, 1]): one_collective,
            ProcessGroupId([1, 2]): one_collective,
        }),
        executable_fingerprint="fp",
        rank=0,
    )

    with self.assertRaisesRegex(ValueError, "not a member"):
      self.consensus.submit(request)

    self.assertEmpty(self.submit(0, 0, 1))
    self.assertEqual(self.submit(1, 0, 1), {0: True, 1: True})

  def test_request_on_a_second_process_group_is_refused(self) -> None:
    # TODO: b/553328238 - Remove once more than one process group is supported.
    self.submit(0, 0, 1)

    with self.assertRaisesRegex(NotImplementedError, "more than one process"):
      self.consensus.submit(_make_request(1, participating_ranks=[1, 2]))

    # The refused request is ignored.
    self.assertEqual(self.submit(1, 0, 1), {0: True, 1: True})


class HandshakeConsensusProcessGroupStateTest(seed_test_utils.RepeatableTest):
  """Unit tests for `handshake_consensus._ProcessGroupState` error handling."""

  def test_claimants_have_to_make_the_same_claim(self) -> None:
    state = _ProcessGroupState(ProcessGroupId([0, 1, 2]))
    state.add_claimant(0, _ExpectedHandshakeClaim(0, 2, "a"))

    with self.assertRaisesRegex(ValueError, "rank 1 cannot claim"):
      state.add_claimant(1, _ExpectedHandshakeClaim(0, 2, "b"))
    with self.assertRaisesRegex(ValueError, "rank 1 cannot claim"):
      state.add_claimant(1, _ExpectedHandshakeClaim(0, 1, "a"))
    self.assertEqual(state.claimants, (0,))


class GetValidatedHandshakePortTest(seed_test_utils.RepeatableTest):
  """Unit tests for _get_validated_port error handling."""

  def test_get_validated_port_in_use_raises_error_for_coordinator(self) -> None:
    """Tests that an exception is raised when the handshake port is in use."""
    # Non-coordinators skip the availability check, which reveals the port the
    # coordinator would bind without depending on how it was configured.
    port = handshake._get_validated_port(is_coordinator=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy_sock:
      try:
        busy_sock.bind(("", port))
        busy_sock.listen(1)
      except OSError:
        # Something else already holds the port, which is just as busy.
        pass
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
      _ZMQClient(port=-1, routing_id=b"rank-1")
    with self.assertRaises(ValueError):
      _ZMQClient(port=65536, routing_id=b"rank-1")

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
      _ZMQClient(port=12345, routing_id=b"rank-1", send_timeout_s=-1)
    with self.assertRaisesRegex(
        ValueError, "recv_timeout_s must be non-negative"
    ):
      _ZMQClient(port=12345, routing_id=b"rank-1", recv_timeout_s=-1)

  def test_zmq_server_close_idempotent_and_closed_operations(self) -> None:
    port = portpicker.pick_unused_port()
    server = _ZMQServer(port=port)
    server.close()
    # Multiple close calls must be safe and idempotent
    server.close()

    async def _test() -> None:
      with self.assertRaises(zmq.ZMQError):
        await server.recv_requests_batch()
      with self.assertRaises(zmq.ZMQError):
        await server.send_reply(
            b"c0", CollectiveHandshakeResponse(success=True)
        )

    asyncio.run(_test())

  def test_zmq_client_close_idempotent_and_closed_operations(self) -> None:
    port = portpicker.pick_unused_port()
    client = _ZMQClient(port=port, routing_id=b"rank-1")
    client.close()
    # Multiple close calls must be safe and idempotent
    client.close()

    msg = _make_request(rank=0)
    with self.assertRaises(zmq.ZMQError):
      client.send(msg)
    with self.assertRaises(zmq.ZMQError):
      client.recv()


class HandshakeServerTest(seed_test_utils.RepeatableTest):
  """Unit tests for _HandshakeServer handling of invalid remote requests."""

  def setUp(self) -> None:
    super().setUp()
    handshake._HandshakeServer._reset_instance()

  def tearDown(self) -> None:
    super().tearDown()
    handshake._HandshakeServer._reset_instance()

  @parameterized.named_parameters(
      dict(testcase_name="invalid_json", payload=b"not-valid-json"),
      dict(testcase_name="invalid_utf8", payload=b"\xff\xfe\xfd"),
      dict(testcase_name="missing_keys", payload=b"{}"),
      dict(testcase_name="json_list", payload=b"[]"),
      dict(
          testcase_name="rank_out_of_range",
          payload=_make_request(rank=5, participating_ranks=[5]).to_bytes(),
      ),
      dict(
          testcase_name="non_integer_rank",
          payload=(
              b'{"executable_fingerprint": "fp", "rank": 1.5,'
              b' "pg_collective_counts": [[{"start": 0, "stop": 2}, 0, 2]]}'
          ),
      ),
      dict(
          testcase_name="participating_rank_out_of_range",
          payload=_make_request(
              rank=0, participating_ranks=[0, 1, 5]
          ).to_bytes(),
      ),
  )
  def test_drops_invalid_request_and_keeps_serving(
      self, payload: bytes
  ) -> None:
    """Tests that an invalid request is dropped without killing server loops."""
    port = portpicker.pick_unused_port()
    server = handshake._HandshakeServer(current_rank=0, port=port, world_size=2)
    # A raw DEALER socket lets us send arbitrary bytes. All requests, including
    # rank 0's, go over this one connection so they are delivered in order: the
    # invalid request is always processed before the valid round. The server
    # replies to rank 0 locally, so `server.recv()` receives rank 0's response.
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.ROUTING_ID, b"rank-1")
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"tcp://localhost:{port}")
    participating_ranks = [0, 1]
    try:
      sock.send(payload)
      for rank in participating_ranks:
        sock.send(
            _make_request(
                rank=rank, participating_ranks=participating_ranks
            ).to_bytes()
        )

      self.assertTrue(
          CollectiveHandshakeResponse.from_bytes(sock.recv()).success
      )
      self.assertTrue(server.recv().success)
      for task in server._get_loop_tasks():
        self.assertFalse(task.done())
    finally:
      sock.close(linger=0)
      ctx.term()
      server.close()


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
          env,
          "get_int_env_once",
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
          env,
          "get_int_env_once",
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
