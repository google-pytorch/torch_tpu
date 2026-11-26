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

"""Tests for the consensus rules of the collective handshake."""

from collections.abc import Mapping
from unittest import mock

from absl.testing import absltest
from torch_tpu._internal.distributed import handshake_consensus
from torch_tpu._internal.distributed import handshake_messages
from torch_tpu._internal.distributed import process_group_utils
from tests import seed_test_utils

CollectiveHandshakeConsensus = handshake_consensus.CollectiveHandshakeConsensus
CollectiveHandshakeRequest = handshake_messages.CollectiveHandshakeRequest
CollectiveHandshakeResponse = handshake_messages.CollectiveHandshakeResponse
ProcessGroupCollectiveCount = handshake_messages.ProcessGroupCollectiveCount
RankCollectiveCounts = handshake_messages.RankCollectiveCounts
ProcessGroupId = process_group_utils.ProcessGroupId

# pylint: disable=protected-access
_Claim = handshake_consensus._ExpectedHandshakeClaim
_ProcessGroupState = handshake_consensus._ProcessGroupState
_ProcessGroupClaim = handshake_consensus._ProcessGroupClaim
_RejectionHistory = handshake_consensus._RejectionHistory
_RequestWithProcessGroupClaims = (
    handshake_consensus._RequestWithProcessGroupClaims
)
_mismatch_rule_applies = handshake_consensus._mismatch_rule_applies
# pylint: enable=protected-access


def _request(
    rank: int, claims: Mapping[ProcessGroupId, _Claim]
) -> CollectiveHandshakeRequest:
  """Returns the request of `rank` that makes `claims`.

  Args:
    rank: The rank sending the request.
    claims: The claim of the request on every process group it claims. All
      claims have the fingerprint of the request.
  """
  (fingerprint,) = {claim.fingerprint for claim in claims.values()}
  counts = {
      process_group: ProcessGroupCollectiveCount(
          collective_count_before=claim.start,
          num_collectives_in_graph=claim.end - claim.start,
      )
      for process_group, claim in claims.items()
  }
  return CollectiveHandshakeRequest(
      pg_collective_counts=RankCollectiveCounts(counts),
      executable_fingerprint=fingerprint,
      rank=rank,
  )


def _handshake_successes(
    responses: Mapping[int, CollectiveHandshakeResponse],
) -> Mapping[int, bool]:
  """Returns whether each response is a success, by rank."""
  return {rank: response.success for rank, response in responses.items()}


class _ConsensusTestCase(seed_test_utils.RepeatableTest):
  """Base class of the tests that submit the requests of valid programs.

  The tests draw their program with one column per rank. A box is a compiled
  graph, with its fingerprint on top and its collectives inside. A rank runs
  its graphs from top to bottom. `0_1[2]` is collective 2 of the process group
  of ranks 0 and 1, which the tests call `group_0_1`.
  """

  def setUp(self) -> None:
    super().setUp()
    # Programs are valid, so every request has to continue the previous one of
    # its rank. Debug mode makes the consensus check that.
    self.enter_context(
        mock.patch.object(
            handshake_messages,
            "is_torch_tpu_debug_mode",
            autospec=True,
            return_value=True,
        )
    )
    self.consensus = CollectiveHandshakeConsensus()

  def submit(
      self, rank: int, claims: Mapping[ProcessGroupId, _Claim]
  ) -> Mapping[int, bool]:
    """Submits the request of `rank` that makes `claims`.

    Args:
      rank: The rank sending the request.
      claims: The claim of the request on every process group it claims.

    Returns:
      Whether each response is a success, by rank.
    """
    return _handshake_successes(self.consensus.submit(_request(rank, claims)))


class FusedProgramsTest(_ConsensusTestCase):
  """Programs whose requests must all be accepted."""

  def test_process_group_without_collectives_in_the_graph_is_not_claimed(
      self,
  ) -> None:
    def request(rank: int) -> CollectiveHandshakeRequest:
      # The request also lists a process group that its rank used in earlier
      # graphs, with no collectives in this graph.
      counts = RankCollectiveCounts({
          ProcessGroupId([0, 1]): ProcessGroupCollectiveCount(
              collective_count_before=0, num_collectives_in_graph=1
          ),
          ProcessGroupId([0, 1, 2]): ProcessGroupCollectiveCount(
              collective_count_before=3, num_collectives_in_graph=0
          ),
      })
      return CollectiveHandshakeRequest(
          pg_collective_counts=counts, executable_fingerprint="a", rank=rank
      )

    self.assertEmpty(self.consensus.submit(request(0)))
    # The requests do not wait for rank 2.
    self.assertEqual(
        _handshake_successes(self.consensus.submit(request(1))),
        {0: True, 1: True},
    )

  def test_spmd_program_on_the_global_group(self) -> None:
    group_0_1_2_3 = ProcessGroupId([0, 1, 2, 3])
    # Every rank runs three graphs, with two, one and two collectives.
    for claim in (_Claim(0, 2, "a"), _Claim(2, 3, "b"), _Claim(3, 5, "c")):
      self.assertEmpty(self.submit(0, {group_0_1_2_3: claim}))
      self.assertEmpty(self.submit(1, {group_0_1_2_3: claim}))
      self.assertEmpty(self.submit(2, {group_0_1_2_3: claim}))
      self.assertEqual(
          self.submit(3, {group_0_1_2_3: claim}),
          {0: True, 1: True, 2: True, 3: True},
      )

  def test_last_rank_to_submit_triggers_the_responses(self) -> None:
    group_0_1_2 = ProcessGroupId([0, 1, 2])
    everyone = {0: True, 1: True, 2: True}

    first = _Claim(0, 1, "a")
    self.assertEmpty(self.submit(2, {group_0_1_2: first}))
    self.assertEmpty(self.submit(0, {group_0_1_2: first}))
    self.assertEqual(self.submit(1, {group_0_1_2: first}), everyone)

    second = _Claim(1, 3, "b")
    self.assertEmpty(self.submit(1, {group_0_1_2: second}))
    self.assertEmpty(self.submit(2, {group_0_1_2: second}))
    self.assertEqual(self.submit(0, {group_0_1_2: second}), everyone)

  def test_spmd_program_with_tensor_and_data_parallel_groups(self) -> None:
    # Every graph runs an all-reduce on the tensor parallel group of its rank,
    # `group_0_1` or `group_2_3`, and one on the data parallel group of its
    # rank, `group_0_2` or `group_1_3`.
    #
    #    rank 0      rank 1      rank 2      rank 3
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[0] |  | 0_1[0] |  | 2_3[0] |  | 2_3[0] |
    #  | 0_2[0] |  | 1_3[0] |  | 0_2[0] |  | 1_3[0] |
    #  +--------+  +--------+  +--------+  +--------+
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[1] |  | 0_1[1] |  | 2_3[1] |  | 2_3[1] |
    #  | 0_2[1] |  | 1_3[1] |  | 0_2[1] |  | 1_3[1] |
    #  +--------+  +--------+  +--------+  +--------+
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[2] |  | 0_1[2] |  | 2_3[2] |  | 2_3[2] |
    #  | 0_2[2] |  | 1_3[2] |  | 0_2[2] |  | 1_3[2] |
    #  +--------+  +--------+  +--------+  +--------+
    group_0_1 = ProcessGroupId([0, 1])
    group_2_3 = ProcessGroupId([2, 3])
    group_0_2 = ProcessGroupId([0, 2])
    group_1_3 = ProcessGroupId([1, 3])
    for step in range(3):
      claim = _Claim(step, step + 1, "a")
      self.assertEmpty(self.submit(0, {group_0_1: claim, group_0_2: claim}))
      self.assertEmpty(self.submit(1, {group_0_1: claim, group_1_3: claim}))
      self.assertEmpty(self.submit(2, {group_2_3: claim, group_0_2: claim}))
      self.assertEqual(
          self.submit(3, {group_2_3: claim, group_1_3: claim}),
          {0: True, 1: True, 2: True, 3: True},
      )

  def test_spmd_program_with_groups_changing_between_graphs(self) -> None:
    # Graphs on the tensor parallel groups, `group_0_1` and `group_2_3`,
    # alternate with graphs on the data parallel groups, `group_0_2` and
    # `group_1_3`.
    #
    #    rank 0      rank 1      rank 2      rank 3
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[0] |  | 0_1[0] |  | 2_3[0] |  | 2_3[0] |
    #  +--------+  +--------+  +--------+  +--------+
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_2[0] |  | 1_3[0] |  | 0_2[0] |  | 1_3[0] |
    #  +--------+  +--------+  +--------+  +--------+
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[1] |  | 0_1[1] |  | 2_3[1] |  | 2_3[1] |
    #  +--------+  +--------+  +--------+  +--------+
    #  +"a"-----+  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_2[1] |  | 1_3[1] |  | 0_2[1] |  | 1_3[1] |
    #  +--------+  +--------+  +--------+  +--------+
    group_0_1 = ProcessGroupId([0, 1])
    group_2_3 = ProcessGroupId([2, 3])
    group_0_2 = ProcessGroupId([0, 2])
    group_1_3 = ProcessGroupId([1, 3])
    for step in range(2):
      claim = _Claim(step, step + 1, "a")
      self.assertEmpty(self.submit(0, {group_0_1: claim}))
      self.assertEqual(self.submit(1, {group_0_1: claim}), {0: True, 1: True})
      self.assertEmpty(self.submit(2, {group_2_3: claim}))
      self.assertEqual(self.submit(3, {group_2_3: claim}), {2: True, 3: True})

      self.assertEmpty(self.submit(0, {group_0_2: claim}))
      self.assertEmpty(self.submit(1, {group_1_3: claim}))
      self.assertEqual(self.submit(2, {group_0_2: claim}), {0: True, 2: True})
      self.assertEqual(self.submit(3, {group_1_3: claim}), {1: True, 3: True})

  def test_spmd_program_with_the_global_group_and_subgroups(self) -> None:
    #      rank 0          rank 1          rank 2          rank 3
    #  +"a"---------+  +"a"---------+  +"a"---------+  +"a"---------+
    #  | 0_1_2_3[0] |  | 0_1_2_3[0] |  | 0_1_2_3[0] |  | 0_1_2_3[0] |
    #  | 0_1[0]     |  | 0_1[0]     |  | 2_3[0]     |  | 2_3[0]     |
    #  +------------+  +------------+  +------------+  +------------+
    #  +"a"---------+  +"a"---------+  +"a"---------+  +"a"---------+
    #  | 0_1_2_3[1] |  | 0_1_2_3[1] |  | 0_1_2_3[1] |  | 0_1_2_3[1] |
    #  | 0_1[1]     |  | 0_1[1]     |  | 2_3[1]     |  | 2_3[1]     |
    #  +------------+  +------------+  +------------+  +------------+
    group_0_1_2_3 = ProcessGroupId([0, 1, 2, 3])
    group_0_1 = ProcessGroupId([0, 1])
    group_2_3 = ProcessGroupId([2, 3])
    for step in range(2):
      claim = _Claim(step, step + 1, "a")
      self.assertEmpty(self.submit(0, {group_0_1_2_3: claim, group_0_1: claim}))
      self.assertEmpty(self.submit(1, {group_0_1_2_3: claim, group_0_1: claim}))
      self.assertEmpty(self.submit(2, {group_0_1_2_3: claim, group_2_3: claim}))
      self.assertEqual(
          self.submit(3, {group_0_1_2_3: claim, group_2_3: claim}),
          {0: True, 1: True, 2: True, 3: True},
      )

  def test_disjoint_sets_of_ranks_are_answered_independently(self) -> None:
    # Ranks 0-1 and ranks 2-3 run unrelated programs with different graph
    # boundaries and fingerprints.
    #
    #    rank 0      rank 1      rank 2      rank 3
    #  +"left"--+  +"left"--+  +"right"-+  +"right"-+
    #  | 0_1[0] |  | 0_1[0] |  | 2_3[0] |  | 2_3[0] |
    #  +--------+  +--------+  | 2_3[1] |  | 2_3[1] |
    #  +"left"--+  +"left"--+  | 2_3[2] |  | 2_3[2] |
    #  | 0_1[1] |  | 0_1[1] |  +--------+  +--------+
    #  +--------+  +--------+
    group_0_1 = ProcessGroupId([0, 1])
    group_2_3 = ProcessGroupId([2, 3])
    right_claim = _Claim(0, 3, "right")
    self.assertEmpty(self.submit(2, {group_2_3: right_claim}))

    # Ranks 0-1 get their responses while rank 3 has not submitted anything.
    first_claim = _Claim(0, 1, "left")
    self.assertEmpty(self.submit(0, {group_0_1: first_claim}))
    self.assertEqual(
        self.submit(1, {group_0_1: first_claim}), {0: True, 1: True}
    )
    second_claim = _Claim(1, 2, "left")
    self.assertEmpty(self.submit(0, {group_0_1: second_claim}))
    self.assertEqual(
        self.submit(1, {group_0_1: second_claim}), {0: True, 1: True}
    )

    self.assertEqual(
        self.submit(3, {group_2_3: right_claim}), {2: True, 3: True}
    )

  def test_rank_busy_with_another_group_is_waited_for(self) -> None:
    # Rank 0 waits for rank 1, which first runs a graph with rank 2. Waiting
    # is not a cycle.
    #
    #    rank 0      rank 1      rank 2
    #  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[0] |  | 1_2[0] |  | 1_2[0] |
    #  +--------+  +--------+  +--------+
    #              +"a"-----+
    #              | 0_1[0] |
    #              +--------+
    group_1_2 = ProcessGroupId([1, 2])
    group_0_1 = ProcessGroupId([0, 1])
    claim = _Claim(0, 1, "a")
    self.assertEmpty(self.submit(0, {group_0_1: claim}))
    self.assertEmpty(self.submit(1, {group_1_2: claim}))
    self.assertEqual(self.submit(2, {group_1_2: claim}), {1: True, 2: True})
    self.assertEqual(self.submit(1, {group_0_1: claim}), {0: True, 1: True})


class RejectedProgramsTest(_ConsensusTestCase):
  """Programs with requests that must be rejected."""

  def test_different_fingerprints(self) -> None:
    group_0_1_2 = ProcessGroupId([0, 1, 2])
    self.assertEmpty(self.submit(0, {group_0_1_2: _Claim(0, 1, "a")}))
    self.assertEmpty(self.submit(1, {group_0_1_2: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(2, {group_0_1_2: _Claim(0, 1, "b")}),
        {0: False, 1: False, 2: False},
    )

  def test_mismatch_on_one_of_the_process_groups_of_a_graph(self) -> None:
    # The graph of rank 0 matches rank 1 on `group_0_1` and mismatches rank 2
    # on `group_0_2`.
    #
    #    rank 0      rank 1      rank 2
    #  +"a"-----+  +"a"-----+  +"b"-----+
    #  | 0_1[0] |  | 0_1[0] |  | 0_2[0] |
    #  | 0_2[0] |  +--------+  +--------+
    #  +--------+
    group_0_1 = ProcessGroupId([0, 1])
    group_0_2 = ProcessGroupId([0, 2])
    claim = _Claim(0, 1, "a")
    self.assertEmpty(self.submit(0, {group_0_1: claim, group_0_2: claim}))
    self.assertEmpty(self.submit(1, {group_0_1: claim}))
    self.assertEqual(
        self.submit(2, {group_0_2: _Claim(0, 1, "b")}),
        {0: False, 1: False, 2: False},
    )

  def test_mismatch_is_answered_without_waiting_for_all_ranks(self) -> None:
    group_0_1_2 = ProcessGroupId([0, 1, 2])
    self.assertEmpty(self.submit(0, {group_0_1_2: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(1, {group_0_1_2: _Claim(0, 1, "b")}), {0: False, 1: False}
    )
    self.assertEqual(
        self.submit(2, {group_0_1_2: _Claim(0, 1, "a")}), {2: False}
    )

  def test_rejected_requests_are_forgotten_once_all_ranks_have_a_response(
      self,
  ) -> None:
    group_0_1 = ProcessGroupId([0, 1])
    self.assertEmpty(self.submit(0, {group_0_1: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(1, {group_0_1: _Claim(0, 1, "b")}), {0: False, 1: False}
    )

    # The rejections can no longer keep a rank from being settled. The history
    # must not keep them, or it would grow with every rejected request.
    # pylint: disable=protected-access
    self.assertEmpty(self.consensus._rejections._rejected_parts)
    # pylint: enable=protected-access

  def test_graph_split_on_one_rank_then_matching_graphs(self) -> None:
    group_0_1_2 = ProcessGroupId([0, 1, 2])
    # Rank 0 has a graph break between the first two collectives, which ranks 1
    # and 2 run in one graph.
    self.assertEmpty(self.submit(0, {group_0_1_2: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(1, {group_0_1_2: _Claim(0, 2, "a")}), {0: False, 1: False}
    )
    self.assertEqual(
        self.submit(2, {group_0_1_2: _Claim(0, 2, "a")}), {2: False}
    )
    self.assertEqual(
        self.submit(0, {group_0_1_2: _Claim(1, 2, "a")}), {0: False}
    )

    # All ranks run the third collective in a graph of its own.
    self.assertEmpty(self.submit(1, {group_0_1_2: _Claim(2, 3, "a")}))
    self.assertEmpty(self.submit(2, {group_0_1_2: _Claim(2, 3, "a")}))
    self.assertEqual(
        self.submit(0, {group_0_1_2: _Claim(2, 3, "a")}),
        {0: True, 1: True, 2: True},
    )

  def test_mismatch_does_not_affect_a_later_graph_of_a_subgroup(self) -> None:
    #     rank 0        rank 1        rank 2
    #  +"a"-------+  +"b"-------+  +"a"-------+
    #  | 0_1_2[0] |  | 0_1_2[0] |  | 0_1_2[0] |
    #  +----------+  +----------+  +----------+
    #  +"a"-------+  +"a"-------+
    #  | 0_1[0]   |  | 0_1[0]   |
    #  +----------+  +----------+
    group_0_1_2 = ProcessGroupId([0, 1, 2])
    group_0_1 = ProcessGroupId([0, 1])
    self.assertEmpty(self.submit(0, {group_0_1_2: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(1, {group_0_1_2: _Claim(0, 1, "b")}), {0: False, 1: False}
    )

    # Ranks 0 and 1 submit their next graph while rank 2 has not submitted the
    # mismatching one yet. It is accepted once rank 2 has its response.
    self.assertEmpty(self.submit(0, {group_0_1: _Claim(0, 1, "a")}))
    self.assertEmpty(self.submit(1, {group_0_1: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(2, {group_0_1_2: _Claim(0, 1, "a")}),
        {0: True, 1: True, 2: False},
    )

  def test_requests_on_a_cycle_get_no_response_yet(self) -> None:
    # TODO: b/553328238 - Expect rejections once such requests are rejected.
    #
    # Fusing the graphs that run `0_1[0]` and `1_2[0]` would deadlock, because
    # rank 2 runs `0_2[0]` before the fused executable and rank 0 runs it
    # after. The requests are not accepted.
    #
    #    rank 0      rank 1      rank 2
    #  +"a"-----+  +"a"-----+  +"a"-----+
    #  | 0_1[0] |  | 0_1[0] |  | 0_2[0] |
    #  +--------+  | 1_2[0] |  +--------+
    #  +"a"-----+  +--------+  +"a"-----+
    #  | 0_2[0] |              | 1_2[0] |
    #  +--------+              +--------+
    group_0_1 = ProcessGroupId([0, 1])
    group_0_2 = ProcessGroupId([0, 2])
    group_1_2 = ProcessGroupId([1, 2])
    claim = _Claim(0, 1, "a")
    self.assertEmpty(self.submit(0, {group_0_1: claim}))
    self.assertEmpty(self.submit(1, {group_0_1: claim, group_1_2: claim}))
    self.assertEmpty(self.submit(2, {group_0_2: claim}))


class ContractViolationsTest(seed_test_utils.RepeatableTest):
  """Requests that do not continue where the previous ones of the rank ended.

  Ranks must not send such requests. Outside of debug mode they are not
  detected; these tests pin that they are then never fused without agreement of
  all ranks and never leave a rank without a response.
  """

  def setUp(self) -> None:
    super().setUp()
    # In debug mode the consensus refuses these requests.
    self.enter_context(
        mock.patch.object(
            handshake_messages,
            "is_torch_tpu_debug_mode",
            autospec=True,
            return_value=False,
        )
    )
    self.consensus = CollectiveHandshakeConsensus()
    self.group_0_1 = ProcessGroupId([0, 1])

  def submit(self, rank: int, claim: _Claim) -> Mapping[int, bool]:
    """Submits a request of `rank` that makes `claim` on `self.group_0_1`."""
    request = _request(rank, {self.group_0_1: claim})
    return _handshake_successes(self.consensus.submit(request))

  def test_requests_restarting_before_answered_collectives_are_not_fused(
      self,
  ) -> None:
    self.submit(0, _Claim(0, 3, "a"))
    self.submit(1, _Claim(0, 3, "a"))

    # Both ranks have a response for collective 2 already.
    self.assertEqual(self.submit(0, _Claim(2, 4, "b")), {0: False})
    self.assertEqual(self.submit(1, _Claim(2, 4, "b")), {1: False})
    # Requests that continue from there are fused again.
    self.assertEmpty(self.submit(0, _Claim(4, 5, "c")))
    self.assertEqual(self.submit(1, _Claim(4, 5, "c")), {0: True, 1: True})

  def test_requests_skipping_collectives_are_fused_if_all_ranks_agree(
      self,
  ) -> None:
    self.assertEmpty(self.submit(0, _Claim(5, 7, "a")))
    self.assertEqual(self.submit(1, _Claim(5, 7, "a")), {0: True, 1: True})

  def test_rank_skipping_past_pending_requests_does_not_leave_them_waiting(
      self,
  ) -> None:
    self.assertEmpty(self.submit(0, _Claim(0, 2, "a")))
    # Rank 1 will never claim collectives 0 and 1.
    self.assertEqual(self.submit(1, _Claim(6, 7, "a")), {0: False, 1: False})

  def test_rank_is_not_settled_by_the_response_to_its_own_rejection(
      self,
  ) -> None:
    group_0_1 = ProcessGroupId([0, 1])
    group_0_2 = ProcessGroupId([0, 2])
    group_1_3 = ProcessGroupId([1, 3])

    def submit(
        rank: int, process_group: ProcessGroupId, claim: _Claim
    ) -> Mapping[int, bool]:
      request = _request(rank, {process_group: claim})
      return _handshake_successes(self.consensus.submit(request))

    # Ranks 0 and 1 are rejected on `group_0_1`, then ranks 0 and 2 on
    # `group_0_2` while rank 0 is not settled.
    submit(0, group_0_1, _Claim(0, 1, "a"))
    submit(1, group_0_1, _Claim(0, 2, "a"))
    submit(0, group_0_2, _Claim(0, 1, "a"))
    submit(2, group_0_2, _Claim(0, 1, "b"))
    # The response to this request gives rank 0 the response that it was missing
    # on `group_0_1`. Rank 0 was not settled when it was rejected, so the
    # rejected cluster part on `group_0_1` keeps the one on `group_0_2` as a
    # previous part.
    self.assertEqual(submit(0, group_0_1, _Claim(1, 3, "a")), {0: False})
    # Rank 2 breaks the contract. It claims collective 0 of `group_0_2` again,
    # and a collective that rank 0 has no response for.
    self.assertEqual(submit(2, group_0_2, _Claim(0, 2, "a")), {2: False})
    self.assertEqual(submit(1, group_0_1, _Claim(2, 3, "a")), {1: False})

    # Rank 1 is not settled while rank 0 has no response on `group_0_2`.
    self.assertEmpty(submit(1, group_1_3, _Claim(0, 1, "a")))
    self.assertEmpty(submit(3, group_1_3, _Claim(0, 1, "a")))
    self.assertEqual(
        submit(0, group_0_2, _Claim(1, 2, "a")), {0: False, 1: True, 3: True}
    )


class ProcessGroupStateTest(seed_test_utils.RepeatableTest):
  """Tests for the state of one process group."""

  def setUp(self) -> None:
    super().setUp()
    self.state = _ProcessGroupState(ProcessGroupId([0, 1, 2]))

  def test_number_of_decided_collectives(self) -> None:
    self.state.advance_rank(0, 3)
    self.state.advance_rank(1, 1)

    self.assertEqual(self.state.num_decided(0), 3)
    self.assertCountEqual(self.state.ranks_without_response_before(2), [1, 2])

  def test_no_claim_is_expected_after_the_last_claimant_is_removed(
      self,
  ) -> None:
    self.state.add_claimant(0, _Claim(0, 2, "a"))
    self.state.remove_claimant(0)

    self.assertEmpty(self.state.claimants)
    self.assertFalse(self.state.mismatch_rule_applies(_Claim(0, 1, "b")))

  def test_advancing_to_an_earlier_collective_changes_nothing(self) -> None:
    self.state.advance_rank(0, 3)
    self.state.advance_rank(0, 1)

    self.assertEqual(self.state.num_decided(0), 3)

  def test_claim_is_expected_while_a_claimant_remains(self) -> None:
    claim = _Claim(0, 2, "a")
    self.state.add_claimant(0, claim)
    self.state.add_claimant(1, claim)
    self.state.remove_claimant(0)

    self.assertEqual(self.state.claimants, (1,))
    self.assertTrue(self.state.mismatch_rule_applies(_Claim(0, 2, "b")))

  def test_ranks_without_claim(self) -> None:
    claim = _Claim(0, 1, "a")
    self.state.add_claimant(2, claim)
    self.state.add_claimant(0, claim)
    self.assertEqual(self.state.ranks_without_claim(), [1])

    self.state.add_claimant(1, claim)
    self.assertEmpty(self.state.ranks_without_claim())
    self.assertEqual(self.state.claimants, (2, 0, 1))

  def test_mismatch_rule_applies_when_first_collective_is_decided(self) -> None:
    self.state.advance_rank(0, 2)

    self.assertTrue(self.state.mismatch_rule_applies(_Claim(1, 3, "a")))
    self.assertFalse(self.state.mismatch_rule_applies(_Claim(2, 3, "a")))


class MismatchRuleTest(seed_test_utils.RepeatableTest):
  """Tests for the condition of R1 Mismatch."""

  def setUp(self) -> None:
    super().setUp()
    self.state = _ProcessGroupState(ProcessGroupId([0, 1, 2]))

  def applies(self, rank: int, claim: _Claim) -> bool:
    """Returns whether R1 applies to a request of `rank` with `claim`."""
    request = _request(rank, {self.state.process_group: claim})
    return _mismatch_rule_applies(
        _RequestWithProcessGroupClaims(
            request, [_ProcessGroupClaim(self.state, claim)]
        )
    )

  def test_applies_once_a_rank_has_a_response_for_the_first_collective(
      self,
  ) -> None:
    self.state.advance_rank(0, 2)

    self.assertTrue(self.applies(1, _Claim(1, 3, "a")))
    self.assertFalse(self.applies(1, _Claim(2, 3, "a")))

  def test_applies_if_a_pending_request_makes_another_claim(self) -> None:
    self.state.add_claimant(0, _Claim(0, 2, "a"))

    self.assertFalse(self.applies(1, _Claim(0, 2, "a")))
    self.assertTrue(self.applies(1, _Claim(0, 1, "a")))
    self.assertTrue(self.applies(1, _Claim(1, 2, "a")))
    self.assertTrue(self.applies(1, _Claim(0, 2, "b")))

  def test_does_not_apply_once_no_request_is_pending(self) -> None:
    self.state.add_claimant(0, _Claim(0, 2, "a"))
    self.state.remove_claimant(0)

    self.assertFalse(self.applies(1, _Claim(0, 1, "b")))


class RejectionHistoryTest(seed_test_utils.RepeatableTest):
  """Tests for which ranks are settled after rejections."""

  def request(
      self, rank: int, claims: Mapping[_ProcessGroupState, _Claim]
  ) -> _RequestWithProcessGroupClaims:
    """Returns a request of `rank`.

    Args:
      rank: The rank of the request.
      claims: The claim of the request on every process group it claims, by the
        state of the process group.
    """
    request = _request(
        rank, {state.process_group: claim for state, claim in claims.items()}
    )
    process_group_claims = [
        _ProcessGroupClaim(state, claim) for state, claim in claims.items()
    ]
    return _RequestWithProcessGroupClaims(request, process_group_claims)

  def reject(
      self,
      history: _RejectionHistory,
      state: _ProcessGroupState,
      rank: int,
      start: int,
      end: int,
  ) -> None:
    """Rejects a request of `rank` for collectives `[start, end)` of `state`."""
    claim = _Claim(start, end, "fingerprint")
    history.record_rejection(self.request(rank, {state: claim}))
    state.advance_rank(rank, end)

  def test_rank_is_settled_once_all_ranks_have_a_response(self) -> None:
    history = _RejectionHistory()
    state = _ProcessGroupState(ProcessGroupId([0, 1, 2]))
    self.assertTrue(history.is_settled(0))

    self.reject(history, state, rank=0, start=0, end=1)
    self.assertFalse(history.is_settled(0))
    self.assertCountEqual(history.lagging_ranks(0), [1, 2])
    self.assertTrue(history.is_settled(1))

    state.advance_rank(1, 1)
    self.assertCountEqual(history.lagging_ranks(0), [2])

    state.advance_rank(2, 1)
    self.assertTrue(history.is_settled(0))

  def test_overlapping_rejections_form_one_rejected_part(self) -> None:
    history = _RejectionHistory()
    state = _ProcessGroupState(ProcessGroupId([0, 1]))

    self.reject(history, state, rank=0, start=0, end=1)
    self.reject(history, state, rank=1, start=0, end=2)

    # Rank 1 has a response for the collective of rank 0, but rank 0 has none
    # for the second collective of rank 1, which is in the same rejected
    # cluster part.
    self.assertCountEqual(history.lagging_ranks(0), [0])
    self.assertCountEqual(history.lagging_ranks(1), [0])

  def test_rank_rejected_before_it_is_settled_gets_a_previous_part(
      self,
  ) -> None:
    history = _RejectionHistory()
    first = _ProcessGroupState(ProcessGroupId([0, 1]))
    second = _ProcessGroupState(ProcessGroupId([0, 2]))

    self.reject(history, first, rank=0, start=0, end=1)
    self.reject(history, second, rank=0, start=0, end=1)
    self.assertCountEqual(history.lagging_ranks(0), [1, 2])

    # The rejected cluster part of the last rejection is closed, its previous
    # part is not.
    second.advance_rank(2, 1)
    self.assertCountEqual(history.lagging_ranks(0), [1])

    first.advance_rank(1, 1)
    self.assertTrue(history.is_settled(0))

  def test_forgetting_closed_rejected_parts_keeps_the_open_ones(self) -> None:
    history = _RejectionHistory()
    closed = _ProcessGroupState(ProcessGroupId([0, 1]))
    still_open = _ProcessGroupState(ProcessGroupId([2, 3]))
    self.reject(history, closed, rank=0, start=0, end=1)
    self.reject(history, still_open, rank=2, start=0, end=1)
    closed.advance_rank(1, 1)

    history.forget_rejected_parts_without_lagging_ranks()

    self.assertTrue(history.is_settled(0))
    self.assertCountEqual(history.lagging_ranks(2), [3])

  def test_merged_rejected_part_keeps_the_claims_of_both(self) -> None:
    history = _RejectionHistory()
    state = _ProcessGroupState(ProcessGroupId([0, 1, 2]))

    self.reject(history, state, rank=0, start=0, end=2)
    self.reject(history, state, rank=1, start=0, end=1)
    state.advance_rank(2, 1)

    # Every rank has a response for the collective of rank 1, but ranks 1 and 2
    # have none for the second collective of rank 0, which is in the same
    # rejected cluster part.
    self.assertCountEqual(history.lagging_ranks(1), [1, 2])

  def test_merged_rejected_part_keeps_the_previous_parts_of_both(
      self,
  ) -> None:
    history = _RejectionHistory()
    first = _ProcessGroupState(ProcessGroupId([0, 1]))
    second = _ProcessGroupState(ProcessGroupId([0, 2]))

    self.reject(history, first, rank=0, start=0, end=1)
    self.reject(history, second, rank=0, start=0, end=1)
    # Rank 2 joins the second rejected cluster part of rank 0, which has the
    # first one as a previous part.
    self.reject(history, second, rank=2, start=0, end=1)

    self.assertCountEqual(history.lagging_ranks(2), [1])

  def test_rejected_part_stays_a_previous_part_when_it_is_merged(self) -> None:
    history = _RejectionHistory()
    first = _ProcessGroupState(ProcessGroupId([0, 1]))
    second = _ProcessGroupState(ProcessGroupId([0, 2]))

    self.reject(history, first, rank=0, start=0, end=1)
    self.reject(history, second, rank=0, start=0, end=1)
    # Rank 1 adds a collective that rank 0 has no response for to the first
    # rejected cluster part, which is a previous part of the second one.
    self.reject(history, first, rank=1, start=0, end=2)
    second.advance_rank(2, 1)

    self.assertCountEqual(history.lagging_ranks(0), [0])

  def test_lagging_ranks_follow_the_chain_of_previous_parts(self) -> None:
    history = _RejectionHistory()
    states = [
        _ProcessGroupState(ProcessGroupId([0, other])) for other in (1, 2, 3)
    ]

    for state in states:
      self.reject(history, state, rank=0, start=0, end=1)
    states[1].advance_rank(2, 1)
    states[2].advance_rank(3, 1)

    self.assertCountEqual(history.lagging_ranks(0), [1])

  def test_forgetting_keeps_closed_rejected_parts_with_an_open_previous_part(
      self,
  ) -> None:
    history = _RejectionHistory()
    first = _ProcessGroupState(ProcessGroupId([0, 1]))
    second = _ProcessGroupState(ProcessGroupId([0, 2]))
    self.reject(history, first, rank=0, start=0, end=1)
    self.reject(history, second, rank=0, start=0, end=1)
    second.advance_rank(2, 1)

    history.forget_rejected_parts_without_lagging_ranks()

    self.assertCountEqual(history.lagging_ranks(0), [1])

  def test_rejection_overlapping_two_rejected_parts_merges_them(self) -> None:
    history = _RejectionHistory()
    first = _ProcessGroupState(ProcessGroupId([0, 2]))
    second = _ProcessGroupState(ProcessGroupId([1, 2]))
    self.reject(history, first, rank=0, start=0, end=2)
    self.reject(history, second, rank=1, start=0, end=1)

    # The request of rank 2 claims a collective of both rejected cluster parts.
    claim = _Claim(0, 1, "fingerprint")
    history.record_rejection(self.request(2, {first: claim, second: claim}))
    first.advance_rank(2, 1)
    second.advance_rank(2, 1)

    # Rank 2 has no response for the second collective of rank 0, which is now
    # in the rejected cluster part of rank 1 as well.
    self.assertCountEqual(history.lagging_ranks(0), [2])
    self.assertCountEqual(history.lagging_ranks(1), [2])


if __name__ == "__main__":
  absltest.main()
