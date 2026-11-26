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
  """Base class of the tests that submit the requests of valid programs."""

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

  def test_mismatch_is_answered_without_waiting_for_all_ranks(self) -> None:
    group_0_1_2 = ProcessGroupId([0, 1, 2])
    self.assertEmpty(self.submit(0, {group_0_1_2: _Claim(0, 1, "a")}))
    self.assertEqual(
        self.submit(1, {group_0_1_2: _Claim(0, 1, "b")}), {0: False, 1: False}
    )
    self.assertEqual(
        self.submit(2, {group_0_1_2: _Claim(0, 1, "a")}), {2: False}
    )

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


if __name__ == "__main__":
  absltest.main()
