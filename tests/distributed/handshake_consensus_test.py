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

from absl.testing import absltest
from torch_tpu._internal.distributed import handshake_consensus
from torch_tpu._internal.distributed import process_group_utils
from tests import seed_test_utils

ProcessGroupId = process_group_utils.ProcessGroupId

# pylint: disable=protected-access
_Claim = handshake_consensus._ExpectedHandshakeClaim
_ProcessGroupState = handshake_consensus._ProcessGroupState
# pylint: enable=protected-access


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


if __name__ == "__main__":
  absltest.main()
