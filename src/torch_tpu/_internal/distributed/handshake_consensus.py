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

"""Consensus rules of the collective handshake for arbitrary process groups.

Two requests *overlap* if both claim a common collective of the same process
group. `_ExpectedHandshakeClaim` defines what a request claims.
"""

from collections.abc import Collection, Sequence
import dataclasses

from torch_tpu._internal.distributed import process_group_utils

ProcessGroupId = process_group_utils.ProcessGroupId


@dataclasses.dataclass(frozen=True)
class _ExpectedHandshakeClaim:
  """The claim of a request on one process group.

  A request *claims* the collectives `[start, end)` of every process group
  that its graph has collectives of.

  Attributes:
    start: First claimed collective of the process group.
    end: One past the last claimed collective of the process group.
    fingerprint: The executable fingerprint of the request.
  """

  start: int
  end: int
  fingerprint: str


class _ProcessGroupState:
  """Per-rank progress of the handshake on one process group.

  A collective of the process group is *decided* for a rank if a request of
  the rank that claims the collective has a response.

  A request is *pending* if it has been submitted and has no response yet. A
  rank is a *claimant* of the process group if it has a pending request that
  claims collectives of the process group.

  Attributes:
    process_group: The process group.
    _num_decided: Number of decided collectives of each rank, by rank.
    _claimants: The claimants, in the order in which they were added.
    _expected_claim: The claim that all claimants make. None without claimants.
  """

  def __init__(self, process_group: ProcessGroupId) -> None:
    self.process_group = process_group
    self._num_decided = dict.fromkeys(process_group.ranks, 0)
    self._claimants: list[int] = []
    self._expected_claim: _ExpectedHandshakeClaim | None = None

  def num_decided(self, rank: int) -> int:
    """Returns the number of collectives `rank` has a response for."""
    return self._num_decided[rank]

  def advance_rank(self, rank: int, end: int) -> None:
    """Records that `rank` has a response for the collectives before `end`."""
    # By contract, every request of a rank continues where its previous
    # request on the process group ended, so `end` is never before the current
    # value. `max` is defensive. It keeps a request that breaks the contract
    # from moving the rank back.
    self._num_decided[rank] = max(self._num_decided[rank], end)

  def ranks_without_response_before(self, end: int) -> Collection[int]:
    """Returns the ranks without a response for a collective before `end`."""
    return [
        rank for rank, decided in self._num_decided.items() if decided < end
    ]

  @property
  def claimants(self) -> Sequence[int]:
    """The claimants, in the order in which they were added."""
    return tuple(self._claimants)

  def ranks_without_claim(self) -> Sequence[int]:
    """Returns the ranks of the process group that are not claimants."""
    return [
        rank for rank in self.process_group.ranks if rank not in self._claimants
    ]

  def mismatch_rule_applies(self, claim: _ExpectedHandshakeClaim) -> bool:
    """Returns whether a condition of R1 Mismatch holds for `claim`.

    R1 Mismatch: reject a request if one of these holds for a process group
    it claims:
    * the pending request of another rank claims different collectives of
      that process group
    * the pending request of another rank claims collectives of that process
      group with a different fingerprint
    * another rank already has a response for the first collective that the
      request claims on that process group

    Args:
      claim: The claim of a request on the process group. The request may be
        pending or not.
    """
    if claim.start < max(self._num_decided.values()):
      # Another rank already has a response for the first claimed collective.
      return True
    # The pending request of another rank claims collectives of the process
    # group, but not exactly the same ones or not with the same fingerprint.
    return self._expected_claim not in (None, claim)

  def add_claimant(self, rank: int, claim: _ExpectedHandshakeClaim) -> None:
    """Records that the pending request of `rank` claims `claim`.

    Args:
      rank: A rank of the process group that is not a claimant.
      claim: The claim of its pending request on the process group.

    Raises:
      ValueError: If R1 Mismatch applies to a request that makes `claim`. Such
        a request must not become pending.
    """
    if self.mismatch_rule_applies(claim):
      raise ValueError(
          f"rank {rank} cannot claim {claim} of {self.process_group!r}"
          " because R1 Mismatch applies."
      )
    self._claimants.append(rank)
    if self._expected_claim is None:
      # The first claimant sets the claim that the later claimants make.
      self._expected_claim = claim

  def remove_claimant(self, rank: int) -> None:
    """Records that the request of the claimant `rank` is no longer pending."""
    self._claimants.remove(rank)
    if not self._claimants:
      self._expected_claim = None
