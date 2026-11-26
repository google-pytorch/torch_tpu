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

A *cluster* is a set of requests connected by overlaps. If all requests keep the
contract of `CollectiveHandshakeConsensus.submit`, the requests of a cluster all
get the same response.

A cluster *precedes* another cluster if some rank submits a request of the
first cluster before a request of the second one. A cluster that contains two
requests of one rank precedes itself, which happens when ranks cut the same
collectives into graphs differently.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses

from torch_tpu._internal.distributed import handshake_messages
from torch_tpu._internal.distributed import process_group_utils

CollectiveHandshakeRequest = handshake_messages.CollectiveHandshakeRequest
CollectiveHandshakeResponse = handshake_messages.CollectiveHandshakeResponse
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


def _claims(
    request: CollectiveHandshakeRequest,
) -> Mapping[ProcessGroupId, _ExpectedHandshakeClaim]:
  """Returns the claim of `request` on every process group it claims."""
  # A request also lists the process groups that its rank used in earlier
  # graphs but not in this one, with zero collectives.
  return {
      process_group: _ExpectedHandshakeClaim(
          start=count.collective_count_before,
          end=count.collective_count_after,
          fingerprint=request.executable_fingerprint,
      )
      for process_group, count in request.pg_collective_counts.items()
      if count.num_collectives_in_graph
  }


class _ProcessGroupState:
  """Per-rank progress of the handshake on one process group.

  A collective of the process group is *decided* for a rank if a request of
  the rank that claims the collective has a response.

  A rank is a *claimant* of the process group if it has a pending request
  (see `_PendingRequests`) that claims collectives of the process group.

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

    R1 Mismatch is defined in `_mismatch_rule_applies`.

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


@dataclasses.dataclass(frozen=True)
class _ProcessGroupClaim:
  """The claim of a request on one process group, with the state of the group.

  Attributes:
    state: The state of the claimed process group.
    claim: The claim.
  """

  state: _ProcessGroupState
  claim: _ExpectedHandshakeClaim


@dataclasses.dataclass(frozen=True, eq=False)
class _RequestWithProcessGroupClaims:
  """A request together with its claims.

  Made once per `submit`, so that the claims of a request and the states of
  the process groups it claims are not looked up again for every rule.

  Attributes:
    request: The request as it was submitted.
    process_group_claims: The claim of the request on every process group it
      claims.
  """

  request: CollectiveHandshakeRequest
  process_group_claims: Sequence[_ProcessGroupClaim]

  @property
  def rank(self) -> int:
    """The rank that submitted the request."""
    return self.request.rank

  def advance_rank(self) -> None:
    """Records that the rank has a response for the claimed collectives."""
    for process_group_claim in self.process_group_claims:
      process_group_claim.state.advance_rank(
          self.rank, process_group_claim.claim.end
      )


class _PendingRequests:
  """The requests without a response.

  A request is *pending* if it has been submitted and has no response yet.

  A rank has at most one pending request. While its request is pending, the
  rank is a claimant of the process groups that the request claims.
  """

  def __init__(self) -> None:
    """Initializes the instance."""
    self._request_of_rank: dict[int, _RequestWithProcessGroupClaims] = {}

  def has_request_of(self, rank: int) -> bool:
    """Returns whether `rank` has a pending request."""
    return rank in self._request_of_rank

  def requests(self) -> Sequence[_RequestWithProcessGroupClaims]:
    """Returns the pending requests, in the order in which they were added."""
    return list(self._request_of_rank.values())

  def add_request_and_claimant(
      self, request: _RequestWithProcessGroupClaims
  ) -> None:
    """Makes `request` pending and its rank a claimant of what it claims.

    Args:
      request: A request that R1 Mismatch does not apply to.
    """
    self._request_of_rank[request.rank] = request
    for process_group_claim in request.process_group_claims:
      process_group_claim.state.add_claimant(
          request.rank, process_group_claim.claim
      )

  def remove_request_and_claimant(
      self, request: _RequestWithProcessGroupClaims
  ) -> None:
    """Undoes `add_request_and_claimant` for the pending `request`."""
    del self._request_of_rank[request.rank]
    for process_group_claim in request.process_group_claims:
      process_group_claim.state.remove_claimant(request.rank)


def _mismatch_rule_applies(request: _RequestWithProcessGroupClaims) -> bool:
  """Returns whether R1 Mismatch applies to `request`.

  R1 Mismatch: reject a request if one of these holds for a process group it
  claims:
  * the pending request of another rank claims different collectives of that
    process group
  * the pending request of another rank claims collectives of that process
    group with a different fingerprint
  * another rank already has a response for the first collective that the
    request claims on that process group

  The request can then no longer match the requests of the other ranks.

  Args:
    request: A request, pending or not.
  """
  return any(
      process_group_claim.state.mismatch_rule_applies(process_group_claim.claim)
      for process_group_claim in request.process_group_claims
  )


def _agreement_rule_applies(request: _RequestWithProcessGroupClaims) -> bool:
  """Returns whether R2 Agreement applies to `request`.

  R2 Agreement: accept a pending request R if, for any process group PG that R
  claims and any rank r of PG, rank r has a pending request that claims PG (the
  pending requests that claim PG claim the same collectives of PG with the same
  fingerprint, or R1 would have rejected them).

  Args:
    request: A pending request.
  """
  return not any(
      process_group_claim.state.ranks_without_claim()
      for process_group_claim in request.process_group_claims
  )


class CollectiveHandshakeConsensus:
  """Decides which compiled graphs of a distributed program are fused.

  The responses are computed as the requests arrive, by applying two rules
  until no other rules apply:
  * R1 Mismatch rejects requests (see `_mismatch_rule_applies`)
  * R2 Agreement accepts requests (see `_agreement_rule_applies`)

  For now, all requests have to claim collectives of the same process group.

  Guarantee: if all requests keep the contract of `submit`, the requests of a
  cluster are accepted if and only if they all have the same fingerprint and
  the cluster is not on a cycle of the precedes relation. The module docstring
  defines cluster and precedes.

  Instances are not thread-safe.

  Example:
    consensus = CollectiveHandshakeConsensus()
    consensus.submit(request_of_rank_0)  # {}
    consensus.submit(request_of_rank_1)  # Same graph on the group {0, 1}.
    # {0: CollectiveHandshakeResponse(success=True),
    #  1: CollectiveHandshakeResponse(success=True)}
  """

  def __init__(self) -> None:
    """Initializes the instance."""
    # The state of every process group ever claimed.
    self._states: dict[ProcessGroupId, _ProcessGroupState] = {}
    self._pending = _PendingRequests()

  def submit(
      self, request: CollectiveHandshakeRequest
  ) -> Mapping[int, CollectiveHandshakeResponse]:
    """Submits a request and returns the responses it enables.

    Contract:
      * a rank has at most one pending request
      * on every process group, the requests of a rank continue where its
        previous request on that group ended

    Args:
      request: The request of a rank that has no pending request. The rank has
        to be a member of the process groups it claims collectives of.

    Returns:
      The responses that can be given now, by rank, for this and for earlier
      requests. A request gets at most one response.

    Raises:
      ValueError: If the rank of the request has a pending request. In debug
        mode (`handshake_messages.is_torch_tpu_debug_mode`), also if the request
        does not continue the previous request of its rank on a process group,
        or if its rank is not a member of a process group it claims. The
        request is then ignored.
      NotImplementedError: If the requests so far, this one included, claim
        collectives of more than one process group. The request is then
        ignored.
    """
    self._validate(request)
    request_with_claims = self._with_process_group_claims(request)
    responses: dict[int, CollectiveHandshakeResponse] = {}
    # R1 is checked before the request becomes pending. A pending request that
    # R1 applies to would break R2. This is because R2 assumes that the pending
    # requests on a process group all make the same claim.
    if _mismatch_rule_applies(request_with_claims):
      responses[request.rank] = self._reject(request_with_claims)
    else:
      self._pending.add_request_and_claimant(request_with_claims)
    responses.update(self._apply_rules())
    return responses

  def _validate(self, request: CollectiveHandshakeRequest) -> None:
    """Raises ValueError if `request` is not valid (see `submit`)."""
    rank = request.rank
    if self._pending.has_request_of(rank):
      raise ValueError(f"rank {rank} already has a pending request.")
    if not handshake_messages.is_torch_tpu_debug_mode():
      return
    for process_group, claim in _claims(request).items():
      if rank not in process_group:
        raise ValueError(
            f"rank {rank} claims collectives of {process_group!r}, which it is"
            " not a member of."
        )
      state = self._states.get(process_group)
      decided = 0 if state is None else state.num_decided(rank)
      if claim.start != decided:
        raise ValueError(
            f"rank {rank} claims collectives of {process_group!r} from"
            f" {claim.start}, but its earlier requests end at {decided}."
        )

  def _with_process_group_claims(
      self, request: CollectiveHandshakeRequest
  ) -> _RequestWithProcessGroupClaims:
    """Returns `request` together with its claims.

    The state of a process group is created when the group is first claimed.

    Raises:
      NotImplementedError: If the requests so far, `request` included, claim
        collectives of more than one process group.
    """
    claims = _claims(request)
    # TODO: b/553328238 - Support requests on more than one process group. R1
    # and R2 only give the guarantee of the class while all requests claim
    # collectives of the same process group.
    if len(self._states.keys() | claims.keys()) > 1:
      raise NotImplementedError(
          "Requests on more than one process group are not supported yet."
      )
    process_group_claims = []
    for process_group, claim in claims.items():
      state = self._states.get(process_group)
      if state is None:
        state = self._states[process_group] = _ProcessGroupState(process_group)
      process_group_claims.append(_ProcessGroupClaim(state, claim))
    return _RequestWithProcessGroupClaims(request, tuple(process_group_claims))

  def _apply_rules(self) -> dict[int, CollectiveHandshakeResponse]:
    """Applies the rules to the pending requests until no other rules apply.

    Returns:
      The responses to the requests that are no longer pending, by rank.
    """
    responses: dict[int, CollectiveHandshakeResponse] = {}
    # The rules are applied in a loop, because a rejection moves its rank past
    # collectives that other pending requests claim, which makes R1 apply to
    # them.
    while True:
      # `or` is evaluated lazily, so only the first rule that applies gives
      # responses in this iteration. R2 can be tried at any point. It never
      # applies to a request that R1 applies to.
      new_responses = (
          self._apply_mismatch_rule() or self._apply_agreement_rule()
      )
      if not new_responses:
        # No rule applies in this iteration, so nothing changes before the
        # next `submit`.
        return responses
      responses.update(new_responses)

  def _apply_mismatch_rule(self) -> dict[int, CollectiveHandshakeResponse]:
    """Applies R1 Mismatch to the pending requests.

    Returns:
      The responses to the rejected requests, by rank.
    """
    mismatched = [
        request
        for request in self._pending.requests()
        if _mismatch_rule_applies(request)
    ]
    responses: dict[int, CollectiveHandshakeResponse] = {}
    for request in mismatched:
      responses[request.rank] = self._reject(request)
    return responses

  def _apply_agreement_rule(self) -> dict[int, CollectiveHandshakeResponse]:
    """Applies R2 Agreement to the pending requests.

    Returns:
      The responses to the accepted requests, by rank.
    """
    in_agreement = [
        request
        for request in self._pending.requests()
        if _agreement_rule_applies(request)
    ]
    responses: dict[int, CollectiveHandshakeResponse] = {}
    for request in in_agreement:
      responses[request.rank] = self._accept(request)
    return responses

  def _accept(
      self, request: _RequestWithProcessGroupClaims
  ) -> CollectiveHandshakeResponse:
    """Accepts the pending `request` and returns its response."""
    self._pending.remove_request_and_claimant(request)
    request.advance_rank()
    return CollectiveHandshakeResponse(success=True)

  def _reject(
      self, request: _RequestWithProcessGroupClaims
  ) -> CollectiveHandshakeResponse:
    """Rejects `request`, pending or not, and returns its response."""
    if self._pending.has_request_of(request.rank):
      self._pending.remove_request_and_claimant(request)
    request.advance_rank()
    return CollectiveHandshakeResponse(success=False)
