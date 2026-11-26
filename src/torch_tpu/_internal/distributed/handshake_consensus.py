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
get the same response. The consensus never holds a whole cluster. It tracks a
cluster through its parts, which are the pending cluster parts
(`_PendingClusterPart`) and the rejected cluster parts (`_RejectedClusterPart`).

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

  def overlaps(self, other: "_ExpectedHandshakeClaim") -> bool:
    """Returns whether the claims have a collective in common.

    Args:
      other: A claim on the same process group.
    """
    return self.start < other.end and other.start < self.end


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

  def overlaps(self, other: "_ProcessGroupClaim") -> bool:
    """Returns whether the claims have a collective in common."""
    return self.state is other.state and self.claim.overlaps(other.claim)

  def ranks_without_response(self) -> Collection[int]:
    """Returns the ranks without a response for a claimed collective."""
    return self.state.ranks_without_response_before(self.claim.end)


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


@dataclasses.dataclass(frozen=True)
class _PendingClusterPart:
  """A part of a cluster whose requests are pending.

  A *pending cluster part* is a set of pending requests that are connected by
  overlaps with other pending requests. It is a part of a cluster (see the
  module docstring) and not the whole cluster, because other requests of the
  cluster may already have a response or may not have been submitted yet.

  A pending cluster part is only valid until the next response, which changes
  the pending requests and what the states of their process groups report.

  Attributes:
    requests: The requests of the pending cluster part.
  """

  requests: Sequence[_RequestWithProcessGroupClaims]

  @property
  def ranks(self) -> Sequence[int]:
    """The ranks of the requests."""
    return [request.rank for request in self.requests]

  def missing_ranks(self) -> Collection[int]:
    """Returns the ranks that still have to claim one of its process groups."""
    missing_ranks: set[int] = set()
    for request in self.requests:
      for process_group_claim in request.process_group_claims:
        missing_ranks.update(process_group_claim.state.ranks_without_claim())
    return missing_ranks

  def is_complete(self) -> bool:
    """Returns whether no rank still has to claim one of its process groups."""
    return not self.missing_ranks()


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

  def pending_parts(self) -> Sequence[_PendingClusterPart]:
    """Returns the pending cluster parts that the pending requests form.

    Pending requests that claim the same process group claim the same
    collectives of it, so they overlap.
    """
    pending_parts = []
    visited: set[int] = set()
    for first in self._request_of_rank.values():
      if first.rank in visited:
        continue
      requests = []
      stack = [first]
      while stack:
        request = stack.pop()
        if request.rank in visited:
          continue
        visited.add(request.rank)
        requests.append(request)
        for process_group_claim in request.process_group_claims:
          stack.extend(
              self._request_of_rank[rank]
              for rank in process_group_claim.state.claimants
          )
      pending_parts.append(_PendingClusterPart(tuple(requests)))
    return pending_parts


class _RejectedClusterPart:
  """A part of a cluster whose requests are rejected.

  A *rejected cluster part* is a set of rejected requests connected by overlaps.
  It is a part of a cluster (see the module docstring) and not the whole
  cluster, because other requests of the cluster may not have been submitted
  yet. The class keeps the claims of the rejected requests, not the requests.

  A rejected cluster part is *open* if some rank of a process group claimed in
  it has no response for one of the claimed collectives.

  A rejected cluster part is *closed* if it is not open.

  A *previous part* of a rejected cluster part is the rejected cluster part of
  the previous rejected request of one of its ranks, if that rank was not
  settled (see `_RejectionHistory`) when it was rejected again.

  The *chain of previous parts* of a rejected cluster part consists of its
  previous parts, their previous parts, and so on.
  """

  def __init__(
      self,
      process_group_claims: Collection[_ProcessGroupClaim],
      previous_parts: Collection["_RejectedClusterPart"],
  ) -> None:
    """Initializes the instance.

    Args:
      process_group_claims: The claims of a rejected request.
      previous_parts: The previous parts of this one.
    """
    self._process_group_claims = list(process_group_claims)
    self._previous_parts = list(previous_parts)

  @property
  def process_group_claims(self) -> Sequence[_ProcessGroupClaim]:
    """The claims of the rejected requests of the rejected cluster part."""
    return tuple(self._process_group_claims)

  @property
  def previous_parts(self) -> Sequence["_RejectedClusterPart"]:
    """The previous parts of this one."""
    return tuple(self._previous_parts)

  def overlaps(
      self, process_group_claims: Collection[_ProcessGroupClaim]
  ) -> bool:
    """Returns whether one of `process_group_claims` overlaps this part."""
    for process_group_claim in process_group_claims:
      for own in self._process_group_claims:
        if process_group_claim.overlaps(own):
          return True
    return False

  def ranks_without_response(self) -> Collection[int]:
    """Returns the ranks that keep the rejected cluster part open."""
    ranks: set[int] = set()
    for process_group_claim in self._process_group_claims:
      ranks.update(process_group_claim.ranks_without_response())
    return ranks

  def absorb(self, other: "_RejectedClusterPart") -> None:
    """Adds the claims and the previous parts of `other` to this part."""
    self._process_group_claims.extend(other.process_group_claims)
    self._previous_parts.extend(other.previous_parts)

  def replace_previous_part(
      self, old: "_RejectedClusterPart", new: "_RejectedClusterPart"
  ) -> None:
    """Makes `new` a previous part wherever `old` is one."""
    self._previous_parts = [
        new if previous_part is old else previous_part
        for previous_part in self._previous_parts
    ]

  def keep_previous_parts(
      self, kept: Collection["_RejectedClusterPart"]
  ) -> None:
    """Removes the previous parts that are not in `kept`."""
    self._previous_parts = [
        previous_part
        for previous_part in self._previous_parts
        if previous_part in kept
    ]


class _RejectionHistory:
  """The rejected cluster parts that can keep a rank from being settled.

  The *lagging ranks* of a rank are the ranks that keep one of these rejected
  cluster parts open:
  * the rejected cluster part of its last rejected request
  * the rejected cluster parts in the chain of previous parts of that rejected
    cluster part

  A rank is *settled* if it has no lagging ranks.
  """

  def __init__(self) -> None:
    """Initializes the instance."""
    # The rejected cluster parts that may keep a rank from being settled.
    self._rejected_parts: list[_RejectedClusterPart] = []
    # The rejected cluster part of the last rejected request of each rank.
    # Without an entry, the rank is settled.
    self._last_rejected_part_per_rank: dict[int, _RejectedClusterPart] = {}

  def record_rejection(self, request: _RequestWithProcessGroupClaims) -> None:
    """Records that `request` was rejected.

    The claims of the request become a new rejected cluster part. If the rank of
    the request is not settled, the last rejected cluster part of the rank
    becomes a previous part of the new one. The rejected cluster parts that the
    request overlaps are merged into the new one.

    This has to be called before `advance_rank` of `request`, so that whether
    the rank is settled is decided without the response to `request`. The
    order only changes responses for requests that break the contract of
    `CollectiveHandshakeConsensus.submit`.
    """
    rank = request.rank
    previous_parts = (
        []
        if self.is_settled(rank)
        else [self._last_rejected_part_per_rank[rank]]
    )
    rejected_part = _RejectedClusterPart(
        request.process_group_claims, previous_parts
    )
    overlapped = [
        other
        for other in self._rejected_parts
        if other.overlaps(request.process_group_claims)
    ]
    self._rejected_parts.append(rejected_part)
    self._last_rejected_part_per_rank[rank] = rejected_part
    for other in overlapped:
      self._merge(rejected_part, other)

  def is_settled(self, rank: int) -> bool:
    """Returns whether `rank` is settled."""
    return not self.lagging_ranks(rank)

  def lagging_ranks(self, rank: int) -> Collection[int]:
    """Returns the ranks that keep `rank` from being settled."""
    rejected_part = self._last_rejected_part_per_rank.get(rank)
    if rejected_part is None:
      return set()
    return self._lagging_ranks_of_rejected_part(rejected_part)

  def forget_rejected_parts_without_lagging_ranks(self) -> None:
    """Forgets the rejected cluster parts that keep no rank from being settled.

    These are the closed rejected cluster parts whose chain of previous parts is
    closed as well. A closed rejected cluster part stays closed for requests
    that keep the contract of `CollectiveHandshakeConsensus.submit`.
    """
    # TODO: b/553328238 - This is suboptimal. The cost is quadratic in the
    # number of rejected cluster parts that are kept. Follow-up changes will
    # look into reducing it.
    self._rejected_parts = [
        rejected_part
        for rejected_part in self._rejected_parts
        if self._lagging_ranks_of_rejected_part(rejected_part)
    ]
    for rejected_part in self._rejected_parts:
      rejected_part.keep_previous_parts(self._rejected_parts)
    self._last_rejected_part_per_rank = {
        rank: rejected_part
        for rank, rejected_part in self._last_rejected_part_per_rank.items()
        if rejected_part in self._rejected_parts
    }

  def _merge(
      self, rejected_part: _RejectedClusterPart, other: _RejectedClusterPart
  ) -> None:
    """Moves the content of `other` into `rejected_part`, which replaces it."""
    rejected_part.absorb(other)
    self._rejected_parts.remove(other)
    for member in self._rejected_parts:
      member.replace_previous_part(other, rejected_part)
    for rank, last_part in self._last_rejected_part_per_rank.items():
      if last_part is other:
        self._last_rejected_part_per_rank[rank] = rejected_part

  def _lagging_ranks_of_rejected_part(
      self, rejected_part: _RejectedClusterPart
  ) -> Collection[int]:
    """Returns the ranks that keep `rejected_part` or its chain open.

    Args:
      rejected_part: A rejected cluster part. Its chain is its chain of previous
        parts.
    """
    lagging_ranks: set[int] = set()
    visited: set[_RejectedClusterPart] = set()
    stack = [rejected_part]
    while stack:
      member = stack.pop()
      if member in visited:
        continue
      visited.add(member)
      stack.extend(member.previous_parts)
      lagging_ranks.update(member.ranks_without_response())
    return lagging_ranks


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


def _agreement_rule_applies(
    pending_part: _PendingClusterPart, rejections: _RejectionHistory
) -> bool:
  """Returns whether R2 Agreement applies to `pending_part`.

  R2 Agreement: accept the requests of a pending cluster part K if both of
  these hold:
  * for any request R in K, any process group PG that R claims, and any rank r
    of PG, rank r has a request in K that claims PG (the requests in K that
    claim PG claim the same collectives of PG with the same fingerprint, or R1
    would have rejected them)
  * every rank of K is settled, as defined in `_RejectionHistory`

  Args:
    pending_part: A pending cluster part.
    rejections: The history of the rejected requests.
  """
  return pending_part.is_complete() and all(
      rejections.is_settled(rank) for rank in pending_part.ranks
  )


class CollectiveHandshakeConsensus:
  """Decides which compiled graphs of a distributed program are fused.

  The responses are computed as the requests arrive, by applying two rules
  until no other rules apply:
  * R1 Mismatch rejects requests (see `_mismatch_rule_applies`)
  * R2 Agreement accepts requests (see `_agreement_rule_applies`)

  TODO: b/553328238 - Reject the requests that R2 can never apply to. For now,
  they stay without a response. This happens to the requests of a cluster that
  is on a cycle of the precedes relation.

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
    self._rejections = _RejectionHistory()

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
    # Keeps the history from growing with every rejected request.
    self._rejections.forget_rejected_parts_without_lagging_ranks()
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
    """
    process_group_claims = []
    for process_group, claim in _claims(request).items():
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
    # The rules are applied in a loop, because a response can make a rule apply
    # to other pending requests:
    # * a rejection moves its rank past collectives that they claim (R1)
    # * any response can settle the ranks that they wait for (R2)
    while True:
      # `or` is evaluated lazily, so only the first rule that applies gives
      # responses in this iteration. R2 can be tried at any point. It never
      # applies to a pending cluster part with a request that R1 applies to.
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
    """Applies R2 Agreement to the pending cluster parts.

    Returns:
      The responses to the accepted requests, by rank.
    """
    in_agreement = [
        pending_part
        for pending_part in self._pending.pending_parts()
        if _agreement_rule_applies(pending_part, self._rejections)
    ]
    responses: dict[int, CollectiveHandshakeResponse] = {}
    for pending_part in in_agreement:
      for request in pending_part.requests:
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
    self._rejections.record_rejection(request)
    request.advance_rank()
    return CollectiveHandshakeResponse(success=False)
