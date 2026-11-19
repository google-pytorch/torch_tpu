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

"""Messages of the distributed handshake protocol.

Ranks send a `CollectiveHandshakeRequest` per compiled graph with collectives
and receive a `CollectiveHandshakeResponse`.
"""

from collections.abc import ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
import functools
import json
from typing import Any

from torch_tpu._internal import env
from torch_tpu._internal.distributed import process_group_utils

ProcessGroupId = process_group_utils.ProcessGroupId


def is_torch_tpu_debug_mode() -> bool:
  """Returns whether expensive debug-only checks are enabled.

  Controlled by the `TORCH_TPU_ENABLE_DEBUG_CHECKS` environment variable.
  Defaults to False.
  """
  return env.get_bool_env_once(
      "TORCH_TPU_ENABLE_DEBUG_CHECKS", default_value=False
  )


class ProcessGroupCollectiveCount:
  """Holds collective counts for a process group.

  Attributes:
    collective_count_before: Number of collectives that occurred on the rank so
      far for this process group.
    num_collectives_in_graph: Number of collectives in the FX graph for the
      compiled region for this process group. Zero for a process group that the
      rank used in earlier graphs but not in this one.
  """

  def __init__(
      self,
      collective_count_before: int,
      num_collectives_in_graph: int,
  ) -> None:
    """Initializes ProcessGroupCollectiveCount.

    Args:
      collective_count_before: Number of collectives that occurred on the rank
        so far for this process group. Must be non-negative.
      num_collectives_in_graph: Number of collectives in the FX graph for the
        compiled region for this process group. Must be non-negative.

    Raises:
      ValueError: If either count is negative.
    """
    if collective_count_before < 0:
      raise ValueError(
          "collective_count_before must be non-negative, got"
          f" {collective_count_before}."
      )

    if num_collectives_in_graph < 0:
      raise ValueError(
          "num_collectives_in_graph must be non-negative, got"
          f" {num_collectives_in_graph}."
      )

    self.collective_count_before = collective_count_before
    self.num_collectives_in_graph = num_collectives_in_graph

  def increment_num_collectives_in_graph(
      self, number_of_collectives: int
  ) -> None:
    """Updates collective counts.

    Adds the current `num_collectives_in_graph` to `collective_count_before`
    and sets `num_collectives_in_graph` to `number_of_collectives`.

    Args:
      number_of_collectives: Number of collectives in the new compiled region.
        Must be greater or equal to 0.

    Raises:
      ValueError: If number_of_collectives < 0.
    """
    if number_of_collectives < 0:
      raise ValueError(
          "number_of_collectives must be greater or equal to 0, got"
          f" {number_of_collectives}."
      )
    self.collective_count_before += self.num_collectives_in_graph
    self.num_collectives_in_graph = number_of_collectives

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ProcessGroupCollectiveCount):
      return False
    return (
        self.collective_count_before == other.collective_count_before
        and self.num_collectives_in_graph == other.num_collectives_in_graph
    )

  def __repr__(self) -> str:
    return (
        "ProcessGroupCollectiveCount("
        f"collective_count_before={self.collective_count_before!r}, "
        f"num_collectives_in_graph={self.num_collectives_in_graph!r})"
    )


class RankCollectiveCounts:
  """Holds collective counts on this rank, mapped by process group ranks."""

  def __init__(
      self,
      pg_to_count: (
          Mapping[ProcessGroupId, ProcessGroupCollectiveCount] | None
      ) = None,
  ) -> None:
    """Initializes RankCollectiveCounts.

    Args:
      pg_to_count: Optional dictionary mapping ProcessGroupId to
        ProcessGroupCollectiveCount. If None, an empty RankCollectiveCounts is
        created.
    """
    self._pg_to_count: dict[ProcessGroupId, ProcessGroupCollectiveCount] = {}
    if pg_to_count is not None:
      for pg, count in pg_to_count.items():
        self._pg_to_count[pg] = count

  def _increment_pg_collective_count(
      self,
      key: ProcessGroupId,
      number_of_collectives: int,
  ) -> None:
    """Increments the collective count for a process group.

    If the process group is already present, its collective counts are updated
    by adding the previous `num_collectives_in_graph` to
    `collective_count_before` and setting `num_collectives_in_graph` to
    `number_of_collectives`. If not present, a new
    `ProcessGroupCollectiveCount` is inserted with `collective_count_before=0`
    and `num_collectives_in_graph=number_of_collectives`.

    Args:
      key: ProcessGroupId identifying the process group.
      number_of_collectives: Number of collectives in the compiled graph for
        this process group. Must be non-negative.
    """
    if key in self._pg_to_count:
      self._pg_to_count[key].increment_num_collectives_in_graph(
          number_of_collectives
      )
    else:
      self._pg_to_count[key] = ProcessGroupCollectiveCount(
          collective_count_before=0,
          num_collectives_in_graph=number_of_collectives,
      )

  def increment_pg_collective_counts(
      self,
      pg_to_num_collectives: Mapping[ProcessGroupId, int],
  ) -> None:
    """Increments collective counts for multiple process groups.

    For each process group in `pg_to_num_collectives`, updates its collective
    counts with the corresponding number of collectives. For process groups
    that are currently in `_pg_to_count` but not in `pg_to_num_collectives`,
    updates their counts with `number_of_collectives=0`.

    Args:
      pg_to_num_collectives: A mapping of ProcessGroupId to the number of
        collectives for that process group.
    """
    for pg in list(self._pg_to_count.keys()):
      # We have to ensure that we progress the collective count even if the
      # process group is not in the new compiled region.
      if pg not in pg_to_num_collectives:
        self._pg_to_count[pg].increment_num_collectives_in_graph(0)

    # We do this loop to ensure new process groups are added to the
    # `_pg_to_count` map if not already present.
    for pg, num_collectives in pg_to_num_collectives.items():
      self._increment_pg_collective_count(pg, num_collectives)

  def to_json(self) -> list[dict[str, Any]]:
    """Serializes RankCollectiveCounts to a JSON-serializable list of dicts.

    Example:
      >>> counts = RankCollectiveCounts()
      >>> counts.increment_pg_collective_counts({ProcessGroupId([0, 1]): 2})
      >>> counts.to_json()
      [{'process_group': [0, 1], 'collective_count_before': 0,
      'num_collectives_in_graph': 2}]

    Returns:
      A list of dictionaries representing the process group collective counts.
    """
    return [
        {
            "process_group": list(pg.ranks),
            "collective_count_before": count.collective_count_before,
            "num_collectives_in_graph": count.num_collectives_in_graph,
        }
        for pg, count in self._pg_to_count.items()
    ]

  @classmethod
  def from_json(
      cls, data: list[dict[str, Any]] | str
  ) -> "RankCollectiveCounts":
    """Deserializes RankCollectiveCounts from a list of dicts or a JSON string.

    Args:
      data: A JSON string or a list of dictionaries representing the counts.

    Returns:
      A RankCollectiveCounts instance.
    """
    if isinstance(data, str):
      items = json.loads(data)
    else:
      items = data
    counts = cls()
    for item in items:
      pg_id = ProcessGroupId(item["process_group"])
      counts._pg_to_count[pg_id] = ProcessGroupCollectiveCount(
          collective_count_before=item["collective_count_before"],
          num_collectives_in_graph=item["num_collectives_in_graph"],
      )
    return counts

  def items(
      self,
  ) -> ItemsView[ProcessGroupId, ProcessGroupCollectiveCount]:
    """Returns an iterator over (key, value) pairs of _pg_to_count."""
    return self._pg_to_count.items()

  def keys(self) -> KeysView[ProcessGroupId]:
    """Returns an iterator over keys of _pg_to_count."""
    return self._pg_to_count.keys()

  def values(
      self,
  ) -> ValuesView[ProcessGroupCollectiveCount]:
    """Returns an iterator over values of _pg_to_count."""
    return self._pg_to_count.values()

  def __getitem__(
      self,
      key: ProcessGroupId,
  ) -> ProcessGroupCollectiveCount:
    """Gets the collective count for the given process group ID.

    Args:
      key: ProcessGroupId identifying the process group.

    Returns:
      ProcessGroupCollectiveCount for the process group.

    Raises:
      KeyError: If key is not present in _pg_to_count.
    """
    return self._pg_to_count[key]

  def __contains__(self, key: ProcessGroupId) -> bool:
    """Checks if key is in collective counts.

    Args:
      key: ProcessGroupId identifying the process group.

    Returns:
      True if key is present, False otherwise.
    """
    return key in self._pg_to_count

  def __len__(self) -> int:
    return len(self._pg_to_count)

  def __iter__(self) -> Iterator[ProcessGroupId]:
    return iter(self._pg_to_count)

  def __eq__(self, other: object) -> bool:
    """Checks whether two RankCollectiveCounts instances match."""
    if not isinstance(other, RankCollectiveCounts):
      return False
    return self._pg_to_count == other._pg_to_count

  def __repr__(self) -> str:
    return f"RankCollectiveCounts(pg_to_count={self._pg_to_count!r})"

  def collective_count(self, key: ProcessGroupId) -> int:
    """Returns total collective count for the given process group.

    Args:
      key: ProcessGroupId identifying the process group.

    Returns:
      The sum of collective_count_before and num_collectives_in_graph for the
      given process group.

    Raises:
      KeyError: If key is not present in _pg_to_count.
    """
    count = self._pg_to_count[key]
    return count.collective_count_before + count.num_collectives_in_graph


class CollectiveHandshakeRequest:
  """Request payload for collective handshake consensus.

  Attributes:
    pg_collective_counts: RankCollectiveCounts tracking collective counts per
      process group of the entire rank.
    executable_fingerprint: Unique identifier for the compiled executable.
    rank: The rank sending this handshake request. Has to be a member of
      participating_ranks.
  """

  def __init__(
      self,
      pg_collective_counts: RankCollectiveCounts,
      executable_fingerprint: str,
      rank: int,
  ) -> None:
    """Initializes CollectiveHandshakeRequest.

    Args:
      pg_collective_counts: Collective counts per process group of the rank.
      executable_fingerprint: Unique identifier for the compiled executable.
      rank: The rank sending this request. Must be non-negative and, when debug
        checks are enabled, a member of `participating_ranks`.

    Raises:
      ValueError: If `rank` is negative or, when debug checks are enabled, if
        `participating_ranks` is empty or does not contain `rank`.
    """
    # TODO(b/542976786): We currently Handshake only on global process
    # groups.
    if rank < 0:
      raise ValueError(f"rank must be non-negative, got {rank}.")

    self.pg_collective_counts = pg_collective_counts
    self.executable_fingerprint = executable_fingerprint
    self.rank = rank

    # These checks are O(world_size) and the coordinator constructs
    # world_size - 1 requests per handshake round (one per incoming message),
    # so running them unconditionally makes every round O(world_size^2) on the
    # coordinator's critical path. Requests are produced by
    # `split_compiler._submit_handshake` from validated ProcessGroupIds, so only
    # run them when debugging.
    if is_torch_tpu_debug_mode():
      if not self.participating_ranks:
        raise ValueError("participating_ranks cannot be empty.")

      if rank not in self.participating_ranks:
        raise ValueError(
            f"rank ({rank}) must be present in participating_ranks"
            f" ({self.participating_ranks})."
        )

  @functools.cached_property
  def participating_ranks(self) -> Sequence[int]:
    """Ranks participating in the handshake.

    This is a sorted list containing unique rank integers of the process groups
    in pg_collective_counts that have collectives in this compiled graph.
    Process groups without collectives in the graph are tracked only to keep
    their counts progressing and do not participate.
    """
    pgs = [
        pg
        for pg, count in self.pg_collective_counts.items()
        if count.num_collectives_in_graph > 0
    ]
    if len(pgs) == 1:
      return list(pgs[0].ranks)
    return sorted(set().union(*pgs))

  def to_bytes(self) -> bytes:
    """Serializes CollectiveHandshakeRequest to JSON UTF-8 encoded bytes.

    `participating_ranks` is derived from the process groups and is not sent.

    Note: The serialized format is not guaranteed to be stable across different
    versions of TorchTPU and should not be persisted or used across different
    TorchTPU versions.

    Returns:
      JSON UTF-8 encoded bytes representing the request.
    """
    return json.dumps({
        "executable_fingerprint": self.executable_fingerprint,
        "pg_collective_counts": [
            [
                pg.to_wire(),
                count.collective_count_before,
                count.num_collectives_in_graph,
            ]
            for pg, count in self.pg_collective_counts.items()
        ],
        "rank": self.rank,
    }).encode("utf-8")

  @classmethod
  def from_bytes(cls, data: bytes) -> "CollectiveHandshakeRequest":
    """Deserializes CollectiveHandshakeRequest from JSON UTF-8 encoded bytes.

    Args:
      data: JSON UTF-8 encoded bytes representing a CollectiveHandshakeRequest.

    Returns:
      A deserialized CollectiveHandshakeRequest instance.

    Raises:
      UnicodeDecodeError: If data is not valid UTF-8.
      json.JSONDecodeError: If data is not valid JSON.
      KeyError: If required fields are missing from the deserialized JSON
        object.
      ValueError: If deserialized values fail parameter validation in
        __init__.
    """
    obj = json.loads(data.decode("utf-8"))
    pg_to_count = {}
    for wire_pg, before, num_in_graph in obj["pg_collective_counts"]:
      pg_to_count[ProcessGroupId.from_wire(wire_pg)] = (
          ProcessGroupCollectiveCount(
              collective_count_before=before,
              num_collectives_in_graph=num_in_graph,
          )
      )
    return cls(
        pg_collective_counts=RankCollectiveCounts(pg_to_count),
        executable_fingerprint=obj["executable_fingerprint"],
        rank=obj["rank"],
    )

  def __repr__(self) -> str:
    return (
        "CollectiveHandshakeRequest("
        f"pg_collective_counts={self.pg_collective_counts!r}, "
        f"executable_fingerprint={self.executable_fingerprint!r}, "
        f"rank={self.rank!r})"
    )


class CollectiveHandshakeResponse:
  """Response payload for collective handshake consensus.

  Attributes:
    success: Whether the collective handshake succeeded (consensus achieved).
  """

  _TRUE_BYTES = b"\x01"
  _FALSE_BYTES = b"\x00"

  def __init__(self, success: bool) -> None:
    """Initializes CollectiveHandshakeResponse.

    Args:
      success: Whether the collective handshake succeeded.
    """
    self.success = success

  def __repr__(self) -> str:
    return f"CollectiveHandshakeResponse(success={self.success!r})"

  def to_bytes(self) -> bytes:
    """Serializes CollectiveHandshakeResponse to bytes.

    Note: The serialized format is not guaranteed to be stable across different
    versions of TorchTPU and should not be persisted or used across different
    TorchTPU versions.


    Returns:
      Bytes representing the response.
    """
    return (
        CollectiveHandshakeResponse._TRUE_BYTES
        if self.success
        else CollectiveHandshakeResponse._FALSE_BYTES
    )

  @classmethod
  def from_bytes(cls, data: bytes) -> "CollectiveHandshakeResponse":
    """Deserializes CollectiveHandshakeResponse from bytes.

    Args:
      data: Bytes representing a CollectiveHandshakeResponse.

    Returns:
      A deserialized CollectiveHandshakeResponse instance.

    Raises:
      ValueError: If data is not valid CollectiveHandshakeResponse bytes.
    """
    if data not in [
        CollectiveHandshakeResponse._TRUE_BYTES,
        CollectiveHandshakeResponse._FALSE_BYTES,
    ]:
      raise ValueError(
          f"Invalid CollectiveHandshakeResponse bytes: {data} (expected"
          f" {CollectiveHandshakeResponse._TRUE_BYTES} or"
          f" {CollectiveHandshakeResponse._FALSE_BYTES})."
      )
    return cls(success=(data == CollectiveHandshakeResponse._TRUE_BYTES))
