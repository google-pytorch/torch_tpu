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

"""Distributed handshake protocol implementation using ZMQ and asyncio."""

import abc
import asyncio
import collections.abc
from collections.abc import Collection
import concurrent.futures
import dataclasses
import functools
import json
import os
import queue
import struct
import threading
from typing import Any, Coroutine, TypeVar, cast

from absl import logging
import portpicker
import torch.distributed as dist
import zmq
from zmq import error
import zmq.asyncio

ZMQError = error.ZMQError
Again = error.Again

_T = TypeVar("_T")

_COORDINATOR_RANK = 0  # Default coordinator rank, can be overridden.


def _get_handshake_timeout_s() -> int:
  """Returns the handshake socket timeout in seconds.

  The timeout specifies the maximum duration (in seconds) for send and
  receive operations on the handshake sockets.

  The timeout can be overridden using the
  `TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S` environment variable. Defaults to 60
  seconds.

  Returns:
    The timeout in seconds.

  Raises:
    ValueError: If `TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S` cannot be parsed as
      an integer or is negative.
  """
  env_val = os.environ.get("TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S", "60")
  try:
    timeout_s = int(env_val)
  except ValueError as e:
    raise ValueError(
        "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S must be an integer, got"
        f" {env_val!r}."
    ) from e
  if timeout_s < 0:
    raise ValueError(
        "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S must be non-negative, got"
        f" {timeout_s}."
    )
  return timeout_s


class ProcessGroupCollectiveCount:
  """Holds collective counts for a process group.

  Attributes:
    collective_count_before: Number of collectives that occurred on the rank so
      far for this process group.
    num_collectives_in_graph: Number of collectives in the FX graph for the
      compiled region for this process group. Must be strictly greater than 0.
  """

  def __init__(
      self,
      collective_count_before: int,
      num_collectives_in_graph: int,
  ) -> None:
    if collective_count_before < 0:
      raise ValueError(
          "collective_count_before must be non-negative, got"
          f" {collective_count_before}."
      )

    if num_collectives_in_graph <= 0:
      raise ValueError(
          "num_collectives_in_graph must be strictly positive, got"
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


class ProcessGroupId:
  """Identifier for a process group consisting of sorted, unique ranks.

  Attributes:
    ranks: A tuple of non-negative rank integers representing the process group
      members, sorted in ascending order with no duplicates.
    world_size: Optional world size. If provided, all ranks must be in [0,
      world_size).
  """

  def __init__(
      self,
      ranks: collections.abc.Sequence[int],
      world_size: int | None = None,
  ) -> None:
    if not ranks:
      raise ValueError("ranks cannot be empty.")

    if any(r < 0 for r in ranks):
      raise ValueError(f"All ranks must be non-negative, got {ranks}.")

    if world_size is not None:
      if world_size <= 0:
        raise ValueError(
            f"world_size must be strictly positive, got {world_size}."
        )
      if any(r >= world_size for r in ranks):
        raise ValueError(
            f"All ranks must be in [0, {world_size}), got {ranks}."
        )

    if len(ranks) != len(set(ranks)):
      raise ValueError(f"ranks must not contain duplicates, got {ranks}.")

    if list(ranks) != sorted(ranks):
      raise ValueError(f"ranks must be sorted in ascending order, got {ranks}.")

    self.ranks = tuple(ranks)
    self.world_size = world_size

  def __hash__(self) -> int:
    return hash(self.ranks)

  def __eq__(self, other: object) -> bool:
    if isinstance(other, ProcessGroupId):
      return self.ranks == other.ranks
    return False

  def __iter__(self) -> collections.abc.Iterator[int]:
    return iter(self.ranks)

  def __len__(self) -> int:
    return len(self.ranks)

  def __getitem__(self, index: int) -> int:
    return self.ranks[index]


class RankCollectiveCounts:
  """Holds collective counts on this rank, mapped by process group ranks."""

  def __init__(
      self,
      pg_to_count: (
          dict[ProcessGroupId, ProcessGroupCollectiveCount] | None
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
      pg_to_num_collectives: dict[ProcessGroupId, int],
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
  ) -> collections.abc.ItemsView[ProcessGroupId, ProcessGroupCollectiveCount]:
    """Returns an iterator over (key, value) pairs of _pg_to_count."""
    return self._pg_to_count.items()

  def keys(self) -> collections.abc.KeysView[ProcessGroupId]:
    """Returns an iterator over keys of _pg_to_count."""
    return self._pg_to_count.keys()

  def values(
      self,
  ) -> collections.abc.ValuesView[ProcessGroupCollectiveCount]:
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

  def __iter__(self) -> collections.abc.Iterator[ProcessGroupId]:
    return iter(self._pg_to_count)

  def __eq__(self, other: object) -> bool:
    """Checks whether two RankCollectiveCounts instances match."""
    if not isinstance(other, RankCollectiveCounts):
      return False
    return self._pg_to_count == other._pg_to_count

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


def should_handshake(
    pg_to_num_collectives: dict[ProcessGroupId, int],
) -> bool:
  """Checks whether a handshake should be performed for the given collective counts.

  Returns True if there is at least one collective and the only process group
  is the global process group. Otherwise returns False. The global process group
  behavior is temporary, and will be generalized in the near future.

  Args:
    pg_to_num_collectives: A dictionary mapping ProcessGroupId to collective
      count.

  Returns:
    True if handshake should be performed, False otherwise.
  """
  if not pg_to_num_collectives:
    return False

  world_size = dist.get_world_size()
  global_pg = ProcessGroupId(range(world_size), world_size=world_size)

  # TODO(b/542976786): We currently Handshake only on global process
  # groups.
  if len(pg_to_num_collectives) != 1 or global_pg not in pg_to_num_collectives:
    logging.warning(
        "We do not support non global PG handshakes: expected only the global"
        " process group, got %s.",
        list(pg_to_num_collectives.keys()),
    )
    return False

  return True


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
    # TODO(b/542976786): We currently Handshake only on global process
    # groups.
    if rank < 0:
      raise ValueError(f"rank must be non-negative, got {rank}.")

    self.pg_collective_counts = pg_collective_counts
    self.executable_fingerprint = executable_fingerprint
    self.rank = rank

    if not self.participating_ranks:
      raise ValueError("participating_ranks cannot be empty.")

    if any(r < 0 for r in self.participating_ranks):
      raise ValueError(
          "All elements in participating_ranks must be non-negative, got"
          f" {self.participating_ranks}."
      )

    if rank not in self.participating_ranks:
      raise ValueError(
          f"rank ({rank}) must be present in participating_ranks"
          f" ({self.participating_ranks})."
      )

  @functools.cached_property
  def participating_ranks(self) -> list[int]:
    """Ranks participating in the handshake.

    This is a sorted list containing unique rank integers derived from the
    process groups in pg_collective_counts.
    """
    return sorted(set().union(*self.pg_collective_counts.keys()))

  def to_bytes(self) -> bytes:
    """Serializes CollectiveHandshakeRequest to JSON UTF-8 encoded bytes.

    Note: The serialized format is not guaranteed to be stable across different
    versions of TorchTPU and should not be persisted or used across different
    TorchTPU versions.

    Returns:
      JSON UTF-8 encoded bytes representing the request.
    """
    return json.dumps({
        "executable_fingerprint": self.executable_fingerprint,
        "participating_ranks": self.participating_ranks,
        "pg_collective_counts": self.pg_collective_counts.to_json(),
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
    return cls(
        pg_collective_counts=RankCollectiveCounts.from_json(
            obj["pg_collective_counts"]
        ),
        executable_fingerprint=obj["executable_fingerprint"],
        rank=obj["rank"],
    )


class CollectiveHandshakeResponse:
  """Response payload for collective handshake consensus.

  Attributes:
    success: Whether the collective handshake succeeded (consensus achieved).
  """

  _TRUE_BYTES = b"\x01"
  _FALSE_BYTES = b"\x00"

  def __init__(self, success: bool) -> None:
    self.success = success

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


def _select_handshake_port(current_rank: int, coordinator_rank: int) -> int:
  """Selects the port for the Handshake server.

  Port selection strategy:
  1. If the default process group is initialized, the coordinator rank
     dynamically finds a free local port using portpicker. The coordinator rank
     broadcasts the chosen port.
  2. Otherwise, we raise a `RuntimeError`.

  Args:
    current_rank: The rank of the current process.
    coordinator_rank: The rank of the coordinator process.

  Returns:
    The selected port for the Handshake server.

  Raises:
    RuntimeError: If no port can be selected.
  """
  if not dist.is_initialized():
    raise RuntimeError(
        "the default process group is not initialized when a collective"
        " operation was invoked, please initialize the distributed process"
        " group by calling `torch.distributed.dist.init_process_group()`."
    )

  if current_rank == coordinator_rank:
    port = portpicker.pick_unused_port()
    port_list = [port]
  else:
    port_list = [0]
  dist.broadcast_object_list(port_list, src=coordinator_rank)
  return port_list[0]


@dataclasses.dataclass
class _CollectiveHandshakeRequestZMQEnvelope:
  """Envelope containing PyZMQ client routing ID and deserialized request.

  Attributes:
    client_id: Opaque routing identifier bytes automatically assigned by PyZMQ
      ROUTER socket representing the client connection.
    request: The deserialized CollectiveHandshakeRequest instance.
  """

  client_id: bytes
  request: CollectiveHandshakeRequest


class _CollectiveHandshakeConsensus:
  """Stores and manages incoming handshake requests from participating ranks.

  This does not have to include the coordinator rank if it isn't participating
  in the handshake.

  Attributes:
    _request_queues: A list of asyncio queues, one per rank, storing incoming
      handshake requests.
    _request_available: An asyncio event set whenever a new request is put into
      any queue.
  """

  def __init__(self, num_queues: int) -> None:
    """Initializes _CollectiveHandshakeConsensus.

    Args:
      num_queues: Total number of rank queues to initialize. Must be strictly
        positive.

    Raises:
      ValueError: If num_queues is not strictly positive.
    """
    if num_queues <= 0:
      raise ValueError(
          f"num_queues must be strictly positive, got {num_queues}."
      )
    self._request_queues: list[
        asyncio.Queue[_CollectiveHandshakeRequestZMQEnvelope]
    ] = [asyncio.Queue() for _ in range(num_queues)]
    self._request_available = asyncio.Event()

  async def put_request(
      self, rank: int, request: _CollectiveHandshakeRequestZMQEnvelope
  ) -> None:
    """Puts a request into the queue corresponding to the given rank.

    Args:
      rank: The rank index from which the request was received. Must be in range
        [0, num_queues).
      request: The request payload to enqueue.

    Raises:
      ValueError: If rank is out of range [0, num_queues).
    """
    if rank < 0 or rank >= len(self._request_queues):
      raise ValueError(
          f"rank ({rank}) must be in range [0, {len(self._request_queues)})."
      )
    await self._request_queues[rank].put(request)
    self._request_available.set()

  async def get_first_request(self) -> _CollectiveHandshakeRequestZMQEnvelope:
    """Waits for and returns the first available request across all queues.

    Returns:
      The request payload from the first non-empty rank queue.
    """
    # We wait until there is a request available on any of the queues.
    while not any(not q.empty() for q in self._request_queues):
      self._request_available.clear()
      await self._request_available.wait()

    # We find which queue has the request available and get only that request.
    first_rank = next(
        r for r, q in enumerate(self._request_queues) if not q.empty()
    )
    return self._request_queues[first_rank].get_nowait()

  async def get_from_ranks(
      self, ranks: Collection[int]
  ) -> list[_CollectiveHandshakeRequestZMQEnvelope]:
    """Concurrently awaits and returns requests from the specified ranks.

    Args:
      ranks: A collection of unique rank indices to await requests from. Each
        rank must be in the range [0, num_queues).

    Returns:
      A list of request payloads corresponding to each rank in `ranks`, where
      the i-th element in the returned list is the request received from the
      i-th rank in `ranks`.

    Raises:
      ValueError: If any rank in `ranks` is out of range [0, num_queues) or if
        `ranks` contains duplicate values.
    """
    if not ranks:
      return []
    if len(ranks) != len(set(ranks)):
      raise ValueError(f"ranks must not contain duplicates, got {ranks}.")
    for r in ranks:
      if r < 0 or r >= len(self._request_queues):
        raise ValueError(
            f"rank ({r}) in ranks must be in range [0,"
            f" {len(self._request_queues)})."
        )
    return await asyncio.gather(*[self._request_queues[r].get() for r in ranks])


class _LoopRunner:
  """Manages a dedicated asyncio event loop running on a background thread.

  Attributes:
    _loop: The asyncio event loop running on the background thread.
    _thread: The background daemon thread executing the event loop.
  """

  def __init__(self, thread_name: str) -> None:
    """Initializes _LoopRunner and starts the background event loop thread.

    Args:
      thread_name: Name assigned to the background thread.
    """
    self._loop_started = threading.Event()
    self._loop = asyncio.new_event_loop()
    self._thread = threading.Thread(
        target=self._run_loop,
        name=thread_name,
        daemon=True,
    )
    self._thread.start()
    # We block on event loop start to ensure that the event loop is ready to
    # accept tasks before returning.
    self._loop_started.wait()

  def _run_loop(self) -> None:
    """Runs the asyncio event loop until stopped and cleans up pending tasks."""
    asyncio.set_event_loop(self._loop)
    self._loop.call_soon(self._loop_started.set)
    try:
      self._loop.run_forever()
    finally:
      # Properly shutdown the event loop.
      try:
        pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
        for task in pending:
          task.cancel()
        if pending:
          self._loop.run_until_complete(
              asyncio.gather(*pending, return_exceptions=True)
          )
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Error cleaning up asyncio event loop: %s", e)
      finally:
        asyncio.set_event_loop(None)
        self._loop.close()

  def run_coroutine_async(
      self, coro: Coroutine[Any, Any, _T]
  ) -> asyncio.Task[_T]:
    """Asynchronously schedules a coroutine to run on the background event loop thread.

    Args:
      coro: The coroutine to schedule on the event loop.

    Returns:
      An asyncio.Task representing the result of the coroutine.
    """
    fut = asyncio.run_coroutine_threadsafe(self._create_task(coro), self._loop)
    return fut.result()

  async def _create_task(
      self, coro: Coroutine[Any, Any, _T]
  ) -> asyncio.Task[_T]:
    """Helper coroutine that creates and returns an asyncio.Task on the running event loop."""
    return self._loop.create_task(coro)

  def run_coroutine(self, coro: Any, timeout_s: float | None = None) -> Any:
    """Executes a coroutine synchronously on the background event loop thread and waits for its result.

    Args:
      coro: The coroutine to execute.
      timeout_s: Optional maximum duration in seconds to wait for the coroutine
        to complete.

    Returns:
      The return value of the completed coroutine.

    Raises:
      concurrent.futures.TimeoutError: If the coroutine does not complete within
        the specified timeout.
      Exception: Any exception raised by the executed coroutine.
    """
    fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
    return fut.result(timeout=timeout_s)

  def close(self, cleanup_coro: Any | None = None) -> None:
    """Stops the event loop, runs optional cleanup, and joins the background thread.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops if the loop is already stopped).

    Args:
      cleanup_coro: Optional coroutine to execute on the event loop before
        stopping it (e.g., to cancel and await background tasks).
    """
    if self._loop.is_running():
      if cleanup_coro is not None:
        try:
          self.run_coroutine(cleanup_coro, timeout_s=1.0)
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              "Error executing cleanup_coro during _LoopRunner shutdown: %s", e
          )
      self._loop.call_soon_threadsafe(self._loop.stop)
    elif cleanup_coro is not None:
      cleanup_coro.close()
    if self._thread.is_alive() and threading.current_thread() != self._thread:
      self._thread.join(timeout=5.0)


class _HandshakeBackend(abc.ABC):
  """Abstract base class for Handshake communication backends.

  Example usage:
    # Coordinator rank:
    server = _HandshakeServer(current_rank=0, port=12345, world_size=2)
    server.send(request)
    response = server.recv()
    server.close()

    # Worker rank:
    client = _HandshakeClient(port=12345)
    client.send(request)
    response = client.recv()
    client.close()
  """

  @abc.abstractmethod
  def send(self, request: CollectiveHandshakeRequest) -> None:
    """Sends request to the backend.

    Args:
      request: The CollectiveHandshakeRequest to send to the backend.
    """

  @abc.abstractmethod
  def recv(self) -> CollectiveHandshakeResponse:
    """Receives consensus result from the backend.

    Returns:
      The consensus CollectiveHandshakeResponse.
    """


def _validate_port_number(port: int) -> None:
  """Validates that the port number is in the valid 16 bit range.

  Args:
    port: The port number to validate.

  Raises:
    ValueError: If the port number is not in the valid 16 bit range.
  """
  try:
    struct.pack("!H", port)
  except struct.error as e:
    raise ValueError(
        f"port number {port} is not a valid 16 bit unsigned integer."
    ) from e


class _ZMQServer:
  """Handles PyZMQ ROUTER socket network communication for _HandshakeServer.

  Example usage:
    # Coordinator server (async):
    server = _ZMQServer(port=12345)
    envelope = await server.recv_request()
    # Process envelope.request...
    await server.send_reply(
        envelope.client_id, CollectiveHandshakeResponse(success=True)
    )

    # Client (synchronous):
    client = _ZMQClient(port=12345)
    client.send(request)
    response = client.recv()  # CollectiveHandshakeResponse(success=True)
    client.close()
    server.close()

  Attributes:
    _port: The TCP port number the server binds to.
    _context: The zmq.asyncio.Context managing socket lifecycle.
    _socket: The PyZMQ ROUTER socket listening for incoming client requests and
      sending replies.
  """

  def __init__(
      self,
      port: int,
      send_timeout_s: int | None = None,
      recv_timeout_s: int | None = None,
  ) -> None:
    """Initializes _ZMQServer and binds the ROUTER socket.

    Args:
      port: TCP port number to bind to.
      send_timeout_s: Optional maximum duration in seconds for send operations.
        If None, the send timeout is not set on the socket (PyZMQ default is no
        timeout).
      recv_timeout_s: Optional maximum duration in seconds for receive
        operations. If None, the receive timeout is not set on the socket (PyZMQ
        default is no timeout).

    Raises:
      ValueError: If `port` is invalid or if `send_timeout_s` /
        `recv_timeout_s` is negative.
    """
    _validate_port_number(port)
    self._port = port
    self._context = zmq.asyncio.Context()
    self._socket = self._context.socket(zmq.ROUTER)
    if send_timeout_s is not None:
      if send_timeout_s < 0:
        raise ValueError(
            f"send_timeout_s must be non-negative, got {send_timeout_s}."
        )
      self._socket.setsockopt(zmq.SNDTIMEO, send_timeout_s * 1000)
    if recv_timeout_s is not None:
      if recv_timeout_s < 0:
        raise ValueError(
            f"recv_timeout_s must be non-negative, got {recv_timeout_s}."
        )
      self._socket.setsockopt(zmq.RCVTIMEO, recv_timeout_s * 1000)
    self._socket.bind(f"tcp://*:{self._port}")

  async def recv_request(self) -> _CollectiveHandshakeRequestZMQEnvelope | None:
    """Receives a raw PyZMQ request and returns a _CollectiveHandshakeRequestZMQEnvelope."""
    frames = await self._socket.recv_multipart()
    if len(frames) != 2:
      logging.warning(
          "ZMQServer expected 2 frames [client_id, payload], got %d frames."
          " Frames: [%s]",
          len(frames),
          frames,
      )
      return None
    client_id = frames[0]
    payload = frames[1]
    req = CollectiveHandshakeRequest.from_bytes(payload)
    return _CollectiveHandshakeRequestZMQEnvelope(
        client_id=client_id, request=req
    )

  async def send_reply(
      self, client_id: bytes, response: CollectiveHandshakeResponse
  ) -> None:
    """Sends a PyZMQ multipart reply [client_id, serialized_response] to a client rank."""
    await self._socket.send_multipart([client_id, response.to_bytes()])

  def close(self) -> None:
    """Closes the server socket and terminates the ZMQ context.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops). Calling network operations (`recv_request`, `send_reply`) after
    `close` will raise a `zmq.ZMQError` because the underlying socket is closed.
    """
    if not self._socket.closed:
      self._socket.close(linger=0)
    if not self._context.closed:
      self._context.term()


class _HandshakeServer(_HandshakeBackend):
  """Singleton server backend for Handshake coordinator rank using _ZMQServer socket handler.

  Attributes:
    _instance: The singleton instance of _HandshakeServer per process, or None
      if not instantiated.
    _instance_lock: A threading lock ensuring thread-safe singleton
      initialization and reset.
    _zmq_server: The _ZMQServer instance handling PyZMQ socket communication
      with worker clients.
    _world_size: Total number of ranks participating in the distributed job.
    _current_rank: The current rank of the process running this server.
    _handshake_consensus: The _CollectiveHandshakeConsensus instance managing
      consensus queues.
    _local_consensus_queue: A queue holding consensus responses for the local
      rank.
    _loop_runner: The _LoopRunner managing the server background event loop.
    _listen_task: The background task running _listen_loop to process client
      requests.
    _consensus_task: The background task running _consensus_loop to process
      consensus requests.
    _closed: Whether the server has been closed.
    _initialized: Whether the server instance has been initialized.
  """

  _instance: "_HandshakeServer | None" = None
  _instance_lock = threading.Lock()

  _zmq_server: _ZMQServer
  _handshake_consensus: _CollectiveHandshakeConsensus
  _local_consensus_queue: queue.Queue[CollectiveHandshakeResponse]
  _world_size: int
  _current_rank: int
  _loop_runner: _LoopRunner
  _listen_task: asyncio.Task[None]
  _consensus_task: asyncio.Task[None]
  _closed: bool
  _initialized: bool

  def __new__(
      cls,
      current_rank: int,
      port: int,
      world_size: int,
  ) -> "_HandshakeServer":
    """Creates or returns the process-wide singleton _HandshakeServer instance.

    We use __new__ instead of __init__ to implement the singleton pattern.
    __new__ intercepts instance allocation under _instance_lock to ensure only
    a single instance is allocated and initialized per process across multiple
    calls.

    Timeouts:
    The server will not timeout on recv so that we allow idle waiting for
    incoming requests.
    The server will timeout on send after _get_handshake_timeout_s() to ensure
    that send operations do not block indefinitely if a client drops.

    Args:
      current_rank: The rank of the current process (must be the coordinator
        rank).
      port: TCP port number to bind the ZMQ server socket to.
      world_size: Total number of ranks participating in the distributed job.

    Returns:
      The singleton _HandshakeServer instance.
    """
    with cls._instance_lock:
      if cls._instance is None:
        inst = super().__new__(cls)
        inst._zmq_server = _ZMQServer(
            port=port,
            send_timeout_s=_get_handshake_timeout_s(),
        )
        inst._handshake_consensus = _CollectiveHandshakeConsensus(
            num_queues=world_size
        )
        # For the coordinator rank, we don't want to send the message over the
        # network, instead we put the response directly into the queue for it to
        # be consumed by the _HandshakeServer.recv() method.
        inst._local_consensus_queue = queue.Queue()
        inst._world_size = world_size
        inst._current_rank = current_rank
        inst._loop_runner = _LoopRunner(thread_name="HandshakeServerLoop")
        # LINT.IfChange(server_loop_tasks)
        inst._listen_task = inst._loop_runner.run_coroutine_async(
            inst._listen_loop()
        )
        inst._consensus_task = inst._loop_runner.run_coroutine_async(
            inst._consensus_loop()
        )
        # LINT.ThenChange(handshake.py:get_loop_tasks)
        inst._closed = False
        inst._initialized = True
        cls._instance = inst
      return cast("_HandshakeServer", cls._instance)

  # LINT.IfChange(get_loop_tasks)
  def _get_loop_tasks(self) -> list[asyncio.Task[Any]]:
    """Returns all background loop tasks managed by the server."""
    return [self._listen_task, self._consensus_task]

  # LINT.ThenChange(handshake.py:server_loop_tasks)

  async def _listen_loop(self) -> None:
    """Continuously receives requests and passes them to the consensus queues.

    Raises:
      Again: If socket receive times out while server is active.
      ZMQError: If socket encounters an error while server is active.
    """
    while True:
      try:
        envelope = await self._zmq_server.recv_request()
        if envelope is None:
          continue
        await self._handshake_consensus.put_request(
            envelope.request.rank, envelope
        )
      except asyncio.CancelledError:
        logging.info("_HandshakeServer _listen_loop cancelled.")
        raise
      except Again as e:
        logging.exception(
            "Socket error in _HandshakeServer _listen_loop: %s", e
        )
        raise e

  def _check_loop_task_errors(self) -> None:
    """Checks if any background loop task failed and raises the exception.

    Raises:
      RuntimeError: If any background loop task failed with an exception.
    """
    for task in self._get_loop_tasks():
      if task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
          raise RuntimeError("Handshake server loop task failed.") from exc

  def send(self, request: CollectiveHandshakeRequest) -> None:
    """Directly adds coordinator message to _CollectiveHandshakeConsensus.

    Args:
      request: Request to be added to the consensus queue.
    """
    self._check_loop_task_errors()
    envelope = _CollectiveHandshakeRequestZMQEnvelope(
        client_id=b"", request=request
    )
    self._loop_runner.run_coroutine(
        self._handshake_consensus.put_request(request.rank, envelope)
    )

  def recv(self) -> CollectiveHandshakeResponse:
    """Awaits consensus result from the background consensus loop.

    Blocks until a response is available or timeout expires.

    Returns:
      The CollectiveHandshakeResponse from the local response queue.

    Raises:
      RuntimeError: If a background loop task failed.
      asyncio.TimeoutError: If no response is received within the timeout.
    """
    self._check_loop_task_errors()
    try:
      return self._local_consensus_queue.get(timeout=_get_handshake_timeout_s())
    except Exception:  # pylint: disable=broad-except
      self._check_loop_task_errors()
      raise

  async def _cleanup_coro(self) -> None:
    """Cancels and awaits all background loop tasks."""
    for task in self._get_loop_tasks():
      if not task.done():
        task.cancel()
        try:
          await task
        except (asyncio.CancelledError, concurrent.futures.CancelledError) as e:
          logging.debug("Loop task cancelled during cleanup: %s", e)

  async def _reply_to_rank(
      self, rank: int, client_id: bytes, response: CollectiveHandshakeResponse
  ) -> None:
    """Sends reply to rank (via _local_consensus_queue if coordinator, else _ZMQServer)."""
    if rank == self._current_rank:
      self._local_consensus_queue.put(response)
    else:
      await self._zmq_server.send_reply(client_id, response)

  async def _gather_handshake_messages(
      self,
  ) -> tuple[dict[int, CollectiveHandshakeRequest], dict[int, bytes]]:
    """Gathers requests and client IDs from all participating ranks for a handshake step."""
    first_envelope = await self._handshake_consensus.get_first_request()
    first_msg = first_envelope.request
    received_msgs = {first_msg.rank: first_msg}
    client_ids = {first_msg.rank: first_envelope.client_id}

    remaining = set(first_msg.participating_ranks) - {first_msg.rank}

    results = await self._handshake_consensus.get_from_ranks(remaining)
    for env in results:
      received_msgs[env.request.rank] = env.request
      client_ids[env.request.rank] = env.client_id

    return received_msgs, client_ids

  async def _exhaust_ranks(
      self,
      received_msgs: dict[int, CollectiveHandshakeRequest],
  ) -> None:
    """Exhausts queues for ranks that have collective count < max collective count."""
    # TODO(b/542976786): The handshake currently supports only the global
    # process group. There is an explicit check against this in the invocation of the `Handshake.submit` function.
    # The reason we do not iterate pg_collective_counts here is that this exhaustion algorithm won't be applicable to non-global process groups.
    global_pg = ProcessGroupId(
        range(self._world_size), world_size=self._world_size
    )

    local_count = {
        r: m.pg_collective_counts.collective_count(global_pg)
        for r, m in received_msgs.items()
    }
    target_count = max(local_count.values())
    ranks_to_exhaust = [
        r for r, count in local_count.items() if count < target_count
    ]
    false_response = CollectiveHandshakeResponse(success=False)

    while ranks_to_exhaust:
      exhaust_envelopes = await self._handshake_consensus.get_from_ranks(
          ranks_to_exhaust
      )
      for envelope in exhaust_envelopes:
        await self._reply_to_rank(
            envelope.request.rank, envelope.client_id, false_response
        )
        count_after = envelope.request.pg_collective_counts.collective_count(
            global_pg
        )
        local_count[envelope.request.rank] = count_after
        target_count = max(target_count, count_after)

      ranks_to_exhaust = [
          r for r, count in local_count.items() if count < target_count
      ]

  async def _consensus_loop(self) -> None:
    """Continuously processes consensus rounds and replies to participating ranks."""
    while True:
      try:
        received_msgs, client_ids = await self._gather_handshake_messages()
        logging.debug("Received handshake messages: %s", received_msgs)
        first_msg = next(iter(received_msgs.values()))

        # Make consensus decision.
        first_fp = first_msg.executable_fingerprint
        all_match = all(
            m.executable_fingerprint == first_fp
            and m.pg_collective_counts == first_msg.pg_collective_counts
            for m in received_msgs.values()
        )
        consensus_response = CollectiveHandshakeResponse(success=all_match)

        for r in received_msgs.keys():
          await self._reply_to_rank(
              r, client_ids.get(r, b""), consensus_response
          )

        await self._exhaust_ranks(received_msgs)
      except asyncio.CancelledError:
        logging.info("_HandshakeServer _consensus_loop cancelled.")
        raise
      except Again as e:
        logging.debug(
            "Timeout expired in _HandshakeServer _consensus_loop: %s", e
        )
        raise e

  def close(self) -> None:
    """Closes server sockets, background tasks, and event loop thread.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops).
    """
    if getattr(self, "_closed", False):
      return
    self._closed = True

    if hasattr(self, "_loop_runner"):
      self._loop_runner.close(cleanup_coro=self._cleanup_coro())

    if hasattr(self, "_zmq_server"):
      self._zmq_server.close()

    self._initialized = False

  @classmethod
  def _reset_instance(cls) -> None:
    """Resets singleton instance for testing purposes."""
    with cls._instance_lock:
      inst = cls._instance
      if inst is not None:
        try:
          inst.close()
        except Exception as e:  # pylint: disable=broad-except
          logging.debug("Error resetting _HandshakeServer instance: %s", e)
        cls._instance = None


class _ZMQClient:
  """Handles PyZMQ DEALER socket network communication for _HandshakeClient.

  Attributes:
    _port: TCP port number of the coordinator handshake server.
    _master_addr: Hostname or IP address of the server to connect to. (defaults
      to `MASTER_ADDR` environment variable or 'localhost').
    _context: The zmq.Context managing socket lifecycle.
    _socket: The PyZMQ DEALER socket connected to the coordinator server.
  """

  def __init__(
      self,
      port: int,
      send_timeout_s: int | None = None,
      recv_timeout_s: int | None = None,
  ) -> None:
    """Initializes _ZMQClient and connects the DEALER socket.

    Args:
      port: TCP port number of the coordinator server.
      send_timeout_s: Optional maximum duration in seconds for send operations.
        If None, the send timeout is not set on the socket (PyZMQ default is no
        timeout).
      recv_timeout_s: Optional maximum duration in seconds for receive
        operations. If None, the receive timeout is not set on the socket (PyZMQ
        default is no timeout).

    Raises:
      ValueError: If `port` is invalid or if `send_timeout_s` /
        `recv_timeout_s` is negative.
    """
    _validate_port_number(port)
    self._port = port
    self._master_addr = os.environ.get("MASTER_ADDR", "localhost")

    self._context = zmq.Context()  # pyrefly: ignore[missing-attribute]
    self._socket = self._context.socket(zmq.DEALER)
    if send_timeout_s is not None:
      if send_timeout_s < 0:
        raise ValueError(
            f"send_timeout_s must be non-negative, got {send_timeout_s}."
        )
      self._socket.setsockopt(zmq.SNDTIMEO, send_timeout_s * 1000)
    if recv_timeout_s is not None:
      if recv_timeout_s < 0:
        raise ValueError(
            f"recv_timeout_s must be non-negative, got {recv_timeout_s}."
        )
      self._socket.setsockopt(zmq.RCVTIMEO, recv_timeout_s * 1000)
    self._socket.connect(f"tcp://{self._master_addr}:{self._port}")

  def send(self, request: CollectiveHandshakeRequest) -> None:
    """Sends serialized request synchronously to the server via PyZMQ socket."""
    self._socket.send(request.to_bytes())

  def recv(self) -> CollectiveHandshakeResponse:
    """Receives response synchronously from the server via PyZMQ socket."""
    return CollectiveHandshakeResponse.from_bytes(self._socket.recv())

  def close(self) -> None:
    """Closes the client socket and terminates the ZMQ context.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops). Calling network operations (`send`, `recv`) after `close` will
    raise
    a `zmq.ZMQError` because the underlying socket is closed.
    """
    if not self._socket.closed:
      self._socket.close(linger=0)
    if not self._context.closed:
      self._context.term()


class _HandshakeClient(_HandshakeBackend):
  """Singleton client backend for Handshake worker ranks using _ZMQClient socket handler.

  Attributes:
    _instance: _HandshakeClient instance.
    _instance_lock: A threading lock ensuring thread-safe access and
      initialization of the singleton instance.
    _zmq_client: The _ZMQClient handling PyZMQ socket communication with the
      coordinator server on that port.
    _closed: Whether this client instance has been closed.
    _initialized: Whether this client instance has been initialized.
  """

  _instance: "_HandshakeClient | None" = None
  _instance_lock = threading.Lock()

  _zmq_client: _ZMQClient
  _closed: bool
  _initialized: bool

  def __new__(
      cls,
      port: int,
  ) -> "_HandshakeClient":
    """Creates or returns the singleton _HandshakeClient instance.

    Timeouts:
    The client will timeout on send and recv after
    _get_handshake_timeout_s() to ensure that synchronous network calls to the
    coordinator server will time out and raise ZMQError (Again) instead of
    blocking indefinitely if the coordinator is unreachable.

    Args:
      port: TCP port number of the coordinator handshake server.

    Returns:
      The singleton _HandshakeClient instance for the specified port.
    """
    with cls._instance_lock:
      if cls._instance is None:
        inst = super().__new__(cls)
        timeout_s = _get_handshake_timeout_s()
        inst._zmq_client = _ZMQClient(
            port=port,
            send_timeout_s=timeout_s,
            recv_timeout_s=timeout_s,
        )
        inst._closed = False
        inst._initialized = True
        cls._instance = inst
      return cls._instance

  def send(self, request: CollectiveHandshakeRequest) -> None:
    """Sends request synchronously to the server.

    Args:
      request: The CollectiveHandshakeRequest to send to the coordinator server.
    """
    self._zmq_client.send(request)

  def recv(self) -> CollectiveHandshakeResponse:
    """Receives response synchronously from the server via PyZMQ socket.

    Returns:
      The CollectiveHandshakeResponse from the coordinator server.
    """
    return self._zmq_client.recv()

  def close(self) -> None:
    """Closes client sockets and marks instance uninitialized.

    Calling `close` multiple times is idempotent and safe (subsequent calls are
    no-ops).
    """
    if getattr(self, "_closed", False):
      return
    self._closed = True
    if hasattr(self, "_zmq_client"):
      self._zmq_client.close()
    self._initialized = False

  @classmethod
  def _reset_instance(cls) -> None:
    """Resets singleton instance for testing purposes."""
    with cls._instance_lock:
      inst = cls._instance
      if inst is not None:
        try:
          inst.close()
        except Exception as e:  # pylint: disable=broad-except
          logging.debug("Error resetting _HandshakeClient instance: %s", e)
        cls._instance = None


class Handshake:
  """Handshake protocol across distributed ranks using ZMQ backend.

  A single entry point for HandshakeServer / HandshakeClient.
  This class coordinates the creation of the correct backend instance based on
  the rank of the current process and the coordinator rank.
  This class allows user code to not have to distinct between the two roles.

  Attributes:
    _port_cache: Dictionary mapping coordinator ranks to handshake ports.
    _port_cache_lock: A threading lock ensuring thread-safe access to the port
      cache.
  """

  _port_cache: dict[int, int] = {}
  _port_cache_lock = threading.Lock()

  def __init__(
      self,
      current_rank: int | None = None,
      coordinator_rank: int = _COORDINATOR_RANK,
  ) -> None:
    if current_rank is None:
      current_rank = dist.get_rank()
    world_size = dist.get_world_size()

    self._current_rank = current_rank
    self._coordinator_rank = coordinator_rank
    self._world_size = world_size

    with self._port_cache_lock:
      if coordinator_rank in self._port_cache:
        port = self._port_cache[coordinator_rank]
      else:
        port = _select_handshake_port(current_rank, coordinator_rank)
        self._port_cache[coordinator_rank] = port

    if self._current_rank == self._coordinator_rank:
      self._backend: _HandshakeBackend = _HandshakeServer(
          current_rank=coordinator_rank, port=port, world_size=world_size
      )
    else:
      self._backend = _HandshakeClient(port=port)

  @classmethod
  def _reset_instances(cls) -> None:
    """Resets singleton instances and port cache for testing purposes."""
    with cls._port_cache_lock:
      cls._port_cache.clear()
    _HandshakeServer._reset_instance()  # pylint: disable=protected-access
    _HandshakeClient._reset_instance()  # pylint: disable=protected-access

  def submit(self, msg: CollectiveHandshakeRequest) -> bool:
    """Submits a CollectiveHandshakeRequest to the handshake protocol.

    Args:
      msg: CollectiveHandshakeRequest object to submit for consensus.

    Returns:
      True if executable fingerprint was identical across all participating
      ranks, else False.

    Raises:
      TypeError: If `msg` is not a `CollectiveHandshakeRequest`.
      RuntimeError: If subgroup communication is attempted.
      ValueError: If `_current_rank` is not in `msg.participating_ranks`.
    """
    if not isinstance(msg, CollectiveHandshakeRequest):
      raise TypeError(
          "Expected msg to be CollectiveHandshakeRequest, got"
          f" {type(msg).__name__}."
      )

    if len(msg.participating_ranks) != self._world_size:
      raise RuntimeError("We currently do not support subgroup communication.")

    if self._current_rank not in msg.participating_ranks:
      raise ValueError(
          f"Current rank {self._current_rank} is not in participating ranks"
          f" {msg.participating_ranks}."
      )

    self._backend.send(msg)
    response = self._backend.recv()
    return response.success

  def close(self) -> None:
    """Closes the underlying backend connection."""
    if hasattr(self, "_backend") and self._backend is not None:
      if hasattr(self._backend, "close"):
        self._backend.close()
