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

"""Distributed handshake protocol implementation."""

import abc
from concurrent import futures
import datetime
import os
import pickle
import socket
import threading
from typing import Any

import torch.distributed as dist
import torch.futures

Future = torch.futures.Future
timedelta = datetime.timedelta  # pylint: disable=invalid-name
ThreadPoolExecutor = futures.ThreadPoolExecutor
_COORDINATOR_RANK = 0  # Default coordinator rank, can be overridden.
_DEFAULT_TIMEOUT_S = 5.0  # Default timeout in seconds for handshake operations.


# Helper functions for formatting topic keys.


def _format_rank_topic_key(key: str, rank: int) -> str:
  """Formats the topic key for a specific rank's submission."""
  return f"{key}/rank_{rank}"


def _format_ack_topic_key(key: str, rank: int) -> str:
  """Formats the topic key for a specific rank's acknowledgment."""
  return f"{key}/ack_{rank}"


def _format_result_topic_key(key: str) -> str:
  """Formats the topic key for the consensus result."""
  return f"{key}/result"


def _select_handshake_port(rank: int, coordinator_rank: int) -> int:
  """Selects the port for the Handshake TCPStore server.

  Port selection strategy:
  1. If `TORCH_TPU_INTERNAL_HANDSHAKE_PORT` is set in the environment, that port
     is used as our first choice.
  2. Otherwise, if the default process group is initialized, the coordinator
     rank dynamically finds a free local port by binding an ephemeral socket
     (port 0). The coordinator rank broadcasts the chosen port.
  3. If neither of the above conditions is met, we raise a `RuntimeError`.

  Args:
    rank: The rank of the current process.
    coordinator_rank: The rank of the coordinator process.

  Returns:
    The selected port for the Handshake TCPStore server.

  Raises:
    RuntimeError: If no port can be selected.
  """
  # TODO(basioli): Add TORCH_TPU_INTERNAL_HANDSHAKE_PORT to documentation in
  # go/tt-knobs if this approach works
  port_str = os.environ.get("TORCH_TPU_INTERNAL_HANDSHAKE_PORT")
  if port_str is not None:
    return int(port_str)

  if dist.is_initialized():
    port_list = [0]
    if rank == coordinator_rank:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port_list = [s.getsockname()[1]]
    dist.broadcast_object_list(port_list, src=coordinator_rank)
    return port_list[0]

  raise RuntimeError(
      "cannot select Handshake port: 'TORCH_TPU_INTERNAL_HANDSHAKE_PORT'"
      " environment variable is not set and the default process group is not"
      " initialized."
  )


class HandshakeBackend(abc.ABC):
  """Abstract base class for Handshake communication backends."""

  @abc.abstractmethod
  def check(self, keys: list[str]) -> bool:
    """Checks if all given keys exist in the backend."""

  @abc.abstractmethod
  def set(self, key: str, value: str | bytes) -> None:
    """Sets a key-value pair in the backend."""

  @abc.abstractmethod
  def get(self, key: str) -> bytes:
    """Retrieves the value for a given key as bytes."""

  @abc.abstractmethod
  def wait(self, keys: list[str], timeout: timedelta) -> None:
    """Waits until all given keys exist in the backend or timeout occurs."""

  @abc.abstractmethod
  def delete(self, key: str) -> bool:
    """Deletes a key from the backend. Returns True if deleted, else False."""


class TCPStoreHandshakeBackend(HandshakeBackend):
  """TCPStore-backed communication for Handshake."""

  _stores: dict[int, dist.Store] = {}
  _store_lock = threading.Lock()

  @classmethod
  def _get_or_init_store(
      cls,
      rank: int,
      coordinator_rank: int,
  ) -> dist.Store:
    """Lazily initializes and returns the class-level TCPStore."""
    with cls._store_lock:
      if coordinator_rank not in cls._stores:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        handshake_port = _select_handshake_port(rank, coordinator_rank)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        cls._stores[coordinator_rank] = dist.TCPStore(
            master_addr,
            handshake_port,
            world_size,
            is_master=(rank == coordinator_rank),
            timeout=timedelta(seconds=_DEFAULT_TIMEOUT_S),
            wait_for_workers=False,
        )

      return cls._stores[coordinator_rank]

  def __init__(
      self,
      rank: int,
      coordinator_rank: int,
  ) -> None:
    self._tcp_store = self._get_or_init_store(rank, coordinator_rank)

  def check(self, keys: list[str]) -> bool:
    return self._tcp_store.check(keys)

  def set(self, key: str, value: str | bytes) -> None:
    self._tcp_store.set(key, value)

  def get(self, key: str) -> bytes:
    return self._tcp_store.get(key)

  def wait(self, keys: list[str], timeout: timedelta) -> None:
    self._tcp_store.wait(keys, timeout)

  def delete(self, key: str) -> bool:
    return self._tcp_store.delete_key(key)


class Handshake:
  """Handshake protocol across distributed ranks.

  If invoked on the coordinator rank, the coordinator rank waits to receive
  key value pairs from ALL participants in the handshake.
  """

  _executor = ThreadPoolExecutor(
      max_workers=8, thread_name_prefix="HandshakeWorker"
  )

  def __init__(
      self,
      current_rank: int | None = None,
      coordinator_rank: int = _COORDINATOR_RANK,
      participating_ranks: list[int] | None = None,
      backend: HandshakeBackend | None = None,
  ) -> None:
    """Initializes the instance and the underlying communication backend.

    Args:
      current_rank: The rank of the current process (defaults to RANK env var).
      coordinator_rank: The rank of the coordinator process (defaults to 0).
      participating_ranks: List of ranks participating in the handshake.
        Defaults to [0, 1, ..., world_size - 1] if None (where world_size is
        taken from WORLD_SIZE env var).
      backend: Optional HandshakeBackend instance (defaults to TCPStore
        backend).
    """
    if current_rank is None:
      current_rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if participating_ranks is None:
      participating_ranks = list(range(world_size))

    if not participating_ranks:
      raise ValueError("participating_ranks cannot be empty.")
    if coordinator_rank not in participating_ranks:
      raise ValueError(
          f"Coordinator rank {coordinator_rank} must be included in"
          f" participating ranks {participating_ranks}."
      )

    self._current_rank = current_rank
    self._coordinator_rank = coordinator_rank
    self._participating_ranks = list(participating_ranks)
    self._timeout_s = _DEFAULT_TIMEOUT_S

    if backend is None:
      backend = TCPStoreHandshakeBackend(
          self._current_rank,
          self._coordinator_rank,
      )
    self._backend = backend

  def submit(self, key: str, value: Any) -> Future[bool]:
    """Submits a key-value pair to the handshake protocol.

    Returns a future that all ranks can await on. The future holds a boolean
    that says if value was identical across all ranks.

    Args:
      key: The identifier for the handshake topic.
      value: The value submitted by this rank.

    Returns:
      A future resolving to True if value was identical across all ranks, else
      False.
    """
    if self._current_rank not in self._participating_ranks:
      raise ValueError(
          f"Rank {self._current_rank} is not in participating ranks"
          f" {self._participating_ranks}."
      )

    key_for_rank = _format_rank_topic_key(key, self._current_rank)
    if self._backend.check([key_for_rank]):
      raise ValueError(
          f"Rank {self._current_rank} has already submitted value for key"
          f" '{key}'."
      )

    self._backend.set(key_for_rank, pickle.dumps(value))

    fut: Future[bool] = Future()
    Handshake._executor.submit(self._handshake_protocol_impl, key, fut)
    return fut

  def _compute_and_broadcast_consensus(self, key: str) -> None:
    """Coordinator waits for all rank submissions and sets consensus result."""
    all_keys = [
        _format_rank_topic_key(key, r) for r in self._participating_ranks
    ]
    self._backend.wait(all_keys, timedelta(seconds=self._timeout_s))
    first_val = pickle.loads(self._backend.get(all_keys[0]))
    all_same = all(
        pickle.loads(self._backend.get(k)) == first_val for k in all_keys[1:]
    )
    self._backend.set(_format_result_topic_key(key), "1" if all_same else "0")

  def _get_consensus_result(self, key: str) -> bool:
    """Waits for and returns the consensus result from the coordinator."""
    result_key = _format_result_topic_key(key)
    self._backend.wait([result_key], timedelta(seconds=self._timeout_s))
    return self._backend.get(result_key) == b"1"

  def _send_result_ack(self, key: str) -> None:
    """Sends an acknowledgment confirming this rank received the result."""
    ack_key = _format_ack_topic_key(key, self._current_rank)
    self._backend.set(ack_key, "1")

  def _wait_for_acks_and_cleanup(self, key: str) -> None:
    """Coordinator waits for all acknowledgments and deletes topic keys."""
    # Coordinator waits for all ACKs before resolving so it never tears
    # down the server while worker ranks are still reading.
    all_acks = [
        _format_ack_topic_key(key, r) for r in self._participating_ranks
    ]
    self._backend.wait(all_acks, timedelta(seconds=self._timeout_s))

    # Clean up all keys associated with this handshake topics.
    self._backend.delete(_format_result_topic_key(key))
    for r in self._participating_ranks:
      self._backend.delete(_format_rank_topic_key(key, r))
    for r in self._participating_ranks:
      self._backend.delete(_format_ack_topic_key(key, r))

  def _handshake_protocol_impl(self, key: str, fut: Future[bool]) -> None:
    """Executes the handshake protocol."""
    try:
      if self._current_rank == self._coordinator_rank:
        self._compute_and_broadcast_consensus(key)

      is_success = self._get_consensus_result(key)
      self._send_result_ack(key)

      if self._current_rank == self._coordinator_rank:
        self._wait_for_acks_and_cleanup(key)

      fut.set_result(is_success)
    except Exception as e:  # pylint: disable=broad-except
      fut.set_exception(e)  # pytype: disable=wrong-arg-types
