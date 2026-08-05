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
import concurrent.futures
import dataclasses
import json
import os
import threading
from typing import Any, Coroutine, TypeVar, cast

from absl import logging
import portpicker
import torch.distributed as dist
import zmq
import zmq.asyncio

_T = TypeVar("_T")
dataclass = dataclasses.dataclass
_COORDINATOR_RANK = 0  # Default coordinator rank, can be overridden.
_DEFAULT_TIMEOUT_MS = int(
    os.environ.get("TORCH_TPU_HANDSHAKE_TIMEOUT_MS", "60000")
)
_TRUE_BYTES = b"\x01"
_FALSE_BYTES = b"\x00"


@dataclass
class HandshakeMessage:
  """Message payload for handshake consensus."""

  executable_fingerprint: str
  is_last_frame_in_compiled_region: bool
  participating_ranks: list[int]
  rank: int

  def to_bytes(self) -> bytes:
    """Serializes HandshakeMessage to JSON UTF-8 encoded bytes."""
    return json.dumps({
        "executable_fingerprint": self.executable_fingerprint,
        "is_last_frame_in_compiled_region": (
            self.is_last_frame_in_compiled_region
        ),
        "participating_ranks": self.participating_ranks,
        "rank": self.rank,
    }).encode("utf-8")

  @classmethod
  def from_bytes(cls, data: bytes) -> "HandshakeMessage":
    """Deserializes HandshakeMessage from JSON UTF-8 encoded bytes."""
    obj = json.loads(data.decode("utf-8"))
    return cls(
        executable_fingerprint=obj["executable_fingerprint"],
        is_last_frame_in_compiled_region=obj[
            "is_last_frame_in_compiled_region"
        ],
        participating_ranks=obj["participating_ranks"],
        rank=obj["rank"],
    )


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
  if dist.is_initialized():
    if current_rank == coordinator_rank:
      port = portpicker.pick_unused_port()
      port_list = [port]
    else:
      port_list = [0]
    dist.broadcast_object_list(port_list, src=coordinator_rank)
    return port_list[0]

  raise RuntimeError(
      "cannot select Handshake port: default process group is not initialized."
  )


@dataclass
class _HandshakeMessageZMQEnvelope:
  """Envelope containing PyZMQ client routing ID and the deserialized HandshakeMessage."""

  client_id: bytes
  message: HandshakeMessage


class _HandshakeConsensus:
  """Stores and manages incoming handshake messages from participating ranks."""

  def __init__(self, num_queues: int) -> None:
    self._message_queues: list[asyncio.Queue[_HandshakeMessageZMQEnvelope]] = [
        asyncio.Queue() for _ in range(num_queues)
    ]
    self._message_available = asyncio.Event()

  async def put_message(
      self, rank: int, envelope: _HandshakeMessageZMQEnvelope
  ) -> None:
    await self._message_queues[rank].put(envelope)
    self._message_available.set()

  async def get_first_message(self) -> _HandshakeMessageZMQEnvelope:
    """Waits for and returns the first available _HandshakeMessageZMQEnvelope across all queues."""
    # We wait until there is a message available on any of the queues.
    while not any(not q.empty() for q in self._message_queues):
      self._message_available.clear()
      await self._message_available.wait()

    # We find which queue has the message available and get only that message.
    first_rank = next(
        r for r, q in enumerate(self._message_queues) if not q.empty()
    )
    return self._message_queues[first_rank].get_nowait()

  async def get_from_ranks(
      self, ranks: list[int] | set[int]
  ) -> list[_HandshakeMessageZMQEnvelope]:
    """Concurrently awaits and returns messages from the specified ranks."""
    if not ranks:
      return []
    return list(
        await asyncio.gather(*[self._message_queues[r].get() for r in ranks])
    )


class _LoopRunner:
  """Manages a dedicated asyncio event loop running on a background thread."""

  def __init__(self, thread_name: str) -> None:
    self._loop = asyncio.new_event_loop()
    self._thread = threading.Thread(
        target=self._run_loop,
        name=thread_name,
        daemon=True,
    )
    self._thread.start()

  def _run_loop(self) -> None:
    asyncio.set_event_loop(self._loop)
    self._loop.run_forever()

  def run_on_event_loop(
      self, coro: Coroutine[Any, Any, _T]
  ) -> concurrent.futures.Future[_T]:
    """Schedules a coroutine to run on the background event loop thread."""
    return asyncio.run_coroutine_threadsafe(coro, self._loop)

  def run_coroutine(self, coro: Any, timeout: float | None = None) -> Any:
    fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
    return fut.result(timeout=timeout)

  def close(self, cleanup_coro: Any | None = None) -> None:
    if self._loop.is_running():
      if cleanup_coro is not None:
        try:
          self.run_coroutine(cleanup_coro, timeout=1.0)
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              "Error executing cleanup_coro during _LoopRunner shutdown: %s", e
          )
      self._loop.call_soon_threadsafe(self._loop.stop)


class _HandshakeBackend(abc.ABC):
  """Abstract base class for Handshake communication backends."""

  @abc.abstractmethod
  def send(self, data: bytes) -> None:
    """Sends serialized data to the backend."""

  @abc.abstractmethod
  def recv(self) -> bytes:
    """Receives consensus result from the backend."""


class _ZMQServer:
  """Handles PyZMQ ROUTER socket network communication for _HandshakeServer."""

  def __init__(self, port: int) -> None:
    self._port = port
    self._master_addr = os.environ.get("MASTER_ADDR", "localhost")

    self._context = zmq.asyncio.Context()
    self._socket = self._context.socket(zmq.ROUTER)
    self._socket.setsockopt(zmq.RCVTIMEO, _DEFAULT_TIMEOUT_MS)
    self._socket.setsockopt(zmq.SNDTIMEO, _DEFAULT_TIMEOUT_MS)
    self._socket.bind(f"tcp://*:{self._port}")

  async def recv_message(self) -> _HandshakeMessageZMQEnvelope | None:
    """Receives a raw PyZMQ message and returns a _HandshakeMessageZMQEnvelope."""
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
    msg = HandshakeMessage.from_bytes(payload)
    return _HandshakeMessageZMQEnvelope(client_id=client_id, message=msg)

  async def send_reply(self, client_id: bytes, data: bytes) -> None:
    """Sends a PyZMQ multipart reply [client_id, data] to a client rank."""
    await self._socket.send_multipart([client_id, data])

  def close(self) -> None:
    if hasattr(self, "_socket") and not self._socket.closed:
      self._socket.close(linger=0)
    if hasattr(self, "_context"):
      self._context.term()


class _HandshakeServer(_HandshakeBackend):
  """Singleton server backend for Handshake coordinator rank using _ZMQServer socket handler."""

  _instance: "_HandshakeServer | None" = None
  _instance_lock = threading.Lock()

  def __new__(
      cls,
      current_rank: int,
      port: int,
      world_size: int,
  ) -> "_HandshakeServer":
    del port, world_size, current_rank
    with cls._instance_lock:
      if cls._instance is None:
        inst = super().__new__(cls)
        inst._initialized = False
        cls._instance = inst
      return cast("_HandshakeServer", cls._instance)

  def __init__(
      self,
      current_rank: int,
      port: int,
      world_size: int,
  ) -> None:
    if getattr(self, "_initialized", False):
      return
    self._zmq_server = _ZMQServer(port=port)
    self.handshake_consensus = _HandshakeConsensus(num_queues=world_size)
    self._world_size = world_size
    self._current_rank = current_rank

    # For the coordinator rank, we don't want to send the message over the
    # network, insteads we put the message directly into the queue for it to be
    # consumed by the _HandshakeServer.recv() method.
    self._local_consensus_queue: asyncio.Queue[bytes] = asyncio.Queue()

    self._loop_runner = _LoopRunner(thread_name="HandshakeServerLoop")
    self._listen_task = self._loop_runner.run_on_event_loop(self._listen_loop())
    self._consensus_task = self._loop_runner.run_on_event_loop(
        self._consensus_loop()
    )
    self._closed = False
    self._initialized = True

  async def _listen_loop(self) -> None:
    while True:
      try:
        envelope = await self._zmq_server.recv_message()
        if envelope is None:
          continue
        await self.handshake_consensus.put_message(
            envelope.message.rank, envelope
        )
      except asyncio.CancelledError:
        logging.info("_HandshakeServer _listen_loop cancelled.")
        break
      except Exception as e:  # pylint: disable=broad-except
        if getattr(self, "_closed", False):
          break
        logging.exception("Error in _HandshakeServer _listen_loop: %s", e)

  def send(self, data: bytes) -> None:
    """Directly adds coordinator message to _HandshakeConsensus."""
    msg = HandshakeMessage.from_bytes(data)
    envelope = _HandshakeMessageZMQEnvelope(client_id=b"", message=msg)
    self._loop_runner.run_coroutine(
        self.handshake_consensus.put_message(msg.rank, envelope)
    )

  def recv(self) -> bytes:
    """Awaits consensus result from the background consensus loop."""
    return self._loop_runner.run_coroutine(self._local_consensus_queue.get())

  async def _reply_to_rank(
      self, rank: int, client_id: bytes, data: bytes
  ) -> None:
    """Sends reply bytes to rank (via _local_consensus_queue if coordinator, else _ZMQServer)."""
    if rank == self._current_rank:
      await self._local_consensus_queue.put(data)
    else:
      await self._zmq_server.send_reply(client_id, data)

  async def _gather_handshake_messages(
      self,
  ) -> tuple[dict[int, HandshakeMessage], dict[int, bytes]]:
    """Gathers messages and client IDs from all participating ranks for a handshake step."""
    first_envelope = await self.handshake_consensus.get_first_message()
    first_msg = first_envelope.message
    received_msgs = {first_msg.rank: first_msg}
    client_ids = {first_msg.rank: first_envelope.client_id}

    participating = set(first_msg.participating_ranks)
    remaining = [r for r in participating if r not in received_msgs]

    results = await self.handshake_consensus.get_from_ranks(remaining)
    for env in results:
      received_msgs[env.message.rank] = env.message
      client_ids[env.message.rank] = env.client_id

    return received_msgs, client_ids

  async def _exhaust_ranks(
      self,
      participating_ranks: set[int],
      received_msgs: dict[int, HandshakeMessage],
  ) -> None:
    """Exhausts queues for ranks that did not send is_last_frame_in_compiled_region=True."""
    if not any(
        m.is_last_frame_in_compiled_region for m in received_msgs.values()
    ):
      return

    ranks_to_exhaust = [
        r
        for r in participating_ranks
        if not received_msgs[r].is_last_frame_in_compiled_region
    ]
    false_bytes = _FALSE_BYTES

    while ranks_to_exhaust:
      exhaust_envelopes = await self.handshake_consensus.get_from_ranks(
          ranks_to_exhaust
      )
      next_ranks_to_exhaust = []
      for envelope in exhaust_envelopes:
        await self._reply_to_rank(
            envelope.message.rank, envelope.client_id, false_bytes
        )
        if not envelope.message.is_last_frame_in_compiled_region:
          next_ranks_to_exhaust.append(envelope.message.rank)
      ranks_to_exhaust = next_ranks_to_exhaust

  async def _consensus_loop(self) -> None:
    while True:
      try:
        received_msgs, client_ids = await self._gather_handshake_messages()
        participating_ranks = set(
            next(iter(received_msgs.values())).participating_ranks
        )

        # Make consensus decision.
        first_fp = next(iter(received_msgs.values())).executable_fingerprint
        all_match = all(
            m.executable_fingerprint == first_fp for m in received_msgs.values()
        )
        consensus_bytes = _TRUE_BYTES if all_match else _FALSE_BYTES

        for r in participating_ranks:
          await self._reply_to_rank(r, client_ids.get(r, b""), consensus_bytes)

        await self._exhaust_ranks(participating_ranks, received_msgs)
      except asyncio.CancelledError:
        logging.info("_HandshakeServer _consensus_loop cancelled.")
        break
      except Exception as e:  # pylint: disable=broad-except
        if getattr(self, "_closed", False):
          break
        logging.exception("Error in _HandshakeServer _consensus_loop: %s", e)

  def close(self) -> None:
    """Closes server sockets, background tasks, and event loop thread."""
    if getattr(self, "_closed", False):
      return
    self._closed = True

    if hasattr(self, "_listen_task"):
      self._listen_task.cancel()
    if hasattr(self, "_consensus_task"):
      self._consensus_task.cancel()

    if hasattr(self, "_zmq_server"):
      self._zmq_server.close()

    if hasattr(self, "_loop_runner"):
      self._loop_runner.close()
    self._initialized = False

  @classmethod
  def _reset_instances(cls) -> None:
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
  """Handles PyZMQ DEALER socket network communication for _HandshakeClient."""

  def __init__(self, port: int) -> None:
    self._port = port
    self._master_addr = os.environ.get("MASTER_ADDR", "localhost")

    self._context = zmq.Context()
    self._socket = self._context.socket(zmq.DEALER)
    self._socket.setsockopt(zmq.RCVTIMEO, _DEFAULT_TIMEOUT_MS)
    self._socket.setsockopt(zmq.SNDTIMEO, _DEFAULT_TIMEOUT_MS)
    self._socket.connect(f"tcp://{self._master_addr}:{self._port}")

  def send(self, data: bytes) -> None:
    """Sends serialized data synchronously to the server via PyZMQ socket."""
    self._socket.send(data)

  def recv(self) -> bytes:
    """Receives response bytes synchronously from the server via PyZMQ socket."""
    return self._socket.recv()

  def close(self) -> None:
    if hasattr(self, "_socket") and not self._socket.closed:
      self._socket.close(linger=0)
    if hasattr(self, "_context"):
      self._context.term()


class _HandshakeClient(_HandshakeBackend):
  """Client backend for Handshake worker ranks using _ZMQClient socket handler."""

  _instances: dict[int, "_HandshakeClient"] = {}
  _instances_lock = threading.Lock()

  def __new__(
      cls,
      port: int,
  ) -> "_HandshakeClient":
    with cls._instances_lock:
      if port not in cls._instances:
        inst = super().__new__(cls)
        inst._initialized = False
        cls._instances[port] = inst
      return cls._instances[port]

  def __init__(
      self,
      port: int,
  ) -> None:
    if getattr(self, "_initialized", False):
      return
    self._zmq_client = _ZMQClient(port=port)
    self._closed = False
    self._initialized = True

  def send(self, data: bytes) -> None:
    """Sends serialized data synchronously to the server."""
    self._zmq_client.send(data)

  def recv(self) -> bytes:
    """Receives response bytes synchronously from the server."""
    return self._zmq_client.recv()

  def close(self) -> None:
    """Closes client sockets and marks instance uninitialized."""
    if getattr(self, "_closed", False):
      return
    self._closed = True
    if hasattr(self, "_zmq_client"):
      self._zmq_client.close()
    self._initialized = False

  @classmethod
  def _reset_instances(cls) -> None:
    """Resets singleton instance for testing purposes."""
    with cls._instances_lock:
      for client in cls._instances.values():
        try:
          client.close()
        except Exception:  # pylint: disable=broad-except # TODO(basioli) this seems dangerous, should we be passing here?
          pass
      cls._instances.clear()


class Handshake:
  """Handshake protocol across distributed ranks using ZMQ backend."""

  _port_cache: dict[int, int] = {}
  _port_cache_lock = threading.Lock()

  def __init__(
      self,
      current_rank: int | None = None,
      coordinator_rank: int = _COORDINATOR_RANK,
  ) -> None:
    if current_rank is None:
      current_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    self._current_rank = current_rank
    self._coordinator_rank = coordinator_rank

    with self._port_cache_lock:
      if coordinator_rank in self._port_cache:
        port = self._port_cache[coordinator_rank]
      else:
        port = _select_handshake_port(current_rank, coordinator_rank)
        self._port_cache[coordinator_rank] = port

    if self._current_rank == self._coordinator_rank:
      self._backend = _HandshakeServer(
          current_rank=coordinator_rank, port=port, world_size=world_size
      )
    else:
      self._backend = _HandshakeClient(port=port)

  @classmethod
  def _reset_instances(cls) -> None:
    """Resets singleton instances and port cache for testing purposes."""
    with cls._port_cache_lock:
      cls._port_cache.clear()
    _HandshakeServer._reset_instances()
    _HandshakeClient._reset_instances()

  def submit(self, msg: HandshakeMessage) -> bool:
    """Submits a HandshakeMessage to the handshake protocol.

    Args:
      msg: HandshakeMessage object to submit for consensus.

    Returns:
      True if executable fingerprint was identical across all participating
      ranks, else False.
    """
    if not isinstance(msg, HandshakeMessage):
      raise TypeError(
          f"Expected msg to be HandshakeMessage, got {type(msg).__name__}."
      )

    if self._current_rank not in msg.participating_ranks:
      raise ValueError(
          f"Current rank {self._current_rank} is not in participating ranks"
          f" {msg.participating_ranks}."
      )

    self._backend.send(msg.to_bytes())
    resp_bytes = self._backend.recv()
    return resp_bytes == _TRUE_BYTES

  def close(self) -> None:
    """Closes the underlying backend connection."""
    if hasattr(self, "_backend") and self._backend is not None:
      if hasattr(self._backend, "close"):
        self._backend.close()
