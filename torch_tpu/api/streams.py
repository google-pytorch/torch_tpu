# Copyright 2025 Google LLC
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

"""Python interface for streams and events on TPU."""

from typing import Optional, Self
from torch_tpu.api import _device_ops_backend

_NOT_IMPLEMENTED_STREAMS_MSG = (
    'Streams and Events are not fully implemented in TorchTPU. Please file a'
    ' feature request describing your use case.'
)


class TpuStream:
  """A stream of operations on a TPU device.

  Streams are a CUDA concept with no clear analog in XLA.
  They refer to a linear sequence of executions on a specific device.
  Along with events they can be used to schedule and synchronize parallel work
  streams. For example, one stream might write to a buffer and record an event
  while a second stream waits for that event before reading from the buffer.
  Because XLA is functional it can schedule operations based on data
  dependencies and does not need streams, at least for the use cases we have
  considered so far around distributed algorithms.

  This class implements dummy methods that are needed to enable
  PyTorch's Fully Sharded Data Parallel. Other methods are left unimplemented
  to help identify new use cases. The CUDA Stream class on which this is
  based can be found in torch/cuda/streams.py.

  TODO: b/452051142 - Explore if and how to make better use of streams.
  """

  def __init__(self, device=None, priority: int = 0, **kwargs):
    pass

  def wait_event(self, event: 'TpuEvent') -> None:  # pylint: disable=unused-argument
    """Makes all future work submitted to the stream wait for an event."""
    event.wait()

  def wait_stream(self, stream: Self) -> None:  # pylint: disable=unused-argument
    """Synchronizes with another stream."""
    self.wait_event(stream.record_event())

  def record_event(self, event: Optional['TpuEvent'] = None) -> 'TpuEvent':
    """Records an event on this stream."""
    if event is None:
      event = TpuEvent()
    event.record(self)
    return event

  def query(self) -> bool:
    """Checks if all work submitted on this stream has been completed."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def synchronize(self) -> None:
    """Waits for all work submitted on this stream to complete."""
    synchronize()

  def priority_range(self):
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def __repr__(self):
    return '<torch.tpu.TpuStream>'


class TpuEvent:
  """Scoped completion marker for async TPU operations.

  Implements the ``torch.xpu.Event`` interface for TPU. Because torch_tpu has
  a single implicit stream per device, events function as point-in-time
  snapshots of pending async futures rather than positional markers in a stream
  timeline.

  Supported operations:
    - ``record()`` snapshots pending async D2H copy futures.
    - ``synchronize()`` blocks until the snapshotted futures complete.
    - ``query()`` checks whether the snapshotted futures have completed.

  Not yet implemented:
    - ``wait(stream)`` requires multi-stream support.
    - ``elapsed_time()`` requires device-side timing support.
    - ``ipc_handle()`` / ``from_ipc_handle()`` require IPC event support.

  TODO: b/452051142 - Explore if and how to make better use of streams.
  """

  def __init__(
      self,
      enable_timing: bool = False,  # pylint: disable=unused-argument
      blocking: bool = False,  # pylint: disable=unused-argument
      interprocess: bool = False,  # pylint: disable=unused-argument
      external: bool = False,  # pylint: disable=unused-argument
  ):
    # Accepted for CUDA API compatibility; TPU currently ignores these flags.
    self._event_id: int | None = None

  def record(self, stream: TpuStream | None = None):  # pylint: disable=unused-argument
    """Snapshot all pending async futures on the current device.

    If this event was previously recorded, the old snapshot is released before
    recording the new one. This matches CUDA semantics where re-recording an
    event overwrites its previous state.

    Args:
      stream: Ignored. Present for CUDA API compatibility. torch_tpu has a
        single implicit stream, so this always snapshots device-wide pending
        futures.

    Returns:
      ``self``, for chaining.
    """
    if self._event_id is not None:
      _device_ops_backend._release_event(self._event_id)  # pylint: disable=protected-access
    self._event_id = _device_ops_backend._record_event()  # pylint: disable=protected-access
    return self

  def wait(self, stream: TpuStream | None = None) -> None:  # pylint: disable=unused-argument
    """No-op placeholder for CUDA stream-wait semantics.

    In CUDA, this inserts a device-side dependency so future work on ``stream``
    waits for this event without blocking the CPU. torch_tpu does not yet
    support multi-stream execution, so this is currently a no-op. For CPU-side
    blocking, use :meth:`synchronize`.

    Args:
      stream: Ignored. Present for CUDA API compatibility.
    """
    return

  def query(self) -> bool:
    """Checks if all work currently captpured by this event has completed."""
    if self._event_id is None:
      return True
    return _device_ops_backend._query_event(self._event_id)  # pylint: disable=protected-access

  def elapsed_time(self, end_event: Self) -> float:  # pylint: disable=unused-argument
    """Returns the time elapsed between recording and completing this event."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def synchronize(self) -> None:
    """Block until all snapshotted futures complete.

    Unlike ``torch.tpu.synchronize()``, which waits for all pending D2H copies
    on the device, this only waits for futures that were pending when
    :meth:`record` was called. If ``record()`` was never called, this returns
    immediately.
    """
    if self._event_id is None:
      return
    _device_ops_backend._wait_event(self._event_id)  # pylint: disable=protected-access

  @classmethod
  def from_ipc_handle(cls, device, handle):
    """Reconstructs an event from an IPC handle on the given device."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def ipc_handle(self):
    """Returns an IPC handle of this event."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def __repr__(self) -> str:
    return '<torch.tpu.TpuEvent>'

  def __del__(self):
    # Best-effort cleanup only.
    # TODO A C++ base event object would give us a safer lifetime model
    if self._event_id is None:
      return
    try:
      _device_ops_backend._release_event(self._event_id)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def synchronize(device: Optional[int] = None) -> None:
  """Waits for all pending d2h copies on a TPU device to complete.

  This function implements `torch.tpu.synchronize()`.

  Args:
    device (int, optional): device for which to wait. Uses the current device if
      device is None (default).
  """
  _device_ops_backend._synchronize(device)  # pylint: disable=protected-access
