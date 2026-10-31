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

from typing import Optional, Self, TypeAlias

import torch
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.utils import annotations

experimental = annotations.experimental

_NOT_IMPLEMENTED_STREAMS_MSG = (
    'Streams and Events are not fully implemented in TorchTPU. Please file a'
    ' feature request describing your use case.'
)


def _get_device_index(device=None) -> int:
  """Returns the device index of the given, or current device."""
  if device is None:
    # pylint: disable=protected-access
    return _device_ops_backend._get_current_device_id()

  if isinstance(device, int):
    return device

  tpu_device = None
  if isinstance(device, str):
    tpu_device = torch.device(device)
  elif isinstance(device, torch.device):
    tpu_device = device

  maybe_index = getattr(tpu_device, 'index', None)
  if maybe_index is not None:
    # A parsed string of "tpu" will not have an index, but a string like
    # "tpu:0" will.
    return maybe_index

  # pylint: disable=protected-access
  return _device_ops_backend._get_current_device_id()


CorePinningMode: TypeAlias = _device_ops_backend.CorePinningMode


def _parse_pinned(pinned: CorePinningMode | str | None) -> CorePinningMode:
  """Parses the "pinned" keyword argument using common strings."""
  if pinned is None:
    return CorePinningMode.UNPINNED
  if isinstance(pinned, CorePinningMode):
    return pinned

  if isinstance(pinned, str):
    lowercase = pinned.lower()
    if lowercase == 'none' or lowercase == 'unpinned':
      return CorePinningMode.UNPINNED
    if lowercase in ('tensorcore', 'tensor_core', 'tc', 'tensor', 'dense'):
      return CorePinningMode.TENSOR_CORE
    if lowercase in (
        'sparse',
        'sparse_core',
        'sparsecore',
        'sc',
        'sparseoffload',
    ):
      return CorePinningMode.SPARSE_CORE
    raise ValueError(f'Unknown device pinning mode: {pinned}')
  raise ValueError(
      f'pinned must be a CorePinningMode or a string, got {type(pinned)}'
  )


class TpuStream(torch.Stream):
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

  TpuStream subclasses torch.Stream so that it can be passed to device-generic
  APIs such as torch.accelerator.set_stream().

  TODO: b/452051142 - Explore if and how to make better use of streams.
  """

  def __new__(
      cls,
      device=None,
      priority: int = 0,
      stream_id: Optional[int] = None,
      *,
      pinned: CorePinningMode | str | None = None,
      **kwargs,
  ):
    # pylint: disable=unused-argument
    # stream_id, device_index and device_type are read-only members of
    # torch.Stream, so they must be set here rather than in __init__.
    device_index = _get_device_index(device)
    if stream_id is None:
      # pylint: disable=protected-access
      stream_id = _device_ops_backend._get_next_stream_id(device_index)
    return super().__new__(
        cls,
        stream_id=stream_id,
        device_index=device_index,
        # pylint: disable=protected-access
        device_type=_device_ops_backend._get_device_type(),
        priority=priority,
    )

  def __init__(
      self,
      device=None,
      priority: int = 0,
      stream_id: Optional[int] = None,
      *,
      pinned: CorePinningMode | str | None = None,
      **kwargs,
  ):
    # pylint: disable=unused-argument
    self.pinned: CorePinningMode = _parse_pinned(pinned)

    if self.pinned != CorePinningMode.UNPINNED:
      # pylint: disable=protected-access
      _device_ops_backend._set_core_pinning_mode(
          self.device_index, self.stream_id, self.pinned
      )

  @experimental(
      'wait_event() is experimental and may change or be removed without'
      ' notice.'
  )
  def wait_event(self, event: 'TpuEvent') -> None:  # pylint: disable=unused-argument
    """Makes all future work submitted to the stream wait for an event."""
    event.wait()

  @experimental(
      'wait_stream() is experimental and may change or be removed without'
      ' notice.'
  )
  def wait_stream(self, stream: Self) -> None:  # pylint: disable=unused-argument
    """Makes all future work submitted to the stream wait for another stream."""
    self.wait_event(stream.record_event())

  @experimental(
      'record_event() is experimental and may change or be removed without'
      ' notice.'
  )
  def record_event(self, event: Optional['TpuEvent'] = None) -> 'TpuEvent':
    """Records an event on this stream."""
    if event is None:
      event = TpuEvent()
    event.record(self)
    return event

  @experimental(
      'query() is experimental and may change or be removed without notice.'
  )
  def query(self) -> bool:
    """Checks if all work submitted on this stream has been completed."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  @experimental(
      'synchronize() is experimental and may change or be removed without'
      ' notice.'
  )
  def synchronize(self) -> None:
    """Waits for all work submitted on this stream to complete."""
    # pylint: disable=protected-access
    _device_ops_backend._synchronize_stream(self.stream_id, self.device_index)

  @experimental(
      'priority_range() is experimental and may change or be removed without'
      ' notice.'
  )
  def priority_range(self):
    """Returns the priority range for streams."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def __repr__(self):
    return '<torch.tpu.TpuStream>'

  @property
  @experimental(
      'is_sparse is experimental and may change or be removed without notice.'
  )
  def is_sparse(self) -> bool:
    return self.pinned == CorePinningMode.SPARSE_CORE

  @property
  @experimental(
      'is_dense is experimental and may change or be removed without notice.'
  )
  def is_dense(self) -> bool:
    return self.pinned == CorePinningMode.TENSOR_CORE


class TpuEvent(torch.Event):
  """Scoped completion marker for async TPU operations.

  Implements the ``torch.tpu.Event`` interface for TPU. Because torch_tpu has
  a single implicit stream per device, events function as point-in-time
  snapshots of pending async futures rather than positional markers in a stream
  timeline.

  TpuEvent subclasses torch.Event so that it can be passed to device-generic
  APIs and conforms to PyTorch's event specification.

  Supported operations:
    - ``record()`` snapshots pending async D2H copy futures.
    - ``synchronize()`` blocks until the snapshotted futures complete.
    - ``query()`` checks whether the snapshotted futures have completed.
    - ``elapsed_time()`` measures the time between two events completing.
    - ``wait(stream)`` causes `stream` to wait for this event before continuing.

  Not yet implemented:
    - ``ipc_handle()`` / ``from_ipc_handle()`` require IPC event support.

  TODO: b/452051142 - Explore if and how to make better use of streams.
  """

  def __new__(
      cls,
      enable_timing: bool = False,
      blocking: bool = False,
      interprocess: bool = False,
      external: bool = False,
      device=None,
  ):
    # pylint: disable=unused-argument
    # external is accepted for CUDA API compatibility; torch.Event does not
    # accept it.
    if device is None:
      device = torch.device('tpu')
    return super().__new__(
        cls,
        device=device,
        enable_timing=enable_timing,
        blocking=blocking,
        interprocess=interprocess,
    )

  @experimental(
      'record() is experimental and may change or be removed without notice.'
  )
  def record(self, stream: torch.Stream | None = None) -> Self:
    """Snapshot all pending async futures on the current device."""
    if stream is None:
      stream = torch.accelerator.current_stream()
    elif not isinstance(stream, torch.Stream):
      raise TypeError(f'expected torch.Stream, but got {type(stream).__name__}')
    super().record(stream)
    return self

  @experimental(
      'wait() is experimental and may change or be removed without notice.'
  )
  def wait(self, stream: torch.Stream | None = None) -> None:
    """Inserts a cross-stream ordering dependency on stream to this event.

    Future work on ``stream`` will wait for this event without blocking the CPU.

    Args:
      stream: the stream that should wait for this event.
    """
    return super().wait(stream)

  @experimental(
      'query() is experimental and may change or be removed without notice.'
  )
  def query(self) -> bool:
    """Checks if all work currently captured by this event has completed."""
    return super().query()

  @experimental(
      'elapsed_time() is experimental and may change or be removed without'
      ' notice.'
  )
  def elapsed_time(self, end_event: torch.Event) -> float:
    """Returns the milliseconds between this event and ``end_event`` completing.

    An event completes when all work captured by :meth:`record` has finished.
    The result is measured on the host, not on the device: it is the interval
    between the moments the host is notified that the work of each event has
    completed, so it can include gaps for dispatch and compilation and the
    delay in delivering those notifications. Both events must be created with
    ``enable_timing=True``, recorded and completed, for example by calling
    :meth:`synchronize` on ``end_event``.

    Args:
      end_event: The event recorded after this one.
    """
    # torch.Event.elapsed_time() only accepts an end event of the exact type
    # torch.Event, so the backend reads the timing of the two events.
    # pylint: disable=protected-access
    return _device_ops_backend._event_elapsed_time(self, end_event)

  @experimental(
      'synchronize() is experimental and may change or be removed without'
      ' notice.'
  )
  def synchronize(self) -> None:
    """Block until all snapshotted futures complete.

    Unlike ``torch.accelerator.synchronize()``, which waits for all pending D2H
    copies on the device, this only waits for futures that were pending when
    :meth:`record` was called. If ``record()`` was never called, this returns
    immediately.
    """
    super().synchronize()

  @classmethod
  @experimental(
      'from_ipc_handle() is experimental and may change or be removed without'
      ' notice.'
  )
  def from_ipc_handle(cls, device, handle):
    """Reconstructs an event from an IPC handle on the given device."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  @experimental(
      'ipc_handle() is experimental and may change or be removed without'
      ' notice.'
  )
  def ipc_handle(self):
    """Returns an IPC handle of this event."""
    raise NotImplementedError(_NOT_IMPLEMENTED_STREAMS_MSG)

  def __repr__(self) -> str:
    return '<torch.tpu.TpuEvent>'


def synchronize(device: torch.device | str | int | None = None) -> None:
  """Waits for all pending d2h copies on a TPU device to complete.

  This function implements `torch.tpu.synchronize()`.

  Args:
    device (torch.device | str | int | None): device for which to wait. Uses the
      current device if device is None (default).
  """
  device_idx = _get_device_index(device)
  _device_ops_backend._synchronize_device(device_idx, wait=True)  # pylint: disable=protected-access
