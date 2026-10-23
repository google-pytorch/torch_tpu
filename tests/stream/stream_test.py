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

import concurrent.futures
import functools
import time

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils


class TpuStreamsTest(seed_test_utils.RepeatableTest):
  """Tests for torch.tpu.Stream and torch.tpu.Event.

  This is the TPU-specific class implementation, which may have additional
  functionality beyond the torch.Stream and torch.Event specification.
  """

  def test_tpu_stream_inherits_from_torch_stream(self):
    """TpuStream can be used wherever a torch.Stream is expected."""
    default_acc_stream = torch.accelerator.current_stream()
    new_stream = torch.tpu.Stream()

    self.assertIsInstance(new_stream, torch.Stream)
    self.assertIsInstance(torch.accelerator.current_stream(), torch.Stream)
    self.assertIsInstance(torch.tpu.default_stream(), torch.Stream)
    self.assertEqual(new_stream.device_type, default_acc_stream.device_type)
    self.assertEqual(new_stream.device, torch.device('tpu', 0))

    torch.accelerator.set_stream(new_stream)
    try:
      self.assertEqual(torch.accelerator.current_stream(), new_stream)
    finally:
      torch.accelerator.set_stream(default_acc_stream)
    self.assertEqual(torch.accelerator.current_stream(), default_acc_stream)

  def test_stream(self):
    """Tests basic usage of streams and events.

    Stream routing is still lightweight, but event recording should be usable.
    """

    default_stream = torch.tpu.default_stream()
    new_stream = torch.tpu.Stream(priority=-1)

    default_acc_stream = torch.accelerator.current_stream()
    torch.accelerator.set_stream(new_stream)
    x = torch.ones((8, 8), device='tpu')
    event = new_stream.record_event()
    torch.accelerator.set_stream(default_acc_stream)

    with torch.tpu.stream(new_stream):
      y = x + x

    default_stream.wait_event(event)
    default_stream.wait_stream(new_stream)
    z = y + y
    z.to('cpu')

  def test_stream_context_manager_and_set_stream(self):
    """Verifies stream context manager and set_stream behavior."""
    default_stream = torch.tpu.default_stream()
    new_stream = torch.tpu.Stream()

    # Initial state: default stream is active
    default_acc_stream = torch.accelerator.current_stream()
    self.assertEqual(default_acc_stream.stream_id, default_stream.stream_id)

    # 1. set_stream updates current_stream
    torch.accelerator.set_stream(new_stream)
    self.assertEqual(
        torch.accelerator.current_stream().stream_id, new_stream.stream_id
    )

    torch.accelerator.set_stream(default_acc_stream)
    self.assertEqual(
        torch.accelerator.current_stream().stream_id, default_stream.stream_id
    )

    # 2. with stream(s): changes current_stream and restores it upon exit
    with torch.tpu.stream(new_stream):
      self.assertEqual(
          torch.accelerator.current_stream().stream_id, new_stream.stream_id
      )
    self.assertEqual(
        torch.accelerator.current_stream().stream_id, default_stream.stream_id
    )

    # 3. with stream(None): is a no-op
    with torch.tpu.stream(None):
      self.assertEqual(
          torch.accelerator.current_stream().stream_id, default_stream.stream_id
      )
    self.assertEqual(
        torch.accelerator.current_stream().stream_id, default_stream.stream_id
    )

  def test_synchronize_default(self):
    """Tests that synchronize() correctly waits for async ops."""

    # Create a large tensor to make the transfer non-instantaneous
    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    # Start a non-blocking copy
    t_cpu.copy_(t_tpu, non_blocking=True)

    # Calling synchronize without arguments should sync the current device
    torch.accelerator.synchronize()

    # Verify data integrity
    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_synchronize_from_multiple_threads(self):

    # Create a large tensor that will take a while to copy
    size = 8192 * 8192
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    # Start a non-blocking copy
    t_cpu.copy_(t_tpu, non_blocking=True)

    def sync_and_check():
      torch.accelerator.synchronize()
      # Verify data integrity
      utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      futures = [
          executor.submit(sync_and_check),
          executor.submit(sync_and_check),
      ]
      # Wait for all futures to complete. Exceptions will be re-raised.
      for future in concurrent.futures.as_completed(futures):
        future.result()

  def test_tpu_event_inherits_from_torch_event(self):
    """TpuEvent can be used wherever a torch.Event is expected."""
    event = torch.tpu.Event()

    self.assertIsInstance(event, torch.Event)
    self.assertEqual(event.device, torch.device('tpu'))
    # An unrecorded event should return True for query and no-op on synchronize.
    self.assertTrue(event.query())
    event.synchronize()

    # Verify CUDA-compatible external keyword argument is accepted.
    event_with_external = torch.tpu.Event(
        enable_timing=False, blocking=False, interprocess=False, external=False
    )
    self.assertIsInstance(event_with_external, torch.Event)

    # Record on specific stream and default accelerator stream.
    s = torch.tpu.Stream()
    recorded_event = event.record(s)
    self.assertIs(recorded_event, event)  # verify chaining: returns self
    s.wait_event(event)
    event.wait(s)
    event.synchronize()
    self.assertTrue(event.query())

    # Verify stream.record_event() returns a torch.Event.
    stream_event = s.record_event()
    self.assertIsInstance(stream_event, torch.Event)
    stream_event.synchronize()
    self.assertTrue(stream_event.query())

    # Verify integer device index like 0 is correctly handled.
    event_device_idx = torch.tpu.Event(device=0)
    self.assertIsInstance(event_device_idx, torch.Event)
    self.assertEqual(event_device_idx.device, torch.device('tpu'))

  def test_event_records_nonblocking_copy(self):

    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    event = torch.tpu.Event()
    event.record()
    event.synchronize()

    self.assertTrue(event.query())
    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_event_snapshot_can_be_released_and_reused(self):

    size = 1024 * 1024
    first_t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    first_t_cpu = torch.empty(size, device='cpu', pin_memory=True)
    second_t_tpu = torch.full((size,), 2.0, device='tpu', dtype=torch.float32)
    second_t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    event = torch.tpu.Event()

    first_t_cpu.copy_(first_t_tpu, non_blocking=True)
    event.record()
    event.synchronize()
    utils.assert_close(first_t_cpu, torch.ones(size, dtype=torch.float32))

    second_t_cpu.copy_(second_t_tpu, non_blocking=True)
    event.record()
    event.synchronize()
    utils.assert_close(
        second_t_cpu, torch.full((size,), 2.0, dtype=torch.float32)
    )

  def test_async_transfer_on_separate_stream(self):
    """Tests async transfers on a separate stream."""
    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    default_stream = torch.accelerator.current_stream()
    transfer_stream = torch.tpu.Stream()

    # Start a non-blocking copy on the transfer stream
    with torch.tpu.stream(transfer_stream):
      current_stream = torch.accelerator.current_stream()
      self.assertEqual(current_stream.stream_id, transfer_stream.stream_id)
      t_cpu.copy_(t_tpu, non_blocking=True)
      transfer_event = torch.tpu.Event()
      transfer_event.record()

    self.assertEqual(
        torch.accelerator.current_stream().stream_id, default_stream.stream_id
    )

    # We should be able to synchronize on the event recorded on the
    # transfer stream.
    transfer_event.synchronize()

    self.assertTrue(transfer_event.query())
    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_materialize_on_stream_switch(self):
    """Tests that a stream begins executing when leaving its context."""
    new_stream = torch.tpu.Stream()

    # Use DEFER_AND_FUSE to avoid automatic materialization.
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Do some work on the default stream.
      x = torch.ones(1, device='tpu')
      self.assertFalse(sync.is_materializing(x))

      # Switch to the new stream.
      with torch.tpu.stream(new_stream):
        # Leaving the default stream should materialize x.
        self.assertTrue(sync.is_materializing(x))

  def test_stream_materialize_on_event(self):
    """Tests that streams materialize on event recording."""
    new_stream = torch.tpu.Stream()
    # Use DEFER_AND_FUSE to avoid automatic materialization.
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Do some work on the new stream.
      with torch.tpu.stream(new_stream):
        x = torch.zeros(1, device='tpu')
        # Since we are in DEFER_AND_FUSE, it should not be materializing.
        self.assertFalse(sync.is_materializing(x))

        # Record an event on the stream. This should trigger materialization.
        event = torch.tpu.Event()
        event.record()

        # Verify that the tensor is now materializing.
        self.assertTrue(sync.is_materializing(x))

    # Wait for the event to complete.
    event.synchronize()

    # Verify that the tensor is now materialized.
    self.assertTrue(sync.is_materialized(x))

  def test_stream_unpinned(self):
    """Tests that streams are unpinned by default."""
    new_stream = torch.tpu.Stream()
    self.assertFalse(new_stream.is_sparse)
    self.assertFalse(new_stream.is_dense)

  @oss_utils.skip_in_oss(
      'OSS libtpu version not high enough to support pinning'
  )
  def test_stream_pinned_to_sparse_core(self):
    """Tests that a stream can be pinned to a sparse core."""
    new_stream = torch.tpu.Stream(pinned='sparse')
    self.assertTrue(new_stream.is_sparse)
    self.assertFalse(new_stream.is_dense)

    x = torch.ones(1, device='cpu').to('tpu')
    y = torch.ones(1, device='cpu').to('tpu')
    with torch.tpu.stream(new_stream):
      actual = (x + y).item()
    self.assertEqual(actual, 2)

  def test_stream_pinned_to_tensor_core(self):
    """Tests that a stream can be pinned to a dense core."""
    new_stream = torch.tpu.Stream(pinned='dense')
    self.assertTrue(new_stream.is_dense)
    self.assertFalse(new_stream.is_sparse)

    x = torch.ones(1, device='cpu').to('tpu')
    y = torch.ones(1, device='cpu').to('tpu')

    with torch.tpu.stream(new_stream):
      actual = (x + y).item()

    self.assertEqual(actual, 2)


class TorchStreamsTest(seed_test_utils.RepeatableTest):
  """Tests for torch.Stream and torch.Event.

  This is the PyTorch device-agnostic interface which lacks any TPU-specific
  behavior.
  """

  def test_stream(self):
    """Tests basic usage of streams and events."""
    # PyTorch bug: torch.Stream(device='tpu') raises UnicodeDecodeError
    # "'utf-8' codec can't decode byte 0xff in position 28: invalid start byte"
    dummy = torch.empty(1, device='tpu')
    new_stream = torch.Stream(device=dummy.device)
    del dummy

    with new_stream:
      x = torch.ones((8, 8), device='tpu')
      event = new_stream.record_event()

    # Exit and re-enter the stream context.
    with new_stream:
      y = x + x

    event.wait()

    new_stream.synchronize()

    z = y + y
    z.to('cpu')

  def test_event_records_nonblocking_copy(self):

    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    event = torch.Event(device='tpu')
    event.record()
    event.synchronize()

    self.assertTrue(event.query())
    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_event_snapshot_can_be_released_and_reused(self):

    size = 1024 * 1024
    first_t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    first_t_cpu = torch.empty(size, device='cpu', pin_memory=True)
    second_t_tpu = torch.full((size,), 2.0, device='tpu', dtype=torch.float32)
    second_t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    event = torch.Event(device='tpu')
    first_t_cpu.copy_(first_t_tpu, non_blocking=True)
    event.record()
    event.synchronize()
    utils.assert_close(first_t_cpu, torch.ones(size, dtype=torch.float32))

    second_t_cpu.copy_(second_t_tpu, non_blocking=True)
    event.record()
    event.synchronize()
    utils.assert_close(
        second_t_cpu, torch.full((size,), 2.0, dtype=torch.float32)
    )

  def test_async_transfer_on_separate_stream(self):
    """Tests async transfers on a separate stream."""
    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    # PyTorch bug: torch.Stream(device='tpu') raises UnicodeDecodeError
    # "'utf-8' codec can't decode byte 0xff in position 28: invalid start byte"
    transfer_stream = torch.Stream(device=t_tpu.device)

    # Start a non-blocking copy on the transfer stream
    with transfer_stream:
      t_cpu.copy_(t_tpu, non_blocking=True)
      transfer_event = transfer_stream.record_event()

    # We should be able to synchronize on the event recorded on the
    # transfer stream.
    transfer_event.synchronize()

    self.assertTrue(transfer_event.query())
    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_materialize_on_stream_switch(self):
    """Tests that a stream begins executing when leaving its context."""
    # PyTorch bug: torch.Stream(device='tpu') raises UnicodeDecodeError
    # "'utf-8' codec can't decode byte 0xff in position 28: invalid start byte"
    dummy = torch.empty(1, device='tpu')
    new_stream = torch.Stream(device=dummy.device)
    del dummy

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Do some work on the default stream.
      x = torch.ones(1, device='tpu')
      self.assertFalse(sync.is_materializing(x))

      # Switch to the new stream.
      with new_stream:
        # Leaving the default stream should materialize x.
        self.assertTrue(sync.is_materializing(x))

  def test_stream_materialize_on_event(self):
    """Tests that streams materialize on event recording."""
    # PyTorch bug: torch.Stream(device='tpu') raises UnicodeDecodeError
    # "'utf-8' codec can't decode byte 0xff in position 28: invalid start byte"
    dummy = torch.empty(1, device='tpu')
    new_stream = torch.Stream(device=dummy.device)
    del dummy

    # Use DEFER_AND_FUSE to avoid automatic materialization.
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      # Do some work on the new stream.
      with new_stream:
        x = torch.zeros(1, device='tpu')
        # Since we are in DEFER_AND_FUSE, it should not be materializing.
        self.assertFalse(sync.is_materializing(x))

        # Record an event on the stream. This should trigger materialization.
        event = new_stream.record_event()

        # Verify that the tensor is now materializing.
        self.assertTrue(sync.is_materializing(x))

    # Wait for the event to complete.
    event.synchronize()

    # Verify that the tensor is now materialized.
    self.assertTrue(sync.is_materialized(x))


# The two public constructors of a TPU event.
_EVENT_CONSTRUCTORS = (
    ('tpu_event', torch.tpu.Event),
    ('torch_event', functools.partial(torch.Event, device='tpu')),
)


class EventElapsedTimeTest(seed_test_utils.RepeatableTest):
  """Tests for elapsed_time() of torch.tpu.Event and torch.Event.

  The parameterized tests run with both constructors, since torch.tpu.Event
  overrides the event methods of torch.Event.
  """

  @parameterized.named_parameters(*_EVENT_CONSTRUCTORS)
  def test_event_elapsed_time(self, make_event):
    """Tests that elapsed_time() lies within the host-observed interval."""
    x = torch.full((2048, 2048), 1 / 2048, device='tpu')
    (x @ x).to('cpu')

    start = make_event(enable_timing=True)
    end = make_event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    y = x
    for _ in range(10):
      y = y @ x
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    host_elapsed_ms = (time.perf_counter() - host_start) * 1000

    self.assertGreater(elapsed_ms, 0.0)
    self.assertLessEqual(elapsed_ms, host_elapsed_ms)
    self.assertEqual(end.elapsed_time(start), -elapsed_ms)
    self.assertEqual(end.elapsed_time(end), 0.0)

  @parameterized.named_parameters(*_EVENT_CONSTRUCTORS)
  def test_event_elapsed_time_ends_when_work_completes(self, make_event):
    """Tests that the end time is taken when the work completes.

    The host waits until the end event has completed, then sleeps before it
    synchronizes, which must not count.
    """
    x = torch.ones((8, 8), device='tpu')
    start = make_event(enable_timing=True)
    end = make_event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    y = x + x
    end.record()
    while not end.query():
      time.sleep(0.001)
    host_completed = time.perf_counter()
    time.sleep(0.1)
    end.synchronize()

    self.assertLessEqual(
        start.elapsed_time(end), (host_completed - host_start) * 1000
    )
    utils.assert_close(y.to('cpu'), torch.full((8, 8), 2.0))

  @parameterized.named_parameters(*_EVENT_CONSTRUCTORS)
  def test_event_elapsed_time_chained_pending_events(self, make_event):
    """Tests events recorded back to back while earlier work is pending.

    The end event completes right after the start event does, so the start
    event must be stamped before the end event is released.
    """
    size = 8192 * 8192
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    for _ in range(20):
      t_cpu.copy_(t_tpu, non_blocking=True)
      start = make_event(enable_timing=True)
      end = make_event(enable_timing=True)
      start.record()
      end.record()
      end.synchronize()
      self.assertGreaterEqual(start.elapsed_time(end), 0.0)

  @parameterized.named_parameters(*_EVENT_CONSTRUCTORS)
  def test_event_elapsed_time_start_during_pending_copy(self, make_event):
    """Tests a start event recorded while a large copy is in flight.

    The start event completes when the copy does, and the elapsed time lies
    within the interval the host observes around the copy and the timed work.
    """
    size = 8192 * 8192
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)
    x = torch.ones((8, 8), device='tpu')
    (x + x).to('cpu')

    host_start = time.perf_counter()
    t_cpu.copy_(t_tpu, non_blocking=True)
    start = make_event(enable_timing=True)
    end = make_event(enable_timing=True)
    start.record()
    y = x + x
    end.record()
    torch.accelerator.synchronize()
    host_elapsed_ms = (time.perf_counter() - host_start) * 1000
    elapsed_ms = start.elapsed_time(end)

    self.assertGreaterEqual(elapsed_ms, 0.0)
    self.assertLessEqual(elapsed_ms, host_elapsed_ms)
    utils.assert_close(y.to('cpu'), torch.full((8, 8), 2.0))

  @parameterized.named_parameters(*_EVENT_CONSTRUCTORS)
  def test_event_elapsed_time_uses_latest_record(self, make_event):
    """Tests that elapsed_time() uses the times of the latest record() calls."""
    x = torch.ones((8, 8), device='tpu')
    start = make_event(enable_timing=True)
    end = make_event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    end.record()
    end.synchronize()
    first_ms = start.elapsed_time(end)
    host_first_end = time.perf_counter()

    # The end event recorded again completes after host_rerecord, while its
    # first completion preceded host_first_end.
    time.sleep(0.01)
    host_rerecord = time.perf_counter()
    y = x + x
    end.record()
    end.synchronize()
    second_ms = start.elapsed_time(end)
    self.assertGreaterEqual(
        second_ms - first_ms, (host_rerecord - host_first_end) * 1000
    )
    self.assertLessEqual(second_ms, (time.perf_counter() - host_start) * 1000)

    # Both events recorded again complete within the latest host interval.
    host_third_start = time.perf_counter()
    start.record()
    y = y + x
    end.record()
    end.synchronize()
    third_ms = start.elapsed_time(end)
    self.assertBetween(
        third_ms, 0.0, (time.perf_counter() - host_third_start) * 1000
    )
    utils.assert_close(y.to('cpu'), torch.full((8, 8), 3.0))

  def test_event_elapsed_time_to_torch_event(self):
    """Tests a torch.tpu.Event timed against a torch.Event end event."""
    x = torch.ones((8, 8), device='tpu')
    start = torch.tpu.Event(enable_timing=True)
    end = torch.Event(device='tpu', enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    y = x + x
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)

    self.assertBetween(
        elapsed_ms, 0.0, (time.perf_counter() - host_start) * 1000
    )
    utils.assert_close(y.to('cpu'), torch.full((8, 8), 2.0))


if __name__ == '__main__':
  absltest.main()
