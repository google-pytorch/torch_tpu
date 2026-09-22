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

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class TpuStreamsTest(seed_test_utils.RepeatableTest):
  """Tests for torch.tpu.Stream and torch.tpu.Event.

  This is the TPU-specific class implementation, which may have additional
  functionality beyond the torch.Stream and torch.Event specification.
  """

  def test_stream(self):
    """Tests basic usage of streams and events.

    Stream routing is still lightweight, but event recording should be usable.
    """

    default_stream = torch.tpu.default_stream()
    new_stream = torch.tpu.Stream(priority=-1)

    torch.tpu.set_stream(new_stream)
    x = torch.ones((8, 8), device='tpu')
    event = new_stream.record_event()
    torch.tpu.set_stream(default_stream)

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
    self.assertEqual(
        torch.tpu.current_stream().stream_id, default_stream.stream_id
    )

    # 1. set_stream updates current_stream
    torch.tpu.set_stream(new_stream)
    self.assertEqual(torch.tpu.current_stream().stream_id, new_stream.stream_id)

    torch.tpu.set_stream(default_stream)
    self.assertEqual(
        torch.tpu.current_stream().stream_id, default_stream.stream_id
    )

    # 2. with stream(s): changes current_stream and restores it upon exit
    with torch.tpu.stream(new_stream):
      self.assertEqual(
          torch.tpu.current_stream().stream_id, new_stream.stream_id
      )
    self.assertEqual(
        torch.tpu.current_stream().stream_id, default_stream.stream_id
    )

    # 3. with stream(None): is a no-op
    with torch.tpu.stream(None):
      self.assertEqual(
          torch.tpu.current_stream().stream_id, default_stream.stream_id
      )
    self.assertEqual(
        torch.tpu.current_stream().stream_id, default_stream.stream_id
    )

  def test_stream_unimplemented_method(self):

    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      default_stream = torch.tpu.default_stream()
      default_stream.query()

  def test_stream_partially_implemented_method(self):
    """Tests that unimplemented stream methods raise NotImplementedError."""

    s = torch.tpu.Stream()
    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      s.query()
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      s.priority_range()

  def test_synchronize_default(self):
    """Tests that synchronize() correctly waits for async ops."""

    # Create a large tensor to make the transfer non-instantaneous
    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    # Start a non-blocking copy
    t_cpu.copy_(t_tpu, non_blocking=True)

    # Calling synchronize without arguments should sync the current device
    torch.tpu.synchronize()

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
      torch.tpu.synchronize()
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

  def test_event_partially_implemented(self):
    """Tests that timing/ipc event methods remain unimplemented."""

    e = torch.tpu.Event()
    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      e.elapsed_time(e)
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      e.ipc_handle()

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

    default_stream = torch.tpu.current_stream()
    transfer_stream = torch.tpu.Stream()

    # Start a non-blocking copy on the transfer stream
    with torch.tpu.stream(transfer_stream):
      current_stream = torch.tpu.current_stream()
      self.assertEqual(current_stream.stream_id, transfer_stream.stream_id)
      t_cpu.copy_(t_tpu, non_blocking=True)
      transfer_event = torch.tpu.Event()
      transfer_event.record()

    self.assertEqual(
        torch.tpu.current_stream().stream_id, default_stream.stream_id
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

  def test_stream_pinned_to_sparse_core(self):
    """Tests that a stream can be pinned to a sparse core."""
    new_stream = torch.tpu.Stream(pinned='sparse')
    self.assertTrue(new_stream.is_sparse)
    self.assertFalse(new_stream.is_dense)

    self.skipTest('XLA compiler failure')
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


if __name__ == '__main__':
  absltest.main()
