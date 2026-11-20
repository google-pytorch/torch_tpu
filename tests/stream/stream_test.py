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
import math
import operator
import time

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import dynamism
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils


# int64 values at the 32 bit carry and sign boundaries.
_INT64_BOUNDARY_VALUES = torch.tensor(
    [0, -1, 2**31 - 1, 2**32 - 1, -(2**63), 2**63 - 1], dtype=torch.int64
)


def _int64_boundary_tensor(shape):
  """Tiles the int64 boundary values over a CPU tensor of `shape`."""
  numel = math.prod(shape)
  repeats = math.ceil(numel / _INT64_BOUNDARY_VALUES.numel())
  return _INT64_BOUNDARY_VALUES.repeat(repeats)[:numel].reshape(shape)


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

  def test_stream_repr_identifies_the_stream(self):
    """Distinct streams have distinct reprs that identify the stream."""
    new_stream = torch.tpu.Stream()

    self.assertEqual(
        repr(new_stream),
        f'<torch.tpu.Stream device=tpu:0 stream_id={new_stream.stream_id}>',
    )
    self.assertNotEqual(repr(new_stream), repr(torch.tpu.default_stream()))

  def test_event_repr_uses_public_name(self):
    """The event repr names the public class and the device."""
    event = torch.tpu.Event()

    self.assertEqual(repr(event), f'<torch.tpu.Event device={event.device}>')
    self.assertEqual(event.device.type, 'tpu')

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

    # Verify TypeError is raised when a non-Stream object is passed to record.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python argument type check
        TypeError, 'expected torch.Stream, but got str'
    ):
      event.record('invalid')

    # Verify integer device index like 0 is correctly handled.
    event_device_idx = torch.tpu.Event(device=0)
    self.assertIsInstance(event_device_idx, torch.Event)
    self.assertEqual(event_device_idx.device, torch.device('tpu'))

  def test_event_partially_implemented(self):
    """Tests that ipc event methods remain unimplemented."""

    e = torch.tpu.Event()
    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
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

  def test_wait_event_waits_on_the_stream_it_is_called_on(self):
    """b.wait_event(ev) holds b, not the stream that is current at the call."""
    a = torch.tpu.Stream()
    b = torch.tpu.Stream()
    x = torch.full((8192, 8192), 1 / 8192, device='tpu')
    ones = torch.ones(1024, device='tpu')
    held = torch.zeros(1024, pin_memory=True)
    (x @ x).to('cpu')

    # Keep a busy for a while, so that ev stays pending below.
    with torch.tpu.stream(a):
      y = x
      for _ in range(100):
        y = y @ x
      ev = a.record_event()
    # The default stream is current: a wait that landed there would hold it.
    b.wait_event(ev)
    with torch.tpu.stream(b):
      held.copy_(ones, non_blocking=True)
      ev_b = b.record_event()

    # The default stream is not held: its copy completes while a is still busy.
    free = ones.to('cpu')
    self.assertFalse(ev.query())
    # b is held: its copy, and the event after it, complete only once ev has.
    self.assertFalse(ev_b.query())
    ev_b.synchronize()
    self.assertTrue(ev.query())
    # utils.assert_close() synchronizes the device, so it only runs from here.
    utils.assert_close(free, torch.ones(1024))
    utils.assert_close(held, torch.ones(1024))
    utils.assert_close(y[0, :2].to('cpu'), torch.full((2,), 1 / 8192))

  def test_record_stream_is_accepted_as_a_no_op(self):
    """record_stream() is accepted on a TPU tensor and leaves it untouched."""
    side = torch.tpu.Stream()
    x = torch.arange(16, dtype=torch.float32, device='tpu')

    self.assertIsNone(x.record_stream(side))
    with torch.tpu.stream(side):
      y = x * 2
      self.assertIsNone(y.record_stream(torch.tpu.default_stream()))
    self.assertIsNone(x.record_stream(torch.accelerator.current_stream()))

    utils.assert_close(x.to('cpu'), torch.arange(16, dtype=torch.float32))
    utils.assert_close(y.to('cpu'), torch.arange(16, dtype=torch.float32) * 2)

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
      z = x + y
      event = new_stream.record_event()
    event.synchronize()
    self.assertTrue(event.query())
    self.assertEqual(z.item(), 2)

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

  @oss_utils.skip_in_oss(
      'OSS libtpu version not high enough to support pinning'
  )
  def test_int64_add_dense_vs_sparse_bitwise(self):
    """Tests static int64 add parity between dense and sparse streams."""
    dense_stream = torch.tpu.Stream(pinned='dense')
    sparse_stream = torch.tpu.Stream(pinned='sparse')
    for shape in [(1,), (2,), (3,), (5,), (6,), (7,), (3, 5)]:
      cpu_x = _int64_boundary_tensor(shape)
      x = cpu_x.to('tpu')
      with torch.tpu.stream(dense_stream):
        dense = (x + 1).cpu()
      with torch.tpu.stream(sparse_stream):
        sparse = (x + 1).cpu()
      self.assertTrue(torch.equal(dense, sparse), msg=f'shape={shape}')
      self.assertTrue(torch.equal(sparse, cpu_x + 1), msg=f'shape={shape}')

  @parameterized.named_parameters(
      ('add', operator.add),
      ('sub', operator.sub),
  )
  @oss_utils.skip_in_oss(
      'OSS libtpu version not high enough to support pinning'
  )
  def test_int64_binary_dense_vs_sparse_bitwise(self, op):
    """Tests int64 tensor op tensor parity between dense and sparse streams."""
    dense_stream = torch.tpu.Stream(pinned='dense')
    sparse_stream = torch.tpu.Stream(pinned='sparse')
    for shape in [(1,), (2,), (3,), (5,), (6,), (7,), (3, 5)]:
      cpu_x = _int64_boundary_tensor(shape)
      # Shifting by one pairs each boundary value with the next one, so the
      # pairs carry and borrow across the 32 bit words and wrap at the sign bit.
      numel = cpu_x.numel()
      cpu_y = _int64_boundary_tensor((numel + 1,))[1:].reshape(shape)
      expected = op(cpu_x, cpu_y)
      x = cpu_x.to('tpu')
      y = cpu_y.to('tpu')
      with torch.tpu.stream(dense_stream):
        dense = op(x, y).cpu()
      with torch.tpu.stream(sparse_stream):
        sparse = op(x, y).cpu()
      self.assertTrue(torch.equal(dense, sparse), msg=f'shape={shape}')
      self.assertTrue(torch.equal(sparse, expected), msg=f'shape={shape}')

  @oss_utils.skip_in_oss(
      'OSS libtpu version not high enough to support pinning'
  )
  def test_int64_fill_from_scalar_dense_vs_sparse_bitwise(self):
    """Tests int64 fill from a rank 0 tensor on dense vs sparse streams."""
    dense_stream = torch.tpu.Stream(pinned='dense')
    sparse_stream = torch.tpu.Stream(pinned='sparse')
    for value in [0, 1, -1, (1 << 32) - 1, (1 << 40) + 7, -(2**63), 2**63 - 1]:
      tpu_scalar = torch.tensor(value, dtype=torch.int64).to('tpu')
      with torch.tpu.stream(dense_stream):
        dense_buf = torch.zeros(1, dtype=torch.int64, device='tpu')
        dense_buf.fill_(tpu_scalar)
        dense = dense_buf.cpu()
      with torch.tpu.stream(sparse_stream):
        sparse_buf = torch.zeros(1, dtype=torch.int64, device='tpu')
        sparse_buf.fill_(tpu_scalar)
        sparse = sparse_buf.cpu()
      expected = torch.tensor([value], dtype=torch.int64)
      self.assertTrue(torch.equal(dense, sparse), msg=f'value={value}')
      self.assertTrue(torch.equal(sparse, expected), msg=f'value={value}')

  @oss_utils.skip_in_oss(
      'OSS libtpu version not high enough to support pinning'
  )
  def test_int64_add_bounded_dynamic_3d_dense_vs_sparse_bitwise(self):
    """Tests 3D foldable bounded dynamic int64 add parity on dense vs sparse."""
    dense_stream = torch.tpu.Stream(pinned='dense')
    sparse_stream = torch.tpu.Stream(pinned='sparse')
    cpu_x = _int64_boundary_tensor((3, 8, 128))
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      x_dense = cpu_x.to('tpu')
      dynamism.mark_dynamic(x_dense, 0, 2, 4)
      with torch.tpu.stream(dense_stream):
        dense = (x_dense + 1).cpu()

      x_sparse = cpu_x.to('tpu')
      dynamism.mark_dynamic(x_sparse, 0, 2, 4)
      with torch.tpu.stream(sparse_stream):
        sparse = (x_sparse + 1).cpu()

    self.assertTrue(torch.equal(dense, sparse))
    self.assertTrue(torch.equal(sparse, cpu_x + 1))

  @oss_utils.skip_in_oss(
      'OSS libtpu version not high enough to support pinning'
  )
  def test_int64_add_bounded_dynamic_1d_dense_vs_sparse_bitwise(self):
    """Tests rank 1 bounded dynamic int64 add parity on dense vs sparse."""
    # `dynamism.mark_dynamic` requires `lower_bound >= 2` because 0 and 1
    # sized dimensions carry PyTorch broadcast semantics; test the smallest
    # unaligned runtime sizes (2, 3, 5, 7).
    dense_stream = torch.tpu.Stream(pinned='dense')
    sparse_stream = torch.tpu.Stream(pinned='sparse')
    for runtime_size in [2, 3, 5, 7]:
      cpu_x = _int64_boundary_tensor((runtime_size,))
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode.DEFER_AND_FUSE
      ):
        x_dense = cpu_x.to('tpu')
        dynamism.mark_dynamic(x_dense, 0, 2, 16)
        with torch.tpu.stream(dense_stream):
          dense = (x_dense + 1).cpu()

        x_sparse = cpu_x.to('tpu')
        dynamism.mark_dynamic(x_sparse, 0, 2, 16)
        with torch.tpu.stream(sparse_stream):
          sparse = (x_sparse + 1).cpu()

      self.assertTrue(
          torch.equal(dense, sparse), msg=f'runtime_size={runtime_size}'
      )
      self.assertTrue(
          torch.equal(sparse, cpu_x + 1), msg=f'runtime_size={runtime_size}'
      )


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

  @absltest.skipIf(
      utils.get_torch_version() < (2, 12),
      'torch.Stream.is_capturing() requires PyTorch >= 2.12.',
  )
  def test_streams_are_not_capturing_without_graph_capture(self):
    """No stream reports that it is capturing a graph outside of capture."""
    device = torch.device('tpu', torch.accelerator.current_device_index())
    new_stream = torch.Stream(device=device)

    self.assertFalse(new_stream.is_capturing())
    self.assertFalse(torch.tpu.Stream().is_capturing())
    self.assertFalse(torch.accelerator.current_stream().is_capturing())
    with new_stream:
      self.assertFalse(torch.accelerator.current_stream().is_capturing())


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

  def test_elapsed_time_reflects_host_time(self):
    # Record two events with host delay but no device-side work between them.
    start = torch.tpu.Event(enable_timing=True)
    end = torch.tpu.Event(enable_timing=True)
    start.record()
    time.sleep(1)
    end.record()
    end.synchronize()

    # The elapsed time should reflect the delay inserted by the host.
    device_time_ms = start.elapsed_time(end)
    self.assertGreater(device_time_ms, 990)  # 1% tolerance

  def test_errors_preserved_after_stream_exit(self):
    new_stream = torch.tpu.Stream()
    with torch.tpu.stream(new_stream):
      # This will fail to compile due to a compile-time OOM (it would require
      # 4 petabytes of HBM).
      x = torch.ones((1024, 1024, 1024, 1024, 1024), device='tpu')

    # Exit the stream, and wait on host. Errors will not be raised here.
    time.sleep(1)

    # The next sync will pick up the error.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests error timing
        RuntimeError, 'out of memory'
    ):
      torch.accelerator.synchronize()

    # Attempting to use the value will also propagate the error.
    y = (x + 1)[0, 0, 0, 0, 0]
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests error timing
        RuntimeError, 'out of memory'
    ):
      y.item()


if __name__ == '__main__':
  absltest.main()
