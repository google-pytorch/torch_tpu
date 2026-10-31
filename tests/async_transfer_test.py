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

import time

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class AsyncTransferTest(seed_test_utils.RepeatableTest):

  def test_non_blocking_returns_immediately(self):
    device = torch.device('tpu')
    size = 8192  # 8192x8192 int32 = 256MB

    # 1. Warmup: compile the 'a + b' graph and drive materialization.
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = a + b
    cpu_dst = torch.empty_like(c, device='cpu', pin_memory=True)
    # Warm up the compilation.
    cpu_dst.copy_(c, non_blocking=True)
    torch.accelerator.synchronize()

    # 2. Main test: measure immediate return of a TPU computation copy.
    a2_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b2_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a2 = a2_cpu.to(device)
    b2 = b2_cpu.to(device)
    c2 = a2 + b2

    t0 = time.time()
    # This should return immediately as it only enqueues the computation and
    # DMA.
    cpu_dst.copy_(c2, non_blocking=True)
    async_duration = time.time() - t0

    torch.accelerator.synchronize()

    # 3. Measure blocking return of the same size
    a3_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b3_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a3 = a3_cpu.to(device)
    b3 = b3_cpu.to(device)
    c3 = a3 + b3

    t1 = time.time()
    # This should block until the computation and DMA are finished.
    cpu_dst.copy_(c3, non_blocking=False)
    sync_duration = time.time() - t1

    # A 256MB transfer + computation takes ~40-60ms if blocking.
    # We expect microsecond-level return for async, so 10ms is a safe, tight
    # threshold.
    print('\n--- Transfer Return Latency (256MB) ---')
    print(f'Non-blocking copy_ duration : {async_duration:.6f}s')
    print(f'Blocking copy_ duration     : {sync_duration:.6f}s')
    print(f'Ratio (Sync/Async)          : {sync_duration/async_duration:.1f}x')
    print('----------------------------------------\n')

    self.assertLess(async_duration, 0.01)  # 10ms threshold
    # The sync duration should be significantly higher than async duration.
    self.assertLess(async_duration, sync_duration)

  def test_non_blocking_returns_immediately_accelerator(self):
    device = torch.device('tpu')
    size = 8192  # 8192x8192 int32 = 256MB

    # 1. Warmup: compile the 'a + b' graph and drive materialization.
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = a + b
    cpu_dst = torch.empty_like(c, device='cpu', pin_memory=True)
    # Warm up the compilation.
    cpu_dst.copy_(c, non_blocking=True)
    torch.accelerator.synchronize()

    # 2. Main test: measure immediate return of a TPU computation copy.
    a2_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b2_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a2 = a2_cpu.to(device)
    b2 = b2_cpu.to(device)
    c2 = a2 + b2

    t0 = time.time()
    cpu_dst.copy_(c2, non_blocking=True)
    async_duration = time.time() - t0

    torch.accelerator.synchronize()

    # 3. Measure blocking return of the same size
    a3_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b3_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a3 = a3_cpu.to(device)
    b3 = b3_cpu.to(device)
    c3 = a3 + b3

    t1 = time.time()
    cpu_dst.copy_(c3, non_blocking=False)
    sync_duration = time.time() - t1

    self.assertLess(async_duration, 0.01)  # 10ms threshold
    self.assertLess(async_duration, sync_duration)

  def test_synchronize_default(self):
    """Tests that synchronize() correctly waits for async ops."""
    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_accelerator_synchronize_default(self):
    """Tests that torch.accelerator.synchronize() correctly waits for async ops."""
    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_to_cpu_non_blocking(self):
    device = torch.device('tpu')

    # Generate on CPU first to avoid lazy evaluation generating different
    # numbers!
    size = 2048
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)

    a = a_cpu.to(device)
    b = b_cpu.to(device)

    # Do some compute
    c = a * 2 + b

    # Non-blocking transfer to CPU. With our custom _to_copy implementation,
    # this will now use pinned memory and the fast async DMA path.
    cpu_tensor = c.to('cpu', non_blocking=True)
    cpu_view_tensor = c.view(-1).to('cpu', non_blocking=True)

    self.assertEqual(cpu_tensor.device.type, 'cpu')
    self.assertTrue(cpu_tensor.is_pinned())
    self.assertEqual(cpu_view_tensor.device.type, 'cpu')
    self.assertTrue(cpu_view_tensor.is_pinned())

    torch.accelerator.synchronize()

    # Now verify the values.
    expected_cpu_tensor = a_cpu * 2 + b_cpu

    utils.assert_close(cpu_tensor, expected_cpu_tensor)
    utils.assert_close(cpu_view_tensor, expected_cpu_tensor.view(-1))

  def test_to_cpu_non_blocking_accelerator(self):
    device = torch.device('tpu')

    size = 2048
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)

    a = a_cpu.to(device)
    b = b_cpu.to(device)

    c = a * 2 + b
    cpu_tensor = c.to('cpu', non_blocking=True)

    self.assertEqual(cpu_tensor.device.type, 'cpu')
    self.assertTrue(cpu_tensor.is_pinned())

    torch.accelerator.synchronize()

    expected_cpu_tensor = a_cpu * 2 + b_cpu
    utils.assert_close(cpu_tensor, expected_cpu_tensor)

  def test_to_cpu_non_blocking_avoids_pin_memory_copy(self):
    """Verifies non_blocking .to('cpu') allocates pinned memory directly."""
    device = torch.device('tpu')
    size = 64
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = a + b

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
      cpu_dst = c.to('cpu', non_blocking=True)

    op_names = [evt.key for evt in prof.key_averages()]
    self.assertNotIn('aten::pin_memory', op_names)
    self.assertNotIn('aten::_pin_memory', op_names)
    self.assertTrue(cpu_dst.is_pinned())
    torch.accelerator.synchronize()
    utils.assert_close(cpu_dst, a_cpu + b_cpu)

  def test_non_blocking_d2h_does_not_wait_for_materialization_start(self):
    """Verifies non_blocking D2H uses kNonBlocking instead of kBlockUntilStarted.

    When a deep deferred DAG is materialized by a non_blocking D2H copy,
    kBlockUntilStarted blocks the host thread until the background
    materialize_thread_ lowers the DAG to MLIR and marks the source tensor as
    pending materialization (sync.is_materializing(c) == True). With
    kNonBlocking, the D2H call enqueues the snapshot request and returns
    before materialize_thread_ finishes lowering the DAG.
    """
    device = torch.device('tpu')
    size = 256
    half = size // 2
    a_cpu = torch.ones((size, size), dtype=torch.int32)
    b_cpu = torch.ones((size, size), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    cpu_dst = torch.empty((size, size), dtype=torch.int32, pin_memory=True)
    # Warm up the pinned allocator for both full and slice sizes.
    _ = torch.empty((size, half), dtype=torch.int32, pin_memory=True)
    torch.accelerator.synchronize()

    # TODO(anuragprakash): There is a theoretical race with the background
    # materialize_thread_ that could flake if it finishes lowering the 200-op
    # DAG before the host thread reaches the assertFalse(is_materializing)
    # checks below. Add a test-only hook to pause materialization so this test
    # is completely deterministic.
    with execution_mode.set_eager_mode(
        execution_mode._EagerMode.INTERNAL_DEFER_ALL
    ):
      # 1. Full-storage path (CopyTpuToCpu::MaterializeBase) via .to('cpu').
      num_ops_1 = 200
      c = a
      for _ in range(num_ops_1):
        c = c + b
      self.assertFalse(sync.is_materializing(c))
      cpu_c = c.to('cpu', non_blocking=True)
      self.assertFalse(sync.is_materializing(c))
      self.assertTrue(cpu_c.is_pinned())
      torch.accelerator.synchronize()
      utils.assert_close(cpu_c, a_cpu + num_ops_1 * b_cpu)

      # 2. Full-storage path via copy_(non_blocking=True).
      num_ops_2 = 201
      c2 = a
      for _ in range(num_ops_2):
        c2 = c2 + b
      self.assertFalse(sync.is_materializing(c2))
      cpu_dst.copy_(c2, non_blocking=True)
      self.assertFalse(sync.is_materializing(c2))
      torch.accelerator.synchronize()
      utils.assert_close(cpu_dst, a_cpu + num_ops_2 * b_cpu)

      # 3. Non-reinterpret view path (CopyTpuToCpu::MaterializeAndReturn).
      num_ops_3 = 202
      c3 = a
      for _ in range(num_ops_3):
        c3 = c3 + b
      self.assertFalse(sync.is_materializing(c3))
      cpu_slice = c3[:, :half].to('cpu', non_blocking=True)
      self.assertFalse(sync.is_materializing(c3))
      self.assertTrue(cpu_slice.is_pinned())
      torch.accelerator.synchronize()
      utils.assert_close(cpu_slice, (a_cpu + num_ops_3 * b_cpu)[:, :half])

  def test_to_cpu_non_blocking_non_reinterpret_views(self):
    """Tests non_blocking .to('cpu') on non-reinterpret views (slice, transpose)."""
    device = torch.device('tpu')
    rows, cols = 512, 256
    a_cpu = torch.randint(1, 10, (rows, cols), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (rows, cols), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)

    c = a * 3 + b
    half_cols = cols // 2
    cpu_slice = c[:, :half_cols].to('cpu', non_blocking=True)
    cpu_transposed = c.t().to('cpu', non_blocking=True)

    self.assertEqual(cpu_slice.device.type, 'cpu')
    self.assertTrue(cpu_slice.is_pinned())
    self.assertEqual(cpu_transposed.device.type, 'cpu')
    self.assertTrue(cpu_transposed.is_pinned())

    torch.accelerator.synchronize()

    expected = a_cpu * 3 + b_cpu
    utils.assert_close(cpu_slice, expected[:, :half_cols])
    utils.assert_close(cpu_transposed, expected.t())

  def test_to_cpu_non_blocking_with_stream_and_event(self):
    """Tests SGLang's async D2H pattern using a secondary Stream and Event."""
    device = torch.device('tpu')
    size = 1024
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)

    # Compute on the default stream.
    c = a * 2 + b

    copy_stream = torch.tpu.Stream()
    copy_event = torch.tpu.Event()
    with torch.tpu.stream(copy_stream):
      cpu_c = c.to('cpu', non_blocking=True)
      copy_event.record(copy_stream)

    self.assertEqual(cpu_c.device.type, 'cpu')
    self.assertTrue(cpu_c.is_pinned())

    # Synchronize only via the recorded event (without full device synchronize).
    copy_event.synchronize()
    self.assertTrue(copy_event.query())
    utils.assert_close(cpu_c, a_cpu * 2 + b_cpu)

  def test_to_cpu_non_blocking_with_dtype_conversion(self):
    """Tests D2H transfer with dtype conversion and non_blocking=True."""
    device = torch.device('tpu')
    size = 512
    a_cpu = torch.randn((size, size), dtype=torch.float32)
    b_cpu = torch.randn((size, size), dtype=torch.float32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)

    c = a + b
    cpu_fp16 = c.to('cpu', dtype=torch.float16, non_blocking=True)
    cpu_bf16 = c.to('cpu', dtype=torch.bfloat16, non_blocking=True)

    self.assertEqual(cpu_fp16.device.type, 'cpu')
    self.assertEqual(cpu_fp16.dtype, torch.float16)
    self.assertTrue(cpu_fp16.is_pinned())
    self.assertEqual(cpu_bf16.device.type, 'cpu')
    self.assertEqual(cpu_bf16.dtype, torch.bfloat16)
    self.assertTrue(cpu_bf16.is_pinned())

    torch.accelerator.synchronize()

    expected = a_cpu + b_cpu
    utils.assert_close(cpu_fp16, expected.to(torch.float16))
    utils.assert_close(cpu_bf16, expected.to(torch.bfloat16))

  def test_pin_memory(self):
    device = torch.device('tpu')
    # Initialize the TPU device by creating a small tensor on it
    _ = torch.zeros(1, device=device)

    t = torch.randn(100)

    # The default pin_memory() will use the current device's pinned
    # allocator if a custom backend like TPU is registered as the accelerator.
    t_pinned = t.pin_memory()

    self.assertTrue(t_pinned.is_pinned())
    utils.assert_close(t, t_pinned)

  def test_h2d_sync_race(self):
    """Verifies that torch.accelerator.synchronize() properly waits for H2D transfers.

    If synchronize() fails to wait, modifying the host tensor immediately
    after the call may corrupt the data transferred to the TPU.
    """
    size = 4096

    # 1. Use pinned memory for async H2D transfer
    cpu_tensor = torch.ones(
        size, size, dtype=torch.float32, device='cpu'
    ).pin_memory()

    # Warmup
    _ = cpu_tensor.to('tpu', non_blocking=True)
    torch.accelerator.synchronize()

    # Reset host tensor to 1s
    cpu_tensor.fill_(1.0)

    # 2. Enqueue async H2D transfer
    tpu_tensor = cpu_tensor.to('tpu', non_blocking=True)

    # 3. Synchronize
    # (Should wait for the H2D future registered via ScheduleHostToDeviceTransfer)
    torch.accelerator.synchronize()

    # 4. Modify the host tensor immediately after synchronize
    cpu_tensor.fill_(2.0)

    # 5. Read back the TPU tensor to verify integrity
    result_tensor = tpu_tensor.cpu()

    # The TPU tensor must contain the original 1.0s.
    mean_val = result_tensor.mean().item()

    self.assertEqual(
        mean_val,
        1.0,
        msg=(
            'H2D race condition detected! synchronize() failed to wait for'
            ' transfer.'
        ),
    )

  def test_h2d_accelerator_sync(self):
    """Verifies that torch.accelerator.synchronize() waits for H2D transfers."""
    size = 4096

    # Use pinned memory for async H2D transfer
    cpu_tensor = torch.ones(
        size, size, dtype=torch.float32, device='cpu'
    ).pin_memory()

    # Warmup
    _ = cpu_tensor.to('tpu', non_blocking=True)
    torch.accelerator.synchronize()

    # Reset host tensor to 1.0
    cpu_tensor.fill_(1.0)

    # Enqueue async H2D
    tpu_tensor = cpu_tensor.to('tpu', non_blocking=True)

    # Synchronize using the generic accelerator API
    torch.accelerator.synchronize()

    # Modify the host tensor immediately after synchronize
    cpu_tensor.fill_(2.0)

    # Read back the TPU tensor.
    result_tensor = tpu_tensor.cpu()

    # If synchronize() worked, the TPU tensor should contain 1.0.
    mean_val = result_tensor.mean().item()
    self.assertEqual(
        mean_val,
        1.0,
        'torch.accelerator.synchronize() failed to wait for H2D transfer.',
    )

  def test_event_wait_delays_host_to_device_transfer(self):
    """Verifies that inserting a stream wait delays H2D transfer."""
    device = torch.device('tpu')

    # 1 byte of data on host, to be transferred to TPU.
    cpu_src = torch.ones(1, dtype=torch.uint8, device='cpu').pin_memory()
    tpu_dst = torch.empty(1, dtype=torch.uint8, device=device)

    work_stream = torch.tpu.Stream()
    transfer_stream = torch.tpu.Stream()

    # Start a bunch of work on the work stream, then take an event at the end
    with torch.tpu.stream(work_stream):
      x = torch.zeros(1024, 1024, dtype=torch.int32, device=device)
      for _ in range(1000):
        x.add_(1)
      work_event = torch.tpu.Event(enable_timing=True)
      work_event.record(work_stream)

    # Start a nonblocking H2D transfer, but make it wait for the work event.
    with torch.tpu.stream(transfer_stream):
      transfer_stream.wait_event(work_event)
      tpu_dst.copy_(cpu_src, non_blocking=True)
      transfer_event = torch.tpu.Event(enable_timing=True)
      transfer_event.record(transfer_stream)

    # The transfer event should not immediately be ready.
    self.assertFalse(transfer_event.query())

    # The work event should be ready before the transfer event; equivalently,
    # there should be no point where the transfer event is complete but the
    # work event is not.
    transfer_done = transfer_event.query()
    work_done = work_event.query()
    while not (transfer_done and work_done):
      transfer_done = transfer_event.query()
      work_done = work_event.query()
      self.assertFalse(transfer_done and not work_done)
      time.sleep(0.01)  # 10ms sleep

    # The elapsed time between the work event and the transfer event should not
    # be negative.
    self.assertGreaterEqual(
        work_event.elapsed_time(transfer_event),
        0,
    )

  def test_event_wait_delays_device_to_host_transfer(self):
    """Verifies that inserting a stream wait delays D2H transfer."""
    device = torch.device('tpu')

    # 1 byte of data on TPU, to be transferred to host.
    tpu_src = torch.ones(1, dtype=torch.uint8, device=device)
    # Pinned memory to copy into.
    cpu_dst = torch.empty(1, dtype=torch.uint8, device='cpu').pin_memory(device)

    work_stream = torch.tpu.Stream()
    transfer_stream = torch.tpu.Stream()

    # Start a bunch of work on the work stream, then take an event at the end
    with torch.tpu.stream(work_stream):
      x = torch.zeros(1024, 1024, dtype=torch.int32, device=device)
      for _ in range(1000):
        x.add_(1)
      work_event = torch.tpu.Event(enable_timing=True)
      work_event.record(work_stream)

    # Start a nonblocking D2H transfer, but make it wait for the work event.
    with torch.tpu.stream(transfer_stream):
      transfer_stream.wait_event(work_event)
      cpu_dst.copy_(tpu_src, non_blocking=True)
      transfer_event = torch.tpu.Event(enable_timing=True)
      transfer_event.record(transfer_stream)

    # The transfer event should not immediately be ready.
    self.assertFalse(transfer_event.query())

    # The work event should be ready before the transfer event; equivalently,
    # there should be no point where the transfer event is complete but the
    # work event is not.
    transfer_done = transfer_event.query()
    work_done = work_event.query()
    while not (transfer_done and work_done):
      transfer_done = transfer_event.query()
      work_done = work_event.query()
      self.assertFalse(transfer_done and not work_done)
      time.sleep(0.01)  # 10ms sleep

    # The elapsed time between the work event and the transfer event should not
    # be negative.
    self.assertGreaterEqual(
        work_event.elapsed_time(transfer_event),
        0,
    )


if __name__ == '__main__':
  absltest.main()
