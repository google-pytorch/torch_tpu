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
from torch_tpu._internal.utils import utils


class AsyncTransferTest(absltest.TestCase):

  def test_non_blocking_returns_immediately(self):
    device = torch.accelerator.current_accelerator()
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
    torch.tpu.synchronize()

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

    torch.tpu.synchronize()

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
    _ = torch.device('tpu')

    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.tpu.synchronize()

    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_accelerator_synchronize_default(self):
    """Tests that torch.accelerator.synchronize() correctly waits for async ops."""
    _ = torch.device('tpu')

    size = 1024 * 1024
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_to_cpu_non_blocking(self):
    device = torch.accelerator.current_accelerator()

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

    self.assertEqual(cpu_tensor.device.type, 'cpu')

    torch.tpu.synchronize()

    # Now verify the values.
    expected_cpu_tensor = a_cpu * 2 + b_cpu

    utils.assert_close(cpu_tensor, expected_cpu_tensor)

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

    torch.accelerator.synchronize()

    expected_cpu_tensor = a_cpu * 2 + b_cpu
    utils.assert_close(cpu_tensor, expected_cpu_tensor)

  def test_pin_memory(self):
    device = torch.accelerator.current_accelerator()
    # Initialize the TPU device by creating a small tensor on it
    _ = torch.zeros(1, device=device)

    t = torch.randn(100)

    # The default pin_memory() will use the current device's pinned
    # allocator if a custom backend like TPU is registered as the accelerator.
    t_pinned = t.pin_memory()

    self.assertTrue(t_pinned.is_pinned())
    utils.assert_close(t, t_pinned)

  def test_h2d_sync_race(self):
    """Verifies that torch.tpu.synchronize() properly waits for H2D transfers.

    If synchronize() fails to wait, modifying the host tensor immediately
    after the call may corrupt the data transferred to the TPU.
    """
    torch.device('tpu')
    size = 4096

    # 1. Use pinned memory for async H2D transfer
    cpu_tensor = torch.ones(
        size, size, dtype=torch.float32, device='cpu'
    ).pin_memory()

    # Warmup
    _ = cpu_tensor.to('tpu', non_blocking=True)
    torch.tpu.synchronize()

    # Reset host tensor to 1s
    cpu_tensor.fill_(1.0)

    # 2. Enqueue async H2D transfer
    tpu_tensor = cpu_tensor.to('tpu', non_blocking=True)

    # 3. Synchronize
    # (Should wait for the H2D future registered via MarkStreamActive)
    torch.tpu.synchronize()

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
    torch.device('tpu')
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


if __name__ == '__main__':
  absltest.main()
