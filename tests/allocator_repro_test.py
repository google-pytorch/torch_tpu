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
import torch.accelerator
from torch_tpu._internal.utils import utils


class AllocatorReproTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
