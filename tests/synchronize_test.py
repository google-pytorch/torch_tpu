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
from torch_tpu._internal.utils import utils


class SynchronizeTest(absltest.TestCase):
  """Tests synchronization APIs: torch.tpu.synchronize and torch.accelerator.synchronize."""

  def setUp(self):
    super().setUp()
    self.old_eager_mode = execution_mode.get_eager_mode()
    execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE)

  def tearDown(self):
    execution_mode.set_eager_mode(self.old_eager_mode)
    super().tearDown()

  def test_synchronize_device_materialized(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y = torch.ones(11, device=torch.device('tpu'))

    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))

    # torch.tpu.synchronize() maps to _device_ops_backend._synchronize(None)
    # which maps to TpuDeviceGuardImpl::synchronizeDevice()
    # which calls SynchronizeAll(true) and SynchronizeStream()
    torch.tpu.synchronize()

    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materialized(y))

  def test_synchronize_device_ready(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y = torch.ones(11, device=torch.device('tpu'))

    self.assertFalse(sync.is_ready(x))
    self.assertFalse(sync.is_ready(y))

    torch.tpu.synchronize()

    self.assertTrue(sync.is_ready(x))
    self.assertTrue(sync.is_ready(y))

  def test_accelerator_synchronize_device_materialized(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y = torch.ones(11, device=torch.device('tpu'))

    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))

    torch.accelerator.synchronize()

    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materialized(y))

  def test_accelerator_synchronize_device_ready(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y = torch.ones(11, device=torch.device('tpu'))

    self.assertFalse(sync.is_ready(x))
    self.assertFalse(sync.is_ready(y))

    torch.accelerator.synchronize()

    self.assertTrue(sync.is_ready(x))
    self.assertTrue(sync.is_ready(y))

  def test_sync_with_zero_sized_tensor_on_tpu(self):
    # Create a zero-sized tensor on the TPU.
    tensor = torch.ones(2, 0, 3, dtype=torch.int32, device=torch.device('tpu'))

    # It is in a deferred state (constant zero-sized).
    self.assertFalse(sync.is_materialized(tensor))
    self.assertFalse(sync.is_ready(tensor))

    sync.synchronize(tensor, wait=True)

    # After synchronization, it should be materialized and ready.
    self.assertTrue(sync.is_materialized(tensor))
    self.assertTrue(sync.is_ready(tensor))

  def test_sync_with_materialized_zero_sized_tensor(self):
    # Create a zero-sized tensor on the CPU.
    tensor_cpu = torch.ones(2, 0, 3, dtype=torch.int32, device='cpu')

    # Send it to the TPU. This should create a deferred zero-sized constant
    # instead of actually transferring 0 bytes.
    tensor = tensor_cpu.to(torch.device('tpu'))
    self.assertFalse(sync.is_materialized(tensor))
    self.assertFalse(sync.is_ready(tensor))

    sync.synchronize(tensor, wait=True)

    # After synchronization, it should be materialized and ready.
    self.assertTrue(sync.is_materialized(tensor))
    self.assertTrue(sync.is_ready(tensor))

  def test_sync_list_with_empty_and_non_empty(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y_cpu = torch.ones(10, 0, device='cpu')
    y = y_cpu.to(torch.device('tpu'))
    # Should not raise error.
    sync.synchronize([x, y], wait=True)
    self.assertTrue(sync.is_ready(x))

  def test_synchronize_from_multiple_threads(self):

    size = 8192 * 8192
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)

    def sync_and_check():
      torch.tpu.synchronize()
      utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      futures = [
          executor.submit(sync_and_check),
          executor.submit(sync_and_check),
      ]
      for future in concurrent.futures.as_completed(futures):
        future.result()

  def test_accelerator_synchronize_from_multiple_threads(self):

    size = 8192 * 8192
    t_tpu = torch.ones(size, device='tpu', dtype=torch.float32)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)

    def sync_and_check():
      torch.accelerator.synchronize()
      utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      futures = [
          executor.submit(sync_and_check),
          executor.submit(sync_and_check),
      ]
      for future in concurrent.futures.as_completed(futures):
        future.result()


if __name__ == '__main__':
  absltest.main()
