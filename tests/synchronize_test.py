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
from typing import TypeAlias

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

EagerMode: TypeAlias = execution_mode.EagerMode


class SynchronizeTest(seed_test_utils.RepeatableTest):
  """Tests synchronization APIs, e.g. torch.accelerator.synchronize."""

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.old_eager_mode = execution_mode.eager_mode
    execution_mode.eager_mode = EagerMode.DEFER_AND_FUSE

  def tearDown(self):
    execution_mode.eager_mode = self.old_eager_mode
    super().tearDown()

  def test_accelerator_synchronize_device_materialized(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y = torch.ones(11, device=torch.device('tpu'))

    self.assertFalse(sync.is_materializing(x))
    self.assertFalse(sync.is_materializing(y))

    torch.accelerator.synchronize()

    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materializing(y))

  def test_accelerator_synchronize_device_ready(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y = torch.ones(11, device=torch.device('tpu'))

    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))

    torch.accelerator.synchronize()

    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materialized(y))

  def test_synchronize_device_ready_defer_never(self):
    execution_mode.eager_mode = EagerMode.DEFER_NEVER
    x = torch.ones(100, 100, device='tpu')
    y = x + 1.0

    torch.accelerator.synchronize()
    self.assertTrue(sync.is_materialized(y))

  def test_synchronize_event_defer_never(self):
    execution_mode.eager_mode = EagerMode.DEFER_NEVER
    x = torch.ones(100, 100, device='tpu')
    y = x + 1.0

    # Record event on current stream (capturing y's ready future)
    event = torch.tpu.Event()
    event.record()

    # Synchronizing the event should block until y is ready
    event.synchronize()
    self.assertTrue(sync.is_materialized(y))

  def test_synchronize_non_default_stream_defer_never(self):
    execution_mode.eager_mode = EagerMode.DEFER_NEVER
    s = torch.tpu.Stream()

    with torch.tpu.stream(s):
      x = torch.ones(100, 100, device='tpu')
      y = x + 1.0

    # Synchronizing the entire device to ensure the stream's pending futures
    # are correctly tracked and waited on
    torch.accelerator.synchronize()
    self.assertTrue(sync.is_materialized(y))

  def test_sync_list_with_empty_and_non_empty(self):
    x = torch.ones(10, device=torch.device('tpu'))
    y_cpu = torch.ones(10, 0, device='cpu')
    y = y_cpu.to(torch.device('tpu'))
    # Should not raise error.
    torch.accelerator.synchronize()
    del y
    self.assertTrue(sync.is_materialized(x))

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

  def test_accelerator_synchronize_compiled_mode(self):
    def func(lhs, rhs):
      return lhs.mm(rhs)

    compiled = torch.compile(func, backend='tpu')

    x = torch.ones(256, 256, dtype=torch.float32, device='tpu')
    y = compiled(x, x)
    torch.accelerator.synchronize()
    self.assertTrue(sync.is_materialized(y))
    expected = torch.full((256, 256), 256.0, dtype=torch.float32, device='cpu')
    utils.assert_close(y.cpu(), expected)


if __name__ == '__main__':
  absltest.main()
