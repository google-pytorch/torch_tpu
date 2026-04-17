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
from torch_tpu import api
from torch_tpu._internal.utils import utils


class StreamsTest(absltest.TestCase):

  def test_stream(self):
    """Tests basic usage of streams and events.

    Stream routing is still lightweight, but event recording should be usable.
    """
    _ = api.tpu_device()
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

  def test_stream_unimplemented_method(self):
    _ = api.tpu_device()
    expected_msg = (
        'Streams and Events are not fully implemented in TorchTPU. Please file'
        ' a feature request describing your use case.'
    )
    with self.assertRaisesWithLiteralMatch(NotImplementedError, expected_msg):
      default_stream = torch.tpu.default_stream()
      default_stream.query()

  def test_stream_partially_implemented_method(self):
    """Tests that unimplemented stream methods raise NotImplementedError."""
    _ = api.tpu_device()
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
    _ = api.tpu_device()

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
    _ = api.tpu_device()

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
    _ = api.tpu_device()
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
    _ = api.tpu_device()

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
    _ = api.tpu_device()

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


if __name__ == '__main__':
  absltest.main()
