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

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class CopyTest(seed_test_utils.RepeatableTest):

  def test_copy_tpu_to_cpu_direct(self):
    """Verifies direct copy to contiguous, matching destination."""
    device = torch.device('tpu')
    size = (4, 4)
    t_tpu = torch.randn(size, device=device)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    # This should use the direct copy loop in tpu_to_cpu.cc
    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, t_tpu.to('cpu'))

  def test_copy_tpu_to_cpu_fallback_non_contiguous(self):
    """Verifies the fallback path when destination is non-contiguous."""
    device = torch.device('tpu')
    t_tpu = torch.randn((4, 4), device=device)
    t_cpu = torch.empty((4, 8), device='cpu', pin_memory=True)[:, ::2]

    self.assertFalse(t_cpu.is_contiguous())

    # This should use the fallback (copy through intermediate)
    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, t_tpu.to('cpu'))

  def test_copy_tpu_to_cpu_fallback_dtype_mismatch(self):
    """Verifies the fallback path when there is a dtype mismatch."""
    device = torch.device('tpu')
    t_tpu = torch.randn((4, 4), device=device, dtype=torch.float32)
    t_cpu = torch.empty(
        (4, 4), device='cpu', dtype=torch.float64, pin_memory=True
    )

    # This should use the fallback (copy through intermediate or ATen copy)
    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, t_tpu.to('cpu').to(torch.float64))

  def test_to_copy_contiguous_requested(self):
    device = torch.device('tpu')
    x = torch.randn(2, 3, device=device).T
    y = x.to(dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    self.assertTrue(y.is_contiguous())
    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = x_cpu.to(
        dtype=torch.bfloat16, memory_format=torch.contiguous_format
    )
    self.assertEqual(y.stride(), y_cpu.stride())

  def test_cpu_to_tpu_non_blocking_dangling_paged(self):
    device = torch.device('tpu')

    def _dangling_transfer(device):
      """Starts a transfer and drops the tensor being copied out of on host."""
      x = torch.arange(1024, dtype=torch.int32, device='cpu')
      return x.to(device, non_blocking=True)

    tpu_values = [_dangling_transfer(device) for _ in range(1024)]
    torch.accelerator.synchronize()

    cpu_values = [t.to('cpu', non_blocking=True) for t in tpu_values]
    torch.accelerator.synchronize()

    expected = torch.arange(1024, dtype=torch.int32, device='cpu')
    for cpu_tensor in cpu_values:
      utils.assert_close(cpu_tensor, expected)

  def test_cpu_to_tpu_non_blocking_dangling_pinned(self):
    device = torch.device('tpu')

    def _dangling_transfer(device):
      """Starts a transfer and drops the tensor being copied out of on host."""
      x = torch.arange(1024, dtype=torch.int32, device='cpu').pin_memory(device)
      return x.to(device, non_blocking=True)

    tpu_values = [_dangling_transfer(device) for _ in range(1024)]
    torch.accelerator.synchronize()

    cpu_values = [t.to('cpu', non_blocking=True) for t in tpu_values]
    torch.accelerator.synchronize()

    expected = torch.arange(1024, dtype=torch.int32, device='cpu')
    for cpu_tensor in cpu_values:
      utils.assert_close(cpu_tensor, expected)

  def test_tpu_to_cpu_drop_pinned_host(self):
    device = torch.device('tpu')
    x = torch.arange(1024, dtype=torch.int32, device=device)

    def _cancelled_transfer(x):
      # Start a transfer to pinned host memory...
      y = torch.empty_like(x, device='cpu').pin_memory(x.device)
      y.copy_(x, non_blocking=True)
      # ...but delete the destination tensor before the transfer completes.
      del y

    for _ in range(1024):
      _cancelled_transfer(x)

    # We can't observe that the transfers complete but we shouldn't crash here.
    torch.accelerator.synchronize()


if __name__ == '__main__':
  absltest.main()
