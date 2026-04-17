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
from torch_tpu import api
from torch_tpu._internal.utils import utils


class CopyTest(absltest.TestCase):

  def test_copy_tpu_to_cpu_direct(self):
    """Verifies direct copy to contiguous, matching destination."""
    device = api.tpu_device()
    size = (4, 4)
    t_tpu = torch.randn(size, device=device)
    t_cpu = torch.empty(size, device='cpu', pin_memory=True)

    # This should use the direct copy loop in tpu_to_cpu.cc
    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.tpu.synchronize()

    utils.assert_close(t_cpu, t_tpu.to('cpu'))

  def test_copy_tpu_to_cpu_fallback_non_contiguous(self):
    """Verifies the fallback path when destination is non-contiguous."""
    device = api.tpu_device()
    t_tpu = torch.randn((4, 4), device=device)
    t_cpu = torch.empty((4, 8), device='cpu', pin_memory=True)[:, ::2]

    self.assertFalse(t_cpu.is_contiguous())

    # This should use the fallback (copy through intermediate)
    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.tpu.synchronize()

    utils.assert_close(t_cpu, t_tpu.to('cpu'))

  def test_copy_tpu_to_cpu_fallback_dtype_mismatch(self):
    """Verifies the fallback path when there is a dtype mismatch."""
    device = api.tpu_device()
    t_tpu = torch.randn((4, 4), device=device, dtype=torch.float32)
    t_cpu = torch.empty(
        (4, 4), device='cpu', dtype=torch.float64, pin_memory=True
    )

    # This should use the fallback (copy through intermediate or ATen copy)
    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.tpu.synchronize()

    utils.assert_close(t_cpu, t_tpu.to('cpu').to(torch.float64))


if __name__ == '__main__':
  absltest.main()
