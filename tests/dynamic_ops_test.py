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

"""Tests ops that precompiled unbounded dynamic kernels serve.

Unbounded dynamic kernels that are enabled by default load unless
TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS is "none", and each test checks
that the op's registered unbounded dynamic kernel serves it, and that the TPU
result matches the CPU one.
"""

from collections.abc import Callable
import math

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests import op_testing


def _unbounded_dynamic_hits() -> int:
  """Returns the number of lookups served by registered dynamic kernels."""
  return torch.tpu._get_cache_stats().num_unbounded_dynamic_hits  # pylint: disable=protected-access


class UnboundedDynamicKernelTestBase(op_testing.TorchTpuVsCpuTestBase):
  """Base class for tests of ops that unbounded dynamic kernels serve."""

  def assert_close_tpu_vs_cpu_with_dynamic_kernel(
      self,
      tensor_from_device: Callable[[torch.device], torch.Tensor],
      *,
      hits: int = 1,
  ) -> None:
    """Checks the TPU result against the CPU one, and the kernels that ran.

    Args:
      tensor_from_device: Returns the op's result on the given device.
      hits: The number of lookups that registered unbounded dynamic kernels must
        serve on TPU.
    """
    hits_before = _unbounded_dynamic_hits()
    self.assert_close_tpu_vs_cpu(tensor_from_device)
    torch.accelerator.synchronize()

    self.assertEqual(
        _unbounded_dynamic_hits() - hits_before,
        hits,
        "Lookups served by unbounded dynamic kernels; was"
        " TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS set to none (or left"
        " at default for an opt-in kernel), or did the kernel fail to build?",
    )


class DynamicOpsTest(UnboundedDynamicKernelTestBase):
  """Tests ops that precompiled unbounded dynamic kernels serve."""

  # The 2-D F32 shapes of the nonzero OpInfo samples that ops_test.py runs:
  # (S, S) with S = 5.
  @parameterized.product(shape=((5, 5), (5, 1025)), dtype=(torch.float32,))
  def test_nonzero(self, shape: tuple[int, ...], dtype: torch.dtype):
    # Both zeros and nonzeros. The kernel requires more than two nonzeros (see
    # nonzero_indices.h), so unlike the OpInfo samples, none is all zeros.
    x = torch.arange(1, math.prod(shape) + 1, dtype=dtype).reshape(shape)
    x[x % 3 == 0] = 0
    self.assert_close_tpu_vs_cpu_with_dynamic_kernel(
        lambda device: torch.nonzero(x.to(device))
    )


if __name__ == "__main__":
  absltest.main()
