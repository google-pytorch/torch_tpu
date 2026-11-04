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

"""Tests ops on their static kernels, with unbounded dynamic kernels disabled.

The static_ops_test target (see BUILD) runs these tests with
TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS=0. The runtime reads the
variable once per process, and with it enabled (the default) the precompiled
unbounded dynamic kernels serve the ops they support instead of the static
kernels, e.g. those nonzero compiles per nonzero count.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import test_utils as utils
from tests import op_testing


class StaticOpsTest(op_testing.TorchTpuVsCpuTestBase):
  """Tests ops on their static kernels."""

  @parameterized.named_parameters(("functional", False), ("out", True))
  def test_nonzero_reuses_static_kernel_for_same_num_nonzero(self, use_out):
    # nonzero compiles one static kernel per nonzero count K > 0 (K == 0 needs
    # none). K is the leading extent of the output, which is already part of the
    # cache key, so the kernels must stay distinct without an op param cache
    # key for it.
    shape = (7, 11)  # Unused by other nonzero tests, so the cache starts cold.

    def make_input(num_nonzero, offset):
      x = torch.zeros(shape).flatten()
      x[offset : offset + 3 * num_nonzero : 3] = 1.0
      return x.reshape(shape)

    def run_nonzero(x):
      """Checks nonzero(x) on TPU and returns the cache (hits, misses) it adds."""
      tpu_x = x.to("tpu")
      tpu_out = torch.empty(0, x.dim(), dtype=torch.int64, device="tpu")
      start_hits = torch.tpu._get_cache_hits()
      start_misses = torch.tpu._get_cache_misses()
      if use_out:
        torch.nonzero(tpu_x, out=tpu_out)
        actual = tpu_out.cpu()
      else:
        actual = torch.nonzero(tpu_x).cpu()
      hits = torch.tpu._get_cache_hits() - start_hits
      misses = torch.tpu._get_cache_misses() - start_misses
      utils.assert_close(actual, torch.nonzero(x))
      return hits, misses

    _, misses = run_nonzero(make_input(num_nonzero=3, offset=0))
    self.assertGreaterEqual(misses, 1, "first count must compile a kernel")

    # Same count at different positions: the count-3 kernel is reused.
    hits, misses = run_nonzero(make_input(num_nonzero=3, offset=1))
    self.assertEqual(misses, 0, "same count must not recompile")
    self.assertGreater(hits, 0)

    # New count: only the count-5 nonzero kernel is compiled. The size kernel
    # depends on the input shape alone, so it is still a hit.
    _, misses = run_nonzero(make_input(num_nonzero=5, offset=0))
    self.assertEqual(misses, 1, "new count must not reuse the count-3 kernel")

    # No nonzeros: the empty result needs no nonzero kernel. The warm-up builds
    # the empty (0, 2) int64 tensor once so that only nonzero kernels count.
    torch.nonzero(torch.zeros(1, 1, device="tpu"))
    _, misses = run_nonzero(make_input(num_nonzero=0, offset=0))
    self.assertEqual(misses, 0, "zero count must not compile a kernel")


if __name__ == "__main__":
  absltest.main()
