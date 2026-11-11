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

"""Tests ops served by precompiled unbounded dynamic kernels."""

from collections.abc import Callable
import math

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests import op_testing


class UnboundedDynamicKernelTestBase(op_testing.TorchTpuVsCpuTestBase):
  """Base class for tests of ops served by unbounded dynamic kernels."""

  def run_and_check_cache(
      self,
      op: Callable[..., torch.Tensor],
      *inputs: torch.Tensor,
      served: bool,
      static_compilations: int = 1,
  ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Checks `op(*inputs)` on TPU vs CPU, input preservation, and cache stats.

    Expects 1 unbounded dynamic hit and 0 compilations if `served`, else 0
    dynamic hits and `static_compilations` cache misses and compilations.
    """
    tpu_inputs = [x.to("tpu") for x in inputs]
    torch.accelerator.synchronize()

    before = torch.tpu._get_cache_stats()  # pylint: disable=protected-access
    result = op(*tpu_inputs)
    torch.accelerator.synchronize()
    after = torch.tpu._get_cache_stats()  # pylint: disable=protected-access

    self.assert_close(golden_result=op(*inputs), torch_tpu_result=result.cpu())

    # Check input has not been modified
    for x, x_tpu in zip(inputs, tpu_inputs):
      self.assertTrue(
          torch.equal(
              x_tpu.cpu().flatten().view(torch.uint8),
              x.flatten().view(torch.uint8),
          ),
          "The op modified its input.",
      )

    hits = after.num_unbounded_dynamic_hits - before.num_unbounded_dynamic_hits
    compilations = after.num_compilations - before.num_compilations
    misses = (after.num_cache_reqs - after.num_cache_hits) - (
        before.num_cache_reqs - before.num_cache_hits
    )
    expected_compilations = 0 if served else static_compilations
    self.assertEqual(hits, 1 if served else 0, "unbounded dynamic hits")
    self.assertEqual(compilations, expected_compilations, "compilations")
    self.assertEqual(misses, expected_compilations, "cache misses")
    return tpu_inputs, result


_ALL_NONZERO = -1


def _nonzero_case(
    shape: tuple[int, ...], k: int, dtype: torch.dtype = torch.float32
) -> tuple[str, torch.Tensor]:
  """Returns a (name, input) case with `k` (or `_ALL_NONZERO`) nonzeros."""
  size = math.prod(shape)
  suffix = "all_nonzero" if k == _ALL_NONZERO else f"k{k}"
  name = f"{'x'.join(map(str, shape))}_{suffix}"
  k = size if k == _ALL_NONZERO else k
  i = torch.arange(size)
  selected = (i + 1) * k // size != i * k // size
  if dtype == torch.float32:
    # Spread inf/-2/subnormal/NaN nonzeros across +0.0 and -0.0 zeros.
    x = torch.where(
        selected,
        torch.tensor([math.inf, -2.0, 1e-40])[i % 3],
        torch.tensor([0.0, -0.0])[i % 2],
    )
    if k:
      x[-1] = math.nan
  else:
    x = selected.to(dtype)
    name += "_" + str(dtype).removeprefix("torch.")
  return name, x.reshape(shape)


_NONZERO_SERVED_CASES = (
    # Partial row block narrower than a 128-lane tile.
    _nonzero_case((5, 5), 3),
    _nonzero_case((5, 5), _ALL_NONZERO),
    # Several 1024-column tiles per row, the last one masked.
    _nonzero_case((9, 1100), 3),
    _nonzero_case((9, 1100), _ALL_NONZERO),
    # Rows wider than the kernel's 16384-column scratch window.
    _nonzero_case((5, 16385), 3),
    _nonzero_case((5, 16385), _ALL_NONZERO),
    # Around the kernel's 128-pair storage rows and 1024-pair flushes.
    _nonzero_case((9, 129), 127),
    _nonzero_case((9, 129), 128),
    _nonzero_case((9, 129), 129),
    _nonzero_case((9, 129), 1023),
    _nonzero_case((9, 129), 1024),
    _nonzero_case((9, 129), 1025),
    # 1-D at the edges of each default tile, T(128) to T(1024).
    _nonzero_case((2,), 2),
    _nonzero_case((16,), 5),
    _nonzero_case((128,), _ALL_NONZERO),
    _nonzero_case((129,), 2),
    _nonzero_case((256,), _ALL_NONZERO),
    _nonzero_case((257,), 3),
    _nonzero_case((512,), _ALL_NONZERO),
    _nonzero_case((513,), _ALL_NONZERO),
    # 1-D: single tile, masked last tile, and multiple 16384-element windows.
    _nonzero_case((1024,), 2),
    _nonzero_case((1024,), _ALL_NONZERO),
    _nonzero_case((3000,), 127),
    _nonzero_case((3000,), 128),
    _nonzero_case((3000,), 129),
    _nonzero_case((3000,), 1023),
    _nonzero_case((3000,), 1024),
    _nonzero_case((3000,), 1025),
    _nonzero_case((16385,), 2),
    _nonzero_case((40000,), 777),
    _nonzero_case((40000,), _ALL_NONZERO),
)

_NONZERO_NOT_SERVED_CASES = (
    # Default 2-D input layout isn't {1,0:T(8,128)}.
    _nonzero_case((3, 7), 5),
    _nonzero_case((5, 1), 5),
    _nonzero_case((5, 3), 5),
    # K = 0 skips the index graph; K <= rank uses row-major S64[K,rank] layout.
    _nonzero_case((17, 257), 0),
    _nonzero_case((17, 257), 1),
    _nonzero_case((17, 257), 2),
    _nonzero_case((3000,), 0),
    _nonzero_case((3000,), 1),
    _nonzero_case((16,), 1),
    # Not 1-D or 2-D F32.
    _nonzero_case((2, 4, 4), 5),
    _nonzero_case((17, 257), 5, torch.bfloat16),
    _nonzero_case((3000,), 5, torch.bfloat16),
    _nonzero_case((16,), 5, torch.bfloat16),
)


class DynamicOpsTest(UnboundedDynamicKernelTestBase):

  @parameterized.named_parameters(*_NONZERO_SERVED_CASES)
  def test_nonzero_served(self, x: torch.Tensor):
    # Warm up the dynamic index kernel and this shape's static count graph
    # (a CPU-to-TPU copy avoids fusing constant ops into either graph).
    torch.nonzero(torch.ones_like(x).to("tpu"))

    (x_tpu,), result = self.run_and_check_cache(torch.nonzero, x, served=True)

    # XLA's default 1-D tile: N rounded up to a power of two in [128, 1024].
    tile = min(max(128, 1 << (x.numel() - 1).bit_length()), 1024)
    # (minor_to_major, tiles, element_size_in_bits); see nonzero_indices.h.
    self.assertEqual(
        torch.tpu.get_device_layout_if_materialized(x_tpu),
        ([1, 0], [[8, 128]], 0) if x.ndim == 2 else ([0], [[tile]], 0),
    )
    self.assertEqual(
        torch.tpu.get_device_layout_if_materialized(result),
        ([0, 1], [[x.ndim, 128]], 0),
    )

  @parameterized.named_parameters(*_NONZERO_NOT_SERVED_CASES)
  def test_nonzero_not_served(self, x: torch.Tensor):
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    torch.nonzero(torch.zeros_like(x).to("tpu"))  # Warm up the count graph.

    self.run_and_check_cache(
        torch.nonzero,
        x,
        served=False,
        static_compilations=1 if x.count_nonzero() else 0,
    )


if __name__ == "__main__":
  absltest.main()
