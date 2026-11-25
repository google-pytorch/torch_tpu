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
import functools
import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import utils
from tests import op_testing
from tests import oss_utils


def _to_cpu(result: Any) -> Any:
  """Moves a tensor, or each tensor of a tuple of tensors, to the CPU."""
  if isinstance(result, tuple):
    return tuple(t.cpu() for t in result)
  return result.cpu()


class UnboundedDynamicKernelTestBase(op_testing.TorchTpuVsCpuTestBase):
  """Base class for tests of ops served by unbounded dynamic kernels."""

  def run_and_check_cache(
      self,
      op: Callable[..., Any],
      *inputs: torch.Tensor,
      served: bool,
      static_compilations: int = 1,
      expected: Any = None,
  ) -> tuple[list[torch.Tensor], Any]:
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

    self.assert_close(
        golden_result=op(*inputs) if expected is None else expected,
        torch_tpu_result=_to_cpu(result),
    )

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
    # 3-D to 8-D: several row blocks with flushes spanning slices, wide rows,
    # then one case per rank. M is a multiple of 8 so that XLA, which picks the
    # input layout that pads least, defaults to row-major T(8,128).
    _nonzero_case((3, 16, 129), 1025),
    _nonzero_case((2, 8, 16385), 5),
    _nonzero_case((2, 3, 8, 129), _ALL_NONZERO),
    _nonzero_case((2, 1, 3, 8, 129), 777),
    _nonzero_case((1, 2, 1, 2, 8, 129), _ALL_NONZERO),
    _nonzero_case((2, 1, 2, 1, 2, 8, 129), 99),
    _nonzero_case((1, 2, 1, 2, 1, 2, 8, 129), _ALL_NONZERO),
    # U8, packed four rows (of 128 for 1-D) to a word: row counts that aren't a
    # multiple of four, several tiles per row, wide rows, flushes, the default
    # 1-D tiles T(512) and T(1024) around their boundary, then one case per
    # rank above two.
    _nonzero_case((5, 5), 3, torch.uint8),
    _nonzero_case((9, 1100), _ALL_NONZERO, torch.uint8),
    _nonzero_case((5, 16385), 3, torch.uint8),
    _nonzero_case((9, 129), 1025, torch.uint8),
    _nonzero_case((2,), 2, torch.uint8),
    _nonzero_case((129,), 2, torch.uint8),
    _nonzero_case((512,), _ALL_NONZERO, torch.uint8),
    _nonzero_case((513,), _ALL_NONZERO, torch.uint8),
    _nonzero_case((3000,), 1025, torch.uint8),
    _nonzero_case((40000,), 777, torch.uint8),
    _nonzero_case((3, 16, 129), 1025, torch.uint8),
    _nonzero_case((2, 8, 16385), 5, torch.uint8),
    _nonzero_case((2, 3, 8, 129), _ALL_NONZERO, torch.uint8),
    _nonzero_case((2, 1, 3, 8, 129), 777, torch.uint8),
    _nonzero_case((1, 2, 1, 2, 8, 129), _ALL_NONZERO, torch.uint8),
    _nonzero_case((2, 1, 2, 1, 2, 8, 129), 99, torch.uint8),
    _nonzero_case((1, 2, 1, 2, 1, 2, 8, 129), _ALL_NONZERO, torch.uint8),
    # bool reuses the U8 kernels via an HLO bitcast from PRED to U8; cover
    # both 1-D tile sizes, 2-D, and N-D without duplicating the U8 edge cases.
    _nonzero_case((129,), 5, torch.bool),
    _nonzero_case((513,), 7, torch.bool),
    _nonzero_case((9, 129), 10, torch.bool),
    _nonzero_case((2, 8, 129), 10, torch.bool),
)

_NONZERO_NOT_SERVED_CASES = (
    # Default input layout isn't row-major with T(8,128) tiles.
    _nonzero_case((3, 7), 5),
    _nonzero_case((5, 1), 5),
    _nonzero_case((5, 3), 5),
    _nonzero_case((2, 4, 4), 5),
    # Nor is it T(8,128)(4,1) for U8 or bool, e.g. T(32,128)(4,1) when M is a
    # multiple of 32.
    _nonzero_case((3, 7), 5, torch.uint8),
    _nonzero_case((32, 129), 5, torch.uint8),
    _nonzero_case((32, 129), 5, torch.bool),
    # K = 0 skips the index graph; K <= rank rounded up to a power of two uses
    # row-major S64[K,rank] layout.
    _nonzero_case((17, 257), 0),
    _nonzero_case((17, 257), 1),
    _nonzero_case((17, 257), 2),
    _nonzero_case((3000,), 0),
    _nonzero_case((3000,), 1),
    _nonzero_case((16,), 1),
    _nonzero_case((2, 8, 129), 4),
    # Neither F32, U8, nor bool, or rank above 8.
    _nonzero_case((17, 257), 5, torch.bfloat16),
    _nonzero_case((3000,), 5, torch.bfloat16),
    _nonzero_case((16,), 5, torch.bfloat16),
    _nonzero_case((1,) * 7 + (5, 5), 10),
)


def _sort_case(n: int, repeated: bool) -> torch.Tensor:
  """Returns `n` int32 keys, distinct unless `repeated`, with the extremes."""
  generator = torch.Generator().manual_seed(n)
  if repeated:
    x = torch.randint(-8, 9, (n,), generator=generator, dtype=torch.int32)
  else:
    x = (torch.randperm(n, generator=generator) - n // 2).to(torch.int32)
  x[1] = torch.iinfo(torch.int32).min
  x[n // 2] = torch.iinfo(torch.int32).max
  return x


# The opt-in sort kernel serves ascending sorts of 1024 int32 elements along
# dim -1. It is stable, so unstable sorts get distinct keys, whose sorted
# indices are unique.
_SORT_SERVED_CASES = (
    ("default", _sort_case(1024, repeated=False), {}),
    ("stable", _sort_case(1024, repeated=True), {"stable": True}),
)

_SORT_NOT_SERVED_CASES = (
    ("descending", _sort_case(1024, repeated=False), {"descending": True}),
    ("dim_0", _sort_case(1024, repeated=False), {"dim": 0}),
    ("n_1023", _sort_case(1023, repeated=False), {}),
    ("n_2048", _sort_case(2048, repeated=False), {}),
    ("rank_2", _sort_case(1024, repeated=False).reshape(8, 128), {}),
    ("float32", _sort_case(1024, repeated=False).float(), {}),
)


def _mm_case(
    m: int, k: int, n: int, dtype: torch.dtype = torch.float32
) -> tuple[str, torch.Tensor, torch.Tensor]:
  """Returns a (name, a, b) case for an [m, k] @ [k, n] product.

  The operands hold small integers, so that the result is exact at any matmul
  precision.
  """
  generator = torch.Generator().manual_seed(m + k + n)
  a = torch.randint(-4, 5, (m, k), generator=generator).to(dtype)
  b = torch.randint(-4, 5, (k, n), generator=generator).to(dtype)
  name = f"{m}x{k}x{n}_" + str(dtype).removeprefix("torch.")
  return name, a, b


# The opt-in matmul kernel serves 2-D float32 and bfloat16 products whose
# operands and result all have the row-major T(8,128) layout. XLA defaults to
# the layout that pads least, e.g. a column-major one for a (600, 700) matrix.
# The kernel computes 512x512 output blocks in 512-deep contraction steps.
_MM_SERVED_CASES = (
    # One block, full or partial.
    _mm_case(512, 512, 512),
    _mm_case(5, 5, 5),
    # Partial edge blocks along M, N and K.
    _mm_case(600, 768, 640),
    _mm_case(300, 300, 1300),
    _mm_case(16, 1100, 1280),
    # bfloat16 rows are padded to 8 in HBM, but to 16 in VMEM.
    _mm_case(512, 512, 512, torch.bfloat16),
    _mm_case(1000, 640, 512, torch.bfloat16),
    _mm_case(24, 256, 128, torch.bfloat16),
    # Mid-lane-tile and mid-row-tile K tail in bfloat16, 3 blocks per program.
    _mm_case(600, 1100, 1280, torch.bfloat16),
)

_MM_NOT_SERVED_CASES = (
    # Default layout of an operand isn't row-major T(8,128).
    _mm_case(3, 128, 128),
    _mm_case(1000, 48, 256),
    # Empty product.
    _mm_case(16, 0, 128),
    # Not float32 or bfloat16.
    _mm_case(16, 128, 128, torch.float64),
)


def _unique_case(
    n: int, u: int, dtype: torch.dtype = torch.int32
) -> tuple[str, torch.Tensor]:
  """Returns a (name, input) case of `n` shuffled elements with `u` uniques."""
  name = f"n{n}_u{u}_{str(dtype).removeprefix('torch.')}"
  generator = torch.Generator().manual_seed(n + u)
  values = torch.arange(u) - u // 2
  x = values[torch.arange(n) % u][torch.randperm(n, generator=generator)]
  return name, x.to(dtype)


# The kernels serve the values graph of int32 inputs of any rank
# (`UniqueSortAndCount` flattens to a 1-D sorted tensor), one kernel per XLA
# default 1-D input tile (N rounded up to a power of two in [128, 1024]).
_UNIQUE_SERVED_CASES = (
    # All four input tiles T(128..1024) and all four output tiles T(128..1024).
    _unique_case(1, 1),
    _unique_case(129, 2),
    _unique_case(257, 200),
    _unique_case(512, 300),
    _unique_case(1025, 1025),
    # Multi-window input with staging flushes and a partial tail.
    _unique_case(40000, 1025),
    # Many unique values, where the XLA path is slowest.
    _unique_case(262144, 262144),
    # Higher-rank input flattened by UniqueSortAndCount.
    (
        "shape_32x128_u300_int32",
        _unique_case(32 * 128, 300)[1].reshape(32, 128),
    ),
)

_UNIQUE_NOT_SERVED_CASES = (
    _unique_case(3000, 777, torch.float32),
    _unique_case(3000, 777, torch.int64),
)


class DynamicOpsTest(UnboundedDynamicKernelTestBase):

  @parameterized.named_parameters(*_NONZERO_SERVED_CASES)
  def test_nonzero_served(self, x: torch.Tensor):
    if (
        x.dtype == torch.bool
        and oss_utils.is_oss()
        and (ver := utils.libtpu_version()) is not None
        and ver < (0, 0, 54)
    ):
      torch.tpu._clear_cache()  # pylint: disable=protected-access
      torch.nonzero(torch.zeros_like(x).to("tpu"))  # Warm up the count graph.
      self.run_and_check_cache(torch.nonzero, x, served=False)
      return

    # Warm up the dynamic index kernel and this shape's static count graph
    # (a CPU-to-TPU copy avoids fusing constant ops into either graph).
    torch.nonzero(torch.ones_like(x).to("tpu"))

    (x_tpu,), result = self.run_and_check_cache(torch.nonzero, x, served=True)

    if x.dtype == torch.float32:
      # XLA's default 1-D tile: N rounded up to a power of two in [128, 1024].
      tiles_1d = [[min(max(128, 1 << (x.numel() - 1).bit_length()), 1024)]]
      tiles = [[8, 128]]
    else:
      # U8 and bool tiles pack four rows (of 128 for 1-D) to a word.
      tiles_1d = [[512 if x.numel() <= 512 else 1024], [128], [4, 1]]
      tiles = [[8, 128], [4, 1]]
    # (minor_to_major, tiles, element_size_in_bits); see nonzero_indices.h.
    self.assertEqual(
        torch.tpu.get_device_layout_if_materialized(x_tpu),
        ([0], tiles_1d, 0)
        if x.ndim == 1
        else (list(range(x.ndim))[::-1], tiles, 0),
    )
    self.assertEqual(
        torch.tpu.get_device_layout_if_materialized(result),
        ([0, 1], [[1 << (x.ndim - 1).bit_length(), 128]], 0),
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

  @parameterized.named_parameters(*_SORT_SERVED_CASES)
  def test_sort_served(self, x: torch.Tensor, kwargs: dict[str, Any]):
    sort = functools.partial(torch.sort, **kwargs)
    sort(torch.zeros_like(x).to("tpu"))  # Warm up the opt-in kernel.

    self.run_and_check_cache(sort, x, served=True)

  def test_sort_shares_the_stable_kernel(self):
    # Unstable sorts are served by the stable sort's kernel, so once a stable
    # sort has loaded it, an unstable sort compiles nothing.
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    x = _sort_case(1024, repeated=False)
    torch.sort(torch.zeros_like(x).to("tpu"), stable=True)

    self.run_and_check_cache(torch.sort, x, served=True)

  @parameterized.named_parameters(*_SORT_NOT_SERVED_CASES)
  def test_sort_not_served(self, x: torch.Tensor, kwargs: dict[str, Any]):
    torch.tpu._clear_cache()  # pylint: disable=protected-access

    self.run_and_check_cache(
        functools.partial(torch.sort, **kwargs), x, served=False
    )

  @parameterized.named_parameters(*_MM_SERVED_CASES)
  def test_mm_served(self, a: torch.Tensor, b: torch.Tensor):
    # Warm up the opt-in kernel on a different shape so the call below proves
    # one executable serves any shape without recompiling.
    torch.mm(
        torch.zeros(16, 128, dtype=a.dtype).to("tpu"),
        torch.zeros(128, 128, dtype=b.dtype).to("tpu"),
    )

    tpu_inputs, result = self.run_and_check_cache(torch.mm, a, b, served=True)

    # (minor_to_major, tiles, element_size_in_bits); see matmul.h.
    tiles = [[8, 128], [2, 1]] if a.dtype == torch.bfloat16 else [[8, 128]]
    for x in (*tpu_inputs, result):
      self.assertEqual(
          torch.tpu.get_device_layout_if_materialized(x), ([1, 0], tiles, 0)
      )

  @parameterized.named_parameters(*_MM_NOT_SERVED_CASES)
  def test_mm_not_served(self, a: torch.Tensor, b: torch.Tensor):
    torch.tpu._clear_cache()  # pylint: disable=protected-access

    self.run_and_check_cache(torch.mm, a, b, served=False)

  @parameterized.named_parameters(
      ("default", torch.tpu.Precision.DEFAULT, 0.0),
      ("high", torch.tpu.Precision.HIGH, None),
      ("highest", torch.tpu.Precision.HIGHEST, None),
  )
  def test_mm_served_at_precision(
      self, precision: torch.tpu.Precision, expected_value: float | None
  ):
    # Each pair of products cancels down to 2**-8, which the single bfloat16
    # pass of the default precision rounds away (giving 0 instead of 1).
    a = torch.tensor([1 + 2**-8, -1.0]).repeat(16, 256)
    b = torch.ones(512, 128)
    expected = (
        None
        if expected_value is None
        else torch.full((16, 128), expected_value)
    )

    with torch.tpu.precision(precision):
      # Warm up the opt-in kernel on a different shape.
      torch.mm(
          torch.zeros(16, 128).to("tpu"),
          torch.zeros(128, 128).to("tpu"),
      )

      self.run_and_check_cache(torch.mm, a, b, served=True, expected=expected)

  def test_mm_bfloat16_shares_the_default_precision_kernel(self):
    # bfloat16 products are served by one kernel at every precision, so once
    # the default precision has loaded it, another precision compiles nothing.
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    _, a, b = _mm_case(24, 256, 128, torch.bfloat16)
    torch.mm(torch.zeros_like(a).to("tpu"), torch.zeros_like(b).to("tpu"))

    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
      self.run_and_check_cache(torch.mm, a, b, served=True)

  def test_mm_served_back_to_back(self):
    # Launches overlap on the device, so a copy running past the end of its
    # buffer corrupts the next launch. 1000 bfloat16 rows end mid VMEM tile.
    _, a, b = _mm_case(1000, 640, 512, torch.bfloat16)
    a_tpu, b_tpu = a.to("tpu"), b.to("tpu")
    before = torch.tpu._get_cache_stats()  # pylint: disable=protected-access

    results = [torch.mm(a_tpu, b_tpu) for _ in range(8)]
    torch.accelerator.synchronize()

    after = torch.tpu._get_cache_stats()  # pylint: disable=protected-access
    self.assertEqual(
        after.num_unbounded_dynamic_hits - before.num_unbounded_dynamic_hits, 8
    )
    for result in results:
      self.assert_close(
          golden_result=torch.mm(a, b), torch_tpu_result=result.cpu()
      )

  @parameterized.named_parameters(
      (f"{name}_inverse{return_inverse}", x, return_inverse)
      for name, x in _UNIQUE_SERVED_CASES
      for return_inverse in (False, True)
  )
  def test_unique_served(self, x: torch.Tensor, return_inverse: bool):
    unique = functools.partial(torch.unique, return_inverse=return_inverse)
    # Warm up this size's sort graph, and the kernel, with one unique value.
    unique(torch.zeros_like(x).to("tpu"))

    _, result = self.run_and_check_cache(unique, x, served=True)
    values = result[0] if return_inverse else result
    out_tile = min(max(128, 1 << (values.numel() - 1).bit_length()), 1024)
    # (minor_to_major, tiles, element_size_in_bits); see unique.h.
    self.assertEqual(
        torch.tpu.get_device_layout_if_materialized(values),
        ([0], [[out_tile]], 0),
    )

  @parameterized.named_parameters(*_UNIQUE_NOT_SERVED_CASES)
  def test_unique_not_served(self, x: torch.Tensor):
    torch.tpu._clear_cache()  # pylint: disable=protected-access
    torch.unique(torch.zeros_like(x).to("tpu"))  # Warm up the sort graph.

    self.run_and_check_cache(torch.unique, x, served=False)


if __name__ == "__main__":
  absltest.main()
