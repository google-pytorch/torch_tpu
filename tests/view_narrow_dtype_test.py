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

"""Tests eager ops through wide-dtype views of a narrow-dtype storage.

Sharding libraries such as FlexShard pack parameters into one uint8 buffer and
use `buffer[off:off + n].view(torch.float32)` as each parameter. Reading or
writing such a view must only bitcast the view's own bytes, so its cost must
not depend on the size of the uint8 buffer.
"""

import gc
import time

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import execution_mode
from tests import seed_test_utils

EagerMode = execution_mode.EagerMode

_MIB = 1024 * 1024
_LARGE_BASE_BYTES = 256 * _MIB
_SMALL_BASE_BYTES = 8 * _MIB
# The view holds 1M fp32 elements (4 MiB).
_VIEW_NUMEL = 1024 * 1024
_VIEW_BYTES = _VIEW_NUMEL * 4
# Allowed (view in large base) / (view in small base) time ratio.
_MAX_BASE_SIZE_RATIO = 2.0

# How the uint8 base and its float32 view are shaped:
#   1d: a 1D uint8 base, viewed as a 1D float32 tensor.
#   2d: a 1D uint8 base, viewed as a 1024 x 1024 float32 tensor.
#   3d: a [rows, 1024, 4] uint8 base, whose middle rows are viewed as a
#       1024 x 1024 float32 tensor.
_VIEW_KINDS = ("1d", "2d", "3d")


def _make_base_cpu(kind: str, base_bytes: int) -> torch.Tensor:
  """Returns a uint8 CPU base of `base_bytes` holding finite fp32 values."""
  base = torch.randn(base_bytes // 4, dtype=torch.float32).view(torch.uint8)
  if kind == "3d":
    return base.view(-1, 1024, 4)
  return base


def _view_byte_offset(base_bytes: int) -> int:
  """Returns the byte offset of the view, centered in the base."""
  return (base_bytes - _VIEW_BYTES) // 2


def _make_view(kind: str, base: torch.Tensor) -> torch.Tensor:
  """Returns the float32 view in the middle of the uint8 `base`."""
  off = _view_byte_offset(base.numel())
  if kind == "3d":
    row = off // 4096
    return base[row : row + 1024].view(torch.float32).view(1024, 1024)
  flat = base[off : off + _VIEW_BYTES].view(torch.float32)
  return flat.view(1024, 1024) if kind == "2d" else flat


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
  """Returns the raw bytes of a CPU tensor."""
  return t.contiguous().view(-1).view(torch.uint8)


class ViewNarrowDtypeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    gc.collect()
    torch.accelerator.synchronize()

  def tearDown(self):
    super().tearDown()
    gc.collect()
    torch.accelerator.synchronize()

  def _operands(self, kind: str):
    """Returns (tensor1, tensor2) shaped like the view, on TPU."""
    shape = (_VIEW_NUMEL,) if kind == "1d" else (1024, 1024)
    t1 = torch.randn(shape, dtype=torch.float32)
    t2 = torch.rand(shape, dtype=torch.float32) + 1.0
    return t1.to(self.device), t2.to(self.device)

  def _apply(self, op: str, view: torch.Tensor, t1, t2):
    """Applies `op` and returns the tensor it wrote.

    These three optimizer-style ops (from AdamW/Muon steps) cover the distinct
    ways a view participates in an op:
      - `mul_`: in-place write on `view` with a scalar (reads `view` via
        forward view decomposition and writes back via inverse decomposition).
      - `addcdiv_`: in-place write on `view` combining multiple tensor
        operands (tests forward read + inverse write-back alongside other
        inputs).
      - `lerp_`: read-only use of `view` as a source operand while mutating a
        standalone tensor `t1` (isolates forward view decomposition without
        inverse write-back on `view`).
    """
    if op == "mul_":
      return view.mul_(0.5)
    if op == "addcdiv_":
      return view.addcdiv_(t1, t2, value=-0.125)
    if op == "lerp_":
      return t1.lerp_(view, 0.25)
    raise ValueError(f"unknown op: {op}")

  @parameterized.product(kind=_VIEW_KINDS, op=("mul_", "addcdiv_", "lerp_"))
  def test_matches_standalone_and_preserves_rest_of_base(self, kind, op):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      base_cpu = _make_base_cpu(kind, _LARGE_BASE_BYTES)
      base = base_cpu.to(self.device)
      view = _make_view(kind, base)
      t1, t2 = self._operands(kind)

      # Same op on a standalone (non-view) TPU tensor holding the same values.
      standalone = _make_view(kind, base_cpu).clone().to(self.device)
      expected = self._apply(op, standalone, t1.clone(), t2.clone()).cpu()
      actual = self._apply(op, view, t1, t2).cpu()

      # Bit-exact equality with the standalone op.
      self.assertTrue(torch.equal(_as_bytes(actual), _as_bytes(expected)))
      expected_base = base_cpu.clone()
      if op != "lerp_":
        _make_view(kind, expected_base).copy_(expected)
      # The rest of the base is unchanged (and, for a read, all of it).
      self.assertTrue(
          torch.equal(_as_bytes(base.cpu()), _as_bytes(expected_base))
      )

  @parameterized.parameters("mul_", "addcdiv_", "lerp_")
  def test_strided_column_view_matches_standalone_and_preserves_base(self, op):
    """Tests a column slice across all rows (GitHub issue #6046)."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      base_cpu = (
          torch.randn(32 * 32, dtype=torch.float32)
          .view(torch.uint8)
          .view(32, 128)
      )
      base = base_cpu.to(self.device)
      view = base.view(torch.float32)[:, 1]
      t1 = torch.randn(32, dtype=torch.float32).to(self.device)
      t2 = (torch.rand(32, dtype=torch.float32) + 1.0).to(self.device)

      standalone = base_cpu.view(torch.float32)[:, 1].clone().to(self.device)
      expected = self._apply(op, standalone, t1.clone(), t2.clone()).cpu()
      actual = self._apply(op, view, t1, t2).cpu()

      self.assertTrue(torch.equal(_as_bytes(actual), _as_bytes(expected)))
      expected_base = base_cpu.clone()
      if op != "lerp_":
        expected_base.view(torch.float32)[:, 1].copy_(expected)
      self.assertTrue(
          torch.equal(_as_bytes(base.cpu()), _as_bytes(expected_base))
      )

  def _best_time_per_op(self, fn, warmup=10, trials=5, iters=20) -> float:
    for _ in range(warmup):
      fn()
    torch.accelerator.synchronize()
    best = float("inf")
    for _ in range(trials):
      start = time.perf_counter()
      for _ in range(iters):
        fn()
      torch.accelerator.synchronize()
      best = min(best, (time.perf_counter() - start) / iters)
    return best

  def _view_time(self, kind: str, op: str, base_bytes: int) -> float:
    base = _make_base_cpu(kind, base_bytes).to(self.device)
    view = _make_view(kind, base)
    t1, t2 = self._operands(kind)
    return self._best_time_per_op(lambda: self._apply(op, view, t1, t2))

  @parameterized.product(kind=_VIEW_KINDS, op=("mul_", "lerp_"))
  def test_cost_independent_of_base_size(self, kind, op):
    """An op on the view costs the same in a 256 MiB and an 8 MiB base."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      small = self._view_time(kind, op, _SMALL_BASE_BYTES)
      large = self._view_time(kind, op, _LARGE_BASE_BYTES)
    ratio = large / small
    print(
        f"[base_size] kind={kind} op={op} base256_us={large * 1e6:.1f} "
        f"base8_us={small * 1e6:.1f} ratio={ratio:.2f}",
        flush=True,
    )
    self.assertLess(ratio, _MAX_BASE_SIZE_RATIO)


if __name__ == "__main__":
  absltest.main()
