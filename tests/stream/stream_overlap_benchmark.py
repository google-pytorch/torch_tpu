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

"""Checks that a side-stream 1 GiB copy overlaps compute on the device.

Saving = (copy + compute - both) / min(copy, compute): 1 full overlap, 0 none.
"""

from collections.abc import Callable
import statistics
import time

from absl import logging
from absl.testing import absltest
import torch
from tests import seed_test_utils

_DEVICE = torch.accelerator.current_accelerator()
_BACKEND = getattr(torch, _DEVICE.type)
_ITERATIONS = 7
# Serialized runs score about 0; overlapped runs measure 0.85 and up.
_MIN_MEDIAN_SAVING = 0.4
_COPY_SIDE = 16384  # 1 GiB of float32.
_MATMUL_SIDE = 8192


def _timed(fn: Callable[[], object]) -> float:
  """Returns the wall time of `fn()` plus the device synchronize after it."""
  start = time.perf_counter()
  keep_alive = fn()
  torch.accelerator.synchronize()
  del keep_alive
  return time.perf_counter() - start


class StreamOverlapBenchmark(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.x = torch.full((_MATMUL_SIDE,) * 2, 1 / _MATMUL_SIDE, device=_DEVICE)
    _timed(lambda: self._chain(4))
    self.per_matmul_s = _timed(lambda: self._chain(10)) / 10

  def _chain(self, k: int) -> torch.Tensor:
    y = self.x
    for _ in range(k):
      y = y @ self.x
    return y

  def _check_overlap(self, name: str, copy_fn: Callable[[], object]) -> None:
    """Asserts that `copy_fn` overlaps a matmul chain of about its duration."""
    _timed(copy_fn)
    t_copy = min(_timed(copy_fn) for _ in range(3))
    k = max(1, round(t_copy / self.per_matmul_s))
    savings = []
    for i in range(_ITERATIONS):
      t_copy = _timed(copy_fn)
      t_compute = _timed(lambda: self._chain(k))
      t_both = _timed(lambda: (copy_fn(), self._chain(k)))
      saving = (t_copy + t_compute - t_both) / min(t_copy, t_compute)
      savings.append(saving)
      logging.info(
          '%s iteration %d: copy %.1f ms, compute %.1f ms (%d matmuls),'
          ' both %.1f ms, saving %.3f',
          name,
          i,
          t_copy * 1e3,
          t_compute * 1e3,
          k,
          t_both * 1e3,
          saving,
      )
    median = statistics.median(savings)
    logging.info(
        '%s saving over %d iterations: min %.3f, median %.3f, max %.3f',
        name,
        _ITERATIONS,
        min(savings),
        median,
        max(savings),
    )
    self.assertGreaterEqual(median, _MIN_MEDIAN_SAVING)

  def test_h2d_prefetch_overlaps_compute(self):
    side = _BACKEND.Stream()
    pinned = torch.ones((_COPY_SIDE,) * 2, device='cpu', pin_memory=True)

    def prefetch() -> torch.Tensor:
      with _BACKEND.stream(side):
        return pinned.to(_DEVICE, non_blocking=True)

    self._check_overlap('H2D prefetch', prefetch)

  def test_d2h_offload_overlaps_compute(self):
    offload = _BACKEND.Stream()
    act = torch.full((_COPY_SIDE,) * 2, 2.0, device=_DEVICE)
    pinned = torch.empty((_COPY_SIDE,) * 2, device='cpu', pin_memory=True)

    def offload_act() -> torch.Tensor:
      offload.wait_stream(torch.accelerator.current_stream())
      with _BACKEND.stream(offload):
        pinned.copy_(act, non_blocking=True)
      return pinned

    self._check_overlap('D2H offload', offload_act)


if __name__ == '__main__':
  absltest.main()
