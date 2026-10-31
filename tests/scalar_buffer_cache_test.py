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

"""Regression tests for the eager scalar device-buffer cache (issue #6036)."""

import gc

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from tests import seed_test_utils

EagerMode = execution_mode.EagerMode

# Must stay above kMaxCachedScalarBuffers in csrc/eager/op_dispatcher.cc and
# below the number of distinct scalars used, so the test distinguishes a
# bounded cache from the unbounded one it replaces.
_MAX_EXTRA_LIVE_ALLOCATIONS = 1024 + 64
_NUM_DISTINCT_SCALARS = 5000


class ScalarBufferCacheTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    gc.collect()
    torch.accelerator.synchronize()

  def tearDown(self):
    super().tearDown()
    gc.collect()
    torch.accelerator.synchronize()

  def _live_allocations(self):
    torch.accelerator.synchronize()
    gc.collect()
    return torch.accelerator.memory_stats()["allocation.all.current"]

  def _assert_distinct_scalars_bounded(self, mode):
    with execution_mode.set_eager_mode(mode):
      x = torch.zeros(8, 128, dtype=torch.float32, device=self.device)
      # Warmup so the program and runtime buffers exist before measuring.
      x.add_(0.5)
      base = self._live_allocations()

      for i in range(_NUM_DISTINCT_SCALARS):
        x.add_(i / 1024)

      extra = self._live_allocations() - base
      # Unbounded cache: one live 0-d buffer per distinct scalar (~5000).
      self.assertLess(extra, _MAX_EXTRA_LIVE_ALLOCATIONS)
      self.assertTrue(torch.isfinite(x.cpu()).all())

  def test_distinct_scalars_bounded_in_defer_never(self):
    self._assert_distinct_scalars_bounded(EagerMode.DEFER_NEVER)

  def test_distinct_scalars_bounded_in_defer_and_fuse(self):
    self._assert_distinct_scalars_bounded(EagerMode.DEFER_AND_FUSE)

  def test_repeated_scalar_reuses_buffer(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      x = torch.zeros(8, 128, dtype=torch.float32, device=self.device)
      x.add_(0.5)
      base = self._live_allocations()

      for _ in range(1000):
        x.add_(0.5)

      self.assertLessEqual(self._live_allocations() - base, 2)


if __name__ == "__main__":
  absltest.main()
