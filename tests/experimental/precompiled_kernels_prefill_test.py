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

"""In-process prefill test for `torch.tpu.experimental.precompiled_kernels`.

Verifies that `load_kernels()` populates the in-memory compilation cache so
that a subsequent unbounded dynamic op is served from the cache instead of
being compiled on the fly.

Persistent Tier-2 / Tier-3 cache behaviour, which needs isolated subprocesses,
is covered by `precompiled_kernels_test.py`.
"""

import time

from absl import logging
from absl.testing import absltest
import torch
from torch.tpu.experimental import precompiled_kernels  # pyrefly: ignore[missing-import]
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


class PrecompiledKernelsPrefillTest(seed_test_utils.RepeatableTest):

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()

  def test_prefill(self) -> None:
    """Prefilled precompiled kernels are served from the cache."""
    tensor = torch.zeros((128, 128), dtype=torch.float32)
    tensor[:64, :64] = 1
    expected = torch.nonzero(tensor)
    tensor = tensor.to("tpu")

    # Without prefill, the op compiles a static kernel: no cache hit.
    torch.tpu._clear_cache()
    hits_before = torch.tpu._get_cache_hits()
    start = time.perf_counter()
    static_res = torch.nonzero(tensor).cpu()
    no_prefill_time = time.perf_counter() - start
    self.assertEqual(torch.tpu._get_cache_hits() - hits_before, 0)

    # With prefill, the op is served from the unbounded dynamic cache.
    torch.tpu._clear_cache()
    precompiled_kernels.load_kernels()
    hits_before = torch.tpu._get_cache_hits()
    start = time.perf_counter()
    dynamic_res = torch.nonzero(tensor).cpu()
    prefill_time = time.perf_counter() - start
    self.assertEqual(torch.tpu._get_cache_hits() - hits_before, 1)

    test_utils.assert_close(static_res, expected)
    test_utils.assert_close(dynamic_res, expected)

    logging.info("No prefill time: %s seconds", no_prefill_time)
    logging.info("Pre-fill time: %s seconds", prefill_time)


if __name__ == "__main__":
  absltest.main()
