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

from collections.abc import Callable
import gc

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


def _count_executed_graphs(fn: Callable[[torch.Tensor], torch.Tensor]) -> int:
  """Returns how many graphs are materialized for `x * 2 -> fn -> * 3`.

  The surrounding ops are deferred in DEFER_AND_FUSE mode, so they are fused
  with `fn`'s ops into a single graph unless `fn` introduces graph boundaries.

  Args:
    fn: The function to run between the surrounding ops.
  """
  x = torch.ones(2, 2).to("tpu")
  torch.accelerator.synchronize()
  # TODO: migrate to shared count_tpu_compilations() helper
  # pylint: disable-next=protected-access
  get_cache_stats = getattr(torch, "tpu")._get_cache_stats
  with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
    reqs_before = get_cache_stats().num_cache_reqs
    out = fn(x * 2) * 3
    out_cpu = out.cpu()
    torch.accelerator.synchronize()
    reqs_after = get_cache_stats().num_cache_reqs
  test_utils.assert_close(out_cpu, torch.full((2, 2), 9.0))
  return reqs_after - reqs_before


class SpmdSafeTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_spmd_safe_isolates_function_graph(self):
    def add_one(a):
      return a + 1

    undecorated = _count_executed_graphs(add_one)
    decorated = _count_executed_graphs(spmd_util.spmd_safe(add_one))
    # Without the decorator, everything fuses into a single graph.
    self.assertEqual(undecorated, 1)
    # The decorator adds exactly two graph boundaries (before and after the
    # function), so no outside op is fused with the function's ops.
    self.assertEqual(decorated - undecorated, 2)

  def test_spmd_safe_no_memory_overhead(self):
    device = torch.device("tpu")
    dtype = torch.float32

    def run_compilation(use_spmd_safe: bool) -> int:
      def simple_fn(a: torch.Tensor, b: torch.Tensor):
        return a + b, a

      if use_spmd_safe:
        simple_fn = spmd_util.spmd_safe(simple_fn)

      compiled_fn = torch.compile(simple_fn, backend="tpu")

      a = torch.randn(8192, 8192, device=device, dtype=dtype)
      b = torch.randn(8192, 8192, device=device, dtype=dtype)

      # Warmup run to compile and initialize runtime buffers
      out1, out2 = compiled_fn(a, b)
      del out1, out2
      gc.collect()
      torch.accelerator.synchronize()

      torch.accelerator.reset_peak_memory_stats()
      mem_before = torch.accelerator.memory_allocated()

      out1, out2 = compiled_fn(a, b)
      torch.accelerator.synchronize()
      peak_mem = torch.accelerator.max_memory_allocated()

      mem_increase = peak_mem - mem_before

      del a, b, out1, out2, compiled_fn
      gc.collect()
      torch.accelerator.synchronize()

      return mem_increase

    mem_without = run_compilation(use_spmd_safe=False)
    mem_with = run_compilation(use_spmd_safe=True)

    self.assertAlmostEqual(mem_with, mem_without)


if __name__ == "__main__":
  absltest.main()
