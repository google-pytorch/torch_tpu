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

"""Compilation benchmark for async compilation."""

import concurrent.futures
import statistics
import time

from absl import flags
from absl.testing import absltest
import torch
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd import autograd_cache
from torch._inductor.codecache import FxGraphCache
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import clear_caches
from torch.compiler._cache import CacheArtifactManager
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.benchmarks import xprof_adapter
from tests import module_registry
from tests import seed_test_utils

_MODEL_NAME = flags.DEFINE_string(
    "model_name", "Qwen/Qwen3-0.6B", "Model name to benchmark."
)

_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials", 1, "Number of trials for each measurement."
)

_NUM_BUCKETS = flags.DEFINE_integer(
    "num_buckets", 3, "Number of different bucket sizes."
)

_SEQ_LEN = flags.DEFINE_integer(
    "seq_len", 32, "Sequence length for input tokens."
)


def _reset() -> None:
  """Resets torch compiler, dynamo, inductor codecache, and AOTAutograd cache."""
  torch.compiler.reset()
  torch.tpu._clear_cache()
  torch._dynamo.reset()
  DynamoCache.clear()
  PrecompileContext.clear()
  counters.clear()
  autograd_cache.AOTAutogradCache.clear()
  CacheArtifactManager.clear()
  PyCodeCache.cache_clear(purge=True)
  FxGraphCache.clear()
  clear_caches()


class AsyncCompileBenchmarkTest(seed_test_utils.RepeatableTest):

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()
    _reset()

  def tearDown(self) -> None:
    _reset()
    super().tearDown()

  def _run_trial(self, lazy: bool) -> tuple[float, float, float]:
    _reset()
    registry = module_registry.ModuleRegistry()
    spec = registry.get_module_spec(
        "transformers",
        _MODEL_NAME.value,
        load_weights=False,
    )
    with torch.device("tpu"):
      model = spec.module_factory()
    model.eval()

    bucket_sizes = [16 * (i + 1) for i in range(_NUM_BUCKETS.value)]
    inputs = [
        spec.sample_inputs_factory((b, _SEQ_LEN.value), device="tpu")
        for b in bucket_sizes
    ]

    torch.accelerator.synchronize()
    start_total = time.perf_counter()

    with xprof_adapter.TraceMe("precompile"):
      if lazy:
        # Precompile phase: Dynamo tracing + MLIR dispatch + async warmup barrier
        compiled_model = tpu_compile.async_compile(
            model,
            inputs,
        )
      else:
        # Synchronous mode calls torch.compile directly and executes all buckets
        compiled_model = torch.compile(
            torch.no_grad()(model),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        for args, kwargs in inputs:
          compiled_model(*args, **kwargs)

    end_precompile = time.perf_counter()

    # Execution phase
    def run_one(i, item):
      args, kwargs = item
      with xprof_adapter.TraceMe("execute", step_num=i):
        with torch.no_grad():
          return compiled_model(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(inputs)
    ) as executor:
      futures = [
          executor.submit(run_one, idx, item) for idx, item in enumerate(inputs)
      ]
      _ = [f.result() for f in futures]

    torch.accelerator.synchronize()
    end_total = time.perf_counter()

    total_time = end_total - start_total
    precompile_time = end_precompile - start_total
    exec_time = end_total - end_precompile
    return total_time, precompile_time, exec_time

  def test_async_compile(self):
    sync_times = []
    sync_precompile_times = []
    sync_exec_times = []
    for i in range(_NUM_TRIALS.value):
      with xprof_adapter.TraceMe("Sync", step_num=i):
        t, p, e = self._run_trial(lazy=False)
        sync_times.append(t)
        sync_precompile_times.append(p)
        sync_exec_times.append(e)

    _reset()

    lazy_times = []
    lazy_precompile_times = []
    lazy_exec_times = []
    for i in range(_NUM_TRIALS.value):
      with xprof_adapter.TraceMe("Async", step_num=i):
        t, p, e = self._run_trial(lazy=True)
        lazy_times.append(t)
        lazy_precompile_times.append(p)
        lazy_exec_times.append(e)

    print(
        f"\nResults (Trials={_NUM_TRIALS.value}, Buckets={_NUM_BUCKETS.value}):"
    )
    print("Precompile metrics (Time to capture graph + dispatch):")
    print(f"  Sync:  {statistics.median(sync_precompile_times):.4f}s")
    print(f"  Async: {statistics.median(lazy_precompile_times):.4f}s")
    print("Execution metrics (Time to run graphs, inc async compile wait):")
    print(f"  Sync:  {statistics.median(sync_exec_times):.4f}s")
    print(f"  Async: {statistics.median(lazy_exec_times):.4f}s")
    print("Total Time:")
    print(f"  Sync:  {statistics.median(sync_times):.4f}s")
    print(f"  Async: {statistics.median(lazy_times):.4f}s")


if __name__ == "__main__":
  absltest.main()
