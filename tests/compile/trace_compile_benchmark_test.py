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

"""Performance benchmark comparing trace_and_compile vs sync/async torch.compile."""

import gc
import statistics
import time

from absl import flags
from absl.testing import absltest
import torch
from torch import nn
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
from torch_tpu._internal.compile import trace_compiler
from torch_tpu._internal.benchmarks import xprof_adapter
from tests import seed_test_utils

_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials", 3, "Number of trials for each measurement."
)

_NUM_BUCKETS = flags.DEFINE_integer(
    "num_buckets", 4, "Number of different batch/shape buckets."
)


class BenchmarkMLP(nn.Module):
  """Multi-layer perceptron for compilation benchmarking."""

  def __init__(self, hidden_dim: int = 1024):
    super().__init__()
    self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = self.norm(x)
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x + residual


def _reset() -> None:
  """Resets torch compiler, dynamo, inductor, AOTAutograd, and TPU device caches."""
  tpu_compile.clear_cache()
  torch.compiler.reset()
  torch._dynamo.reset()
  DynamoCache.clear()
  PrecompileContext.clear()
  counters.clear()
  autograd_cache.AOTAutogradCache.clear()
  CacheArtifactManager.clear()
  PyCodeCache.cache_clear(purge=True)
  FxGraphCache.clear()
  clear_caches()
  gc.collect()
  if hasattr(torch, "tpu") and hasattr(torch.tpu, "_clear_cache"):
    torch.tpu._clear_cache()
  tt_testing.reset_eager_state()
  gc.collect()
  torch.accelerator.synchronize()


class TraceCompileBenchmarkTest(seed_test_utils.RepeatableTest):

  def setUp(self) -> None:
    super().setUp()
    _reset()

  def tearDown(self) -> None:
    _reset()
    super().tearDown()

  def _run_trial_sync_torch_compile(
      self, model: nn.Module, inputs: list[tuple[torch.Tensor]]
  ) -> tuple[float, float, float, float, float]:
    _reset()
    start_total = time.perf_counter()

    with xprof_adapter.TraceMe("sync_torch_compile.compile_and_execute_warmup"):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
      )
      with torch.no_grad():
        for (x,) in inputs:
          compiled(x)
      torch.accelerator.synchronize()

    end_compile = time.perf_counter()

    with xprof_adapter.TraceMe("sync_torch_compile.execute"):
      start_exec = time.perf_counter()
      with torch.no_grad():
        for (x,) in inputs:
          compiled(x)
      torch.accelerator.synchronize()
      end_exec = time.perf_counter()

    del compiled
    _reset()

    warm_exec_time = end_exec - start_exec
    total_time = end_compile - start_total
    compile_time = max(0.0, total_time - warm_exec_time)
    dispatch_time = (
        compile_time  # In sync mode, dispatch and compile are synchronous
    )
    first_exec_time = warm_exec_time

    return (
        total_time,
        dispatch_time,
        compile_time,
        first_exec_time,
        warm_exec_time,
    )

  def _run_trial_async_torch_compile(
      self, model: nn.Module, inputs: list[tuple[torch.Tensor]]
  ) -> tuple[float, float, float, float, float]:
    _reset()
    start_total = time.perf_counter()

    with xprof_adapter.TraceMe("async_torch_compile.compile"):
      compiled = tpu_compile.async_compile(
          model,
          inputs,
          no_grad=True,
      )
      torch.accelerator.synchronize()

    end_compile = time.perf_counter()

    with xprof_adapter.TraceMe("async_torch_compile.first_execute"):
      start_first_exec = time.perf_counter()
      with torch.no_grad():
        for (x,) in inputs:
          compiled(x)
      torch.accelerator.synchronize()
      end_first_exec = time.perf_counter()

    end_total = time.perf_counter()

    with xprof_adapter.TraceMe("async_torch_compile.execute"):
      start_warm_exec = time.perf_counter()
      with torch.no_grad():
        for (x,) in inputs:
          compiled(x)
      torch.accelerator.synchronize()
      end_warm_exec = time.perf_counter()

    del compiled
    _reset()

    compile_time = end_compile - start_total
    first_exec_time = end_first_exec - start_first_exec
    warm_exec_time = end_warm_exec - start_warm_exec
    dispatch_time = compile_time
    total_time = end_total - start_total

    return (
        total_time,
        dispatch_time,
        compile_time,
        first_exec_time,
        warm_exec_time,
    )

  def _run_trial_trace_and_compile(
      self, model: nn.Module, inputs: list[tuple[torch.Tensor]]
  ) -> tuple[float, float, float, float, float]:
    _reset()
    start_total = time.perf_counter()

    with xprof_adapter.TraceMe("trace_and_compile.dispatch"):
      start_dispatch = time.perf_counter()
      futures = [
          trace_compiler.trace_and_compile(
              model,
              (x,),
              module_name=f"trace_mlp_b{x.shape[0]}",
          )
          for (x,) in inputs
      ]
      end_dispatch = time.perf_counter()

    with xprof_adapter.TraceMe("trace_and_compile.wait"):
      executables = [f.result() for f in futures]
      torch.accelerator.synchronize()
      end_compile = time.perf_counter()

    with xprof_adapter.TraceMe("trace_and_compile.first_execute"):
      start_first_exec = time.perf_counter()
      with torch.no_grad():
        for exe, (x,) in zip(executables, inputs):
          exe(x)
      torch.accelerator.synchronize()
      end_first_exec = time.perf_counter()

    end_total = time.perf_counter()

    with xprof_adapter.TraceMe("trace_and_compile.execute"):
      start_warm_exec = time.perf_counter()
      with torch.no_grad():
        for exe, (x,) in zip(executables, inputs):
          exe(x)
      torch.accelerator.synchronize()
      end_warm_exec = time.perf_counter()

    del executables, futures
    _reset()

    dispatch_time = end_dispatch - start_dispatch
    compile_time = end_compile - start_total
    first_exec_time = end_first_exec - start_first_exec
    warm_exec_time = end_warm_exec - start_warm_exec
    total_time = end_total - start_total

    return (
        total_time,
        dispatch_time,
        compile_time,
        first_exec_time,
        warm_exec_time,
    )

  def test_benchmark_compilation_deltas(self):
    hidden_dim = 1024
    model = BenchmarkMLP(hidden_dim=hidden_dim).to("tpu").eval()

    bucket_sizes = [16 * (i + 1) for i in range(_NUM_BUCKETS.value)]
    inputs = [(torch.randn(b, hidden_dim, device="tpu"),) for b in bucket_sizes]

    sync_totals, sync_dispatches, sync_compiles, sync_1st, sync_warm = (
        [],
        [],
        [],
        [],
        [],
    )
    async_totals, async_dispatches, async_compiles, async_1st, async_warm = (
        [],
        [],
        [],
        [],
        [],
    )
    trace_totals, trace_dispatches, trace_compiles, trace_1st, trace_warm = (
        [],
        [],
        [],
        [],
        [],
    )

    for trial_idx in range(_NUM_TRIALS.value):
      if trial_idx > 0:
        _reset()
      t, d, c, e1, ew = self._run_trial_sync_torch_compile(model, inputs)
      sync_totals.append(t)
      sync_dispatches.append(d)
      sync_compiles.append(c)
      sync_1st.append(e1)
      sync_warm.append(ew)

      t, d, c, e1, ew = self._run_trial_async_torch_compile(model, inputs)
      async_totals.append(t)
      async_dispatches.append(d)
      async_compiles.append(c)
      async_1st.append(e1)
      async_warm.append(ew)

      t, d, c, e1, ew = self._run_trial_trace_and_compile(model, inputs)
      trace_totals.append(t)
      trace_dispatches.append(d)
      trace_compiles.append(c)
      trace_1st.append(e1)
      trace_warm.append(ew)

    med_trace_d = statistics.median(trace_dispatches)

    med_sync_c = statistics.median(sync_compiles)
    med_async_c = statistics.median(async_compiles)
    med_trace_c = statistics.median(trace_compiles)

    med_async_1st = statistics.median(async_1st)
    med_trace_1st = statistics.median(trace_1st)

    med_sync_warm = statistics.median(sync_warm)
    med_async_warm = statistics.median(async_warm)
    med_trace_warm = statistics.median(trace_warm)

    med_sync_t = statistics.median(sync_totals)
    med_async_t = statistics.median(async_totals)
    med_trace_t = statistics.median(trace_totals)

    def _row(
        candidate: str,
        dispatch: str,
        compile_time: str,
        first_exec: str,
        warm_exec: str,
        total_time: str,
        speedup: str,
    ) -> str:
      return (
          f"| {candidate:<24} | {dispatch:>20} | {compile_time:>9} |"
          f" {first_exec:>9} | {warm_exec:>9} | {total_time:>9} |"
          f" {speedup:>12} |"
      )

    sep = f"+{'-'*26}+{'-'*22}+{'-'*11}+{'-'*11}+{'-'*11}+{'-'*11}+{'-'*14}+"
    border = "=" * len(sep)

    report_lines = [
        border,
        (
            "Benchmark Results"
            f" (Trials={_NUM_TRIALS.value}, Buckets={_NUM_BUCKETS.value}):"
        ),
        border,
        _row(
            "Candidate",
            "Compilation Dispatch",
            "Compile",
            "1st Exec",
            "Warm Exec",
            "Total E2E",
            "E2E vs Sync",
        ),
        sep,
        _row(
            "Sync torch.compile",
            "N/A",
            f"{med_sync_c:.4f}s",
            "N/A",
            f"{med_sync_warm * 1000:.2f}ms",
            f"{med_sync_t:.4f}s",
            "baseline",
        ),
        _row(
            "Async torch.compile",
            "N/A",
            f"{med_async_c:.4f}s",
            f"{med_async_1st * 1000:.2f}ms",
            f"{med_async_warm * 1000:.2f}ms",
            f"{med_async_t:.4f}s",
            f"{med_sync_t / max(med_async_t, 1e-6):.2f}x",
        ),
        _row(
            "trace_and_compile (New)",
            f"{med_trace_d:.4f}s",
            f"{med_trace_c:.4f}s",
            f"{med_trace_1st * 1000:.2f}ms",
            f"{med_trace_warm * 1000:.2f}ms",
            f"{med_trace_t:.4f}s",
            f"{med_sync_t / max(med_trace_t, 1e-6):.2f}x",
        ),
        border,
    ]
    print("\n" + "\n".join(report_lines))

    self.assertGreater(len(trace_totals), 0)


if __name__ == "__main__":
  absltest.main()
