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
from torch import nn
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing

from torch_tpu._internal.shims.xprof import traceme

_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials", 1, "Number of trials for each measurement."
)

_NUM_BUCKETS = flags.DEFINE_integer(
    "num_buckets", 3, "Number of different bucket sizes."
)


class RMSNorm(nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return norm * self.weight


class Qwen3MLP(nn.Module):

  def __init__(self, emb_dim: int, hidden_dim: int):
    super().__init__()
    self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
    self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)
    self.act = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):

  def __init__(
      self,
      emb_dim: int,
      n_heads: int,
      n_kv_groups: int,
      head_dim: int,
  ):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_groups = n_kv_groups
    self.head_dim = head_dim
    self.q_proj = nn.Linear(emb_dim, n_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(emb_dim, n_kv_groups * head_dim, bias=False)
    self.v_proj = nn.Linear(emb_dim, n_kv_groups * head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)
    self.q_norm = RMSNorm(head_dim)
    self.k_norm = RMSNorm(head_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, s, _ = x.shape
    q = self.q_norm(
        self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
    ).transpose(1, 2)
    k = self.k_norm(
        self.k_proj(x).view(b, s, self.n_kv_groups, self.head_dim)
    ).transpose(1, 2)
    v = (
        self.v_proj(x)
        .view(b, s, self.n_kv_groups, self.head_dim)
        .transpose(1, 2)
    )
    num_repeat = self.n_heads // self.n_kv_groups
    if num_repeat > 1:
      k = k.repeat_interleave(num_repeat, dim=1)
      v = v.repeat_interleave(num_repeat, dim=1)
    scale = self.head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn = torch.matmul(attn_weights, v)
    attn = attn.transpose(1, 2).contiguous().view(b, s, -1)
    return self.o_proj(attn)


class Qwen3Block(nn.Module):

  def __init__(
      self,
      emb_dim: int,
      n_heads: int,
      n_kv_groups: int,
      head_dim: int,
      hidden_dim: int,
  ):
    super().__init__()
    self.attn_norm = RMSNorm(emb_dim)
    self.attn = Qwen3Attention(emb_dim, n_heads, n_kv_groups, head_dim)
    self.mlp_norm = RMSNorm(emb_dim)
    self.mlp = Qwen3MLP(emb_dim, hidden_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.attn_norm(x))
    x = x + self.mlp(self.mlp_norm(x))
    return x


class Qwen3Model(nn.Module):
  """A Qwen3-level complexity model for compilation benchmarking."""

  def __init__(
      self,
      n_layers: int = 16,
      emb_dim: int = 2048,
      n_heads: int = 16,
      n_kv_groups: int = 4,
      head_dim: int = 128,
      hidden_dim: int = 5120,
  ):
    super().__init__()
    self.blocks = nn.ModuleList([
        Qwen3Block(emb_dim, n_heads, n_kv_groups, head_dim, hidden_dim)
        for _ in range(n_layers)
    ])
    self.final_norm = RMSNorm(emb_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      x = block(x)
    return self.final_norm(x)


def _reset() -> None:
  """Resets torch compiler and TPU cache."""
  torch.compiler.reset()
  if hasattr(torch, "tpu") and hasattr(torch.tpu, "_clear_cache"):
    torch.tpu._clear_cache()
  torch._dynamo.reset()
  try:
    from torch._functorch._aot_autograd import autograd_cache  # pylint: disable=g-import-not-at-top

    autograd_cache.AOTAutogradCache.clear()
  except (ImportError, AttributeError):
    pass


class AsyncCompileBenchmarkTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    tt_testing.reset_eager_state()

  def _run_trial(self, lazy: bool) -> tuple[float, float, float]:
    _reset()
    model = Qwen3Model().to("tpu")

    bucket_sizes = [16 * (i + 1) for i in range(_NUM_BUCKETS.value)]
    inputs = [torch.randn(b, 32, 2048, device="tpu") for b in bucket_sizes]

    torch.accelerator.synchronize()
    start_total = time.perf_counter()

    with traceme.TraceMe("precompile"):
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
        for x in inputs:
          compiled_model(x)

    end_precompile = time.perf_counter()

    # Execution phase
    def run_one(i, x):
      with traceme.TraceMe("execute", step_num=i):
        with torch.no_grad():
          return compiled_model(x)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(inputs)
    ) as executor:
      futures = [
          executor.submit(run_one, idx, x) for idx, x in enumerate(inputs)
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
      with traceme.TraceMe("Sync", step_num=i):
        t, p, e = self._run_trial(lazy=False)
        sync_times.append(t)
        sync_precompile_times.append(p)
        sync_exec_times.append(e)

    _reset()

    lazy_times = []
    lazy_precompile_times = []
    lazy_exec_times = []
    for i in range(_NUM_TRIALS.value):
      with traceme.TraceMe("Async", step_num=i):
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
