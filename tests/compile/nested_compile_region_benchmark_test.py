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

"""Benchmark comparing compilation time and execution latency with and without nested_compile_region."""

import time
from absl.testing import absltest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class RMSNorm(nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return x * norm * self.weight


class SwiGLUMLP(nn.Module):

  def __init__(self, hidden_dim: int, intermediate_dim: int):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
    self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
    self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerAttention(nn.Module):

  def __init__(self, hidden_dim: int, num_heads: int):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads
    self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, s, d = x.shape
    q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, d)
    return self.out_proj(out)


class TransformerBlock(nn.Module):

  def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int):
    super().__init__()
    self.norm1 = RMSNorm(hidden_dim)
    self.attn = TransformerAttention(hidden_dim, num_heads)
    self.norm2 = RMSNorm(hidden_dim)
    self.mlp = SwiGLUMLP(hidden_dim, intermediate_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = x + self.attn(self.norm1(x))
    return h + self.mlp(self.norm2(h))


class DeepTransformer(nn.Module):

  def __init__(
      self,
      num_layers: int,
      hidden_dim: int,
      intermediate_dim: int,
      num_heads: int,
  ):
    super().__init__()
    self.layers = nn.ModuleList([
        TransformerBlock(hidden_dim, intermediate_dim, num_heads)
        for _ in range(num_layers)
    ])
    self.final_norm = RMSNorm(hidden_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return self.final_norm(x)


class NestedCompileRegionBenchmarkTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def _run_benchmark(
      self,
      num_layers: int = 16,
      hidden_dim: int = 256,
      intermediate_dim: int = 512,
      num_heads: int = 4,
      seq_len: int = 64,
      batch_size: int = 2,
      num_eval_iters: int = 10,
  ):
    x_cpu = torch.randn(batch_size, seq_len, hidden_dim)

    # 1. Baseline: Standard torch.compile (nested_compile_region DISABLED)
    orig_forward = TransformerBlock.forward
    if hasattr(orig_forward, "_torch_dynamo_orig_callable"):
      orig_forward = orig_forward._torch_dynamo_orig_callable
    TransformerBlock.forward = orig_forward

    model_baseline = DeepTransformer(
        num_layers, hidden_dim, intermediate_dim, num_heads
    )
    x_tpu = _backend.to_device(x_cpu, "tpu")
    model_baseline_tpu = _backend.to_device(model_baseline, "tpu").eval()

    b_baseline = _backend.TpuBackend(debug=True)
    compiled_baseline = torch.compile(model_baseline_tpu, backend=b_baseline)

    torch._dynamo.reset()
    t0 = time.perf_counter()
    with torch.no_grad():
      out_baseline = compiled_baseline(x_tpu).to("cpu")
    baseline_compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
      for _ in range(num_eval_iters):
        _ = compiled_baseline(x_tpu).to("cpu")
    baseline_exec_time = (time.perf_counter() - t0) / num_eval_iters

    baseline_mlir_lines = sum(
        len(txt.splitlines())
        for txt in b_baseline._compiled_executables[0].mlir_texts
    )

    # 2. Enabled: nested_compile_region ENABLED
    TransformerBlock.forward = torch.compiler.nested_compile_region(
        orig_forward
    )

    model_nested = DeepTransformer(
        num_layers, hidden_dim, intermediate_dim, num_heads
    )
    model_nested.load_state_dict(model_baseline.state_dict())
    model_nested_tpu = _backend.to_device(model_nested, "tpu").eval()

    b_nested = _backend.TpuBackend(debug=True)
    compiled_nested = torch.compile(model_nested_tpu, backend=b_nested)

    torch._dynamo.reset()
    t0 = time.perf_counter()
    with torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False
    ), torch.no_grad():
      out_nested = compiled_nested(x_tpu).to("cpu")
    nested_compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False
    ), torch.no_grad():
      for _ in range(num_eval_iters):
        _ = compiled_nested(x_tpu).to("cpu")
    nested_exec_time = (time.perf_counter() - t0) / num_eval_iters

    nested_mlir_lines = sum(
        len(txt.splitlines())
        for txt in b_nested._compiled_executables[0].mlir_texts
    )

    # Restore original forward
    TransformerBlock.forward = orig_forward

    # Verify accuracy
    utils.assert_close(out_nested, out_baseline, rtol=1e-3, atol=1e-4)

    speedup = (
        (baseline_compile_time / nested_compile_time)
        if nested_compile_time > 0
        else 0
    )
    reduction = (
        (1.0 - (nested_mlir_lines / baseline_mlir_lines)) * 100.0
        if baseline_mlir_lines > 0
        else 0
    )
    diff_pct = (
        (nested_exec_time - baseline_exec_time) / baseline_exec_time
    ) * 100.0

    report = f"""
================================================================================
BENCHMARK RESULTS: Deep Transformer (num_layers={num_layers}, hidden_dim={hidden_dim}, seq_len={seq_len})
================================================================================
Metric                           | Baseline (Disabled) | Nested Compile Region (Enabled) | Improvement / Delta
---------------------------------+---------------------+---------------------------------+--------------------
Compilation Time (1st run)       | {baseline_compile_time:17.4f} s | {nested_compile_time:29.4f} s | {speedup:6.2f}x faster ({((baseline_compile_time - nested_compile_time)/baseline_compile_time)*100:.1f}% reduction)
Steady-State Execution Latency   | {baseline_exec_time*1000:15.3f} ms | {nested_exec_time*1000:27.3f} ms | {diff_pct:+6.2f}%
MLIR Generated Lines             | {baseline_mlir_lines:17d}   | {nested_mlir_lines:29d}   | {reduction:6.1f}% smaller
================================================================================
"""
    print(report, flush=True)

    return {
        "num_layers": num_layers,
        "baseline_compile_time": baseline_compile_time,
        "nested_compile_time": nested_compile_time,
        "compile_speedup": speedup,
        "baseline_exec_time": baseline_exec_time,
        "nested_exec_time": nested_exec_time,
        "baseline_mlir_lines": baseline_mlir_lines,
        "nested_mlir_lines": nested_mlir_lines,
    }

  def test_benchmark_transformer_16_layers(self):
    results = self._run_benchmark(num_layers=16, num_eval_iters=10)
    self.assertGreater(results["compile_speedup"], 0.5)

  def test_benchmark_transformer_24_layers(self):
    results = self._run_benchmark(num_layers=24, num_eval_iters=10)
    self.assertGreater(results["compile_speedup"], 0.5)


if __name__ == "__main__":
  absltest.main()
