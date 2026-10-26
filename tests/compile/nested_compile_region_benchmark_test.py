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

"""Compilation benchmark for torch.compiler.nested_compile_region."""

from collections.abc import Sequence
import dataclasses
import statistics
import time
from typing import Any
from absl import flags
from absl import logging
from absl.testing import absltest
import torch
import torch.fx
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from torch_tpu._internal import compile as compile_lib
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import nested_compile_region as tt_ncr
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_layers", 64, "Number of Transformer layers.")
flags.DEFINE_integer("num_trials", 1, "Number of benchmark trials.")
flags.DEFINE_integer("batch_size", 4, "Batch size.")
flags.DEFINE_integer("seq_len", 64, "Sequence length.")
flags.DEFINE_integer("hidden_dim", 128, "Hidden dimension.")
flags.DEFINE_integer("num_heads", 4, "Number of attention heads.")
flags.DEFINE_integer("ffn_dim", 512, "Feed-forward hidden dimension.")
flags.DEFINE_integer("warmup_iters", 2, "Number of warmup iterations.")
flags.DEFINE_integer(
    "steady_state_iters", 5, "Number of steady-state execution iterations."
)


class MultiHeadAttention(torch.nn.Module):
  """Multi-Head Attention block."""

  def __init__(self, dim: int, num_heads: int):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.q_proj = torch.nn.Linear(dim, dim)
    self.k_proj = torch.nn.Linear(dim, dim)
    self.v_proj = torch.nn.Linear(dim, dim)
    self.out_proj = torch.nn.Linear(dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, s, d = x.shape
    q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    scale = 1.0 / (self.head_dim**0.5)
    attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    out = (attn @ v).transpose(1, 2).contiguous().view(b, s, d)
    return self.out_proj(out)


class FeedForward(torch.nn.Module):
  """Feed-Forward Network (FFN) block."""

  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.w1 = torch.nn.Linear(dim, hidden_dim)
    self.w2 = torch.nn.Linear(hidden_dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(torch.nn.functional.gelu(self.w1(x)))


class TransformerLayer(torch.nn.Module):
  """Standard Transformer Encoder Layer (monolithic base case)."""

  def __init__(self, dim: int, num_heads: int, ffn_dim: int):
    super().__init__()
    self.norm1 = torch.nn.LayerNorm(dim)
    self.attn = MultiHeadAttention(dim, num_heads)
    self.norm2 = torch.nn.LayerNorm(dim)
    self.ffn = FeedForward(dim, ffn_dim)

  def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward_impl(x)


class NestedTransformerLayer(TransformerLayer):
  """Transformer Layer with forward decorated as a nested compile region."""

  @torch.compiler.nested_compile_region
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward_impl(x)


class ScanTransformerLayer(TransformerLayer):
  """Transformer Layer whose consecutive calls TorchTPU rolls into a scan."""

  @tt_ncr.nested_compile_region(scan_unroll=1)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward_impl(x)


class TransformerEncoder(torch.nn.Module):
  """Transformer Encoder stack composed of N layers."""

  def __init__(
      self,
      layer_cls: type[torch.nn.Module],
      num_layers: int,
      dim: int,
      num_heads: int,
      ffn_dim: int,
  ):
    super().__init__()
    self.layers = torch.nn.ModuleList(
        [layer_cls(dim, num_heads, ffn_dim) for _ in range(num_layers)]
    )
    self.final_norm = torch.nn.LayerNorm(dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return self.final_norm(x)


class TimingBackendWrapper:
  """Instruments TpuBackend to measure Dynamo trace time and XLA compilation time."""

  def __init__(self, inner_backend: Any):
    self.inner_backend = inner_backend
    self.first_backend_entry_time: float | None = None
    self.xla_compile_time: float = 0.0
    self._invocation_start_time: float | None = None

  def mark_invocation_start(self) -> None:
    self._invocation_start_time = time.perf_counter()
    self.first_backend_entry_time = None
    self.xla_compile_time = 0.0

  @property
  def dynamo_trace_time(self) -> float:
    if (
        self._invocation_start_time is not None
        and self.first_backend_entry_time is not None
    ):
      return max(
          0.0, self.first_backend_entry_time - self._invocation_start_time
      )
    return 0.0

  def __call__(
      self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any]
  ) -> Any:
    entry_time = time.perf_counter()
    if self.first_backend_entry_time is None:
      self.first_backend_entry_time = entry_time

    t0 = time.perf_counter()
    compiled_runner = self.inner_backend(gm, example_inputs)
    t1 = time.perf_counter()
    self.xla_compile_time += t1 - t0
    return compiled_runner


def reset_all_caches() -> None:
  """Hermetically clears all Dynamo and TorchTPU compilation caches."""
  tt_testing.reset_eager_state()
  torch.compiler.reset()
  torch._dynamo.reset()
  compile_lib.clear_cache()
  if hasattr(torch, "tpu") and hasattr(torch.tpu, "_clear_cache"):
    torch.tpu._clear_cache()
  if torch.accelerator.is_available():
    torch.accelerator.synchronize()


@dataclasses.dataclass
class TrialMetrics:
  overall_compile_time_s: float
  dynamo_trace_time_s: float
  xla_compile_time_s: float
  steady_state_exec_time_ms: float
  output_tensor: torch.Tensor


def run_benchmark_trial(
    config_id: int,
    num_layers: int,
    dim: int,
    num_heads: int,
    ffn_dim: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> TrialMetrics:
  """Executes a single benchmark trial for the specified configuration."""
  if config_id not in (1, 2, 3, 4):
    raise ValueError(f"Invalid config_id: {config_id}. Must be 1, 2, 3 or 4.")

  reset_all_caches()

  torch.manual_seed(seed)
  if config_id == 1:
    layer_cls = TransformerLayer
  elif config_id == 4:
    layer_cls = ScanTransformerLayer
  else:
    layer_cls = NestedTransformerLayer

  model = TransformerEncoder(layer_cls, num_layers, dim, num_heads, ffn_dim).to(
      "tpu"
  )
  model.eval()

  torch.manual_seed(seed + 999)
  x = torch.randn(batch_size, seq_len, dim, device="tpu")

  raw_backend = _backend.TpuBackend(debug=True)
  timing_wrapper = TimingBackendWrapper(raw_backend)

  # Prepare Dynamo patch configuration based on config_id
  patch_kwargs: dict[str, Any] = {}
  if config_id == 2:
    if hasattr(torch._dynamo.config, "inline_invoke_subgraph"):
      patch_kwargs["inline_invoke_subgraph"] = True
  elif config_id in (3, 4):
    if hasattr(torch._dynamo.config, "inline_invoke_subgraph"):
      patch_kwargs["inline_invoke_subgraph"] = False
    if hasattr(torch._dynamo.config, "inline_single_use_invoke_subgraph"):
      patch_kwargs["inline_single_use_invoke_subgraph"] = False
    if hasattr(torch._dynamo.config, "enable_invoke_subgraph_regional_compile"):
      patch_kwargs["enable_invoke_subgraph_regional_compile"] = True

  with torch.no_grad(), torch._dynamo.config.patch(patch_kwargs):
    compiled_model = torch.compile(model, backend=timing_wrapper)

    if torch.accelerator.is_available():
      torch.accelerator.synchronize()

    timing_wrapper.mark_invocation_start()
    t_start = time.perf_counter()
    out = compiled_model(x)
    if torch.accelerator.is_available():
      torch.accelerator.synchronize()
    t_end = time.perf_counter()
    first_call_total = t_end - t_start

    # Steady-state execution timing
    for _ in range(FLAGS.warmup_iters):
      _ = compiled_model(x)
    if torch.accelerator.is_available():
      torch.accelerator.synchronize()

    exec_times = []
    for _ in range(FLAGS.steady_state_iters):
      t0 = time.perf_counter()
      _ = compiled_model(x)
      if torch.accelerator.is_available():
        torch.accelerator.synchronize()
      t1 = time.perf_counter()
      exec_times.append(t1 - t0)

    median_exec_s = statistics.median(exec_times)
    steady_state_ms = median_exec_s * 1000.0
    overall_compile_s = max(0.0, first_call_total - median_exec_s)

    return TrialMetrics(
        overall_compile_time_s=overall_compile_s,
        dynamo_trace_time_s=timing_wrapper.dynamo_trace_time,
        xla_compile_time_s=timing_wrapper.xla_compile_time,
        steady_state_exec_time_ms=steady_state_ms,
        output_tensor=out.detach().cpu(),
    )


class NestedCompileRegionBenchmarkTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_nested_compile_region_compilation_benchmark(self):
    num_layers = FLAGS.num_layers
    num_trials = FLAGS.num_trials
    batch_size = FLAGS.batch_size
    seq_len = FLAGS.seq_len
    hidden_dim = FLAGS.hidden_dim
    num_heads = FLAGS.num_heads
    ffn_dim = FLAGS.ffn_dim

    logging.info(
        "Starting nested_compile_region benchmark with %d layers, %d trials"
        " (batch_size=%d, seq_len=%d, dim=%d, heads=%d, ffn=%d)...",
        num_layers,
        num_trials,
        batch_size,
        seq_len,
        hidden_dim,
        num_heads,
        ffn_dim,
    )

    configs = [
        (1, "1. Monolithic Base Case (No NCR)"),
    ]
    # Config 2 requires Dynamo support for inlining invoke_subgraph (PyTorch >= 2.13)
    if hasattr(torch._dynamo.config, "inline_invoke_subgraph"):
      configs.append((2, "2. Vanilla NCR (Dynamo Inlining)"))
    configs.append((3, "3. NCR (Regional Compile)"))
    configs.append((4, "4. NCR + Scan (stablehlo.while)"))

    all_metrics: dict[int, list[TrialMetrics]] = {cid: [] for cid, _ in configs}

    base_seed = 42
    for trial_idx in range(num_trials):
      trial_seed = base_seed + trial_idx * 100
      logging.info("--- Running Trial %d/%d ---", trial_idx + 1, num_trials)

      for config_id, config_name in configs:
        logging.info(
            "Benchmarking [%s] - Trial %d...", config_name, trial_idx + 1
        )
        metrics = run_benchmark_trial(
            config_id=config_id,
            num_layers=num_layers,
            dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=trial_seed,
        )
        all_metrics[config_id].append(metrics)
        logging.info(
            "[%s] Trial %d: Overall Compile: %.3fs | Dynamo Trace: %.3fs | XLA"
            " Compile: %.3fs | Steady-State: %.2fms",
            config_name,
            trial_idx + 1,
            metrics.overall_compile_time_s,
            metrics.dynamo_trace_time_s,
            metrics.xla_compile_time_s,
            metrics.steady_state_exec_time_ms,
        )

      # Verify numerical output correctness across configs for this trial
      out_ref = all_metrics[1][-1].output_tensor
      for cid, _ in configs[1:]:
        out_cfg = all_metrics[cid][-1].output_tensor
        utils.assert_close(out_cfg, out_ref, rtol=1e-3, atol=1e-3)

    # Compute medians across trials
    medians: dict[int, dict[str, float]] = {}
    for config_id, _ in configs:
      m_list = all_metrics[config_id]
      medians[config_id] = {
          "overall": statistics.median(
              m.overall_compile_time_s for m in m_list
          ),
          "dynamo": statistics.median(m.dynamo_trace_time_s for m in m_list),
          "xla": statistics.median(m.xla_compile_time_s for m in m_list),
          "steady": statistics.median(
              m.steady_state_exec_time_ms for m in m_list
          ),
      }

    base_overall = medians[1]["overall"]
    base_xla = medians[1]["xla"]

    # Format summary table
    table_width = 144
    title = "torch.compiler.nested_compile_region Compilation Benchmark Results"
    subtitle = (
        f"(Layers: {num_layers}, Hidden Dim: {hidden_dim}, Heads: {num_heads},"
        f" Batch: {batch_size}, SeqLen: {seq_len}, Trials: {num_trials})"
    )

    table_header = (
        "\n"
        + "=" * table_width
        + "\n"
        + f"{title:^{table_width}}\n"
        + f"{subtitle:^{table_width}}\n"
        + "=" * table_width
        + "\n"
        + f"{'Configuration':<38} {'Overall Compile (s)':>20}"
        f" {'Overall Speedup':>16} {'Dynamo Trace (s)':>18}"
        f" {'XLA Compile (s)':>16} {'Steady-State (ms)':>18}"
        f" {'XLA Speedup':>12}\n"
        + "-" * table_width
    )

    rows = []
    for config_id, config_name in configs:
      m = medians[config_id]
      overall_speedup = (
          (base_overall / m["overall"]) if m["overall"] > 0 else 1.0
      )
      xla_speedup = (base_xla / m["xla"]) if m["xla"] > 0 else 1.0
      row = (
          f"{config_name:<38} "
          f"{m['overall']:>20.3f} "
          f"{overall_speedup:>15.2f}x "
          f"{m['dynamo']:>18.3f} "
          f"{m['xla']:>16.3f} "
          f"{m['steady']:>18.2f} "
          f"{xla_speedup:>11.2f}x"
      )
      rows.append(row)

    table_footer = "=" * table_width + "\n"
    full_table = table_header + "\n" + "\n".join(rows) + "\n" + table_footer

    logging.info("%s", full_table)

    # Sanity checks
    for config_id, _ in configs:
      self.assertGreater(medians[config_id]["overall"], 0.0)
      self.assertGreater(medians[config_id]["xla"], 0.0)
      self.assertGreater(medians[config_id]["steady"], 0.0)


if __name__ == "__main__":
  absltest.main()
