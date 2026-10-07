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

"""Speed benchmarks, profiler breakdown, and parity tests for FLUX on TorchTPU.

================================================================================
OVERVIEW: BENCHMARKING & PROFILING METHODOLOGY ON TPU
================================================================================
This test suite verifies numerical equivalence across optimization milestones
and measures end-to-end training step performance (Forward + Flow Matching MSE
Loss + Backward + AdamW Optimizer Step) on Cloud TPU hardware.

Key Benchmarking Principles Demonstrated:
1. Warmup Isolation (`--num_warmup_steps=2`):
   `torch.compile(backend="tpu")` performs Just-In-Time (JIT) graph tracing and
   XLA HLO compilation during the initial steps. Running dedicated warmup steps
   before starting the benchmark timer ensures that `Mean Step Time` reflects
   true steady-state TPU execution rather than one-time compiler overhead.

2. Asynchronous Device Profiling (`torch.profiler.profile`):
   Inserting explicit host synchronizations (e.g., printing tensor values mid-step)
   breaks the asynchronous TPU execution pipeline. Instead, this benchmark uses
   `torch.profiler.profile(activities=[CPU, TPU])` with `prof.step()` markers and
   post-processes `self_device_time_total` microseconds via `format_tpu_operator_table`
   to inspect hardware-level kernel execution times cleanly.

================================================================================
EXPECTED BENCHMARK RESULTS ON CLOUD TPU (batch_size=2, seq_len=640, heads=8)
================================================================================
  1. Baseline (Eager Reference):              1576.01 ms (1.00x)
  2. Zero-Copy Views Only (.view+unbind):      578.03 ms (2.72x speedup)
  3. Compile Only (Unoptimized Graph):         344.50 ms (4.57x speedup)
  4. Zero-Copy Views + Compile:                113.31 ms (13.91x speedup)
  5. Zero-Copy + Fused 2D RoPE + Compile:       69.16 ms (22.79x speedup)

================================================================================
HOW TO RUN
================================================================================
Run test suite via Bazel:
  bazel test \\
    //examples/tutorials/FLUX_optimization:flux_test

Run interactive benchmark with custom flags:
  bazel run \\
    //examples/tutorials/FLUX_optimization:flux_test -- \\
    --batch_size=2 --img_seq_len=512 --txt_seq_len=128 --hidden_size=1024 \\
    --use_zero_copy_view=true --use_fused_rope=true --compile=true \\
    --num_warmup_steps=2 --num_benchmark_steps=10
"""

import contextlib
from dataclasses import dataclass, field
import time
from absl import flags
from absl import logging
from absl.testing import absltest
import torch
import torch.nn.functional as F
import torch.profiler
import torch_tpu
from examples.tutorials.FLUX_optimization import flux

_NUM_WARMUP_STEPS = flags.DEFINE_integer(
    "num_warmup_steps",
    2,
    "Number of warmup steps prior to performance measurement.",
)

_NUM_BENCHMARK_STEPS = flags.DEFINE_integer(
    "num_benchmark_steps",
    5,
    "Number of timed benchmark steps.",
)

_ENABLE_PROFILER = flags.DEFINE_bool(
    "enable_profiler",
    True,
    "Whether to capture in-memory execution traces for operator breakdown"
    " logging.",
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    1,
    "Batch size for the FLUX block benchmark.",
)

_IMG_SEQ_LEN = flags.DEFINE_integer(
    "img_seq_len",
    128,
    "Image sequence length (number of image patch tokens).",
)

_TXT_SEQ_LEN = flags.DEFINE_integer(
    "txt_seq_len",
    32,
    "Text sequence length (number of text tokens from T5).",
)

_HIDDEN_SIZE = flags.DEFINE_integer(
    "hidden_size",
    512,
    "Hidden dimension size of the transformer block (must be divisible by"
    " num_heads).",
)

_NUM_HEADS = flags.DEFINE_integer(
    "num_heads",
    8,
    "Number of attention heads.",
)

_USE_ZERO_COPY_VIEW = flags.DEFINE_bool(
    "use_zero_copy_view",
    False,
    "Whether to enable the zero-copy view optimization (.view + torch.unbind)"
    " for QKV splitting.",
)

_USE_FUSED_ROPE = flags.DEFINE_bool(
    "use_fused_rope",
    False,
    "Whether to enable the fused algebraic 2D RoPE optimization.",
)

_COMPILE = flags.DEFINE_bool(
    "compile",
    False,
    "Whether to compile the DoubleStreamBlock using"
    " torch.compile(backend='tpu').",
)


# ==============================================================================
# STEP 1: BENCHMARK METRICS & DEVICE PROFILER FORMATTING
# ==============================================================================


@dataclass
class BenchmarkMetrics:
  """Structured container for end-to-end training step timings on TPU."""

  mode: str = "Baseline (rearrange)"
  warmup_time_ms: float = 0.0
  step_times_ms: list[float] = field(default_factory=list)
  total_execution_time_s: float = 0.0

  @property
  def mean_step_time_ms(self) -> float:
    return (
        sum(self.step_times_ms) / len(self.step_times_ms)
        if self.step_times_ms
        else 0.0
    )

  def print_summary(self) -> None:
    """Prints a formatted ASCII summary of benchmark speed results."""
    border = "=" * 70
    logging.info("\n%s\nFLUX TPU SPEED BENCHMARK SUMMARY\n%s", border, border)
    logging.info("  Optimization Mode:      %s", self.mode)
    logging.info("  Warmup Execution Time:  %.2f ms", self.warmup_time_ms)
    logging.info(
        "  Total Measured Runtime: %.4f s (%d steps)",
        self.total_execution_time_s,
        len(self.step_times_ms),
    )
    logging.info("  Mean Step Time:         %.2f ms", self.mean_step_time_ms)
    logging.info("%s\n", border)


def format_tpu_operator_table(
    prof: torch.profiler.profile, row_limit: int = 50
) -> str:
  """Formats a clean summary table showing strictly TPU device execution metrics.

  Filters `prof.key_averages()` to events with non-zero `self_device_time_total`
  so that host-side dispatch wrappers are excluded and actual TPU hardware
  operators (such as XLA fused clusters or copy kernels) are ranked by time.
  """
  key_averages = prof.key_averages()
  # Filter events that executed on the TPU device
  tpu_events = [
      evt
      for evt in key_averages
      if getattr(evt, "self_device_time_total", 0) > 0
  ]
  if not tpu_events:
    return "No TPU device events recorded."

  # Sort descending by self TPU device time
  tpu_events.sort(
      key=lambda e: getattr(e, "self_device_time_total", 0), reverse=True
  )
  total_self_tpu_us = sum(evt.self_device_time_total for evt in tpu_events)

  lines = []
  header = (
      f"{'Op Name':<50} {'Self TPU (us)':>14} {'Self TPU %':>11} "
      f"{'TPU Total (us)':>15} {'Avg TPU (us)':>13} {'Calls':>7}"
  )
  sep = "-" * len(header)
  lines.append(sep)
  lines.append(header)
  lines.append(sep)

  for evt in tpu_events[:row_limit]:
    name = evt.key or "unknown"
    if len(name) > 48:
      name = name[:45] + "..."
    self_tpu_us = evt.self_device_time_total
    total_tpu_us = evt.device_time_total
    pct = (
        (self_tpu_us / total_self_tpu_us * 100)
        if total_self_tpu_us > 0
        else 0.0
    )
    avg_tpu_us = getattr(
        evt, "device_time", self_tpu_us / evt.count if evt.count else 0.0
    )
    lines.append(
        f"{name:<50} {self_tpu_us:>14.2f} {pct:>10.2f}% "
        f"{total_tpu_us:>15.2f} {avg_tpu_us:>13.2f} {evt.count:>7}"
    )

  lines.append(sep)
  lines.append(
      f"Total Self TPU Time: {total_self_tpu_us / 1000.0:.2f} ms across"
      f" {len(tpu_events)} TPU operators"
  )
  return "\n".join(lines)


# ==============================================================================
# STEP 2: NUMERICAL PARITY & END-TO-END SPEED BENCHMARK SUITE
# ==============================================================================


class FluxBaselineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(42)

    self.device = torch.device("tpu")
    logging.info(
        "Running FLUX performance benchmark on device: %s", self.device
    )

    self.batch_size = _BATCH_SIZE.value
    self.img_seq_len = _IMG_SEQ_LEN.value
    self.txt_seq_len = _TXT_SEQ_LEN.value
    self.hidden_size = _HIDDEN_SIZE.value
    self.num_heads = _NUM_HEADS.value
    self.head_dim = self.hidden_size // self.num_heads
    # Multi-axis RoPE dimensions (time/frame, height, width) summing to head_dim.
    # For standard FLUX head_dim=128, uses (16, 56, 56).
    if self.head_dim == 128:
      self.axes_dim = (16, 56, 56)
    else:
      dim1 = self.head_dim // 4
      dim2 = (self.head_dim - dim1) // 2
      dim3 = self.head_dim - dim1 - dim2
      self.axes_dim = (dim1, dim2, dim3)

    logging.info(
        "Model configuration: hidden_size=%d, num_heads=%d, head_dim=%d,"
        " seq_len=%d (img=%d, txt=%d)",
        self.hidden_size,
        self.num_heads,
        self.head_dim,
        self.img_seq_len + self.txt_seq_len,
        self.img_seq_len,
        self.txt_seq_len,
    )

  def test_baseline_rope_and_embed_shapes(self):
    """Verifies that reference 5D RoPE and EmbedNDRef produce expected tensor shapes on TPU."""
    total_seq_len = self.img_seq_len + self.txt_seq_len
    ids = torch.randint(
        0,
        1000,
        (self.batch_size, total_seq_len, 3),
        device=self.device,
        dtype=torch.long,
    )
    embed = flux.EmbedNDRef(
        dim=self.head_dim, theta=10000, axes_dim=self.axes_dim
    ).to(self.device)
    pe = embed(ids)

    # Reference 5D RoPE frequency shape: (B, L, 1, D/2, 2, 2)
    self.assertEqual(
        pe.shape,
        (self.batch_size, total_seq_len, 1, self.head_dim // 2, 2, 2),
    )
    self.assertEqual(pe.device.type, "tpu")

  def test_zero_copy_view_numerical_parity(self):
    """Verifies that `.view() + torch.unbind()` produces numerically identical output to `einops.rearrange`."""
    img, txt, vec, pe = flux.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
    )

    # 1. Instantiate baseline block using einops.rearrange (BEFORE)
    block_base = flux.DoubleStreamBlock(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(use_zero_copy_view=False),
    ).to(self.device)

    # 2. Instantiate optimized block using zero-copy .view() + unbind() (AFTER)
    block_opt = flux.DoubleStreamBlock(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(use_zero_copy_view=True),
    ).to(self.device)

    # 3. Copy exact weights and biases
    block_opt.load_state_dict(block_base.state_dict())

    with torch.no_grad():
      out_img_base, out_txt_base = block_base(img=img, txt=txt, vec=vec, pe=pe)
      out_img_opt, out_txt_opt = block_opt(img=img, txt=txt, vec=vec, pe=pe)

    torch.testing.assert_close(out_img_base, out_img_opt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_txt_base, out_txt_opt, atol=1e-5, rtol=1e-5)
    logging.info(
        "Numerical parity verified between rearrange and zero-copy view."
    )

  def test_compile_numerical_parity(self):
    """Verifies that `torch.compile(backend='tpu')` matches eager execution within XLA float tolerances."""
    img, txt, vec, pe = flux.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
    )

    block_eager = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(use_zero_copy_view=True, compile=False),
    ).to(self.device)

    block_compiled = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=flux.OptimizationConfig(use_zero_copy_view=True, compile=True),
    ).to(self.device)

    orig_mod = getattr(block_compiled, "_orig_mod", block_compiled)
    orig_mod.load_state_dict(block_eager.state_dict())

    with torch.no_grad():
      out_img_eager, out_txt_eager = block_eager(
          img=img, txt=txt, vec=vec, pe=pe
      )
      out_img_compiled, out_txt_compiled = block_compiled(
          img=img, txt=txt, vec=vec, pe=pe
      )

    # Note: XLA compilation fuses multiply-adds (FMA) and reorders floating-point
    # summation trees during reductions (LayerNorm, RMSNorm, Softmax). Tolerances
    # of atol=2e-3, rtol=1e-2 account for standard associative differences.
    torch.testing.assert_close(
        out_img_eager, out_img_compiled, atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(
        out_txt_eager, out_txt_compiled, atol=2e-3, rtol=1e-2
    )
    logging.info(
        "Numerical parity verified between eager and compiled block on TPU."
    )

  def test_baseline_double_stream_block_training_step_and_profile(self):
    """Runs a full training step benchmark and profiles TPU device operators."""
    img, txt, vec, pe = flux.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
        use_fused_rope=_USE_FUSED_ROPE.value,
    )

    opt_config = flux.OptimizationConfig(
        use_zero_copy_view=_USE_ZERO_COPY_VIEW.value,
        use_fused_rope=_USE_FUSED_ROPE.value,
        compile=_COMPILE.value,
    )
    block = flux.build_double_stream_block(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        config=opt_config,
    ).to(self.device)

    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)

    # Synthetic targets for Flow Matching MSE loss
    target_img = torch.randn_like(img)
    target_txt = torch.randn_like(txt)

    # --------------------------------------------------------------------------
    # 1. Warmup Iterations (Absorbs XLA JIT Compilation & Memory Allocation)
    # --------------------------------------------------------------------------
    warmup_start = time.perf_counter()
    for _ in range(_NUM_WARMUP_STEPS.value):
      optimizer.zero_grad()
      out_img, out_txt = block(img=img, txt=txt, vec=vec, pe=pe)
      loss = F.mse_loss(out_img, target_img) + F.mse_loss(out_txt, target_txt)
      loss.backward()
      optimizer.step()
    warmup_time_ms = (time.perf_counter() - warmup_start) * 1000

    modes = []
    modes.append(
        "Zero-Copy View (.view+unbind)"
        if _USE_ZERO_COPY_VIEW.value
        else "Baseline Rearrange (einops)"
    )
    if _USE_FUSED_ROPE.value:
      modes.append("Fused 2D RoPE")
    if _COMPILE.value:
      modes.append("torch.compile (TPU)")
    mode_str = " + ".join(modes)

    metrics = BenchmarkMetrics(
        mode=mode_str,
        warmup_time_ms=warmup_time_ms,
    )

    # --------------------------------------------------------------------------
    # 2. Steady-State Timed Benchmark Loop with Native TPU Profiler
    # --------------------------------------------------------------------------
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "TPU"):
      activities.append(torch.profiler.ProfilerActivity.TPU)

    bench_start_total = time.perf_counter()

    # Pure end-to-end execution without disk trace dumps or graph-breaking host syncs
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) if _ENABLE_PROFILER.value else contextlib.nullcontext() as prof:

      for step_idx in range(_NUM_BENCHMARK_STEPS.value):
        step_start = time.perf_counter()

        optimizer.zero_grad()
        with torch.profiler.record_function("FLUX_Forward"):
          out_img, out_txt = block(img=img, txt=txt, vec=vec, pe=pe)

        with torch.profiler.record_function("FLUX_Backward"):
          loss = F.mse_loss(out_img, target_img) + F.mse_loss(
              out_txt, target_txt
          )
          loss.backward()

        with torch.profiler.record_function("FLUX_Optimizer"):
          optimizer.step()

        step_total_ms = (time.perf_counter() - step_start) * 1000
        metrics.step_times_ms.append(step_total_ms)

        if prof is not None:
          prof.step()

        logging.info("Step %d: total=%.2f ms", step_idx + 1, step_total_ms)

    metrics.total_execution_time_s = time.perf_counter() - bench_start_total

    # --------------------------------------------------------------------------
    # 3. Output In-Memory TPU Operator Breakdown & Benchmark Summary
    # --------------------------------------------------------------------------
    if prof is not None:
      try:
        table_summary = format_tpu_operator_table(prof)
        logging.info(
            "\nPyTorch Profiler Top Operators on TPU:\n%s", table_summary
        )
      except Exception as e:
        logging.warning("Could not read profiler summary: %s", e)

    # Output structured summary table
    metrics.print_summary()

    # Functional Assertions
    self.assertEqual(len(metrics.step_times_ms), _NUM_BENCHMARK_STEPS.value)
    self.assertTrue(torch.isfinite(loss.detach().cpu()).item())
    self.assertGreater(metrics.mean_step_time_ms, 0.0)


if __name__ == "__main__":
  absltest.main()
