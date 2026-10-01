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

"""Speed benchmarks and profiling for baseline FLUX exclusively targeting TorchTPU."""

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
from examples.tutorials.FLUX_optimization import flux_baseline

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
    2,
    "Batch size for the FLUX block benchmark.",
)

_IMG_SEQ_LEN = flags.DEFINE_integer(
    "img_seq_len",
    512,
    "Image sequence length (number of image patch tokens).",
)

_TXT_SEQ_LEN = flags.DEFINE_integer(
    "txt_seq_len",
    128,
    "Text sequence length (number of text tokens from T5).",
)

_HIDDEN_SIZE = flags.DEFINE_integer(
    "hidden_size",
    1024,
    "Hidden dimension size of the transformer block (must be divisible by"
    " num_heads).",
)

_NUM_HEADS = flags.DEFINE_integer(
    "num_heads",
    8,
    "Number of attention heads.",
)


@dataclass
class BenchmarkMetrics:
  """Structured container for end-to-end training step timings on TPU."""

  warmup_time_ms: float
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
    logging.info(
        "\n%s\nFLUX BASELINE TPU SPEED BENCHMARK SUMMARY\n%s", border, border
    )
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
  """Formats a clean, focused summary table showing strictly TPU device execution metrics."""
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
    """Verifies that reference RoPE and EmbedND produce expected tensor shapes on TPU."""
    total_seq_len = self.img_seq_len + self.txt_seq_len
    ids = torch.randint(
        0,
        1000,
        (self.batch_size, total_seq_len, 3),
        device=self.device,
        dtype=torch.long,
    )
    embed = flux_baseline.EmbedNDRef(
        dim=self.head_dim, theta=10000, axes_dim=self.axes_dim
    ).to(self.device)
    pe = embed(ids)

    # Reference RoPE frequency shape: (B, L, 1, D/2, 2, 2)
    self.assertEqual(
        pe.shape,
        (self.batch_size, total_seq_len, 1, self.head_dim // 2, 2, 2),
    )
    self.assertEqual(pe.device.type, "tpu")

  def test_baseline_double_stream_block_training_step_and_profile(self):
    """Verifies a full training step and profiles step times on TPU without host synchronization."""
    img, txt, vec, pe = flux_baseline.generate_baseline_inputs(
        batch_size=self.batch_size,
        img_seq_len=self.img_seq_len,
        txt_seq_len=self.txt_seq_len,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        axes_dim=self.axes_dim,
        device=self.device,
    )

    block = flux_baseline.DoubleStreamBlockBaseline(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
    ).to(self.device)

    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)

    # Synthetic targets for Flow Matching MSE loss
    target_img = torch.randn_like(img)
    target_txt = torch.randn_like(txt)

    # --- Step 1: Warmup Iterations ---
    warmup_start = time.perf_counter()
    for _ in range(_NUM_WARMUP_STEPS.value):
      optimizer.zero_grad()
      out_img, out_txt = block(img=img, txt=txt, vec=vec, pe=pe)
      loss = F.mse_loss(out_img, target_img) + F.mse_loss(out_txt, target_txt)
      loss.backward()
      optimizer.step()
    warmup_time_ms = (time.perf_counter() - warmup_start) * 1000

    metrics = BenchmarkMetrics(
        warmup_time_ms=warmup_time_ms,
    )

    # --- Step 2: Configure Native torch.profiler for TPU ---
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ]

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
        with torch.profiler.record_function("FLUX_Baseline_Forward"):
          out_img, out_txt = block(img=img, txt=txt, vec=vec, pe=pe)

        with torch.profiler.record_function("FLUX_Baseline_Backward"):
          loss = F.mse_loss(out_img, target_img) + F.mse_loss(
              out_txt, target_txt
          )
          loss.backward()

        with torch.profiler.record_function("FLUX_Baseline_Optimizer"):
          optimizer.step()

        step_total_ms = (time.perf_counter() - step_start) * 1000
        metrics.step_times_ms.append(step_total_ms)

        if prof is not None:
          prof.step()

        logging.info("Step %d: total=%.2f ms", step_idx + 1, step_total_ms)

    metrics.total_execution_time_s = time.perf_counter() - bench_start_total

    # Log in-memory operator breakdown from profiler
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
