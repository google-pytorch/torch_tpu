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

"""The benchmarks that opt in to the batch sweep (`--batch_sweep`).

The batch sweep measures a benchmark at more batch sizes than its default
workload's, which costs a compile and a measurement per batch size, so only the
benchmarks listed here are swept. The list is central, rather than a property
of each registration, because the generated `*_gen` benchmarks are registered
from configs rather than by hand-written `add_benchmark` calls.

The batch sizes of a benchmark's sweep are those of its `registry.BatchSweep`
if it sets them, or else those of `--batch_sweep_sizes`, without those above
the `BatchSweepOptIn.max_batch_size` of the benchmark here. A benchmark listed
here can still be kept out of the sweep, with a reason, with
`registry.BatchSweep.disabled`.
"""

from collections.abc import Mapping
import dataclasses
import types


@dataclasses.dataclass(frozen=True)
class BatchSweepOptIn:
  """How a benchmark that opts in to the batch sweep is swept.

  Attributes:
    max_batch_size: The largest batch size to sweep the benchmark at, or None to
      sweep it at every batch size. The larger batch sizes get no cases, except
      the default batch size of the benchmark, which is always measured.
    max_batch_size_reason: Why the sweep of the benchmark stops at
      `max_batch_size`. Required if and only if `max_batch_size` is set.
  """

  max_batch_size: int | None = None
  max_batch_size_reason: str | None = None

  def __post_init__(self):
    if (self.max_batch_size is not None) != bool(self.max_batch_size_reason):
      raise ValueError(
          "BatchSweepOptIn requires a max_batch_size_reason if and only if it"
          f" sets a max_batch_size, got max_batch_size={self.max_batch_size},"
          f" max_batch_size_reason={self.max_batch_size_reason!r}"
      )
    if self.max_batch_size is not None and self.max_batch_size < 1:
      raise ValueError(
          "BatchSweepOptIn.max_batch_size must be a positive batch size, got"
          f" {self.max_batch_size}"
      )


# The benchmarks of the batch sweep trial, by name: modern, representative
# forward (prefill) benchmarks that run on both TPUs and GPUs.
# tests/batch_sweep_benchmarks_test.py checks that each is registered and can
# be swept.
BATCH_SWEEP_BENCHMARKS: Mapping[str, BatchSweepOptIn] = types.MappingProxyType({
    "gemma_4_26b_a4b_6_layers_forward": BatchSweepOptIn(),
    "gemma_4_31b_6_layers_forward": BatchSweepOptIn(),
    "gpt_oss_120b_4_layers_forward": BatchSweepOptIn(),
    "llama_3_1_8b_forward": BatchSweepOptIn(),
    "modernbert_base_forward_gen": BatchSweepOptIn(),
    "qwen3_8b_forward": BatchSweepOptIn(),
    "siglip_so400m_patch14_384_forward_gen": BatchSweepOptIn(),
    # Video diffusion serves a few videos per request, and each sample of this
    # workload (5 frames of 704x1280) is already 1760 video tokens (2 latent
    # frames x 22 x 40 patches) that attend to 512 text tokens, so throughput
    # saturates at small batch sizes. Sweeping to batch size 1024 took ~1.5
    # machine-hours per run on TPU (~1.2 on B200), about a third of the trial,
    # at batch sizes that do not occur in serving.
    "wan_2_2_ti2v_5b_forward": BatchSweepOptIn(
        max_batch_size=64,
        max_batch_size_reason=(
            "video diffusion serves a few videos per request and each sample"
            " is already 1760 video tokens, so larger batches are not"
            " representative"
        ),
    ),
    "whisper_large_v3_turbo_forward_gen": BatchSweepOptIn(),
})
