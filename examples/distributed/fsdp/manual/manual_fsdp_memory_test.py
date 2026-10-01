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

"""Verifies ZeRO-3 memory behavior of the manual FSDP model under torch.compile.

Confirms that full unsharded weights (32.0 MiB for 4096x4096 and 64.0 MiB for
8192x4096 bf16 layers) are freed during the forward pass rather than retained
across the forward-backward boundary.
"""

from absl import logging
import torch
from torch import distributed as dist
from torch import nn
from examples.distributed.fsdp.manual import model
from tests.distributed import distributed_utils

logging.set_verbosity(logging.INFO)

BATCH_SIZE = 16
IN_FEATURES = 4096
OUT_FEATURES = 8192
NUM_STEPS = 4
RANDOM_SEED = 0


def memory_worker_fn() -> None:
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  torch.set_default_dtype(torch.bfloat16)
  torch.manual_seed(RANDOM_SEED + rank)

  with torch.device("tpu"):
    sharded_model = model.ManualSimpleNN(
        in_features=IN_FEATURES, out_features=OUT_FEATURES
    )

  compiled_model = torch.compile(sharded_model, backend="tpu")

  x = torch.randn(BATCH_SIZE, IN_FEATURES, device="tpu")
  y = torch.randn(BATCH_SIZE, OUT_FEATURES, device="tpu")

  # Run multiple steps to observe both initial compilation/warmup memory
  # (steps 0-1) and long-term steady-state memory (steps 2+).
  step_stats = []
  for step in range(NUM_STEPS):
    sharded_model.zero_grad(set_to_none=True)
    torch.tpu.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    mem_before_fwd = torch.accelerator.memory_allocated()

    y_pred = compiled_model(x)

    torch.tpu.synchronize()
    mem_after_fwd = torch.accelerator.memory_allocated()
    fwd_retained_bytes = mem_after_fwd - mem_before_fwd
    fwd_peak_bytes = torch.accelerator.max_memory_allocated() - mem_before_fwd
    step_stats.append((
        step,
        mem_before_fwd,
        mem_after_fwd,
        fwd_retained_bytes,
        fwd_peak_bytes,
    ))

    loss = nn.MSELoss()(y_pred, y) / world_size
    loss.backward()
    del y_pred, loss

  if rank == 0:
    lines = ["Memory summary across steps:"]
    for (
        step,
        mem_before_fwd,
        mem_after_fwd,
        fwd_retained_bytes,
        fwd_peak_bytes,
    ) in step_stats:
      lines.append(
          f"  step {step}: mem_before_fwd={mem_before_fwd / (1024 * 1024):.2f}"
          f" MiB, mem_after_fwd={mem_after_fwd / (1024 * 1024):.2f} MiB,"
          f" fwd_retained={fwd_retained_bytes / (1024 * 1024):.2f} MiB,"
          f" fwd_peak_delta={fwd_peak_bytes / (1024 * 1024):.2f} MiB"
      )
    logging.info("\n".join(lines))

  # Full bf16 weights are 32.0 MiB (linear1: 4096x4096) and 64.0 MiB
  # (linear2: 8192x4096). Sharded weights are 4.0 MiB + 8.0 MiB = 12.0 MiB.
  # Assert that retained memory across the forward-backward boundary is
  # < 20 MiB, proving that full weights (32.0 MiB / 64.0 MiB) were freed
  # during forward rather than saved for backward.
  for step, _, _, fwd_retained_bytes, _ in step_stats:
    assert fwd_retained_bytes < 20 * 1024 * 1024, (
        f"Step {step}: expected < 20 MiB retained after forward, got"
        f" {fwd_retained_bytes / (1024 * 1024):.2f} MiB"
    )

  dist.destroy_process_group()


def main():
  world_size = 8
  distributed_utils.dist_run(world_size, memory_worker_fn)


if __name__ == "__main__":
  main()
