# Copyright 2025 Google LLC
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

"""GPU-specific boilerplate we need to run on multiple GPUs."""
import os

from absl import logging
import torch
import torch.distributed as dist


def run_in_workers(worker_fn, *worker_args):
  """Worker function for CUDA launched by gdist.torchrun.

  Args:
    worker_fn: The function to be executed by each worker. This function will
      receive `rank` and `world_size` as its first two arguments, followed by
      any arguments provided in `worker_args`.
    *worker_args: Additional arguments to be passed to `worker_fn`.
  """
  # Retrieve rank and world_size from environment variables
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))

  logging.info(
      "CUDA gdist worker %d/%d (local %d) starting...",
      rank,
      world_size,
      local_rank,
  )
  try:
    if not torch.cuda.is_available():
      raise RuntimeError(f"CUDA is not available on rank {rank}.")

    # Determine the specific device for this process.
    # We use local_rank to assign devices, assuming one GPU per process.
    # If using more processes than GPUs, adjust device_id logic.
    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
      logging.warning(
          "world_size (%d) is greater than available CUDA devices (%d)."
          " Multiple processes will share GPUs.",
          world_size,
          num_gpus,
      )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    logging.info("CUDA worker rank %d assigned to device: %s", rank, device)

    # Initialize the process group. dist.init_process_group
    # uses environment variables by default.
    dist.init_process_group(backend="nccl", device_id=local_rank)
    logging.info(
        "CUDA Distributed setup complete for rank %d/%d", rank, world_size
    )
    assert dist.is_initialized()

    worker_fn(rank, world_size, *worker_args)

  finally:
    dist.destroy_process_group()
    logging.info("CUDA gdist worker %d finished and group destroyed.", rank)
