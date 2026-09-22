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

"""Distributed testing utilities for TorchTPU."""

import os
from typing import Any, Callable

from absl import logging
import portpicker
import torch.multiprocessing as mp
from tests import seed_test_utils


def _worker_fn(
    local_rank: int,
    nproc_per_node: int,
    default_master_port: int,
    base_seed: int,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
  """Worker function for mp.spawn."""
  node_rank = int(os.environ.get("NODE_RANK", "0"))
  if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
  elif node_rank > 0:
    raise ValueError(
        f"NODE_RANK: {node_rank} > 0 suggests multihost job. Expected"
        " WORLD_SIZE environment variable to be set"
    )
  else:
    world_size = nproc_per_node
  global_rank = node_rank * nproc_per_node + local_rank
  if global_rank >= world_size:
    raise ValueError(
        f"Calculated global_rank {global_rank} >= world_size {world_size}"
    )

  if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
  if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(default_master_port)

  os.environ["RANK"] = str(global_rank)
  os.environ["LOCAL_RANK"] = str(local_rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_WORLD_SIZE"] = str(nproc_per_node)
  os.environ["GROUP_RANK"] = str(node_rank)

  if "XLA_FLAGS" in os.environ:
    os.environ["XLA_FLAGS"] = os.path.expandvars(os.environ["XLA_FLAGS"])

  seed_test_utils.seed_rngs(base_seed + global_rank)

  fn(*args, **kwargs)


def dist_run(
    nproc_per_node: int,
    fn: Callable[..., Any],
    *args: Any,
    base_seed: int | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> None:
  """Runs the given function in a distributed environment using mp.spawn.

  Think of dist_run(n, foo, *args, **kwargs) as running
  n copies of foo(*args, **kwargs) in parallel, each with a different rank.
  Preserves multi-host environment variables (MASTER_ADDR, MASTER_PORT,
  NODE_RANK, WORLD_SIZE) if already present in os.environ.

  Args:
    nproc_per_node: The number of processes to spawn on the current node.
    fn: The function to be executed by each distributed worker.
    *args: Positional arguments to pass to the function.
    base_seed: The base RNG seed for worker processes. Defaults to the seed
      chosen by RepeatableTest.choose_seed().
    max_retries: Maximum number of attempts if a port collision (EADDRINUSE)
      occurs when auto-selecting a port.
    **kwargs: Keyword arguments to pass to the function.
  """
  if base_seed is None:
    base_seed = seed_test_utils.RepeatableTest.choose_seed()

  for attempt in range(max_retries):
    default_master_port = (
        int(os.environ["MASTER_PORT"])
        if "MASTER_PORT" in os.environ
        else portpicker.pick_unused_port()
    )
    try:
      mp.spawn(
          _worker_fn,
          args=(
              nproc_per_node,
              default_master_port,
              base_seed,
              fn,
              args,
              kwargs,
          ),
          nprocs=nproc_per_node,
          join=True,
      )
      return
    except mp.ProcessRaisedException as e:
      # Only retry if:
      # 1. The failure was a port collision (EADDRINUSE).
      # 2. MASTER_PORT was not explicitly set in os.environ (if pinned by the
      #    caller or a multi-host orchestrator, picking a new port is not
      #    possible and re-trying the same port would fail).
      # 3. We have remaining attempts.
      if (
          "EADDRINUSE" in str(e)
          and "MASTER_PORT" not in os.environ
          and attempt < max_retries - 1
      ):
        logging.warning(
            "Port collision (EADDRINUSE) on port %d on attempt %d/%d; retrying"
            " with a new port.",
            default_master_port,
            attempt + 1,
            max_retries,
        )
        continue
      raise
