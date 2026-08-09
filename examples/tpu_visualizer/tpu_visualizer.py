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

"""TPU Visualizer Tool for PyTorch Distributed Backend."""

import os
import re

from absl import logging
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import hardware
from tests.distributed import distributed_utils


def _get_coords_from_debug_string(debug_str: str) -> tuple[int, int, int, int]:
  # Format: TPU_<id>(process=<proc>,(<x>,<y>,<z>,<core>))
  match = re.search(r"\((\d+),(\d+),(\d+),(\d+)\)", debug_str)
  if match:
    return tuple(map(int, match.groups()))
  raise ValueError(f"Failed to parse coords from debug string: {debug_str}")


def _get_topology_string(
    coords_list: list[tuple[int, int, int, int]], my_rank: int
) -> str:
  """Returns a text-based 3D visualization of the TPU topology."""
  xs = [c[0] for c in coords_list]
  ys = [c[1] for c in coords_list]
  zs = [c[2] for c in coords_list]
  cores = [c[3] for c in coords_list]

  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  min_z, max_z = min(zs), max(zs)
  min_core, max_core = min(cores), max(cores)

  # Map from (x, y, z, core) to rank
  coord_to_rank = {}
  for rank, coord in enumerate(coords_list):
    coord_to_rank[coord] = rank

  lines = []
  lines.append("=" * 60)
  lines.append(f"TPU Topology Visualization (My Rank: {my_rank})")
  lines.append(
      f"Bounds: X:[{min_x}..{max_x}], Y:[{min_y}..{max_y}],"
      f" Z:[{min_z}..{max_z}], Core:[{min_core}..{max_core}]"
  )
  lines.append("=" * 60)

  # Print Z-slices. For each Z, print a grid of Y (rows) and X (columns).
  # If there are multiple cores per chip, print them side-by-side inside the
  # cell. Make each cell represent a chip (X, Y, Z). Inside the cell, show the
  # ranks on its cores.

  for z in range(min_z, max_z + 1):
    lines.append(f"\n--- Z-Slice: {z} ---")

    # Header row for X coordinates
    header = "Y \\ X | "
    for x in range(min_x, max_x + 1):
      header += f"   X={x}   | "
    lines.append(header)
    lines.append("-" * len(header))

    for y in range(min_y, max_y + 1):
      row_str = f" Y={y} | "
      for x in range(min_x, max_x + 1):
        # For this chip (x, y, z), find all cores
        chip_cores_str = []
        for core in range(min_core, max_core + 1):
          rank = coord_to_rank.get((x, y, z, core))
          if rank is not None:
            # Highlight my rank
            if rank == my_rank:
              chip_cores_str.append(f"*{rank}*")
            else:
              chip_cores_str.append(f" {rank} ")
          else:
            chip_cores_str.append(" . ")
        cell_content = "/".join(chip_cores_str)
        row_str += f"[{cell_content}] | "
      lines.append(row_str)

  lines.append("=" * 60)
  return "\n".join(lines)


def _run_visualizer(return_list: list[str] | None = None) -> None:
  assert "RANK" in os.environ
  assert "WORLD_SIZE" in os.environ

  world_size = int(os.environ["WORLD_SIZE"])
  rank = int(os.environ["RANK"])

  dist.init_process_group(backend="tpu_dist")

  # Get device debug string
  debug_str = tpu_distributed.device_debug_string()
  my_coords = _get_coords_from_debug_string(debug_str)

  # Gather coordinates from all ranks
  # We use a tensor on TPU to perform all_gather
  my_coords_tensor = torch.tensor(my_coords, dtype=torch.int32, device="tpu")
  gathered_coords = [
      torch.zeros_like(my_coords_tensor) for _ in range(world_size)
  ]

  dist.all_gather(gathered_coords, my_coords_tensor)

  # Convert gathered tensors to list of tuples
  coords_list = []
  for t in gathered_coords:
    coords_list.append(tuple(t.cpu().tolist()))

  # Log the visualization
  topology_str = _get_topology_string(coords_list, rank)
  logging.info("\n%s", topology_str)
  if return_list is not None:
    return_list[rank] = topology_str

  dist.barrier()
  dist.destroy_process_group()


def visualize_topology(world_size: int | None = None) -> list[str]:
  """Runs TPU visualizer on all ranks, logs topology, and returns it per rank.

  Args:
    world_size: The number of TPU devices to use. If None, auto-detects from
      hardware.

  Returns:
    A list of topology visualization strings, where index i corresponds to the
    topology string generated and logged by rank i.
  """
  if world_size is None:
    world_size = hardware.get_tpu_device_count()

  manager = mp.Manager()
  return_list = manager.list([None] * world_size)

  distributed_utils.dist_run(
      nproc_per_node=world_size,
      fn=singlehost_wrapper.tpu_env_wrapper(
          _run_visualizer, world_size=world_size
      ),
      return_list=return_list,
  )

  return list(return_list)
