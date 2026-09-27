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

"""Environment setup and launcher helpers for multihost distributed TorchTPU.

This module sets up environment variables required for running distributed
TorchTPU jobs across multiple hosts. The wrapper's contract is standard shell
environment exports and is decoupled from `torchrun`.

CLI Eval Usage:
    In launch scripts (e.g. bash scripts or cluster launcher templates), run:

    ```bash
    eval "$(python -m torch_tpu._internal.distributed.launchers.multihost_wrapper)"
    ```

    This executes `prepare_tpu_environment()` and prints shell `export` statements
    to stdout:
      - `export TORCH_TPU_TOPOLOGY="..."`
      - `export TORCH_TPU_SLICEBUILDER_ADDRESSES="..."`
      - `export TORCH_TPU_XPROF_SESSION_ID="..."`
      - `export NODE_RANK="..."`
      - `export NNODES="..."`
      - `export MASTER_ADDR="..."`
      - `export MASTER_PORT="..."`

    After evaluating the exports, launch `torchrun` or a custom multi-host
    orchestrator on each host.

    Example: Using with `torchrun`:

    ```bash
    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        --nproc_per_node=8 \
        my_script.py
    ```

    Example: Using with SGLang or custom multi-host orchestrators:

    ```bash
    eval "$(python -m torch_tpu._internal.distributed.launchers.multihost_wrapper)"

    python -m sglang.launch_server \
        --nnodes="${NNODES}" \
        --node-rank="${NODE_RANK}" \
        --dist-init-addr="${MASTER_ADDR}:${MASTER_PORT}"
    ```

Coordinate Sorting Invariants:
    TPU slice host coordinates are sorted reverse-lexicographically:
    `(Hx, Hy, Hz)` with `key=lambda h: h[::-1]` (Z-major, Y-intermediate, X-minor).
    This matches JAX `mesh_utils` device coordinate sorting conventions,
    ensuring deterministic and topologically contiguous node rank assignments
    across hosts in multi-host TPU slices.
"""

from collections.abc import Sequence
import os
import re
import time

from absl import logging
from torch_tpu._internal import env
from torch_tpu._internal import env_utils


def parse_topology(topology_str: str) -> tuple[int, int, int]:
  """Parses a topology string into (Mx, My, Mz) mesh bounds.

  Args:
    topology_str: A string representing the topology, e.g. "4x4x4", "2,2,1",
      "2,2,1,2", "2x4".

  Returns:
    A tuple of 3 integers representing (Mx, My, Mz).

  Raises:
    ValueError: If the topology string is invalid or cannot be parsed.
  """
  parts = [int(p) for p in re.split(r"[,xX]", topology_str) if p]
  if not parts:
    raise ValueError(f"Invalid topology string: '{topology_str}'")
  if len(parts) < 3:
    parts = parts + [1] * (3 - len(parts))
  return (parts[0], parts[1], parts[2])


def get_slice_host_coordinates(
    topology_str: str,
    chips_per_host: tuple[int, int, int] = (2, 2, 1),
) -> list[tuple[int, int, int]]:
  """Computes and returns all host coordinates in the slice, topologically sorted.

  Host coordinates (Hx, Hy, Hz) are derived from mesh bounds (Mx, My, Mz) and
  per-host chip bounds (Cx, Cy, Cz):
    Bx = max(1, Mx // Cx)
    By = max(1, My // Cy)
    Bz = max(1, Mz // Cz)

  The host coordinates are sorted reverse-lexicographically with
  `key=lambda h: h[::-1]` (Z-major, Y-intermediate, X-minor), matching JAX
  mesh_utils convention.

  Note:
    `NODE_RANK` provides physical coordinate linear mapping across hosts
    (Z-major, X-minor) to establish deterministic host-level identity.
    Multi-dimensional device mesh permutations (such as ring/snake collectives
    in JAX `create_device_mesh` or PyTorch `init_device_mesh`) operate on
    device meshes within application runtimes, rather than altering host
    `NODE_RANK` assignment.

  Args:
    topology_str: A string representing the topology, e.g. "4x4x4", "2,2,1",
      "2,2,1,2", "2x4".
    chips_per_host: Per-host chip bounds (Cx, Cy, Cz), default (2, 2, 1).

  Returns:
    A list of 3-tuples representing sorted host coordinates (Hx, Hy, Hz).
  """
  m_x, m_y, m_z = parse_topology(topology_str)
  c_x, c_y, c_z = chips_per_host
  b_x = max(1, m_x // c_x)
  b_y = max(1, m_y // c_y)
  b_z = max(1, m_z // c_z)

  hosts = [
      (x, y, z) for x in range(b_x) for y in range(b_y) for z in range(b_z)
  ]
  hosts.sort(key=lambda h: h[::-1])
  return hosts


def calculate_topological_node_rank(
    coords: Sequence[int] | None,
    topology_str: str | None,
    fallback_index: str | int | None = None,
    chips_per_host: tuple[int, int, int] = (2, 2, 1),
) -> int:
  """Calculates the topologically sorted NODE_RANK from device coordinates.

  Maps local chip coordinates to local host coordinates (x // Cx, y // Cy, z //
  Cz)
  and finds the host's rank by locating it in `get_slice_host_coordinates()`.
  If coordinates are out of bounds or unavailable, falls back to
  `fallback_index`.

  Note:
    `NODE_RANK` provides physical coordinate linear mapping across hosts
    (Z-major, X-minor) to establish deterministic host-level identity.
    Multi-dimensional device mesh permutations (such as ring/snake collectives
    in JAX `create_device_mesh` or PyTorch `init_device_mesh`) operate on
    device meshes within application runtimes rather than redefining host
    `NODE_RANK`.

  Args:
    coords: Sequence of chip coordinates [X, Y, Z, ...] or None.
    topology_str: Topology specification string, e.g. "4x4x4" or "2,2,1".
    fallback_index: Optional fallback rank (e.g. JOB_COMPLETION_INDEX) if coords
      or topology_str are unavailable or out of bounds.
    chips_per_host: Per-host chip bounds (Cx, Cy, Cz), default (2, 2, 1).

  Returns:
    The integer NODE_RANK.

  Raises:
    ValueError: If neither coordinates/topology nor a fallback index is
    available, or if coordinates are out of bounds and no fallback index is
    available.
  """
  if not coords or not topology_str:
    if fallback_index is not None:
      return int(fallback_index)
    raise ValueError(
        "Cannot determine NODE_RANK: neither valid TPU coordinates/topology"
        " nor fallback index available."
    )

  try:
    sorted_hosts = get_slice_host_coordinates(
        topology_str, chips_per_host=chips_per_host
    )
    padded_coords = list(coords) + [0] * max(0, 3 - len(coords))
    c_x, c_y, c_z = chips_per_host
    local_host = (
        padded_coords[0] // c_x,
        padded_coords[1] // c_y,
        padded_coords[2] // c_z,
    )
    return sorted_hosts.index(local_host)
  except (ValueError, IndexError, ZeroDivisionError) as e:
    if fallback_index is not None:
      return int(fallback_index)
    raise ValueError(
        f"Cannot determine NODE_RANK: coords {coords} or topology"
        f" '{topology_str}' invalid or out of bounds: {e}"
    ) from e


def prepare_tpu_environment() -> None:
  """Prepares the environment variables required for TPU distributed execution.

  To be called in the main process before spawning workers.

  Raises:
    ValueError: If `TPU_WORKER_HOSTNAMES` is unset or `NODE_RANK` cannot be
      determined.
  """
  env_utils.get_and_set_str_env_once(
      "TORCH_TPU_XPROF_SESSION_ID", lambda: str(time.time_ns())
  )

  # Get open ports on this host for slicebuilder.
  worker_hostnames_str = env.get_str_env_once("TPU_WORKER_HOSTNAMES")
  if not worker_hostnames_str:
    raise ValueError("No TPU_WORKER_HOSTNAMES found.")
  worker_hostnames = worker_hostnames_str.split(",")

  def build_slicebuilder_addresses() -> str:
    """Returns one address per device across all worker hosts."""
    base_port = env.get_int_env_once("TORCH_TPU_BASE_PORT", default_value=8070)
    # 8 devices per host is true for all current platforms.
    devices_per_host = env.get_int_env_once(
        "TORCH_TPU_DEVICES_PER_HOST", default_value=8
    )
    return ",".join(
        f"{host}:{base_port + i}"
        for host in worker_hostnames
        for i in range(devices_per_host)
    )

  env_utils.get_and_set_str_env_once(
      "TORCH_TPU_SLICEBUILDER_ADDRESSES", build_slicebuilder_addresses
  )

  # Set multihost distributed configuration.
  env_utils.get_and_set_str_env_once(
      "NNODES", lambda: str(len(worker_hostnames))
  )

  # If TPU device coordinates and topology are available, compute topologically
  # sorted NODE_RANK. Otherwise fallback to JOB_COMPLETION_INDEX.
  completion_index = env.get_str_env_once("JOB_COMPLETION_INDEX")
  topology_str = env.get_str_env_once("TORCH_TPU_TOPOLOGY")

  coords = None
  if (
      "TPU_LIBRARY_PATH" in os.environ  # PY_ENVIRON_OK=external env var.
      or "PJRT_DEVICE" in os.environ  # PY_ENVIRON_OK=external env var.
  ):
    try:
      from torch_tpu._internal.distributed import tpu_distributed  # pylint: disable=g-import-not-at-top

      coords = tpu_distributed.local_device_coords()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Could not query local_device_coords: %s", e)

  if coords and topology_str:
    node_rank = str(
        calculate_topological_node_rank(coords, topology_str, completion_index)
    )
  elif completion_index is not None:
    node_rank = str(completion_index)
  else:
    raise ValueError(
        "Cannot determine NODE_RANK: neither TPU device coordinates with"
        " TORCH_TPU_TOPOLOGY nor JOB_COMPLETION_INDEX found in environment."
    )

  env_utils.set_str_env("NODE_RANK", node_rank)

  # Master address defaults to the first worker in the list.
  env_utils.get_and_set_str_env_once("MASTER_ADDR", lambda: worker_hostnames[0])
  env_utils.get_and_set_str_env_once("MASTER_PORT", lambda: "29500")


def print_shell_exports() -> None:
  """Prints environment export statements to stdout for shell evaluation."""
  host_topology = env.get_str_env_once("TORCH_TPU_TOPOLOGY", default_value="")
  host_sb_addresses = env.get_str_env_once(
      "TORCH_TPU_SLICEBUILDER_ADDRESSES", default_value=""
  )
  host_xprof_session_id = env.get_str_env_once(
      "TORCH_TPU_XPROF_SESSION_ID", default_value=""
  )
  host_node_rank = env.get_str_env_once("NODE_RANK", default_value="0")
  host_nnodes = env.get_str_env_once("NNODES")

  # Only these prints should go to stdout for shell eval
  print(f'export TORCH_TPU_TOPOLOGY="{host_topology}"')
  print(f'export TORCH_TPU_SLICEBUILDER_ADDRESSES="{host_sb_addresses}"')
  print(f'export TORCH_TPU_XPROF_SESSION_ID="{host_xprof_session_id}"')
  print(f'export NODE_RANK="{host_node_rank}"')

  if host_nnodes is not None:
    host_master_addr = env.get_str_env_once(
        "MASTER_ADDR", default_value="localhost"
    )
    host_master_port = env.get_str_env_once(
        "MASTER_PORT", default_value="29500"
    )
    print(f'export NNODES="{host_nnodes}"')
    print(f'export MASTER_ADDR="{host_master_addr}"')
    print(f'export MASTER_PORT="{host_master_port}"')


if __name__ == "__main__":
  # Ensure absl logging doesn't go to stdout if it's somehow configured that way
  logging.get_absl_logger().propagate = False
  prepare_tpu_environment()
  print_shell_exports()
