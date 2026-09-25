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

"""TPU environment bootstrap for PyTorch distributed launchers.

TorchTPU needs three environment variables to be decided in the *parent*
process, before any launcher spawns workers, because every rank must agree on
the same values:

  TORCH_TPU_SLICEBUILDER_ADDRESSES  one `host:port` per rank, same list on all
  TORCH_TPU_TOPOLOGY                physical mesh, e.g. "2,2,1"
  TORCH_TPU_XPROF_SESSION_ID        shared profiling session id

Nothing upstream populates these, so TorchTPU has to own the step. What it does
*not* need to own is spawning: `set_tpu_launch_env()` only mutates `os.environ`
and then gets out of the way, so it composes with `torchrun`, `elastic_launch`,
`torch.multiprocessing.spawn`, or any other launcher:

```python
from torch.tpu.distributed import environment

environment.set_tpu_launch_env(nproc_per_node=8)
torch.multiprocessing.spawn(user_fn, nprocs=8)
```

Single-host and multi-host differ only in where the addresses come from, so
they are one function rather than two modules. Which layout to populate is
read off `TPU_WORKER_HOSTNAMES` (the list of workers in the job) and
`JOB_COMPLETION_INDEX`:

  * Two or more hostnames (or `nnodes > 1` or `JOB_COMPLETION_INDEX` set):
    addresses are `<host>:<base_port + i>` across all of them, and the PyTorch
    rendezvous variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`)
    are derived from that list. `NODE_RANK` is derived from the physical
    coordinates of this host's TPU devices within the slice described by
    `TORCH_TPU_TOPOLOGY` when both are available, and from
    `JOB_COMPLETION_INDEX` otherwise.
  * Anything else, a single hostname without `JOB_COMPLETION_INDEX` or no
    variable at all: addresses are `localhost:<unused port>`.

Counting the hostnames rather than just testing for `TPU_WORKER_HOSTNAMES`
matters, because the variable is ambient. A single-host TPU VM sets it to one
name, and `oss/.bazelrc` forwards whatever it finds into every test, so
treating its mere presence as multi-host would put a single-host job on the
rendezvous path and hang it.

Topological `NODE_RANK`:
  Host coordinates `(Hx, Hy, Hz)` in the slice are sorted reverse-
  lexicographically, with `key=lambda h: h[::-1]` (Z-major, Y-intermediate,
  X-minor). This matches the JAX `mesh_utils` device coordinate ordering, so
  that node ranks are deterministic and topologically contiguous across the
  hosts of a multi-host slice. Only a `TORCH_TPU_TOPOLOGY` supplied from the
  outside describes the whole slice; the topology auto-detected from the local
  PCI devices describes this host alone, so it is never used to rank hosts.

  Querying the device coordinates initializes the PJRT client, which claims the
  TPU devices of the process that does it. The query therefore runs in a
  short-lived child process, so that the parent calling `set_tpu_launch_env()`
  leaves the devices free for the workers it is about to spawn.
"""

from collections.abc import Sequence
import json
import os
import re
import subprocess
import sys
import time

from absl import logging
import portpicker
from torch_tpu._internal import env
from torch_tpu._internal import env_utils
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import hardware

# Per-host chip bounds (Cx, Cy, Cz) of the TPU slices that multi-host jobs run
# on.
_CHIPS_PER_HOST: tuple[int, int, int] = (2, 2, 1)

# Module to run with `python -m` in the child process. Spelled out, because
# `__name__` is "__main__" when this module is itself run with `python -m`.
_MODULE_NAME = "torch_tpu._internal.distributed.environment"

# Command-line flag that makes `python -m` of this module print the physical
# coordinates of the local TPU device instead of the launch environment.
_PRINT_LOCAL_DEVICE_COORDS_FLAG = "--print_local_device_coords"

# Prefix of the stdout line on which the child process reports the local TPU
# device coordinates, as a JSON list. Importing TorchTPU may print to stdout
# too, so the line is found by its prefix rather than by its position.
_LOCAL_DEVICE_COORDS_PREFIX = "TORCH_TPU_LOCAL_DEVICE_COORDS="

# How long to wait for the child process to report the local TPU device
# coordinates before falling back to `JOB_COMPLETION_INDEX`.
_LOCAL_DEVICE_COORDS_TIMEOUT_SECONDS = 300


def _parse_topology(topology_str: str) -> tuple[int, int, int]:
  """Parses a topology string into (Mx, My, Mz) mesh bounds.

  Dimensions beyond the third, such as the cores per chip in "2,2,1,2", are
  dropped, and missing dimensions default to 1.

  Args:
    topology_str: A string representing the topology, e.g. "4x4x4", "2,2,1",
      "2,2,1,2", "2x4".

  Returns:
    A tuple of 3 integers representing (Mx, My, Mz).

  Raises:
    ValueError: If the topology string is empty or has a dimension that is not
      an integer.
  """
  try:
    parts = [int(p) for p in re.split(r"[,xX]", topology_str) if p]
  except ValueError:
    parts = []
  if not parts:
    raise ValueError(
        "set_tpu_launch_env(): expected environment variable"
        " TORCH_TPU_TOPOLOGY to be integer mesh dimensions separated by 'x'"
        f" or ',', e.g. '4x4x4' or '2,2,1', got '{topology_str}'"
    )
  if len(parts) < 3:
    parts = parts + [1] * (3 - len(parts))
  return (parts[0], parts[1], parts[2])


def _get_slice_host_coordinates(
    topology_str: str,
    chips_per_host: tuple[int, int, int] = _CHIPS_PER_HOST,
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

  Raises:
    ValueError: If the topology string cannot be parsed.
  """
  m_x, m_y, m_z = _parse_topology(topology_str)
  c_x, c_y, c_z = chips_per_host
  b_x = max(1, m_x // c_x)
  b_y = max(1, m_y // c_y)
  b_z = max(1, m_z // c_z)

  hosts = [
      (x, y, z) for x in range(b_x) for y in range(b_y) for z in range(b_z)
  ]
  hosts.sort(key=lambda h: h[::-1])
  return hosts


def _calculate_topological_node_rank(
    coords: Sequence[int] | None,
    topology_str: str | None,
    fallback_index: str | int | None = None,
    chips_per_host: tuple[int, int, int] = _CHIPS_PER_HOST,
) -> int:
  """Calculates the topologically sorted NODE_RANK from device coordinates.

  Maps local chip coordinates to local host coordinates
  (x // Cx, y // Cy, z // Cz) and finds the host's rank by locating it in
  `_get_slice_host_coordinates()`. If coordinates are unavailable or out of
  bounds, or the topology cannot be parsed, falls back to `fallback_index`.

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
      or topology_str are unavailable, invalid, or out of bounds.
    chips_per_host: Per-host chip bounds (Cx, Cy, Cz), default (2, 2, 1).

  Returns:
    The integer NODE_RANK.

  Raises:
    ValueError: If there is no fallback index and either the coordinates or
      topology are unavailable, the topology cannot be parsed, or the
      coordinates lie outside the slice.
  """
  if not coords or not topology_str:
    if fallback_index is not None:
      return int(fallback_index)
    raise ValueError(
        "set_tpu_launch_env(): expected either TPU device coordinates together"
        " with environment variable TORCH_TPU_TOPOLOGY, or environment"
        " variable JOB_COMPLETION_INDEX to be a 0-based index of this host"
        " (Kubernetes Indexed Jobs set this automatically), to determine"
        " NODE_RANK, got neither"
    )

  try:
    sorted_hosts = _get_slice_host_coordinates(
        topology_str, chips_per_host=chips_per_host
    )
  except ValueError:
    if fallback_index is not None:
      return int(fallback_index)
    raise

  padded_coords = list(coords) + [0] * max(0, 3 - len(coords))
  c_x, c_y, c_z = chips_per_host
  local_host = (
      padded_coords[0] // c_x,
      padded_coords[1] // c_y,
      padded_coords[2] // c_z,
  )
  if local_host in sorted_hosts:
    return sorted_hosts.index(local_host)
  if fallback_index is not None:
    return int(fallback_index)
  raise ValueError(
      "set_tpu_launch_env(): expected TPU device coordinates within the slice"
      " described by environment variable TORCH_TPU_TOPOLOGY"
      f' ("{topology_str}"), got {list(coords)}'
  )


def _query_local_device_coords() -> list[int] | None:
  """Returns the physical coordinates of this host's TPU device, if available.

  The coordinates are queried in a short-lived child process, because querying
  them initializes the PJRT client, which would otherwise hold on to the TPU
  devices in this process and keep the workers it spawns from claiming them.

  Returns:
    The coordinates [X, Y, Z, ...] of the local TPU device, or None if no TPU
    devices are present on this host or the child process fails to report them.
  """
  if not hardware.get_tpu_device_count():
    return None

  child_env = dict(os.environ)  # PY_ENVIRON_OK=inherited by the child.
  # Hermetic Python binaries set up `sys.path` at startup rather than through
  # `PYTHONPATH`, so pass it along for the child to find TorchTPU.
  child_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
  try:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            _MODULE_NAME,
            _PRINT_LOCAL_DEVICE_COORDS_FLAG,
        ],
        capture_output=True,
        check=True,
        env=child_env,
        text=True,
        timeout=_LOCAL_DEVICE_COORDS_TIMEOUT_SECONDS,
    )
  except (OSError, subprocess.SubprocessError) as e:
    logging.warning("Could not query local TPU device coordinates: %s", e)
    return None

  for line in reversed(result.stdout.splitlines()):
    if line.startswith(_LOCAL_DEVICE_COORDS_PREFIX):
      try:
        coords = json.loads(line.removeprefix(_LOCAL_DEVICE_COORDS_PREFIX))
      except json.JSONDecodeError as e:
        logging.warning("Could not parse local TPU device coordinates: %s", e)
        return None
      return [int(c) for c in coords]
  logging.warning(
      "Could not find local TPU device coordinates in child process output: %s",
      result.stdout,
  )
  return None


def _print_local_device_coords() -> None:
  """Prints the local TPU device coordinates for `_query_local_device_coords`."""
  from torch_tpu._internal.distributed import tpu_distributed  # pylint: disable=g-import-not-at-top

  coords = list(tpu_distributed.local_device_coords())
  print(f"{_LOCAL_DEVICE_COORDS_PREFIX}{json.dumps(coords)}")


@annotations.experimental(
    "torch.tpu.distributed.environment.set_tpu_launch_env is experimental"
    " and subject to change."
)
def set_tpu_launch_env(
    *,
    nnodes: int | None = None,
    nproc_per_node: int | None = None,
) -> None:
  """Sets the environment variables a TPU job needs, before workers spawn.

  Call this once in the parent process, before handing anything to a launcher.
  It is idempotent and never overwrites a variable that is already set, so an
  operator can pin any of them from the outside and a second call is a no-op.

  Args:
    nnodes: Expected number of hosts. Optional; when given it is checked against
      `TPU_WORKER_HOSTNAMES` so that a mismatched launcher configuration fails
      here rather than as a hang during rendezvous.
    nproc_per_node: Ranks to run on each host. Defaults to the number of
      attached TPU devices on a single host, and to `TORCH_TPU_DEVICES_PER_HOST`
      or 8 on multiple hosts.

  Raises:
    ValueError: If `nnodes` and `TPU_WORKER_HOSTNAMES` disagree on the number of
      hosts, if no TPU devices are found on single-host, or, on multi-host, if
      `TPU_WORKER_HOSTNAMES` is unset or `NODE_RANK` is unset and can be derived
      neither from the TPU device coordinates and `TORCH_TPU_TOPOLOGY` nor from
      `JOB_COMPLETION_INDEX`.
  """
  xprof_session_id = env_utils.get_and_set_str_env_once(
      "TORCH_TPU_XPROF_SESSION_ID", lambda: str(time.time_ns())
  )
  logging.info("TORCH_TPU_XPROF_SESSION_ID: %s", xprof_session_id)

  worker_hostnames_str = env.get_str_env_once("TPU_WORKER_HOSTNAMES")
  worker_hostnames = (
      [h for h in worker_hostnames_str.split(",") if h]
      if worker_hostnames_str
      else []
  )
  if (
      nnodes is not None
      and worker_hostnames
      and nnodes != len(worker_hostnames)
  ):
    raise ValueError(
        "set_tpu_launch_env(): expected nnodes to match the number of hosts in "
        f"environment variable TPU_WORKER_HOSTNAMES ({len(worker_hostnames)}: "
        f'"{worker_hostnames_str}"), got {nnodes}'
    )

  completion_index = env.get_str_env_once("JOB_COMPLETION_INDEX")
  is_multihost = (
      (nnodes is not None and nnodes > 1)
      or len(worker_hostnames) > 1
      or completion_index is not None
  )

  if is_multihost:
    if not worker_hostnames:
      raise ValueError(
          "set_tpu_launch_env(): expected environment variable "
          "TPU_WORKER_HOSTNAMES to be a comma-separated list of worker "
          "hostnames or IP addresses, got it as unset"
      )

    # Only a topology supplied from the outside describes the whole slice, so
    # read it before the auto-detected, host-local one below is filled in. Read
    # it through `os.environ`, because `env.get_str_env_once()` would memoize
    # it as unset before that write.
    slice_topology = os.environ.get(  # PY_ENVIRON_OK=must not memoize.
        "TORCH_TPU_TOPOLOGY"
    )

    def build_slicebuilder_addresses() -> str:
      """Returns one address per device across all worker hosts."""
      base_port = env.get_int_env_once(
          "TORCH_TPU_BASE_PORT", default_value=8070
      )
      default_devices = nproc_per_node or hardware.get_tpu_device_count() or 8
      devices_per_host = env.get_int_env_once(
          "TORCH_TPU_DEVICES_PER_HOST", default_value=default_devices
      )
      return ",".join(
          f"{host}:{base_port + i}"
          for host in worker_hostnames
          for i in range(devices_per_host)
      )

    env_utils.get_and_set_str_env_once(
        "TORCH_TPU_SLICEBUILDER_ADDRESSES", build_slicebuilder_addresses
    )

    resolved_nnodes = nnodes if nnodes is not None else len(worker_hostnames)
    env_utils.get_and_set_str_env_once("NNODES", lambda: str(resolved_nnodes))

    def resolve_node_rank() -> str:
      """Returns this host's rank, by physical coordinates when available."""
      coords = _query_local_device_coords() if slice_topology else None
      return str(
          _calculate_topological_node_rank(
              coords, slice_topology, completion_index
          )
      )

    env_utils.get_and_set_str_env_once("NODE_RANK", resolve_node_rank)

    env_utils.get_and_set_str_env_once(
        "MASTER_ADDR", lambda: worker_hostnames[0]
    )
    env_utils.get_and_set_str_env_once("MASTER_PORT", lambda: "29500")

    topology = hardware.get_tpu_topology()
    if topology is not None:
      env_utils.get_and_set_str_env_once("TORCH_TPU_TOPOLOGY", lambda: topology)
  else:
    topology = hardware.get_tpu_topology(nproc_per_node)
    count = (
        nproc_per_node
        if nproc_per_node is not None
        else hardware.get_tpu_device_count()
    )

    def pick_slicebuilder_addresses() -> str:
      """Returns addresses for `count` unused ports on this host."""
      return ",".join(
          f"localhost:{portpicker.pick_unused_port()}" for _ in range(count)
      )

    def require_topology() -> str:
      """Returns the detected topology, which must not be empty."""
      if topology is None:
        raise ValueError("set_tpu_launch_env(): expected TPU devices, got none")
      return topology

    env_utils.get_and_set_str_env_once(
        "TORCH_TPU_SLICEBUILDER_ADDRESSES", pick_slicebuilder_addresses
    )
    env_utils.get_and_set_str_env_once("TORCH_TPU_TOPOLOGY", require_topology)


def _print_launch_env() -> None:
  """Bootstraps TPU launch env and prints shell variable assignments to stdout."""
  logging.get_absl_logger().propagate = False
  set_tpu_launch_env()
  host_topology = env.get_str_env_once("TORCH_TPU_TOPOLOGY", default_value="")
  host_sb_addresses = env.get_str_env_once(
      "TORCH_TPU_SLICEBUILDER_ADDRESSES", default_value=""
  )
  host_xprof_session_id = env.get_str_env_once(
      "TORCH_TPU_XPROF_SESSION_ID", default_value=""
  )
  host_count = (
      len(host_sb_addresses.split(","))
      if host_sb_addresses
      else hardware.get_tpu_device_count()
  )
  host_node_rank = env.get_str_env_once("NODE_RANK", default_value="0")
  print(f'TORCH_TPU_TOPOLOGY="{host_topology}"')
  print(f'TORCH_TPU_SLICEBUILDER_ADDRESSES="{host_sb_addresses}"')
  print(f'TORCH_TPU_XPROF_SESSION_ID="{host_xprof_session_id}"')
  print(f"WORLD_SIZE={host_count}")
  # Always print NODE_RANK, so that launch scripts can pass
  # `--node_rank="${NODE_RANK}"` on single-host too.
  print(f'NODE_RANK="{host_node_rank}"')
  for var in ("NNODES", "MASTER_ADDR", "MASTER_PORT"):
    val = env.get_str_env_once(var)
    if val is not None:
      print(f'{var}="{val}"')


if __name__ == "__main__":
  if sys.argv[1:] == [_PRINT_LOCAL_DEVICE_COORDS_FLAG]:
    _print_local_device_coords()
  else:
    _print_launch_env()
