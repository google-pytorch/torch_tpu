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
from torch_tpu._internal.distributed.launchers import environment

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
    are derived from that list plus `JOB_COMPLETION_INDEX`.
  * Anything else, a single hostname without `JOB_COMPLETION_INDEX` or no
    variable at all: addresses are `localhost:<unused port>`.

Counting the hostnames rather than just testing for `TPU_WORKER_HOSTNAMES`
matters, because the variable is ambient. A single-host TPU VM sets it to one
name, and `oss/.bazelrc` forwards whatever it finds into every test, so
treating its mere presence as multi-host would put a single-host job on the
rendezvous path and hang it.

This module is meant to take over from `singlehost_wrapper` and
`multihost_wrapper`. Both are left untouched for now; rewriting them as shims
over this one and moving their callers across is follow-up work.
"""

import time

from absl import logging
import portpicker
from torch_tpu._internal import env
from torch_tpu._internal import env_utils
from torch_tpu._internal.utils import hardware


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
      hosts, if no TPU devices are found on single-host, or if required
      multi-host variables (`TPU_WORKER_HOSTNAMES`, `JOB_COMPLETION_INDEX`) are
      unset on multi-host.
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

  node_rank = env.get_str_env_once("JOB_COMPLETION_INDEX")
  is_multihost = (
      (nnodes is not None and nnodes > 1)
      or len(worker_hostnames) > 1
      or node_rank is not None
  )

  if is_multihost:
    if not worker_hostnames:
      raise ValueError(
          "set_tpu_launch_env(): expected environment variable "
          "TPU_WORKER_HOSTNAMES to be a comma-separated list of worker "
          "hostnames or IP addresses, got it as unset"
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

    if node_rank is None:
      raise ValueError(
          "set_tpu_launch_env(): expected environment variable "
          "JOB_COMPLETION_INDEX to be a 0-based index of this host "
          "(Kubernetes Indexed Jobs set this automatically), got it as unset"
      )
    env_utils.get_and_set_str_env_once("NODE_RANK", lambda: node_rank)

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
