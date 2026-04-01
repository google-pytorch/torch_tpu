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

"""An example wrapper for using torchrun with distributed TorchTPU.

This sets up the environment variables expected by the `torch_tpu` library
for distributed TorchTPU tests. Enables tests to be run with torchrun.

The torchrun command is called on a wrapper around user logic:

```
g3_distributed.torchrun(
    tpu_env_wrapper(user_logic, world_size=N),
    nproc_per_node=N,
)()
```

The tpu_env_wrapper performs the following setup in the main process before
torchrun spawns worker processes:
1. Sets TORCH_TPU_SLICEBUILDER_ADDRESSES.
   - If TPU_WORKER_HOSTNAMES is set, generates addresses for each host with 8
     ports starting from BASE_PORT.
2. Detects and sets TORCH_TPU_TOPOLOGY.
3. Sets TORCH_TPU_XPROF_SESSION_ID.
4. Sets standard distributed variables for multihost (NNODES, NODE_RANK, etc).

Then, torchrun spawns worker processes, each of which runs `user_logic` with
RANK, LOCAL_RANK, and WORLD_SIZE set in environment variables.
"""

import os
import time

from absl import logging


def prepare_tpu_environment() -> None:
  """Prepares the environment variables required for TPU distributed execution.

  To be called in the main process before spawning workers.
  """
  xprof_session_id = os.environ.get("TORCH_TPU_XPROF_SESSION_ID", None)
  if xprof_session_id is None:
    xprof_session_id = str(time.time_ns())
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id

  # Get open ports on this host for slicebuilder.
  worker_hostnames_str = os.environ.get("TPU_WORKER_HOSTNAMES")
  if not worker_hostnames_str:
    raise ValueError("No TPU_WORKER_HOSTNAMES found.")
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    sb_addresses = []
    base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
    # 8 devices per host is true for all current platforms.
    devices_per_host = int(os.environ.get("TORCH_TPU_DEVICES_PER_HOST", 8))
    for host in worker_hostnames_str.split(","):
      for i in range(devices_per_host):
        sb_addresses.append(f"{host}:{base_port + i}")
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(sb_addresses)

  # Set multihost distributed configuration.
  worker_hostnames = worker_hostnames_str.split(",")
  num_nodes = len(worker_hostnames)
  os.environ.setdefault("NNODES", str(num_nodes))

  # Identify node rank.
  # JOB_COMPLETION_INDEX is standard for Kubernetes Indexed Jobs.
  node_rank = os.environ.get("JOB_COMPLETION_INDEX")
  if node_rank is None:
    raise ValueError("No JOB_COMPLETION_INDEX found in environment.")

  os.environ["NODE_RANK"] = node_rank

  # Master address defaults to the first worker in the list.
  os.environ.setdefault("MASTER_ADDR", worker_hostnames[0])
  os.environ.setdefault("MASTER_PORT", "29500")


if __name__ == "__main__":
  # Ensure absl logging doesn't go to stdout if it's somehow configured that way
  logging.get_absl_logger().propagate = False
  prepare_tpu_environment()

  host_topology = os.environ.get("TORCH_TPU_TOPOLOGY", "")
  host_sb_addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")

  # Only these prints should go to stdout for shell eval
  print(f'export TORCH_TPU_TOPOLOGY="{host_topology}"')
  print(f'export TORCH_TPU_SLICEBUILDER_ADDRESSES="{host_sb_addresses}"')
  print(
      "export"
      f' TORCH_TPU_XPROF_SESSION_ID="{os.environ.get("TORCH_TPU_XPROF_SESSION_ID", "")}"'
  )

  if "NNODES" in os.environ:
    print(f'export NNODES="{os.environ["NNODES"]}"')
    print(f'export NODE_RANK="{os.environ.get("NODE_RANK", "0")}"')
    print(f'export MASTER_ADDR="{os.environ.get("MASTER_ADDR", "localhost")}"')
    print(f'export MASTER_PORT="{os.environ.get("MASTER_PORT", "29500")}"')
