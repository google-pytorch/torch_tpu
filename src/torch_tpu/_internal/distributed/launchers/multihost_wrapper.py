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

import time

from absl import logging
from torch_tpu._internal import env
from torch_tpu._internal import env_utils


def prepare_tpu_environment() -> None:
  """Prepares the environment variables required for TPU distributed execution.

  To be called in the main process before spawning workers.

  Raises:
    ValueError: If `TPU_WORKER_HOSTNAMES` or `JOB_COMPLETION_INDEX` is unset.
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

  # Identify node rank.
  # JOB_COMPLETION_INDEX is standard for Kubernetes Indexed Jobs.
  node_rank = env.get_str_env_once("JOB_COMPLETION_INDEX")
  if node_rank is None:
    raise ValueError("No JOB_COMPLETION_INDEX found in environment.")

  env_utils.set_str_env("NODE_RANK", node_rank)

  # Master address defaults to the first worker in the list.
  env_utils.get_and_set_str_env_once("MASTER_ADDR", lambda: worker_hostnames[0])
  env_utils.get_and_set_str_env_once("MASTER_PORT", lambda: "29500")


if __name__ == "__main__":
  # Ensure absl logging doesn't go to stdout if it's somehow configured that way
  logging.get_absl_logger().propagate = False
  prepare_tpu_environment()

  host_topology = env.get_str_env_once("TORCH_TPU_TOPOLOGY", default_value="")
  host_sb_addresses = env.get_str_env_once(
      "TORCH_TPU_SLICEBUILDER_ADDRESSES", default_value=""
  )
  host_xprof_session_id = env.get_str_env_once(
      "TORCH_TPU_XPROF_SESSION_ID", default_value=""
  )
  host_nnodes = env.get_str_env_once("NNODES")

  # Only these prints should go to stdout for shell eval
  print(f'export TORCH_TPU_TOPOLOGY="{host_topology}"')
  print(f'export TORCH_TPU_SLICEBUILDER_ADDRESSES="{host_sb_addresses}"')
  print(f'export TORCH_TPU_XPROF_SESSION_ID="{host_xprof_session_id}"')

  if host_nnodes is not None:
    host_node_rank = env.get_str_env_once("NODE_RANK", default_value="0")
    host_master_addr = env.get_str_env_once(
        "MASTER_ADDR", default_value="localhost"
    )
    host_master_port = env.get_str_env_once(
        "MASTER_PORT", default_value="29500"
    )
    print(f'export NNODES="{host_nnodes}"')
    print(f'export NODE_RANK="{host_node_rank}"')
    print(f'export MASTER_ADDR="{host_master_addr}"')
    print(f'export MASTER_PORT="{host_master_port}"')
