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

"""An example wrapper for using torchrun with single-host distributed TorchTPU.

This sets up the environment variables expected by the `torch_tpu` library
for distributed TorchTPU tests on a single host. Enables tests to be run
with torchrun as follows:

The torchrun command is called on a wrapper around user logic:

```
g3_distributed.torchrun(
    tpu_env_wrapper(user_logic, world_size=N),
    nproc_per_node=N,
)()
```

The tpu_env_wrapper performs the following setup in the main process before
torchrun spawns worker processes:
1. Sets TORCH_TPU_SLICEBUILDER_ADDRESSES by finding N unused ports on localhost.
2. Detects and sets TORCH_TPU_TOPOLOGY.
3. Sets TORCH_TPU_XPROF_SESSION_ID.

Then, torchrun spawns N worker processes, each of which runs `user_logic` with
RANK, LOCAL_RANK, and WORLD_SIZE set in environment variables.
"""

import os
import time
from typing import Any, Callable

from absl import logging
import portpicker
from torch_tpu._internal.utils import hardware


def prepare_tpu_environment(world_size: int | None = None) -> None:
  """Prepares the environment variables required for TPU distributed execution.

  To be called in the main process before spawning workers.

  Args:
    world_size: If provided, use this as the target device count instead of
      auto-detecting from hardware. Allows sub-slicing only on single-host, in
      supported topologies.
  """
  xprof_session_id = os.environ.get("TORCH_TPU_XPROF_SESSION_ID", None)
  if xprof_session_id is None:
    xprof_session_id = str(time.time_ns())
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
  logging.info("TORCH_TPU_XPROF_SESSION_ID: %s", xprof_session_id)

  topology = hardware.get_tpu_topology(world_size)
  count = (
      world_size if world_size is not None else hardware.get_tpu_device_count()
  )

  # Get open ports on this host for slicebuilder.
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    sb_ports = [portpicker.pick_unused_port() for _ in range(count)]
    sb_addresses = ",".join([f"localhost:{p}" for p in sb_ports])
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses

  # Detect TPU topology.
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    if topology is None:
      raise ValueError("No TPU devices found.")
    os.environ["TORCH_TPU_TOPOLOGY"] = topology


class WorkerWrapper:
  """A picklable wrapper around the user function for distributed execution."""

  def __init__(self, func: Callable[..., Any]):
    self.func = func
    # torch.distributed requires entrypoint to have __name__ and __qualname__.
    self.__name__ = getattr(func, "__name__", str(func))
    self.__qualname__ = getattr(func, "__qualname__", str(func))

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    # Execute user function with any positional and keyword args.
    self.func(*args, **kwargs)


def tpu_env_wrapper(func, *, world_size=None) -> Any:
  """Internal wrapper to initialize worker and run user function.

  Args:
    func: The user's callable to be executed by each distributed worker.
    world_size: The number of TPU devices to use. If None, auto-detects based on
      available hardware. This MUST be passed as a keyword argument.

  Returns:
    A WorkerWrapper instance that can be called by torchrun.
  """
  prepare_tpu_environment(world_size)
  return WorkerWrapper(func)


if __name__ == "__main__":
  host_topology = hardware.get_tpu_topology()
  host_count = hardware.get_tpu_device_count()
  host_sb_ports = [portpicker.pick_unused_port() for _ in range(host_count)]
  host_sb_addresses = ",".join([f"localhost:{p}" for p in host_sb_ports])

  print(f'TORCH_TPU_TOPOLOGY="{host_topology or ""}"')
  print(f'TORCH_TPU_SLICEBUILDER_ADDRESSES="{host_sb_addresses}"')
  print(f"WORLD_SIZE={host_count}")
