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
# pylint: skip-file

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Distributed Training 101: Scaling to Multiple Cores

    This section covers how to train up to 8 TPUs on a single host. TorchTPU support both MPMD (Multiple Program, Multiple Data) and SPMD workloads.

    > **This is a distributed training example.** The code in this notebook is for **reference only** and cannot be executed interactively. Distributed code requires a process launcher (like `torchrun`) to manage multiple processes, which is not supported within a single notebook kernel.
    >
    > To run these examples, copy the code into a `.py` file and follow the execution instructions in Section 3.

    In this tutorial, you will learn:
    1.  **The `torchrun` Orchestrator**: Distributed execution on the TPU mesh.
    2.  **The Worker Function**: Initializing DDP and the TPU backend.
    3.  **Launching**: Executing the script safely.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. Model

    The current best practice for distributed training is to use standard PyTorch `torchrun`. On TPU, this is combined with a launch wrapper (covered in Section 3) to initialize the TPUs before the workers are spawned.
    """)
  return


@app.cell(disabled=True)
def _():
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch_tpu import api
  import torch.distributed as dist

  class ToyModel(nn.Module):

    def __init__(self):
      super(ToyModel, self).__init__()
      self.net1 = nn.Linear(10, 10)
      self.relu = nn.ReLU()
      self.net2 = nn.Linear(10, 5)

    def forward(self, x):
      return self.net2(self.relu(self.net1(x)))

  return DDP, ToyModel, api, dist, nn, optim, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. The Worker Function

    The `main` function contains the logic that will execute on *every* rank.

    **Crucial Differences from GPU:**
    1.  **Process Isolation:** Unlike GPUs where all processes can see all available devices, on TPU, each process *only* sees the single TPU core it has been assigned.
    2.  **Device Indexing:** Because a process only sees one core, the device index is always `0` (i.e., `tpu:0`), regardless of the process's rank.
    3.  **Process Group Backend:** When initializing the distributed process group, use `backend="tpu_dist"` instead of the standard `nccl` or `gloo`.

    Because of this, we **do not** pass `device_ids` to the DDP constructor.
    """)
  return


@app.cell(disabled=True)
def _(DDP, ToyModel, api, dist, nn, optim, torch):
  def main():
    # 1. Discover hardware for this process
    # This initializes the TPU and makes the "tpu" device string available.
    api.tpu_device()

    # 2. Initialize process group
    # Crucial Difference: Use "tpu_dist" instead of "nccl" or "gloo"
    dist.init_process_group(backend="tpu_dist")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Initializing Rank {rank}/{world_size}")

    # 3. Move model to TPU and wrap in DDP
    # Note: On TPU, we do NOT pass device_ids to DDP. Passing device_ids may cause errors.
    model = ToyModel().to("tpu")
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 4. Training Step with dummy data
    inputs = torch.randn(20, 10, device="tpu")
    labels = torch.randn(20, 5, device="tpu")

    # Initialize loss accumulator on device to avoid stalling the CPU
    epoch_loss = torch.zeros((), device="tpu")

    for step in range(5):
      optimizer.zero_grad()
      outputs = ddp_model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      # Detach the loss tensor and accumulate on TPU
      epoch_loss += loss.detach()

      # 5. Materialization Trigger
      # Bring the aggregated loss back to CPU once at the end
      avg_loss = (epoch_loss / 5).item()

      print(f"Rank {rank} Step Complete. Average Loss: {avg_loss:.4f}")
      dist.destroy_process_group()

    return

  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Launching the Workers

    Finally, we execute the script using `torchrun` and the `singlehost_wrapper` from the command line.

    *Note: The `_internal` wrapper is used to automatically configure the necessary environment variables for distributed execution on TPU. For more details on what these variables are doing under the hood, see the [Environment Variables](?file=environment_variables.py) reference.*

    ```bash
    # Run this in your TPU VM terminal:
    eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    torchrun --nproc_per_node=$WORLD_SIZE your_script.py
    ```
    """)
  return


if __name__ == "__main__":
  app.run()
