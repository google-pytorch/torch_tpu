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

    Scaling from a single TPU core to an 8-core VM (or a multi-host Pod) is one
    of the primary reasons to use TPUs. TorchTPU handles this via a SPMD (Single
    Program, Multiple Data) model.

    > ** This is a distributed training example.** The code in this notebook is for **reference only** and cannot be executed interactively. Distributed code requires a process launcher (like `torchrun`) to manage multiple processes, which is not supported within a single notebook kernel.
    >
    > To run these examples, copy the code into a `.py` file and follow the execution instructions in Section 3.

    In this tutorial, you will learn:
    1.  **The `torchrun` Orchestrator**: Distributed execution on the TPU mesh.
    2.  **The Worker Function**: Initializing DDP and the TPU backend.
    3.  **Launching & "Strict Sync"**: Executing the script and avoiding deadlocks.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. The `torchrun` Orchestrator

    The current best practice for distributed training is to use standard PyTorch `torchrun`. On TPU, this is combined with the `singlehost_wrapper` (which we'll cover in Section 3) to initialize the TPU mesh and slice builder ports before the workers are spawned.
    """)
  return


@app.cell(disabled=True)
def _():
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.nn.parallel import DistributedDataParallel as DDP
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
    **Crucial Difference:** On TPU, each process only "sees" its assigned core as `tpu:0`. We do not pass `device_ids` to DDP.
    """)
  return


@app.cell(disabled=True)
def _(DDP, ToyModel, api, dist, nn, optim, torch):
  def main():
    # 1. Discover hardware for this process
    # This also registers the "tpu_dist" backend
    device = torch.device("tpu")

    # 2. Initialize process group with tpu_dist backend
    if not dist.is_initialized():
      dist.init_process_group(backend="tpu_dist")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Initializing Rank {rank}/{world_size}")

    # 3. Move model to TPU and wrap in DDP
    # Note: On TPU, we don't pass device_ids to DDP
    model = ToyModel().to("tpu").to(torch.bfloat16)
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 4. Training Step with dummy data
    # Ensure inputs/labels match the model's bfloat16 dtype
    inputs = torch.randn(20, 10).to(device).to(torch.bfloat16)
    labels = torch.randn(20, 5).to(device).to(torch.bfloat16)

    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    # 5. Mandatory Materialization Trigger
    # This ensures all ranks stay in lock-step
    current_loss = loss.item()

    print(f"Rank {rank} Step Complete. Loss: {current_loss:.4f}")
    dist.destroy_process_group()

  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Launching the Workers

    Finally, we execute the script using `torchrun` and the `singlehost_wrapper` from the command line. This setup is mandatory as it defines the hardware topology before the PyTorch processes start.

    ```bash
    # Run this in your TPU VM terminal:
    eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    torchrun --nproc_per_node=$WORLD_SIZE your_script.py
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ###  Golden Rules of Distributed TPU
    1.  **Never Diverge**: Avoid `if rank == 0` blocks around operations that trigger materialization (like `.item()`). If one rank runs ahead, the mesh deadlocks.
    2.  **Shared Seeds**: Ensure all ranks start with the same weights by using a shared seed.
    3.  **Flush at Exit**: Call `torch.tpu.manual_seed_all(0)` after your loop finishes to ensure the final weight update is executed.
    """)
  return


if __name__ == "__main__":
  app.run()
