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

"""Tests for distributed_training_101.py.

Makes sure that torchrun can be used to run the code in distributed mode on TPU.
"""

import os
import re
import subprocess
import sys
import tempfile
import pytest


def get_distributed_env():
  """Sets up the environment for distributed TPU training."""
  env = os.environ.copy()

  # Try to use singlehost_wrapper to initialize TPU environment variables
  try:
    # We try to run it using the same executable that is running pytest
    wrapper_cmd = [
        sys.executable,
        "-m",
        "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
    ]
    result = subprocess.run(
        wrapper_cmd, capture_output=True, text=True, env=env
    )
    if result.returncode == 0:
      for line in result.stdout.splitlines():
        if "=" in line:
          key, val = line.split("=", 1)
          # Strip whitespace and literal quotes
          env[key] = val.strip().strip('"').strip("'")
  except Exception:
    pass  # Fallback to existing environment if wrapper fails

  return env


def test_distributed_training_run():
  """Test that the distributed training code from distributed_training_101.py

  can be run as a standalone script using torchrun on TPU.
  """

  # Standalone script content assembled from tutorial examples
  script_content = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import sys

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

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
    model = ToyModel().to(device).to(torch.bfloat16)
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

if __name__ == "__main__":
    main()
"""

  # Create a temporary script file
  with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(script_content)
    temp_script_path = f.name

  try:
    env = get_distributed_env()
    nproc = 4  # change to match number of TPUs

    # Run using torch.distributed.run module
    # We specify a custom master_port to avoid conflicts with other tests
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master_port=29509",
        f"--nproc_per_node={nproc}",
        temp_script_path,
    ]

    # Print env for debugging if needed (pytest -s)
    for k in [
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "WORLD_SIZE",
    ]:
      print(f"DEBUG ENV: {k}={env.get(k)}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Log output for debugging in case of failure
    if result.returncode != 0:
      print("STDOUT:", result.stdout)
      print("STDERR:", result.stderr)

    assert result.returncode == 0, (
        f"torchrun failed with return code {result.returncode}\nSTDOUT:"
        f" {result.stdout}\nSTDERR: {result.stderr}"
    )

    # 6. Verify Granular Output from the model

    # Find initialization messages: "Initializing Rank {rank}/{world_size}"
    init_regex = r"Initializing Rank (\d+)/(\d+)"
    init_matches = re.findall(init_regex, result.stdout)

    ranks_initialized = {int(m[0]) for m in init_matches}
    expected_ranks = set(range(nproc))
    assert ranks_initialized == expected_ranks, (
        f"Expected ranks {expected_ranks} to initialize, but found"
        f" {ranks_initialized}.\nSTDOUT: {result.stdout}"
    )

    # Find completion messages: "Rank {rank} Step Complete. Loss: {loss}"
    complete_regex = r"Rank (\d+) Step Complete\. Loss: ([\d\.]+)"
    complete_matches = re.findall(complete_regex, result.stdout)

    ranks_completed = {int(m[0]) for m in complete_matches}
    assert ranks_completed == expected_ranks, (
        f"Expected ranks {expected_ranks} to complete, but found"
        f" {ranks_completed}.\nSTDOUT: {result.stdout}"
    )

    # Verify loss values are valid floats
    for rank_str, loss_str in complete_matches:
      loss_val = float(loss_str)
      assert not (
          loss_val != loss_val
      ), f"Loss for rank {rank_str} is NaN."  # Simple NaN check
      print(f"Verified Rank {rank_str} Loss: {loss_val}")

  finally:
    if os.path.exists(temp_script_path):
      os.remove(temp_script_path)
