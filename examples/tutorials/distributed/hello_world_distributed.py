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

"""Tests torch_tpu distributed example.

NOTE: You should not be running this script directly. Instead, run via:
  - `run_script.sh` or
  - `run_script.py` if using Bazel.
"""

import torch
from torch import distributed as dist


def run_all_reduce():
  # Initialize tpu device.

  # Initialize the process group. The "tpu_dist" backend is used for distributed
  # TPU communication. Environment variables (RANK, WORLD_SIZE, etc.) are
  # typically set by the launcher.
  dist.init_process_group(backend="tpu_dist")
  ws = dist.get_world_size()
  x = torch.tensor([0.0, 1.0, ws], device="tpu")
  torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM)

  expected = torch.tensor([0.0, ws, ws * ws])
  torch.testing.assert_close(x.cpu(), expected)
  print("result: ", x)
  print("✅ Distributed ran successfully!")


if __name__ == "__main__":
  run_all_reduce()
