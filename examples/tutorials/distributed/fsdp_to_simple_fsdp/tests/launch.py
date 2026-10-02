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

"""Bazel test runner for the FSDPv2 to SimpleFSDP tutorial.

This file is a test harness used to execute `fsdp_to_simple_fsdp_worker` in
automated test suites (`bazel test`). To run this tutorial directly from the
command line, use `../run_fsdp_to_simple_fsdp.sh` instead.
"""

import os
from torch.distributed import launcher
from torch.tpu import distributed  # pyrefly: ignore[missing-import]
from examples.tutorials.distributed.fsdp_to_simple_fsdp import fsdp_to_simple_fsdp_worker


def main():
  world_size = int(os.environ.get("WORLD_SIZE", "8"))
  distributed.set_tpu_launch_env(nproc_per_node=world_size)
  config = launcher.LaunchConfig(
      min_nodes=1,
      max_nodes=1,
      nproc_per_node=world_size,
      rdzv_backend="c10d",
      rdzv_endpoint="localhost:0",
      max_restarts=0,
  )
  launcher.elastic_launch(config, fsdp_to_simple_fsdp_worker.worker_fn)()


if __name__ == "__main__":
  main()
