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
from torch.google import distributed as g3_distributed
import torch.multiprocessing as mp
from torch.tpu import distributed  # pyrefly: ignore[missing-import]
from examples.tutorials.distributed.fsdp_to_simple_fsdp import fsdp_to_simple_fsdp_worker
from torch_tpu._internal.distributed import multiprocessing


def main(_):
  distributed.set_tpu_launch_env()
  world_size = int(os.environ.get("WORLD_SIZE", "8"))
  g3_distributed.torchrun(
      fsdp_to_simple_fsdp_worker.worker_fn,
      nproc_per_node=world_size,
  )()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_main(main)
