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

"""A test runner for the manual FSDP example."""

from torch.distributed import launcher
from torch.tpu import distributed  # pyrefly: ignore[missing-import]
from examples.fsdp.manual import manual_fsdp_worker


def main():
  world_size = 8
  distributed.set_tpu_launch_env(nproc_per_node=world_size)
  config = launcher.LaunchConfig(
      min_nodes=1,
      max_nodes=1,
      nproc_per_node=world_size,
      rdzv_backend="c10d",
      rdzv_endpoint="localhost:0",
      max_restarts=0,
  )
  launcher.elastic_launch(config, manual_fsdp_worker.worker_fn)()


if __name__ == "__main__":
  main()
