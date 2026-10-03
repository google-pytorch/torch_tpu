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

from examples.distributed.fsdp.manual import manual_fsdp_worker
from tests.distributed import distributed_utils


def main():
  world_size = 8
  distributed_utils.dist_run(world_size, manual_fsdp_worker.worker_fn)


if __name__ == "__main__":
  main()
