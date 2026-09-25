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

"""A wrapper to run the tensor parallel worker via g3_distributed.torchrun."""

import os
from torch.google import distributed as g3_distributed
import torch.multiprocessing as mp
from torch.tpu.distributed import environment  # pyrefly: ignore[missing-import]
from examples.tutorials.distributed import hello_world_distributed
from torch_tpu._internal.distributed import multiprocessing


def main(_):
  environment.set_tpu_launch_env()
  world_size = int(os.environ["WORLD_SIZE"])
  g3_distributed.torchrun(
      hello_world_distributed.run_all_reduce,
      nproc_per_node=world_size,
  )()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_main(main)
