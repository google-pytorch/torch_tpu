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

"""A wrapper to run the distributed tutorial worker via distributed_utils.dist_run."""

import os
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from examples.tutorials.distributed import hello_world_distributed
from tests.distributed import distributed_utils

try:
  from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
except (ImportError, ModuleNotFoundError):
  g3_multiprocessing = None


def main(_=None):
  world_size = int(os.environ["WORLD_SIZE"])
  singlehost_wrapper.prepare_tpu_environment(world_size=world_size)
  distributed_utils.dist_run(world_size, hello_world_distributed.run_all_reduce)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  if g3_multiprocessing is not None:
    g3_multiprocessing.handle_main(main)
  else:
    main()
