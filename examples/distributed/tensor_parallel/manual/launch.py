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

"""A test runner for the tensor parallel example."""

import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import multihost_wrapper
from examples.distributed.tensor_parallel.manual import tp_worker
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def main(_):
  world_size = 8
  multihost_wrapper.prepare_tpu_environment()
  distributed_utils.dist_run(world_size, tp_worker.worker_fn)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_main(main)
