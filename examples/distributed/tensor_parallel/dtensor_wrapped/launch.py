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

"""A test runner for the tensor parallel wrappers example."""

from torch.google import distributed as gdist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import environment
from examples.distributed.tensor_parallel.dtensor_wrapped import dtp_worker
from torch_tpu._internal.distributed import multiprocessing


def main(_):
  environment.set_tpu_launch_env()
  gdist.torchrun(dtp_worker.worker_fn, nproc_per_node=8)()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_main(main)
