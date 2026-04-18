# Copyright 2025 Google LLC
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

"""Tests PyTorch collective all_gather errors on multiple TPUs."""

import os
import re

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from tests import error_testing as et
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def run_all_gather_dtype_error() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  input_tensor = torch.ones((1, 2), dtype=torch.float64, device="tpu")
  output_tensors = [
      torch.empty((1, 2), dtype=torch.int64, device="tpu")
      for _ in range(world_size)
  ]
  # This message comes from native PyTorch.
  expected_msg = re.compile("Invalid usage of tensors with different dtypes.*")
  with et.assert_raises_message(ValueError, tpu=expected_msg):
    dist.all_gather(output_tensors, input_tensor)


def run_all_gather_wrong_num_tensors() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  input_tensor = torch.ones((1, 2), dtype=torch.float32, device="tpu")
  output_tensors = [torch.empty((1, 2), dtype=torch.float32, device="tpu")]
  # This message comes from native PyTorch.
  expected_msg = re.compile(
      r"distributed\.all_gather\(\): output tensor list must have one tensor"
      r" per process, got 1 tensor and 8 processes.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    dist.all_gather(output_tensors, input_tensor)


def run_all_gather_dispatch_failure() -> None:
  _ = api.tpu_device()
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  input_tensor = torch.ones((1, 2), dtype=torch.float32, device="tpu")
  output_tensors = [
      torch.empty((1, 2), dtype=torch.float32, device="tpu")
      for _ in range(world_size)
  ]
  tt_testing.set_op_dispatch_failure(
      "distributed.all_gather", "forced dispatch failure"
  )

  expected_msg = re.compile(
      r"distributed\.all_gather\(\): forced dispatch failure.*"
  )
  with et.assert_raises_message(RuntimeError, tpu=expected_msg):
    dist.all_gather(output_tensors, input_tensor)


class AllGatherErrorsTest(et.TpuOnlyErrorTestBase):
  _world_size = 8

  def test_invalid_dtype(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_dtype_error, world_size=self._world_size
        ),
    )

  def test_wrong_num_of_tensors(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_wrong_num_tensors, world_size=self._world_size
        ),
    )

  def test_dispatch_failure(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_dispatch_failure, world_size=self._world_size
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
