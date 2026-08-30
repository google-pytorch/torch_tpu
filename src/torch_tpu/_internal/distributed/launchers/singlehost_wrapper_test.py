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

"""Tests the singlehost_wrapper to check all mandatory variables are set."""

import os

from absl.testing import absltest
from torch.google import distributed as g3_distributed
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils


WORLD_SIZE = 8


def worker_func(rank, world_size):
  mandatory_vars = [
      "WORLD_SIZE",
      "RANK",
      "LOCAL_RANK",
      "MASTER_ADDR",
      "MASTER_PORT",
      "TORCH_TPU_SLICEBUILDER_ADDRESSES",
      "TORCH_TPU_SLICEBUILDER_PORT",
      "TORCH_TPU_TOPOLOGY",
  ]
  for var in mandatory_vars:
    if var not in os.environ:
      raise RuntimeError(f"Mandatory variable {var} not set in worker {rank}")

  # Basic validation of values
  if int(os.environ["WORLD_SIZE"]) != world_size:
    raise ValueError(
        f"WORLD_SIZE mismatch: expected {world_size}, got"
        f" {os.environ['WORLD_SIZE']}"
    )
  if int(os.environ["RANK"]) != rank:
    raise ValueError(
        f"RANK mismatch: expected {rank}, got {os.environ['RANK']}"
    )
  if int(os.environ["LOCAL_RANK"]) != rank:
    raise ValueError(
        f"LOCAL_RANK mismatch: expected {rank}, got {os.environ['LOCAL_RANK']}"
    )
  if os.environ["MASTER_ADDR"] != "localhost":
    raise ValueError(
        "MASTER_ADDR mismatch: expected localhost, got"
        f" {os.environ['MASTER_ADDR']}"
    )

  # Ensure ports are set and represent integers
  int(os.environ["MASTER_PORT"])
  int(os.environ["TORCH_TPU_SLICEBUILDER_PORT"])

  # Ensure topology and addresses are non-empty
  if not os.environ["TORCH_TPU_TOPOLOGY"]:
    raise ValueError("TORCH_TPU_TOPOLOGY is empty")
  if not os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"]:
    raise ValueError("TORCH_TPU_SLICEBUILDER_ADDRESSES is empty")


def dummy_worker_func():
  pass


def dummy_worker_func_with_arg(arg1):  # pylint: disable=unused-argument
  pass


class SingleHostTestLauncherTest(seed_test_utils.RepeatableTest):

  def test_mandatory_variables(self):
    if "TEST_UNDECLARED_OUTPUTS_DIR" not in os.environ:
      os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = (
          self.create_tempdir().full_path
      )

    g3_distributed.torchrun(
        singlehost_wrapper.tpu_env_wrapper(worker_func, world_size=WORLD_SIZE),
        nproc_per_node=8,
    )

  def test_tpu_env_wrapper_no_args(self):
    singlehost_wrapper.tpu_env_wrapper(dummy_worker_func, world_size=8)


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  multiprocessing.handle_test_main(absltest.main)
