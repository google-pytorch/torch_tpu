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

import glob
import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal import execution_mode
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.sync import sync
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def _test_wrapper(test_fn, *args, **kwargs):
  dist.init_process_group(backend="tpu_dist")
  try:
    test_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def run_spmd_safe_decorator_test():
  @spmd_util.spmd_safe
  def eager_fused_all_reduce(x):
    y = x * 5
    dist.all_reduce(y)
    return y

  x = torch.ones((2, 2), device="cpu").to("tpu")

  with execution_mode.eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
    res = eager_fused_all_reduce(x)
    sync.synchronize(res, wait=True)


class SpmdSafeDecoratorTest(absltest.TestCase):
  _world_size = 4

  def test_spmd_safe_decorator(self):

    dump_dir = self.create_tempdir(name="xla_dump_fused_collective").full_path

    # We set XLA_FLAGS to dump the MLIR modules.
    # We set it here so it's inherited by the spawned worker processes.
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={dump_dir}"

    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_spmd_safe_decorator_test,
    )

    # Search for .mlir files in the dump directory.
    mlir_files = glob.glob(os.path.join(dump_dir, "**/*.mlir"), recursive=True)

    found_fused_module = False
    fusion_details = []

    for fpath in mlir_files:
      with open(fpath, "r") as f:
        mlir = f.read()
        has_all_reduce = "stablehlo.all_reduce" in mlir
        has_multiply = "stablehlo.multiply" in mlir

        if has_all_reduce and has_multiply:
          found_fused_module = True
          break
        elif has_all_reduce:
          fusion_details.append(f"Found all_reduce WITHOUT multiply in {fpath}")
    if not found_fused_module:
      details_str = "\n".join(fusion_details)
      raise RuntimeError(
          "Could not find an MLIR module containing both 'multiply' and "
          "'all_reduce'. "
          f"Dumps were in {dump_dir}. Files found: {mlir_files}\n"
          f"Details:\n{details_str}"
      )


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
