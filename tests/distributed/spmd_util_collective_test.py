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
import hashlib
import os
from typing import TypeAlias
from unittest import mock

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal import execution_mode
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.sync import sync
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

EagerMode: TypeAlias = execution_mode.EagerMode


def _test_wrapper(test_fn, *args, **kwargs):
  dist.init_process_group(backend="tpu_dist")
  try:
    test_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def run_spmd_safe_decorator_test(compile_test: bool = False):
  @spmd_util.spmd_safe
  def eager_fused_all_reduce(x):
    y = x * 5
    dist.all_reduce(y)
    return y

  x = torch.ones((2, 2), device="cpu").to("tpu")

  if compile_test:
    compiled_fn = torch.compile(eager_fused_all_reduce)
    res = compiled_fn(x)
  else:
    with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
      res = eager_fused_all_reduce(x)
  sync.synchronize(res, wait=True)


def run_spmd_safe_decorator_backward_test():
  @spmd_util.spmd_safe
  def eager_fused_all_reduce(x):
    y = x * 5
    dist.all_reduce(y)
    return y

  x = torch.ones((2, 2), device="tpu", requires_grad=True)
  compiled_fn = torch.compile(eager_fused_all_reduce)
  res = compiled_fn(x)
  loss = res.sum()
  loss.backward()
  sync.synchronize(x.grad, wait=True)


class SpmdSafeDecoratorTest(absltest.TestCase):
  _world_size = 4

  def _check_fused_mlir(self, dump_dir, expected_count=1):
    """Checks there is exactly `expected_count` unique fused modules with all_reduce and multiply."""
    # Search for .mlir files in the dump directory.
    mlir_files = glob.glob(os.path.join(dump_dir, "**/*.mlir"), recursive=True)

    unique_fused_hashes = set()
    fusion_details = []

    for fpath in mlir_files:
      with open(fpath, "r") as f:
        mlir = f.read()
        has_all_reduce = "stablehlo.all_reduce" in mlir
        has_multiply = "stablehlo.multiply" in mlir

        if has_all_reduce and has_multiply:
          h = hashlib.sha256(mlir.encode()).hexdigest()
          unique_fused_hashes.add(h)
        elif has_all_reduce:
          fusion_details.append(f"Found all_reduce WITHOUT multiply in {fpath}")

    if len(unique_fused_hashes) != expected_count:
      details_str = "\n".join(fusion_details)
      raise RuntimeError(
          f"Expected exactly {expected_count} UNIQUE fused modules, but only"
          f" found {len(unique_fused_hashes)}. Unique hashes:"
          f" {unique_fused_hashes}\nDumps were in {dump_dir}. Files found:"
          f" {mlir_files}\nDetails:\n{details_str}"
      )

  def test_spmd_safe_decorator(self):
    dump_dir = self.create_tempdir(name="xla_dump_fused_collective").full_path
    with mock.patch.dict(
        os.environ, {"XLA_FLAGS": f"--xla_dump_to={dump_dir}"}
    ):
      distributed_utils.dist_run(
          nproc_per_node=self._world_size,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _test_wrapper, world_size=self._world_size
          ),
          test_fn=run_spmd_safe_decorator_test,
      )
    self._check_fused_mlir(dump_dir)

  def test_spmd_safe_decorator_compile(self):
    dump_dir = self.create_tempdir(
        name="xla_dump_fused_collective_compile"
    ).full_path
    with mock.patch.dict(
        os.environ, {"XLA_FLAGS": f"--xla_dump_to={dump_dir}"}
    ):
      distributed_utils.dist_run(
          nproc_per_node=self._world_size,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _test_wrapper, world_size=self._world_size
          ),
          test_fn=run_spmd_safe_decorator_test,
          compile_test=True,
      )
    self._check_fused_mlir(dump_dir)

  def test_spmd_safe_decorator_compile_backward(self):
    dump_dir = self.create_tempdir(
        name="xla_dump_fused_collective_compile_backward"
    ).full_path
    with mock.patch.dict(
        os.environ, {"XLA_FLAGS": f"--xla_dump_to={dump_dir}"}
    ):
      distributed_utils.dist_run(
          nproc_per_node=self._world_size,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _test_wrapper, world_size=self._world_size
          ),
          test_fn=run_spmd_safe_decorator_backward_test,
      )
    self._check_fused_mlir(dump_dir, expected_count=2)


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
