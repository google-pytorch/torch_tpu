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
import re
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


def _update_xla_dump_to(xla_flags: str, dump_dir: str) -> str:
  pattern = r"--xla_dump_to=\S+"
  new_flag = f"--xla_dump_to={dump_dir}"
  if re.search(pattern, xla_flags):
    return re.sub(pattern, new_flag, xla_flags)
  return f"{xla_flags} {new_flag}".strip()


def _test_wrapper(test_fn, *args, **kwargs):
  dump_dir = kwargs.pop("dump_dir", None)
  if dump_dir is not None:
    rank = os.environ.get("RANK", "0")
    rank_dump_dir = os.path.join(dump_dir, f"rank_{rank}")
    os.makedirs(rank_dump_dir, exist_ok=True)
    os.environ["XLA_FLAGS"] = _update_xla_dump_to(
        os.environ.get("XLA_FLAGS", ""), rank_dump_dir
    )

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


def run_uncoordinated_op_stream_test():
  """Repro test for interaction of SPMD safe and repeated ops heuristic."""
  rank = dist.get_rank()
  x = torch.ones((2, 2), device="cpu").to("tpu")

  with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
    for i in range(10):
      if rank == 0 and i % 2 == 0:
        x = x + 1

      @spmd_util.spmd_safe
      def foo(x):
        for _ in range(10):
          x = x + 1
        dist.all_reduce(x)
        return x

      x = foo(x)

    sync.synchronize(x, wait=True)


def run_spmd_safe_dtensor_compile_test():
  world_size = dist.get_world_size()
  mesh = torch.distributed.tensor.DeviceMesh("tpu", torch.arange(world_size))

  @spmd_util.spmd_safe
  def my_func(x):
    return x * 2

  compiled_func = torch.compile(my_func)

  param = torch.randn(4, 4, device="tpu")
  dtensor = torch.distributed.tensor.DTensor.from_local(
      param, mesh, [torch.distributed.tensor.Shard(0)], run_check=False
  )

  out = compiled_func(dtensor)
  assert isinstance(out, torch.distributed.tensor.DTensor)


def run_spmd_safe_multi_dtensor_compile_test():
  world_size = dist.get_world_size()
  mesh = torch.distributed.tensor.DeviceMesh("tpu", torch.arange(world_size))

  @spmd_util.spmd_safe
  def my_func(x, y):
    return x + y

  compiled_func = torch.compile(my_func)

  param1 = torch.randn(4, 4, device="tpu")
  dtensor1 = torch.distributed.tensor.DTensor.from_local(
      param1, mesh, [torch.distributed.tensor.Shard(0)], run_check=False
  )
  param2 = torch.randn(4, 4, device="tpu")
  dtensor2 = torch.distributed.tensor.DTensor.from_local(
      param2, mesh, [torch.distributed.tensor.Shard(0)], run_check=False
  )

  out = compiled_func(dtensor1, dtensor2)
  assert isinstance(out, torch.distributed.tensor.DTensor)


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

  def _check_all_reduce_mlir_matching(self, dump_dir):
    """Verifies that all MLIR files containing all_reduce match across all ranks."""
    rank_mlirs = []
    for r in range(self._world_size):
      rank_dir = os.path.join(dump_dir, f"rank_{r}")
      mlir_files = glob.glob(
          os.path.join(rank_dir, "**/*.mlir"), recursive=True
      )

      all_reduce_contents = []
      for fpath in sorted(mlir_files):
        with open(fpath, "r") as f:
          content = f.read()
          if "all_reduce" in content:
            all_reduce_contents.append(content)

      rank_mlirs.append(all_reduce_contents)

    rank_0_mlirs = rank_mlirs[0]
    self.assertNotEmpty(
        rank_0_mlirs, f"No all_reduce MLIR files found for rank 0 in {dump_dir}"
    )

    for r in range(1, self._world_size):
      self.assertEqual(
          len(rank_mlirs[r]),
          len(rank_0_mlirs),
          f"Rank {r} produced {len(rank_mlirs[r])} all_reduce MLIR modules, "
          f"but Rank 0 produced {len(rank_0_mlirs)}.",
      )
      for idx, (mlir_0, mlir_r) in enumerate(zip(rank_0_mlirs, rank_mlirs[r])):
        self.assertEqual(
            mlir_0,
            mlir_r,
            f"MLIR content mismatch for all_reduce module #{idx} between rank 0"
            f" and rank {r}!",
        )

  def test_spmd_safe_decorator(self):
    dump_dir = self.create_tempdir(name="xla_dump_fused_collective").full_path
    with mock.patch.dict(
        os.environ,
        {
            "XLA_FLAGS": _update_xla_dump_to(
                os.environ.get("XLA_FLAGS", ""), dump_dir
            )
        },
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
        os.environ,
        {
            "XLA_FLAGS": _update_xla_dump_to(
                os.environ.get("XLA_FLAGS", ""), dump_dir
            )
        },
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
        os.environ,
        {
            "XLA_FLAGS": _update_xla_dump_to(
                os.environ.get("XLA_FLAGS", ""), dump_dir
            )
        },
    ):
      distributed_utils.dist_run(
          nproc_per_node=self._world_size,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _test_wrapper, world_size=self._world_size
          ),
          test_fn=run_spmd_safe_decorator_backward_test,
      )
    self._check_fused_mlir(dump_dir, expected_count=2)

  def test_uncoordinated_op_stream_deadlock(self):
    dump_dir = self.create_tempdir(name="xla_dump_deadlock_test").full_path
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_uncoordinated_op_stream_test,
        dump_dir=dump_dir,
    )
    self._check_all_reduce_mlir_matching(dump_dir)

  def test_spmd_safe_dtensor_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_spmd_safe_dtensor_compile_test,
    )

  def test_spmd_safe_multi_dtensor_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_spmd_safe_multi_dtensor_compile_test,
    )


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
