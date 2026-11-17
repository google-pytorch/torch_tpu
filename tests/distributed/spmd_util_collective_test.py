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
import shutil
from typing import Any, TypeAlias
from unittest import mock

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal import execution_mode
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils

EagerMode: TypeAlias = execution_mode.EagerMode

_DUMP_DIR = os.path.join(absltest.get_default_test_tmpdir(), "xla_dump")


def _update_xla_dump_to(xla_flags: str, dump_dir: str) -> str:
  pattern = r"--xla_dump_to=\S+"
  new_flag = f"--xla_dump_to={dump_dir}"
  if re.search(pattern, xla_flags):
    return re.sub(pattern, new_flag, xla_flags)
  return f"{xla_flags} {new_flag}".strip()


def run_spmd_safe_decorator_test(compile_test: bool = False):

  @torch.tpu.distributed.spmd_safe
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
  torch.accelerator.synchronize()
  del res


def run_spmd_safe_async_work_wait_test() -> None:
  """Runs an op after wait() on the Work of a fused async collective."""

  @torch.tpu.distributed.spmd_safe
  def eager_fused_async_all_reduce(x: torch.Tensor) -> torch.Tensor:
    y = x + 1
    work = dist.all_reduce(y, async_op=True)
    work.wait()
    return y * 5

  x = torch.ones((2, 2), device="cpu").to("tpu")
  with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
    res = eager_fused_async_all_reduce(x)
  torch.accelerator.synchronize()
  del res


def run_deferred_async_collective_test() -> None:
  """Issues a deferred async collective while a temporary is still alive."""
  x = torch.ones((2, 2), device="cpu").to("tpu")
  with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
    y = x + 1
    unused = y * 5
    work = dist.all_reduce(y, async_op=True)
    del unused
  torch.accelerator.synchronize()
  del work


def run_spmd_safe_decorator_backward_test():

  @torch.tpu.distributed.spmd_safe
  def eager_fused_all_reduce(x):
    y = x * 5
    dist.all_reduce(y)
    return y

  x = torch.ones((2, 2), device="tpu", requires_grad=True)
  compiled_fn = torch.compile(eager_fused_all_reduce)
  res = compiled_fn(x)
  loss = res.sum()
  loss.backward()
  torch.accelerator.synchronize()


def run_uncoordinated_op_stream_test():
  """Repro test for interaction of SPMD safe and repeated ops heuristic."""
  rank = dist.get_rank()
  x = torch.ones((2, 2), device="cpu").to("tpu")
  torch.accelerator.synchronize()

  with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
    for i in range(10):
      if rank == 0 and i % 2 == 0:
        x = x + 1

      @torch.tpu.distributed.spmd_safe
      def foo(x):
        for _ in range(10):
          x = x + 1
        dist.all_reduce(x)
        return x

      x = foo(x)

    torch.accelerator.synchronize()


def run_spmd_safe_dtensor_compile_test():
  world_size = dist.get_world_size()
  mesh = torch.distributed.tensor.DeviceMesh("tpu", torch.arange(world_size))

  @torch.tpu.distributed.spmd_safe
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

  @torch.tpu.distributed.spmd_safe
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


class SpmdSafeDecoratorTest(continuous_test_utils.TpuMultiProcContinuousTest):

  # Using world_size = 4 has flakiness on 8-chip TPU v5e CI runners:
  # initializing only 4 chips leaves the rest of the physical interconnect ring
  # idle, causing process group init to intermittently hang for ~20s and fail
  # with SIGTERM (ProcessExitedException). Setting this to 8 ensures all chips
  # on the host participate, allowing the physical ring to close and sync
  # deterministically.
  world_size = 8

  @classmethod
  def _spawn_processes(cls, world_size: int) -> None:
    # Sets per-rank dump dirs so workers don't collide when
    # verifying MLIR fusions.
    xla_flags = _update_xla_dump_to(
        os.environ.get("XLA_FLAGS", ""), f"{_DUMP_DIR}/rank_${{RANK}}"
    )
    with mock.patch.dict(os.environ, {"XLA_FLAGS": xla_flags}):
      super()._spawn_processes(world_size)

  @classmethod
  def _run_test_given_id(cls, test_id: str, **kwargs: Any) -> None:
    if dist.is_initialized():
      # Warm up `dist.barrier()` so its 1-element int64 all_reduce is cached in
      # the C++ compilation cache before clearing `_DUMP_DIR`. Otherwise, when
      # non-zero ranks finish a test and enter `finally: dist.barrier()` while
      # rank 0 is still inspecting `_DUMP_DIR`, `dist.barrier()` compiles and
      # dumps an extra `all_reduce` MLIR module on those ranks.
      torch.accelerator.synchronize()
      dist.barrier()
      torch.accelerator.synchronize()
    # Clear compiler MLIR dumps from previous tests on this worker rank.
    shutil.rmtree(f"{_DUMP_DIR}/rank_{cls.rank}", ignore_errors=True)
    # Flush Dynamo cache so subsequent tests trigger fresh compilation passes.
    torch._dynamo.reset()
    super()._run_test_given_id(test_id, **kwargs)

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
    for r in range(self.world_size):
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

    for r in range(1, self.world_size):
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
    run_spmd_safe_decorator_test()
    if self.rank == 0:
      self._check_fused_mlir(_DUMP_DIR)

  def test_spmd_safe_async_work_wait_stays_fused(self):
    run_spmd_safe_async_work_wait_test()
    if self.rank == 0:
      self._check_fused_mlir(_DUMP_DIR)

  def test_deferred_async_collective_is_not_flushed_at_issue(self):
    run_deferred_async_collective_test()
    # A flush at issue would have computed the temporary that was alive then.
    dumps = glob.glob(f"{_DUMP_DIR}/rank_{self.rank}/**/*.mlir", recursive=True)
    for dump in dumps:
      with open(dump, "r") as f:
        self.assertNotIn("stablehlo.multiply", f.read(), dump)

  def test_spmd_safe_decorator_compile(self):
    run_spmd_safe_decorator_test(compile_test=True)
    if self.rank == 0:
      self._check_fused_mlir(_DUMP_DIR)

  def test_spmd_safe_decorator_compile_backward(self):
    run_spmd_safe_decorator_backward_test()
    if self.rank == 0:
      self._check_fused_mlir(_DUMP_DIR, expected_count=2)

  def test_uncoordinated_op_stream_deadlock(self):
    run_uncoordinated_op_stream_test()
    if self.rank == 0:
      self._check_all_reduce_mlir_matching(_DUMP_DIR)


class SpmdSafeDTensorTest(continuous_test_utils.TpuMultiProcContinuousTest):

  world_size = 8

  def test_spmd_safe_dtensor_compile(self):
    run_spmd_safe_dtensor_compile_test()

  def test_spmd_safe_multi_dtensor_compile(self):
    run_spmd_safe_multi_dtensor_compile_test()


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
