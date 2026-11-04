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

"""End-to-end tests for distributed handshake protocol."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import torch
from torch import distributed as dist
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils


class _HandshakeE2ETestMixin(metaclass=parameterized.TestGeneratorMetaclass):
  """Common end-to-end handshake tests shared across handshake stages.

  `TORCH_TPU_SPMD_HANDSHAKE_STAGE` and `TORCH_TPU_INTERNAL_HANDSHAKE_PORT` are
  memoized once per process on first read, so each handshake stage runs in its
  own `TpuMultiProcContinuousTest` subclass with its own worker pool.
  """

  rank: int
  world_size: int
  _handshake_stage: str

  @classmethod
  def _spawn_processes(cls, world_size: int) -> None:
    with mock.patch.dict(
        os.environ,
        {
            "TORCH_TPU_SPMD_HANDSHAKE_STAGE": cls._handshake_stage,
            # We patch the port with an unused one to ensure the test doesn't
            # fail due to a port conflict with another test.
            "TORCH_TPU_INTERNAL_HANDSHAKE_PORT": str(
                portpicker.pick_unused_port()
            ),
        },
    ):
      super()._spawn_processes(world_size)  # pyrefly: ignore[missing-attribute]

  def test_handshake_no_rank_divergence(self) -> None:
    rank = self.rank
    debugs: list[tpu_compile.TpuCompileDebug] = []
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    def no_rank_divergence(x, y):
      z = torch.add(x, y)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        no_rank_divergence,
        backend="tpu",
        fullgraph=True,
        options={"serializable": False, "debug_callback": debugs.append},
    )
    _ = compiled_fn(x, y)
    compiled_executables = [e for x in debugs for e in x.compiled_executables]
    assert len(compiled_executables) > 0
    assert len(debugs[0].stablehlo_forward_text) == 1, (
        "Expected 1 entry in stablehlo_forward_text, got "
        f"{len(debugs[0].stablehlo_forward_text)}"
    )

  def test_handshake_rank_divergence(self) -> None:
    rank = self.rank
    debugs: list[tpu_compile.TpuCompileDebug] = []
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    def rank_divergence(x, y):
      z = torch.add(x, y)
      if rank == 0:
        z = torch.mul(z, 5.0)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        rank_divergence,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )
    _ = compiled_fn(x, y)
    compiled_executables = [e for x in debugs for e in x.compiled_executables]
    assert len(compiled_executables) > 0
    exec_artifact = compiled_executables[0]
    assert len(exec_artifact.mlir_texts) == 3, (
        f"Expected 3 executables on rank {rank}, got "
        f"{len(exec_artifact.mlir_texts)}"
    )
    assert "stablehlo.all_reduce" in exec_artifact.mlir_texts[1], (
        "Expected index 1 to be a skinny collective, got "
        f"{exec_artifact.mlir_texts[1]}"
    )

  def test_handshake_divergence_after_iteration(self) -> None:
    rank = self.rank
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    iteration = 0

    def divergence_after_iteration(x, y):
      z = torch.add(x, y)
      if iteration == 1:
        z = torch.mul(z, 5.0)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        divergence_after_iteration,
        backend="tpu",
        options={"serializable": False},
    )
    _ = compiled_fn(x, y)
    iteration += 1
    _ = compiled_fn(x, y)

  @parameterized.named_parameters(("rank_0", 0), ("rank_1", 1))
  def test_handshake_divergence_after_iteration_on_a_single_rank(
      self, divergent_rank: int
  ) -> None:
    if self._handshake_stage == "COMPILE_STAGE":
      self.skipTest(  # pyrefly: ignore[missing-attribute]
          "If a single rank enters a handshake due to rank specific"
          " recompilation it will timeout and throw an exception. The other"
          " ranks that haven't recompiled will get stuck on an enqueued"
          " collective. This results in a test timeout which happens because"
          " the subprocess exception won't get propagated as the process"
          " which raised the exception will wait on the global barrier."
      )

    rank = self.rank
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    iteration = 0

    def divergence_after_iteration(x, y):
      z = torch.add(x, y)
      if rank == divergent_rank and iteration == 1:
        z = torch.mul(z, 5.0)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        divergence_after_iteration,
        backend="tpu",
        options={"serializable": False},
    )
    _ = compiled_fn(x, y)
    iteration += 1
    _ = compiled_fn(x, y)

  @parameterized.named_parameters(("rank_0", 0), ("rank_1", 1))
  def test_handshake_uneven_executables_between_ranks(
      self, divergent_rank: int
  ) -> None:
    rank = self.rank
    debugs: list[tpu_compile.TpuCompileDebug] = []
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    def uneven_executables(x, y):
      z = torch.add(x, y)
      dist.all_reduce(z)
      if rank == divergent_rank:
        torch._dynamo.graph_break()
      z = torch.mul(z, 2.0)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        uneven_executables,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )
    _ = compiled_fn(x, y)
    compiled_executables = [e for x in debugs for e in x.compiled_executables]
    if rank == divergent_rank:
      assert len(compiled_executables) == 2, (
          f"Expected 2 executables on rank {rank}, got"
          f" {len(compiled_executables)}"
      )
    else:
      assert len(compiled_executables) == 1, (
          f"Expected 1 executable on rank {rank}, got"
          f" {len(compiled_executables)}"
      )

  @parameterized.named_parameters(("rank_0", 0), ("rank_1", 1))
  def test_handshake_last_frame_on_divergent_rank_is_not_collective_test(
      self, divergent_rank: int
  ) -> None:
    rank = self.rank
    debugs: list[tpu_compile.TpuCompileDebug] = []
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    def last_frame_on_divergent_rank_is_not_collective_test(x, y):
      z = torch.add(x, y)
      dist.all_reduce(z)
      if rank == divergent_rank:
        return z
      else:
        # We break so that the last frame is not a collective, so it doesn't
        # participate in the handshake.
        torch._dynamo.graph_break()
      z = torch.mul(z, 2.0)

      return z

    def spmd_function(x, y):
      z = torch.sub(x, y)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        last_frame_on_divergent_rank_is_not_collective_test,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )
    _ = compiled_fn(x, y)

    compiled_spmd_fn = torch.compile(
        spmd_function,
        backend="tpu",
        options={"serializable": False, "debug_callback": debugs.append},
    )
    _ = compiled_spmd_fn(x, y)

    compiled_executables = [e for x in debugs for e in x.compiled_executables]
    if rank == divergent_rank:
      # 1 for spmd_function, 1 for
      # last_frame_on_divergent_rank_is_not_collective_test
      assert len(compiled_executables) == 2, (
          f"Expected 2 executables on rank {rank}, got"
          f" {len(compiled_executables)}"
      )
      spmd_function_index = 1
    else:
      # Non divergent ranks will have an extra executable due to a dynamo break
      # in last_frame_on_divergent_rank_is_not_collective_test
      assert len(compiled_executables) == 3, (
          f"Expected 3 executables on rank {rank}, got"
          f" {len(compiled_executables)}"
      )
      spmd_function_index = 2

    # First executable should have no internal splits because of the handshake.
    assert len(debugs[0].stablehlo_forward_text) == 1, (
        f"Expected no internal splits on rank {rank}, got "
        f"{len(debugs[0].stablehlo_forward_text)}"
    )
    # SPMD function should have no internal splits because of the handshake.
    assert len(debugs[spmd_function_index].stablehlo_forward_text) == 1, (
        f"Expected no internal splits for SPMD function on rank {rank}, got "
        f"{len(debugs[spmd_function_index].stablehlo_forward_text)}"
    )

  @parameterized.named_parameters(("rank_0", 0), ("rank_1", 1))
  def test_handshake_separate_vs_composed_compiled_functions(
      self, divergent_rank: int
  ) -> None:
    rank = self.rank
    debugs: list[tpu_compile.TpuCompileDebug] = []
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    def foo1(x, y):
      z = torch.add(x, y)
      dist.all_reduce(z)
      return z

    def foo2(x, y):
      z = torch.mul(x, y)
      dist.all_reduce(z)
      return z

    def foo(x, y):
      z = foo1(x, y)
      return foo2(z, y)

    if rank == divergent_rank:
      compiled_foo1 = torch.compile(
          foo1,
          backend="tpu",
          options={"serializable": False, "debug_callback": debugs.append},
      )
      res1 = compiled_foo1(x, y)
      compiled_foo2 = torch.compile(
          foo2,
          backend="tpu",
          options={"serializable": False, "debug_callback": debugs.append},
      )
      res = compiled_foo2(res1, y)
    else:
      compiled_foo = torch.compile(
          foo,
          backend="tpu",
          options={"serializable": False, "debug_callback": debugs.append},
      )
      res = compiled_foo(x, y)

    world_size = self.world_size
    foo1_expected = sum((r + 1) + 2 for r in range(world_size))
    foo2_expected = (foo1_expected * 2) * world_size
    utils.assert_close(res, torch.ones(4, device="tpu") * foo2_expected)

    compiled_executables = [e for x in debugs for e in x.compiled_executables]
    if rank == divergent_rank:
      assert len(compiled_executables) == 2, (
          f"Expected 2 executables on rank {rank}, got"
          f" {len(compiled_executables)}"
      )
    else:
      assert len(compiled_executables) == 1, (
          f"Expected 1 executable on rank {rank}, got"
          f" {len(compiled_executables)}"
      )

  def test_handshake_no_rank_divergence_async_compile(self) -> None:
    rank = self.rank
    world_size = self.world_size
    debugs: list[tpu_compile.TpuCompileDebug] = []
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    def no_rank_divergence(x, y):
      z = torch.add(x, y)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        no_rank_divergence,
        backend="tpu",
        fullgraph=True,
        options={
            "async_compile": True,
            "serializable": False,
            "debug_callback": debugs.append,
        },
    )
    try:
      _ = compiled_fn(x, y)
      raise AssertionError(
          f"Expected {tpu_compile.AsyncCompilationSubmitted.__name__} to be"
          " raised"
      )
    except tpu_compile.AsyncCompilationSubmitted as cm:
      art = cm.artifact
      assert art is not None

    res = compiled_fn(x, y)
    expected = sum((r + 1) + 2 for r in range(world_size))
    utils.assert_close(res, torch.ones(4, device="tpu") * expected)
    compiled_executables = [e for x in debugs for e in x.compiled_executables]
    assert len(compiled_executables) > 0
    assert len(debugs[0].stablehlo_forward_text) == 1, (
        "Expected 1 entry in stablehlo_forward_text, got "
        f"{len(debugs[0].stablehlo_forward_text)}"
    )

  def test_handshake_divergence_after_iteration_async_compile(self) -> None:
    rank = self.rank
    world_size = self.world_size
    x = torch.ones(4, device="tpu") * (rank + 1)
    y = torch.ones(4, device="tpu") * 2

    iteration = 0

    def divergence_after_iteration(x, y):
      z = torch.add(x, y)
      if iteration == 1:
        z = torch.mul(z, 5.0)
      dist.all_reduce(z)
      return z

    compiled_fn = torch.compile(
        divergence_after_iteration,
        backend="tpu",
        options={"async_compile": True, "serializable": False},
    )
    try:
      _ = compiled_fn(x, y)
      raise AssertionError(
          f"Expected {tpu_compile.AsyncCompilationSubmitted.__name__} to be"
          " raised"
      )
    except tpu_compile.AsyncCompilationSubmitted as cm:
      art = cm.artifact
      assert art is not None
    res0 = compiled_fn(x, y)
    expected_iter0 = sum((r + 1) + 2 for r in range(world_size))
    utils.assert_close(res0, torch.ones(4, device="tpu") * expected_iter0)

    iteration += 1
    try:
      _ = compiled_fn(x, y)
      raise AssertionError(
          f"Expected {tpu_compile.AsyncCompilationSubmitted.__name__} to be"
          " raised"
      )
    except tpu_compile.AsyncCompilationSubmitted as cm:
      art = cm.artifact
      assert art is not None
    res1 = compiled_fn(x, y)
    expected_iter1 = expected_iter0 * 5.0
    utils.assert_close(res1, torch.ones(4, device="tpu") * expected_iter1)


class HandshakeCompileStageE2ETest(
    _HandshakeE2ETestMixin,
    continuous_test_utils.TpuMultiProcContinuousTest,
):
  _handshake_stage = "COMPILE_STAGE"


class HandshakeDispatchStageE2ETest(
    _HandshakeE2ETestMixin,
    continuous_test_utils.TpuMultiProcContinuousTest,
):
  _handshake_stage = "DISPATCH_STAGE"


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
