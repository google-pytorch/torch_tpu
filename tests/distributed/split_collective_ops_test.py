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
import dataclasses
import os
import pickle
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.distributed import collective_ops
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

CompiledArtifact = torch_tpu_compiled_executable.CompiledArtifact
TpuBackend = _backend.TpuBackend
_SplitCompiledExecutable = split_compiler._SplitCompiledExecutable
_COLLECTIVE_OPS = collective_ops.COLLECTIVE_OPS
SplitCompiler = split_compiler.SplitCompiler
TorchTpuCompiledExecutable = (
    torch_tpu_compiled_executable.TorchTpuCompiledExecutable
)


# pylint: disable=protected-access
@dataclasses.dataclass(frozen=True)
class CollectiveOpSpec:
  op: torch._ops.OpOverloadPacket
  extra_args_fn: Callable[[int], tuple[Any, ...]]


@dataclasses.dataclass(frozen=True)
class OutCollectiveOpSpec:
  op: torch._ops.OpOverloadPacket
  extra_args_fn: Callable[[int], tuple[Any, ...]]
  input_shape_fn: Callable[[int], tuple[int, ...]]
  out_shape_fn: Callable[[int], tuple[int, ...]]


_SINGLE_TENSOR_COLLECTIVES = (
    CollectiveOpSpec(
        torch.ops._c10d_functional.all_reduce,
        extra_args_fn=lambda ws: ("sum",),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.all_reduce_,
        extra_args_fn=lambda ws: ("sum",),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.reduce_scatter_tensor,
        extra_args_fn=lambda ws: ("sum", ws),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.all_gather_into_tensor,
        extra_args_fn=lambda ws: (ws,),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.broadcast,
        extra_args_fn=lambda ws: (0,),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.broadcast_,
        extra_args_fn=lambda ws: (0,),
    ),
)
_COALESCED_COLLECTIVES = (
    CollectiveOpSpec(
        torch.ops._c10d_functional.all_reduce_coalesced,
        extra_args_fn=lambda ws: ("sum",),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.all_reduce_coalesced_,
        extra_args_fn=lambda ws: ("sum",),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.reduce_scatter_tensor_coalesced,
        extra_args_fn=lambda ws: ("sum", ws),
    ),
    CollectiveOpSpec(
        torch.ops._c10d_functional.all_gather_into_tensor_coalesced,
        extra_args_fn=lambda ws: (ws,),
    ),
)
_OUT_COLLECTIVES = (
    OutCollectiveOpSpec(
        torch.ops._c10d_functional.reduce_scatter_tensor_out,
        extra_args_fn=lambda ws: ("sum", ws),
        input_shape_fn=lambda ws: (ws * 2, 2),
        out_shape_fn=lambda ws: (2, 2),
    ),
    OutCollectiveOpSpec(
        torch.ops._c10d_functional.all_gather_into_tensor_out,
        extra_args_fn=lambda ws: (ws,),
        input_shape_fn=lambda ws: (2, 2),
        out_shape_fn=lambda ws: (ws * 2, 2),
    ),
)
_ALL_TO_ALL_COLLECTIVES = (torch.ops._c10d_functional.all_to_all_single,)
# pylint: enable=protected-access


def stablehlo_instructions_count(mlir: str) -> int:
  # Subtract occurrences of '#stablehlo.' (used for attributes like
  # '#stablehlo.channel_handle') to get the exact instruction count.
  return mlir.count("stablehlo.") - mlir.count("#stablehlo.")


def _test_wrapper(test_fn, *args, **kwargs):
  dist.init_process_group(backend="tpu_dist")
  try:
    test_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def run_compile_all_reduce_and_serdes_test():
  backend = TpuBackend(debug=True)

  def f(x):
    y = x * 2
    y = y + 5
    dist.all_reduce(y)
    z = y + 3
    return z

  compiled_f = torch.compile(f, backend=backend)

  x = torch.ones((2, 2), device="tpu")
  res = compiled_f(x)

  # Expected: (1.0 * 2.0 + 5.0) * world_size + 3.0 = 7.0 * world_size + 3.0
  expected = torch.full((2, 2), 59.0, device="cpu")
  utils.assert_close(res.cpu(), expected)

  # Verify splitting in backend.
  assert (
      len(backend._compiled_executables) == 1
  ), f"Expected 1 compiled executable, got {len(backend._compiled_executables)}"

  wrapper = backend._compiled_executables[0]
  assert isinstance(
      wrapper, _SplitCompiledExecutable
  ), f"Expected _SplitCompiledExecutable, got {type(wrapper)}"
  split_gm = wrapper._split_gm

  # The split module partitions the graph into 3 submodules:
  # 1. Compute before collective (x * 2) + 5
  # 2. Collective (all_reduce + wait_tensor)
  # 3. Compute after collective (y + 3)
  children = list(split_gm.children())
  assert (
      len(children) == 3
  ), f"Expected exactly 3 submodules, got {len(children)}"

  for i, child in enumerate(children):
    assert isinstance(child.submod, TorchTpuCompiledExecutable), (
        f"Expected submodule {i} to be TorchTpuCompiledExecutable, got"
        f" {type(child.submod)}"
    )

  all_reduce_graph = children[1]  # We expect the second child to be all_reduce

  assert "stablehlo.all_reduce" in all_reduce_graph.submod.mlir_text, (
      "Expected all_reduce submodule to contain stablehlo.all_reduce, got"
      f" {all_reduce_graph.submod.mlir_text}"
  )

  # We expect the all_reduce submodule to contain 3 StableHLO instructions:
  # 1. stablehlo.all_reduce
  # 2. stablehlo.add (reduction region)
  # 3. stablehlo.return (reduction region)
  assert stablehlo_instructions_count(all_reduce_graph.submod.mlir_text) == 3, (
      "Expected all_reduce submodule to contain 3 StableHLO instructions,"
      f" got {stablehlo_instructions_count(all_reduce_graph.submod.mlir_text)}"
  )

  # Test pickle/unpickle of the split executable.

  serialized_wrapper = pickle.dumps(wrapper)
  deserialized_wrapper = pickle.loads(serialized_wrapper)

  # Run the deserialized executable and verify correctness.
  res_deserialized = deserialized_wrapper(x)
  if isinstance(res_deserialized, (tuple, list)) and len(res_deserialized) == 1:
    res_deserialized = res_deserialized[0]
  utils.assert_close(res_deserialized.cpu(), expected)


def run_compile_no_splits_when_env_zero_test():
  backend = TpuBackend(debug=True)

  def f(x):
    y = x * 2
    y = y + 5
    dist.all_reduce(y)
    z = y + 3
    return z

  compiled_f = torch.compile(f, backend=backend)

  x = torch.ones((2, 2), device="tpu")
  res = compiled_f(x)

  # Expected: (1.0 * 2.0 + 5.0) * world_size + 3.0 = 7.0 * world_size + 3.0
  expected = torch.full((2, 2), 59.0, device="cpu")
  utils.assert_close(res.cpu(), expected)

  assert (
      len(backend._compiled_executables) == 1
  ), f"Expected 1 compiled executable, got {len(backend._compiled_executables)}"

  wrapper = backend._compiled_executables[0]
  assert isinstance(
      wrapper, _SplitCompiledExecutable
  ), f"Expected _SplitCompiledExecutable, got {type(wrapper)}"
  assert len(wrapper.compiled_executables) == 1


def run_compile_two_collectives_test():
  """Test that two collectives are compiled separately."""

  backend = TpuBackend(debug=True)

  def f(x, y):
    dist.all_reduce(x)
    dist.all_reduce(y)
    return x + y + 3

  compiled_f = torch.compile(f, backend=backend)

  x = torch.ones((2, 2), device="tpu")
  y = torch.ones((2, 2), device="tpu")
  res = compiled_f(x, y)

  # Expected: (1 * world_size) + (1 * world_size) + 3 = 2 * world_size + 3
  expected = torch.full((2, 2), 19.0, device="cpu")
  utils.assert_close(res.cpu(), expected)

  assert len(backend._compiled_executables) == 1
  wrapper = backend._compiled_executables[0]
  assert isinstance(wrapper, _SplitCompiledExecutable)

  split_gm = wrapper._split_gm

  all_reduce_submods_count = 0

  for child in split_gm.children():
    submod = child.submod
    assert isinstance(submod, TorchTpuCompiledExecutable)
    mlir = submod.mlir_text

    shlo_count = stablehlo_instructions_count(mlir)
    ar_count = mlir.count("stablehlo.all_reduce")

    assert (
        ar_count <= 1
    ), f"Expected at most 1 all_reduce in submod, got {ar_count}"

    if ar_count == 1:
      # We expect the all_reduce submodule to contain 3 StableHLO instructions:
      # 1. stablehlo.all_reduce
      # 2. stablehlo.add (reduction region)
      # 3. stablehlo.return (reduction region)
      assert shlo_count == 3, (
          "Expected exactly 3 StableHLO instructions for all_reduce submod, got"
          f" {shlo_count}"
      )
      assert mlir.count("stablehlo.add") == 1
      assert mlir.count("stablehlo.return") == 1
      all_reduce_submods_count += 1

  assert (
      all_reduce_submods_count == 2
  ), f"Expected exactly 2 all_reduce submods, got {all_reduce_submods_count}"


class CollectiveModule(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.parameter = torch.nn.Parameter(torch.ones(2, 2))

  def forward(self, x):
    y = x + self.parameter
    dist.all_reduce(y)
    z = y * 2
    return z


def run_torch_compile_fullgraph_no_break_test():
  backend = TpuBackend(debug=True)
  with torch.device("tpu"):
    model = CollectiveModule()
  compiled_model = torch.compile(model, backend=backend, fullgraph=True)

  x = torch.ones((2, 2), device="tpu")
  res = compiled_model(x)

  # Expected: ((1.0 + 1.0) * world_size) * 2.0 = 4.0 * world_size
  expected = torch.full((2, 2), 32.0, device="cpu")
  utils.assert_close(res.cpu(), expected)

  # Verify splitting in backend.
  assert (
      len(backend._compiled_executables) == 1
  ), f"Expected 1 compiled executable, got {len(backend._compiled_executables)}"

  wrapper = backend._compiled_executables[0]
  assert isinstance(
      wrapper, _SplitCompiledExecutable
  ), f"Expected _SplitCompiledExecutable, got {type(wrapper)}"
  split_gm = wrapper._split_gm

  # The split module partitions the graph into 3 submodules:
  # 1. Compute before collective (x + parameter)
  # 2. Collective (all_reduce + wait_tensor)
  # 3. Compute after collective (y * 2)
  children = list(split_gm.children())
  assert (
      len(children) == 3
  ), f"Expected exactly 3 submodules, got {len(children)}"


def _check_split_submod(wrapper, op):
  assert isinstance(
      wrapper, _SplitCompiledExecutable
  ), f"Expected _SplitCompiledExecutable, got {type(wrapper)}"
  split_gm = wrapper._split_gm
  children = list(split_gm.children())
  assert (
      len(children) == 3
  ), f"Expected exactly 3 submodules for {op}, got {len(children)}"
  collective_submod = children[1].submod
  assert isinstance(collective_submod, _DummyCompiledExecutable)
  node_ops = [
      getattr(n.target, "overloadpacket", n.target)
      for n in collective_submod.gm.graph.nodes
      if n.op == "call_function"
  ]
  assert (
      op in node_ops
  ), f"Expected {op} in collective submodule, got {node_ops}"
  assert (
      torch.ops._c10d_functional.wait_tensor in node_ops
  ), f"Expected wait_tensor in collective submodule, got {node_ops}"


def _check_single_tensor_op(spec: CollectiveOpSpec):
  world_size = dist.get_world_size()
  group_name = dist._get_process_group_name(dist.group.WORLD)
  extra_args = spec.extra_args_fn(world_size)

  def f(x):
    x = x * 2.0
    work = spec.op(x, *extra_args, group_name)
    return torch.ops._c10d_functional.wait_tensor(work) + 1.0

  x = torch.ones((world_size * 2, 2), device="tpu")
  graph_module = torch.fx.symbolic_trace(f)
  split_compiler_instance = SplitCompiler(DummyBaseCompiler())
  _check_split_submod(split_compiler_instance(graph_module, (x,)), spec.op)


def _check_coalesced_op(spec: CollectiveOpSpec):
  world_size = dist.get_world_size()
  group_name = dist._get_process_group_name(dist.group.WORLD)
  extra_args = spec.extra_args_fn(world_size)

  def f(x, y):
    x = x * 2.0
    y = y * 3.0
    coalesced = spec.op([x, y], *extra_args, group_name)
    x_out = torch.ops._c10d_functional.wait_tensor(coalesced[0])
    y_out = torch.ops._c10d_functional.wait_tensor(coalesced[1])
    return x_out + y_out + 1.0

  x = torch.ones((world_size * 2, 2), device="tpu")
  y = torch.ones((world_size * 2, 2), device="tpu")
  graph_module = torch.fx.symbolic_trace(f)
  split_compiler_instance = SplitCompiler(DummyBaseCompiler())
  _check_split_submod(split_compiler_instance(graph_module, (x, y)), spec.op)


def _check_out_op(spec: OutCollectiveOpSpec):
  world_size = dist.get_world_size()
  group_name = dist._get_process_group_name(dist.group.WORLD)
  extra_args = spec.extra_args_fn(world_size)

  def f(x, out):
    x = x * 2.0
    work = spec.op(x, *extra_args, group_name, out=out)
    return torch.ops._c10d_functional.wait_tensor(work) + 1.0

  x = torch.ones(spec.input_shape_fn(world_size), device="tpu")
  out = torch.empty(spec.out_shape_fn(world_size), device="tpu")
  graph_module = torch.fx.symbolic_trace(f)
  split_compiler_instance = SplitCompiler(DummyBaseCompiler())
  _check_split_submod(split_compiler_instance(graph_module, (x, out)), spec.op)


def _check_all_to_all_op(op: torch._ops.OpOverloadPacket):
  world_size = dist.get_world_size()
  group_name = dist._get_process_group_name(dist.group.WORLD)

  def f(x):
    x = x * 2.0
    splits = [1] * world_size
    work = op(x, splits, splits, group_name)
    return torch.ops._c10d_functional.wait_tensor(work) + 1.0

  x = torch.ones((world_size, 2), device="tpu")
  graph_module = torch.fx.symbolic_trace(f)
  split_compiler_instance = SplitCompiler(DummyBaseCompiler())
  _check_split_submod(split_compiler_instance(graph_module, (x,)), op)


def run_single_tensor_collectives_test():
  # We run a loop over _SINGLE_TENSOR_COLLECTIVES because parametrizing the test
  # fails dues to ops not being pickleable.
  for spec in _SINGLE_TENSOR_COLLECTIVES:
    _check_single_tensor_op(spec)


def run_coalesced_collectives_test():
  # See comment above for why we run a loop instead of parametrizing.
  for spec in _COALESCED_COLLECTIVES:
    _check_coalesced_op(spec)


def run_out_collectives_test():
  # See comment above for why we run a loop instead of parametrizing.
  for spec in _OUT_COLLECTIVES:
    _check_out_op(spec)


def run_all_to_all_collectives_test():
  # See comment above for why we run a loop instead of parametrizing.
  for op in _ALL_TO_ALL_COLLECTIVES:
    _check_all_to_all_op(op)


def run_rank_variable_dead_collective_test():
  backend = TpuBackend(debug=True)
  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x):
    y = x.clone()
    dist.all_reduce(y, dist.ReduceOp.SUM)
    if rank == 0:
      return y
    return x

  x = torch.tensor([1.0], device="tpu")
  compiled_f = torch.compile(func, backend=backend)
  actual = compiled_f(x)
  expected = (
      torch.tensor([float(world_size)]) if rank == 0 else torch.tensor([1.0])
  )
  utils.assert_close(actual.cpu(), expected)

  # Verify splitting in backend.
  assert (
      len(backend._compiled_executables) == 1
  ), f"Expected 1 compiled executable, got {len(backend._compiled_executables)}"

  wrapper = backend._compiled_executables[0]
  assert isinstance(
      wrapper, _SplitCompiledExecutable
  ), f"Expected _SplitCompiledExecutable, got {type(wrapper)}"
  split_gm = wrapper._split_gm

  children = list(split_gm.children())
  if rank == 0:
    # The non-dead rank should have 3 submodules:
    # 1. Clone before collective
    # 2. Collective (all_reduce + wait_tensor)
    # 3. Clone after collective
    assert (
        len(children) == 3
    ), f"Expected exactly 3 submodules for rank 0, got {len(children)}"
  else:
    # The dead ranks should have 2 submodules:
    # 1. Clone before collective
    # 2. Collective (all_reduce + wait_tensor)
    # without the third submodule (clone after collective) as it is dead.
    assert (
        len(children) == 2
    ), f"Expected exactly 2 submodules for rank {rank}, got {len(children)}"

  for i, child in enumerate(children):
    assert isinstance(child.submod, TorchTpuCompiledExecutable), (
        f"Expected submodule {i} to be TorchTpuCompiledExecutable, got"
        f" {type(child.submod)}"
    )

  all_reduce_graph = children[1]  # We expect the second child to be all_reduce

  assert "stablehlo.all_reduce" in all_reduce_graph.submod.mlir_text, (
      "Expected all_reduce submodule to contain stablehlo.all_reduce, got"
      f" {all_reduce_graph.submod.mlir_text}"
  )

  # We expect the all_reduce submodule to contain 3 StableHLO instructions:
  # 1. stablehlo.all_reduce
  # 2. stablehlo.add (reduction region)
  # 3. stablehlo.return (reduction region)
  assert stablehlo_instructions_count(all_reduce_graph.submod.mlir_text) == 3, (
      "Expected all_reduce submodule to contain 3 StableHLO instructions,"
      f" got {stablehlo_instructions_count(all_reduce_graph.submod.mlir_text)}"
  )


class _DummyCompiledExecutable(CompiledArtifact):

  def __init__(self, gm):
    self.gm = gm
    self._updates_default_generator_state = False

  def updates_default_generator_state(self) -> bool:
    return self._updates_default_generator_state

  def __reduce__(self):
    def _unpickle_dummy_compiled_executable(gm):
      return _DummyCompiledExecutable(gm)

    return (_unpickle_dummy_compiled_executable, (self.gm,))

  def __call__(self, *args, **kwargs):
    return self.gm(*args, **kwargs)


class DummyBaseCompiler(compiler.Compiler):

  def __init__(self):
    super().__init__(debug=True)
    self.compiler_fn = lambda gm, args: gm

  def __call__(self, gm, args, is_fwd=True, module_name=None):
    return _DummyCompiledExecutable(gm)


class SplitCollectiveOpsTest(seed_test_utils.MultiProcessRepeatableTest):
  # Using _world_size = 4 has flakiness on 8-chip TPU v5e CI runners:
  # initializing only 4 chips leaves the rest of the physical interconnect ring
  # idle, causing process group init to intermittently hang for ~20s and fail
  # with SIGTERM (ProcessExitedException). Setting this to 8 ensures all chips
  # on the host participate, allowing the physical ring to close and sync
  # deterministically.
  _world_size = 8

  def test_compile_all_reduce_and_serdes(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_compile_all_reduce_and_serdes_test,
    )

  def test_compile_no_splits_when_env_zero(self):
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "0"}
    ):
      distributed_utils.dist_run(
          nproc_per_node=self._world_size,
          fn=singlehost_wrapper.tpu_env_wrapper(
              _test_wrapper, world_size=self._world_size
          ),
          test_fn=run_compile_no_splits_when_env_zero_test,
      )

  def test_compile_two_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_compile_two_collectives_test,
    )

  def test_torch_compile_fullgraph_no_break(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_torch_compile_fullgraph_no_break_test,
    )

  def test_compile_single_tensor_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_single_tensor_collectives_test,
    )

  def test_compile_coalesced_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_coalesced_collectives_test,
    )

  def test_compile_out_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_out_collectives_test,
    )

  def test_compile_all_to_all_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_all_to_all_collectives_test,
    )

  def test_collective_ops_tuple_equality(self):
    all_grouped_ops = (
        tuple(spec.op for spec in _SINGLE_TENSOR_COLLECTIVES)
        + tuple(spec.op for spec in _COALESCED_COLLECTIVES)
        + tuple(spec.op for spec in _OUT_COLLECTIVES)
        + _ALL_TO_ALL_COLLECTIVES
    )
    self.assertEqual(set(all_grouped_ops), set(_COLLECTIVE_OPS))
    self.assertLen(all_grouped_ops, len(_COLLECTIVE_OPS))

  def test_rank_variable_dead_collective(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_rank_variable_dead_collective_test,
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
