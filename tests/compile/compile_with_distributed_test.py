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

"""Tests PyTorch distributed ops with torch.compile()."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch.distributed import tensor
import torch.multiprocessing as mp
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal.compile import split_compiler
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.device import _device_module as tpu_device
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

TorchTpuCompiledExecutable = (
    torch_tpu_compiled_executable.TorchTpuCompiledExecutable
)
_WrapperModule = split_compiler._WrapperModule
_SplitCompiledExecutable = split_compiler._SplitCompiledExecutable


def get_all_compiled_executables(execs) -> list[TorchTpuCompiledExecutable]:
  flat_execs = []
  for exe in execs:
    if isinstance(exe, TorchTpuCompiledExecutable):
      flat_execs.append(exe)
    elif isinstance(exe, _SplitCompiledExecutable):
      for child in exe._split_gm.children():
        if isinstance(child, _WrapperModule) and isinstance(
            child.submod, TorchTpuCompiledExecutable
        ):
          flat_execs.append(child.submod)
  return flat_execs


def compile_and_assert_outputs(func, inputs, expected_outputs=None):
  """Compiles a function, runs it, and returns compiled executables."""
  backend = tt_compile.TpuBackend(debug=True)
  compiled = torch.compile(func, backend=backend)
  output_compiled = compiled(*inputs)

  if expected_outputs is not None:
    if not isinstance(output_compiled, (list, tuple)):
      output_compiled = (output_compiled,)
    for actual, expected in zip(output_compiled, expected_outputs):
      utils.assert_close(actual.to("cpu"), expected.to("cpu"))

  return get_all_compiled_executables(backend._compiled_executables)


def run_all_reduce_with_torch_compile() -> None:
  """Tests all-reduce functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])

  def func(x):
    x = x + torch.zeros_like(x)
    # TODO: b/478944489 - Force a graph break to avoid the crash in the
    # distributed runtime in OSS.
    torch.distributed.all_reduce(x, torch.distributed.ReduceOp.SUM)
    x = x + torch.abs(x)
    x = x + torch.ones_like(x)
    return x

  input_tpu = torch.tensor(
      [0.0, 1.0, float(rank), float(rank**2)], device="tpu"
  )
  expected = torch.tensor([1.0, 17.0, 57.0, 281.0])
  execs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(expected,)
  )
  assert len(execs) == 3, f"Expected 3 graphs, got {len(execs)}"

  # debug mode enabled so expect graphs to be set and in plaintext
  assert "torch.ops.aten.abs" not in execs[0].graph_module_debug_str
  assert "stablehlo.abs" not in execs[0].mlir_text

  assert (
      "torch.ops._c10d_functional.all_reduce" in execs[1].graph_module_debug_str
  )

  assert "stablehlo.all_reduce" in execs[1].mlir_text

  assert "torch.ops.aten.abs" in execs[2].graph_module_debug_str
  assert "stablehlo.abs" in execs[2].mlir_text

  assert len({e.graph_module_debug_str for e in execs}) == 3
  assert len({e.mlir_text for e in execs}) == 3


def run_all_gather_into_tensor_with_torch_compile() -> None:
  """Tests all-gather-into-tensor functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x):
    x = x + torch.zeros_like(x)
    output = torch.empty(
        x.shape[0] * world_size, dtype=x.dtype, device=x.device
    )
    torch.distributed.all_gather_into_tensor(output, x)
    output = output + torch.abs(output)
    output = output + torch.ones_like(output)
    return output

  expected = torch.tensor([float(2 * i + 1) for i in range(world_size)])
  execs = compile_and_assert_outputs(
      func,
      inputs=(torch.tensor([float(rank)], device="tpu"),),
      expected_outputs=(expected,),
  )
  assert len(execs) == 3, f"Expected 3 graphs, got {len(execs)}"


def run_all_gather_with_torch_compile() -> None:
  """Tests all-gather functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x):
    x = x + torch.zeros_like(x)
    output_list = [torch.empty_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, x)
    output = torch.cat(output_list, dim=0)
    output = output + torch.abs(output)
    output = output + torch.ones_like(output)
    return output

  expected = torch.tensor([float(2 * i + 1) for i in range(world_size)])
  execs = compile_and_assert_outputs(
      func,
      inputs=(torch.tensor([float(rank)], device="tpu"),),
      expected_outputs=(expected,),
  )
  assert len(execs) == 3, f"Expected 3 graphs, got {len(execs)}"


def run_all_to_all_single_with_torch_compile() -> None:
  """Tests all-to-all-single functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x):
    x = x + torch.zeros_like(x)
    output = torch.empty_like(x)
    torch.distributed.all_to_all_single(output, x)
    output = output + torch.abs(output)
    return output

  expected = torch.tensor(
      [float(j * world_size + rank) for j in range(world_size)]
  )
  expected = expected + torch.abs(expected)
  execs = compile_and_assert_outputs(
      func,
      inputs=(
          (
              torch.arange(world_size, dtype=torch.float32, device="tpu")
              + rank * world_size
          ),
      ),
      expected_outputs=(expected,),
  )
  assert len(execs) == 3, f"Expected 3 graphs, got {len(execs)}"


def run_all_to_all_with_torch_compile() -> None:
  """Tests all-to-all functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(inputs):
    outputs = [torch.empty_like(t) for t in inputs]
    torch.distributed.all_to_all(outputs, inputs)
    return torch.cat(outputs, dim=0)

  inputs = [
      torch.tensor([float(rank * world_size + i)], device="tpu")
      for i in range(world_size)
  ]
  expected = torch.tensor(
      [float(j * world_size + rank) for j in range(world_size)]
  )
  execs = compile_and_assert_outputs(
      func, inputs=(inputs,), expected_outputs=(expected,)
  )
  assert len(execs) == 2, f"Expected 2 graphs, got {len(execs)}"


def run_reduce_scatter_with_torch_compile() -> None:
  """Tests reduce-scatter functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(inputs):
    output = torch.zeros(2, 3, device="tpu")
    torch.distributed.reduce_scatter(output, inputs)
    output = output + torch.abs(output)
    return output

  inputs = []
  for i in range(world_size):
    inputs.append(
        torch.tensor(
            [
                [1.0, float(i), float(i**2)],
                [float(rank), float(rank**2), float(i * rank)],
            ],
            device="tpu",
        )
    )

  expected = torch.tensor([
      [world_size, rank * world_size, rank**2 * world_size],
      [28.0, 140.0, rank * 28.0],
  ])
  expected = expected + torch.abs(expected)

  execs = compile_and_assert_outputs(
      func, inputs=(inputs,), expected_outputs=(expected,)
  )
  assert len(execs) == 2, f"Expected 2 graphs, got {len(execs)}"


def run_gather_with_torch_compile() -> None:
  """Tests gather functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()
  dst = 0

  def func(x):
    x = x + 1
    if rank == dst:
      gather_list = [torch.empty_like(x) for _ in range(world_size)]
      torch.distributed.gather(x, gather_list=gather_list, dst=dst)
    else:
      torch.distributed.gather(x, dst=dst)
    x = x + 1
    return x

  input_tpu = torch.tensor([float(rank)], device="tpu")
  expected = torch.tensor([float(rank) + 2.0])

  execs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(expected,)
  )
  assert len(execs) == 2, f"Expected 2 graphs, got {len(execs)}"


def run_barrier_with_torch_compile() -> None:
  """Tests barrier functionality."""

  dist.init_process_group(backend="tpu_dist")

  def func(x):
    x = x + 1
    torch.distributed.barrier()
    x = torch.sin(x)
    torch.distributed.barrier()
    x = torch.cos(x)
    return x

  input_tpu = torch.tensor([1.0], device="tpu")
  execs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(torch.tensor([0.6143]),)
  )
  assert len(execs) == 3, f"Expected 3 graphs, got {len(execs)}"


def run_reduce_scatter_tensor_with_torch_compile() -> None:
  """Tests reduce-scatter-tensor functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x):
    x = x + torch.zeros_like(x)
    output = torch.empty(1, dtype=x.dtype, device=x.device)
    torch.distributed.reduce_scatter_tensor(output, x)
    output = output + torch.abs(output)
    output = output + torch.ones_like(output)
    return output

  input_tpu = torch.tensor([float(rank)] * world_size, device="tpu")
  execs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(torch.tensor([57.0]),)
  )
  assert len(execs) == 3, f"Expected 3 graphs, got {len(execs)}"


def run_broadcast_with_torch_compile() -> None:
  """Tests broadcast functionality."""

  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])

  def func(x):
    x = x + torch.zeros_like(x)
    torch.distributed.broadcast(x, src=0)
    x = x + torch.abs(x)
    x = x + torch.ones_like(x)
    return x

  input_tpu = torch.tensor([float(rank)], device="tpu")
  execs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(torch.tensor([1.0]),)
  )
  assert len(execs) == 2, f"Expected 2 graphs, got {len(execs)}"


def run_fake_tensor_side_effect_pruning_with_torch_compile() -> None:
  """Tests that fake tensors from compile tracing are not incorrectly anchored."""
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()
  mesh = tensor.DeviceMesh("tpu", list(range(world_size)))

  def my_func(x):
    dt = tensor.DTensor.from_local(x, mesh, [tensor.Shard(0)])
    dt_repl = dt.redistribute(mesh, [tensor.Replicate()])
    return dt_repl.to_local()

  backend = tt_compile.TpuBackend(debug=True)
  compiled_func = torch.compile(my_func, backend=backend, fullgraph=True)

  input_tpu = torch.tensor([float(rank)], dtype=torch.float32, device="tpu")

  output = compiled_func(input_tpu)

  tpu_device.get_device_module("tpu").synchronize()

  expected = torch.tensor(
      [float(i) for i in range(world_size)], dtype=torch.float32
  )
  utils.assert_close(output.to("cpu"), expected)


class MultiTpuTorchCompileTest(absltest.TestCase):

  def test_fake_tensor_side_effect_pruning_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_fake_tensor_side_effect_pruning_with_torch_compile, world_size=8
        ),
    )

  def test_all_reduce_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_reduce_with_torch_compile, world_size=8
        ),
    )

  def test_all_gather_into_tensor_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_into_tensor_with_torch_compile, world_size=8
        ),
    )

  def test_all_to_all_single_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_single_with_torch_compile, world_size=8
        ),
    )

  def test_all_to_all_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_to_all_with_torch_compile, world_size=8
        ),
    )

  def test_all_gather_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_all_gather_with_torch_compile, world_size=8
        ),
    )

  def test_barrier_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_barrier_with_torch_compile, world_size=8
        ),
    )

  def test_reduce_scatter_tensor_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_reduce_scatter_tensor_with_torch_compile, world_size=8
        ),
    )

  def test_broadcast_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_broadcast_with_torch_compile, world_size=8
        ),
    )

  def test_reduce_scatter_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_reduce_scatter_with_torch_compile, world_size=8
        ),
    )

  def test_gather_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_gather_with_torch_compile, world_size=8
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
