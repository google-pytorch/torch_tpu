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
import unittest

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch.distributed import tensor
import torch.multiprocessing as mp
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import oss_utils
from tests import seed_test_utils
from tests.distributed import distributed_utils


def compile_and_assert_outputs(
    func, inputs, expected_outputs=None
) -> list[tt_compile.TpuCompileDebug]:
  """Compiles a function, runs it, and returns debug objects."""
  debugs: list[tt_compile.TpuCompileDebug] = []
  compiled = torch.compile(
      func,
      backend="tpu",
      options={"serializable": False, "debug_callback": debugs.append},
  )
  output_compiled = compiled(*inputs)

  if expected_outputs is not None:
    if not isinstance(output_compiled, (list, tuple)):
      output_compiled = (output_compiled,)
    for actual, expected in zip(output_compiled, expected_outputs):
      utils.assert_close(actual.to("cpu"), expected.to("cpu"))

  return debugs


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
  debugs = compile_and_assert_outputs(
      func,
      inputs=(input_tpu,),
      expected_outputs=(expected,),
  )

  # debug mode enabled so expect graphs to be set and in plaintext
  assert len(debugs) == 1
  assert len(debugs[0].stablehlo_forward_text) == 3

  assert (
      "torch.ops._c10d_functional.all_reduce"
      in debugs[0].post_autograd_fx_forward_code[0]
  )

  assert "stablehlo.abs" not in debugs[0].stablehlo_forward_text[0]
  assert "stablehlo.all_reduce" not in debugs[0].stablehlo_forward_text[0]

  assert "stablehlo.all_reduce" in debugs[0].stablehlo_forward_text[1]
  assert "stablehlo.abs" not in debugs[0].stablehlo_forward_text[1]

  assert "stablehlo.abs" in debugs[0].stablehlo_forward_text[2]
  assert "stablehlo.all_reduce" not in debugs[0].stablehlo_forward_text[2]

  assert len(set(debugs[0].stablehlo_forward_text)) == 3


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
  debugs = compile_and_assert_outputs(
      func,
      inputs=(torch.tensor([float(rank)], device="tpu"),),
      expected_outputs=(expected,),
  )
  assert (
      len(debugs[0].stablehlo_forward_text) == 3
  ), f"Expected 3 graphs, got {len(debugs[0].stablehlo_forward_text)}"


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
  debugs = compile_and_assert_outputs(
      func,
      inputs=(torch.tensor([float(rank)], device="tpu"),),
      expected_outputs=(expected,),
  )
  assert (
      len(debugs[0].stablehlo_forward_text) == 3
  ), f"Expected 3 graphs, got {len(debugs[0].stablehlo_forward_text)}"


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
  debugs = compile_and_assert_outputs(
      func,
      inputs=(
          (
              torch.arange(world_size, dtype=torch.float32, device="tpu")
              + rank * world_size
          ),
      ),
      expected_outputs=(expected,),
  )
  assert (
      len(debugs[0].stablehlo_forward_text) == 3
  ), f"Expected 3 graphs, got {len(debugs[0].stablehlo_forward_text)}"


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
  debugs = compile_and_assert_outputs(
      func, inputs=(inputs,), expected_outputs=(expected,)
  )
  assert len(debugs) == 2, f"Expected 2 graphs, got {len(debugs)}"


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

  debugs = compile_and_assert_outputs(
      func, inputs=(inputs,), expected_outputs=(expected,)
  )
  assert (
      len(debugs[0].stablehlo_forward_text) == 3
  ), f"Expected 3 graphs, got {len(debugs[0].stablehlo_forward_text)}"


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

  debugs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(expected,)
  )
  assert len(debugs) == 2, f"Expected 2 graphs, got {len(debugs)}"


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
  debugs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(torch.tensor([0.6143]),)
  )
  assert len(debugs) == 3, f"Expected 3 graphs, got {len(debugs)}"


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
  debugs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(torch.tensor([57.0]),)
  )
  assert (
      len(debugs[0].stablehlo_forward_text) == 3
  ), f"Expected 3 graphs, got {len(debugs[0].stablehlo_forward_text)}"


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
  debugs = compile_and_assert_outputs(
      func, inputs=(input_tpu,), expected_outputs=(torch.tensor([1.0]),)
  )
  assert len(debugs) == 2, f"Expected 2 graphs, got {len(debugs)}"


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

  compiled_func = torch.compile(
      my_func,
      backend="tpu",
      options={"serializable": False},
      fullgraph=True,
  )

  input_tpu = torch.tensor([float(rank)], dtype=torch.float32, device="tpu")

  output = compiled_func(input_tpu)

  torch.accelerator.synchronize()

  expected = torch.tensor(
      [float(i) for i in range(world_size)], dtype=torch.float32
  )
  utils.assert_close(output.to("cpu"), expected)


class MultiTpuTorchCompileTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_fake_tensor_side_effect_pruning_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_fake_tensor_side_effect_pruning_with_torch_compile,
    )

  def test_all_reduce_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_all_reduce_with_torch_compile,
    )

  def test_all_gather_into_tensor_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_all_gather_into_tensor_with_torch_compile,
    )

  def test_all_to_all_single_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_all_to_all_single_with_torch_compile,
    )

  def test_all_to_all_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_all_to_all_with_torch_compile,
    )

  def test_all_gather_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_all_gather_with_torch_compile,
    )

  def test_barrier_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_barrier_with_torch_compile,
    )

  def test_reduce_scatter_tensor_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_reduce_scatter_tensor_with_torch_compile,
    )

  def test_broadcast_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_broadcast_with_torch_compile,
    )

  # TODO(b/493050035): Enable once OSS PyTorch supports functional collective
  # tracing for list-of-tensor reduce_scatter, which currently falls back to
  # eager and produces 2 subgraphs instead of the expected 3.
  @unittest.skipIf(
      oss_utils.is_oss(),
      "The test fails in OSS with 'AssertionError: Expected 3 graphs, got 2'.",
  )
  def test_reduce_scatter_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_reduce_scatter_with_torch_compile,
    )

  def test_gather_with_torch_compile(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_gather_with_torch_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
