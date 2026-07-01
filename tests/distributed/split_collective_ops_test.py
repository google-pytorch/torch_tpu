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
import pickle

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.compile._backend import TpuBackend
from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable
from torch_tpu._internal.compile.torch_tpu_compiled_executable import TorchTpuCompiledExecutable
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


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

  expected = torch.full((2, 2), 31.0, device="cpu")
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

  # Expected: (1 * 4) + (1 * 4) + 3 = 11.0
  expected = torch.full((2, 2), 11.0, device="cpu")
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

  expected = torch.full((2, 2), 16.0, device="cpu")
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


class SplitCollectiveOpsTest(absltest.TestCase):
  _world_size = 4

  def test_compile_all_reduce_and_serdes(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=self._world_size
        ),
        test_fn=run_compile_all_reduce_and_serdes_test,
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


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
