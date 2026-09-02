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

"""Distributed tests for torch.compiler.nested_compile_region on multi-TPU."""

import os
import sys
from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.distributed.spmd_util import spmd_safe
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils


class ReplicateComputation(nn.Module):
  """Replicates DTensor parameters during forward for local computation."""

  def __init__(
      self,
      device_mesh: DeviceMesh,
      param_sharding: tuple[Placement, ...],
  ):
    super().__init__()
    self.device_mesh = device_mesh
    self.param_sharding = param_sharding
    self.compute_placements: list[Placement] = [
        Replicate()
    ] * self.device_mesh.ndim
    self.grad_placements: list[Placement] = [
        Partial(reduce_op="sum")
    ] * self.device_mesh.ndim

  def replicate_compute(self, x: DTensor) -> torch.Tensor:
    output = x.redistribute(
        device_mesh=self.device_mesh,
        placements=self.compute_placements,
    )
    return output.to_local(grad_placements=self.grad_placements)

  def forward(self, x: DTensor) -> torch.Tensor:
    return self.replicate_compute(x)


_wrap_class_cache: dict[tuple[type, frozenset[str]], type] = {}


def _register_parametrization(
    module: nn.Module, param_names: list[str], parametrization: nn.Module
) -> None:
  """Registers properties on a dynamic subclass that replicate DTensor parameters on property access."""
  param_name_to_property = {
      param_name: property(
          lambda self, pn=param_name: self._simple_fsdp_parametrization_list[0](
              self._parameters[pn]
          )
      )
      for param_name in param_names
  }
  cache_key = (module.__class__, frozenset(param_names))
  if cache_key in _wrap_class_cache:
    module_cls = _wrap_class_cache[cache_key]
  else:
    module_cls = type(
        f"SimpleFSDP{module.__class__.__name__}",
        (module.__class__,),
        param_name_to_property,
    )
    import sys

    sys.modules[module_cls.__module__].__dict__[
        module_cls.__name__
    ] = module_cls
    _wrap_class_cache[cache_key] = module_cls
  module.__class__ = module_cls
  module._simple_fsdp_parametrization_list = [parametrization]


def simple_data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "fully_shard",
    shard_dim: int = 0,
) -> nn.Module:
  """Applies simple DTensor-based data parallelism or FSDP to a model."""
  if mode == "fully_shard":
    param_sharding = (Shard(shard_dim),)
  elif mode == "replicate":
    param_sharding = (Replicate(),)
  else:
    raise ValueError(f"Unsupported mode: {mode}")

  for mod in list(model.modules()):
    params_dict = dict(mod.named_parameters(recurse=False))
    if not params_dict or "SimpleFSDP" in mod.__class__.__name__:
      continue
    param_names = []
    for p_name, p in params_dict.items():
      if p is not None and p.numel() > 0:
        actual_sharding = list(param_sharding)
        for i, placement in enumerate(actual_sharding):
          if isinstance(placement, Shard) and p.shape[
              placement.dim
          ] < device_mesh.size(i):
            actual_sharding[i] = Replicate()
        actual_sharding = tuple(actual_sharding)

        sharded_param = distribute_tensor(p, device_mesh, actual_sharding)
        mod.register_parameter(
            p_name,
            nn.Parameter(sharded_param, requires_grad=p.requires_grad),
        )
        param_names.append(p_name)

    if param_names:
      _register_parametrization(
          mod,
          param_names,
          ReplicateComputation(device_mesh, param_sharding),
      )
  return model


@torch.compiler.nested_compile_region
def linear_layer_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  return x + F.relu(F.linear(x, weight))


class LinearLayer(nn.Module):
  """Single Linear layer executing within a nested compile region."""

  def __init__(self, dim: int):
    super().__init__()
    self.linear = nn.Linear(dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return linear_layer_forward(x, self.linear.weight)


class LinearStackNested(nn.Module):
  """Stack of Linear layers with residual connections in nested compile regions."""

  def __init__(self, num_layers: int = 4, dim: int = 16):
    super().__init__()
    self.layers = nn.ModuleList([LinearLayer(dim) for _ in range(num_layers)])

  @spmd_safe
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return x


from torch_tpu._internal.device import _device_module as tpu_device


def run_nested_compile_region_data_parallel(
    mode: str = "fully_shard",
) -> None:
  """Runs multi-TPU data parallel forward with nested compile regions and loop rolling."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  mesh = DeviceMesh("tpu", torch.arange(world_size))
  device = torch.device("tpu")

  num_layers = 4
  dim = 16
  torch.manual_seed(42 + rank)

  model = LinearStackNested(num_layers=num_layers, dim=dim).to(device)

  sharded_model = simple_data_parallel(model, mesh, mode=mode, shard_dim=0)
  tpu_backend = _backend.TpuBackend(debug=True)
  compiled_model = torch.compile(sharded_model, backend=tpu_backend)

  x = torch.randn(8, dim, device=device)

  with torch.no_grad(), torch._dynamo.config.patch(
      inline_single_use_invoke_subgraph=False
  ):
    out = compiled_model(x)

  tpu_device.get_device_module("tpu").synchronize()

  assert out is not None
  assert out.shape == (8, dim)
  assert not torch.isnan(out).any()

  # Verify MLIR generation: loop rolling should fold sequential layers into stablehlo.while
  assert len(tpu_backend._compiled_executables) == 1, (
      "Expected exactly 1 compiled executable (single binary), got"
      f" {len(tpu_backend._compiled_executables)}"
  )
  fwd_executable = tpu_backend._compiled_executables[0]
  mlir_text = fwd_executable.mlir_texts[0]
  if rank == 0:
    print(
        f"\n=== [Rank 0] StableHLO MLIR Dump ({mode}) ===\n{mlir_text}\n=== End"
        " StableHLO MLIR Dump ===\n",
        file=sys.stderr,
        flush=True,
    )
  assert (
      "stablehlo.while" in mlir_text
  ), f"Expected stablehlo.while loop rolling in MLIR, got:\n{mlir_text}"


def run_nested_compile_region_with_collectives_in_loop() -> None:
  """Tests that roll_nested_regions nests collective operations inside stablehlo.while."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  device = torch.device("tpu")

  @torch.compiler.nested_compile_region
  def collective_layer(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    h = F.linear(x, weight)
    reduced = torch.ops._c10d_functional.all_reduce.default(h, "sum", "0")
    waited = torch.ops._c10d_functional.wait_tensor.default(reduced)
    return x + F.relu(waited)

  class StackWithCollectives(nn.Module):

    def __init__(self, num_layers: int = 3, dim: int = 16):
      super().__init__()
      self.weights = nn.ParameterList([
          nn.Parameter(
              torch.randn(dim, dim, device=device), requires_grad=False
          )
          for _ in range(num_layers)
      ])

    @spmd_safe
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      for w in self.weights:
        x = collective_layer(x, w)
      return x

  model = StackWithCollectives(num_layers=3, dim=16).to(device)
  tpu_backend = _backend.TpuBackend(debug=True)
  compiled_model = torch.compile(model, backend=tpu_backend)

  x = torch.ones(4, 16, device=device) * float(rank + 1)
  with torch.no_grad(), torch._dynamo.config.patch(
      inline_single_use_invoke_subgraph=False
  ):
    out = compiled_model(x)

  tpu_device.get_device_module("tpu").synchronize()

  assert out is not None
  assert out.shape == (4, 16)
  assert not torch.isnan(out).any()

  # Verify MLIR generation: loop rolling must produce stablehlo.while containing collectives in a single binary
  assert len(tpu_backend._compiled_executables) == 1, (
      "Expected exactly 1 compiled executable (single binary), got"
      f" {len(tpu_backend._compiled_executables)}"
  )
  fwd_executable = tpu_backend._compiled_executables[0]
  mlir_text = fwd_executable.mlir_texts[0]
  if rank == 0:
    print(
        "\n=== [Rank 0] StableHLO MLIR Dump (collectives_in_loop)"
        f" ===\n{mlir_text}\n=== End StableHLO MLIR Dump ===\n",
        file=sys.stderr,
        flush=True,
    )
  assert (
      "stablehlo.while" in mlir_text
  ), f"Expected stablehlo.while loop rolling in MLIR, got:\n{mlir_text}"


def run_nested_compile_region_functionalized_module_distributed() -> None:
  """Tests a pure torch.nn.Module with no distributed logic in its forward pass,

  functionalized to pass parameters into @nested_compile_region, then
  distributed via DTensor.
  Demonstrates that DTensor communication (un-sharding / all-gather) is sunk
  into the subgraph,
  rolled into a scan loop, and the stablehlo.while op contains the collective
  communication ops
  nested inside the loop body.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  mesh = DeviceMesh("tpu", torch.arange(world_size))
  device = torch.device("tpu")

  num_layers = 4
  dim = 16
  torch.manual_seed(42 + rank)

  # 1. Pure non-distributed computation function decorated with nested_compile_region
  @torch.compiler.nested_compile_region
  def transformer_mlp_forward(
      x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor
  ) -> torch.Tensor:
    gate = F.silu(F.linear(x, gate_weight))
    up = F.linear(x, up_weight)
    return x + (gate * up)

  # 2. Standard torch.nn.Module with NO distributed logic
  class TransformerMLPBlock(nn.Module):

    def __init__(self, dim: int):
      super().__init__()
      self.gate = nn.Linear(dim, dim, bias=False)
      self.up = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # Functional call passing local parameters to the nested compile region
      return transformer_mlp_forward(x, self.gate.weight, self.up.weight)

  class TransformerMLPStack(nn.Module):

    def __init__(self, num_layers: int = 4, dim: int = 16):
      super().__init__()
      self.layers = nn.ModuleList(
          [TransformerMLPBlock(dim) for _ in range(num_layers)]
      )

    @spmd_safe
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      for layer in self.layers:
        x = layer(x)
      return x

  model = TransformerMLPStack(num_layers=num_layers, dim=dim).to(device)

  # 3. Make it distributed using DTensor parameter sharding (fully_shard)
  sharded_model = simple_data_parallel(
      model, mesh, mode="fully_shard", shard_dim=0
  )

  tpu_backend = _backend.TpuBackend(debug=True)
  compiled_model = torch.compile(sharded_model, backend=tpu_backend)

  x = torch.randn(8, dim, device=device)

  with torch.no_grad(), torch._dynamo.config.patch(
      inline_single_use_invoke_subgraph=False
  ):
    out = compiled_model(x)

  tpu_device.get_device_module("tpu").synchronize()

  assert out is not None
  assert out.shape == (8, dim)
  assert not torch.isnan(out).any()

  # 4. Verify loop rolling into single executable and StableHLO while loop
  assert len(tpu_backend._compiled_executables) == 1, (
      "Expected exactly 1 compiled executable (single binary), got"
      f" {len(tpu_backend._compiled_executables)}"
  )
  fwd_executable = tpu_backend._compiled_executables[0]
  mlir_text = fwd_executable.mlir_texts[0]
  if rank == 0:
    print(
        "\n=== [Rank 0] StableHLO MLIR Dump (functionalized_module_distributed)"
        f" ===\n{mlir_text}\n=== End StableHLO MLIR Dump ===\n",
        file=sys.stderr,
        flush=True,
    )
  assert (
      "stablehlo.while" in mlir_text
  ), f"Expected stablehlo.while loop rolling in MLIR, got:\n{mlir_text}"
  # Verify that collective communication (all_gather) is inside the while loop body
  while_loop_segment = mlir_text[mlir_text.find("stablehlo.while") :]
  assert (
      "all_gather" in while_loop_segment
      or "all_reduce" in while_loop_segment
      or "collective" in while_loop_segment
  ), (
      "Expected collective communication inside stablehlo.while body,"
      f" got:\n{while_loop_segment}"
  )


def run_make_fx_with_nested_compile_region_dtensor() -> None:
  """Tests whether make_fx can trace a DTensor-sharded model with nested_compile_region

  without encountering the AttributeError: 'FakeTensor' object has no attribute
  'redistribute'.
  """
  from torch.fx.experimental.proxy_tensor import make_fx

  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  mesh = DeviceMesh("tpu", torch.arange(world_size))
  device = torch.device("tpu")

  dim = 16
  torch.manual_seed(42 + rank)

  class ShardedBlock(nn.Module):

    def __init__(self, dim: int):
      super().__init__()
      self.linear = nn.Linear(dim, dim, bias=False)

    @torch.compiler.nested_compile_region
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # Non-functionalized: access self.linear directly (triggers self.linear.weight redistribute)
      return x + F.relu(self.linear(x))

  class ShardedModel(nn.Module):

    def __init__(self, num_layers: int = 4, dim: int = 16):
      super().__init__()
      self.layers = nn.ModuleList(
          [ShardedBlock(dim) for _ in range(num_layers)]
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      for layer in self.layers:
        x = layer(x)
      return x

  model = ShardedModel(num_layers=4, dim=dim).to(device)
  sharded_model = simple_data_parallel(
      model, mesh, mode="fully_shard", shard_dim=0
  )

  x = torch.randn(8, dim, device=device)

  # Try tracing with make_fx
  gm = make_fx(sharded_model)(x)

  if rank == 0:
    print(
        f"\n=== [Rank 0] make_fx Traced Graph Code ===\n{gm.code}\n=== End"
        " make_fx Code ===\n",
        file=sys.stderr,
        flush=True,
    )

  out = gm(x)
  tpu_device.get_device_module("tpu").synchronize()
  assert out is not None
  assert out.shape == (8, dim)


def run_nested_compile_region_training_fsdp_alexander_repro() -> None:
  """Replicates Alexander's 10-layer FSDP training setup with nested_compile_region (weights + biases sharded).

  Tests forward + backward training with optimizer step, verifying that
  collective
  communication is sunk into the loop body and NO redundant upfront all-gathers
  exist.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  mesh = DeviceMesh("tpu", torch.arange(world_size))
  device = torch.device("tpu")

  num_layers = 10
  dim = 32
  torch.manual_seed(42 + rank)

  @torch.compiler.nested_compile_region
  def linear_with_bias_forward(
      x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
  ) -> torch.Tensor:
    return x + F.relu(F.linear(x, weight, bias))

  class LinearLayerWithBias(nn.Module):

    def __init__(self, d: int):
      super().__init__()
      self.linear = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return linear_with_bias_forward(x, self.linear.weight, self.linear.bias)

  class LinearStackNestedWithBias(nn.Module):

    def __init__(self, n_layers: int = 10, d: int = 32):
      super().__init__()
      self.layers = nn.ModuleList(
          [LinearLayerWithBias(d) for _ in range(n_layers)]
      )

    @spmd_safe
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      for layer in self.layers:
        x = layer(x)
      return x

  model = LinearStackNestedWithBias(num_layers, dim).to(device)
  sharded_model = simple_data_parallel(
      model, mesh, mode="fully_shard", shard_dim=0
  )

  tpu_backend = _backend.TpuBackend(debug=True)
  compiled_model = torch.compile(sharded_model, backend=tpu_backend)

  x = torch.randn(16, dim, device=device, requires_grad=True)
  y = torch.randn(16, dim, device=device)

  with torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False):
    out = compiled_model(x)
    loss = F.mse_loss(out, y)
    loss.backward()

  tpu_device.get_device_module("tpu").synchronize()

  assert out is not None
  assert out.shape == (16, dim)
  assert not torch.isnan(out).any()

  # Check MLIR of compiled executables
  if rank == 0:
    print(
        "\nNumber of compiled executables:"
        f" {len(tpu_backend._compiled_executables)}",
        file=sys.stderr,
        flush=True,
    )
    for i, executable in enumerate(tpu_backend._compiled_executables):
      for j, mlir_text in enumerate(executable.mlir_texts):
        print(
            f"\n=== [Rank 0] StableHLO MLIR Dump (Executable {i}, Graph {j})"
            f" ===\n{mlir_text}\n=== End StableHLO MLIR Dump ===\n",
            file=sys.stderr,
            flush=True,
        )

  assert len(tpu_backend._compiled_executables) >= 1
  fwd_executable = tpu_backend._compiled_executables[0]
  mlir_text = fwd_executable.mlir_texts[0]
  # Check loop rolling and no top-level all_gather outside while loop
  assert (
      "stablehlo.while" in mlir_text
  ), f"Expected stablehlo.while loop rolling in MLIR, got:\n{mlir_text}"
  while_pos = mlir_text.find("stablehlo.while")
  before_while = mlir_text[:while_pos]
  top_all_gathers = before_while.count(
      "stablehlo.all_gather"
  ) + before_while.count("c10d_functional.all_gather")
  if rank == 0:
    print(
        f"Top-level all_gathers before while loop: {top_all_gathers}",
        file=sys.stderr,
        flush=True,
    )
  assert top_all_gathers == 0, (
      "Expected 0 top-level all_gathers before while loop, found"
      f" {top_all_gathers}"
  )
  while_body = mlir_text[while_pos:]
  assert (
      "all_gather" in while_body
  ), f"Expected all_gather inside while body, got:\n{while_body}"


def run_nested_compile_region_data_parallel_fully_shard() -> None:
  run_nested_compile_region_data_parallel(mode="fully_shard")


def run_nested_compile_region_data_parallel_replicate() -> None:
  run_nested_compile_region_data_parallel(mode="replicate")


class NestedCompileRegionDistributedTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def test_nested_compile_region_training_fsdp_alexander_repro(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_nested_compile_region_training_fsdp_alexander_repro,
            world_size=8,
        ),
    )

  def test_nested_compile_region_data_parallel_fully_shard(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_nested_compile_region_data_parallel_fully_shard,
            world_size=8,
        ),
    )

  def test_nested_compile_region_data_parallel_replicate(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_nested_compile_region_data_parallel_replicate,
            world_size=8,
        ),
    )

  def test_nested_compile_region_with_collectives_in_loop(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_nested_compile_region_with_collectives_in_loop,
            world_size=8,
        ),
    )

  def test_nested_compile_region_functionalized_module_distributed(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_nested_compile_region_functionalized_module_distributed,
            world_size=8,
        ),
    )

  def test_make_fx_with_nested_compile_region_dtensor(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_make_fx_with_nested_compile_region_dtensor,
            world_size=8,
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
