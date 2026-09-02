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

"""Unit tests and benchmarks for compute/comms overlap in loop-rolled nested compile regions."""

import os
import sys
import time
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
from torch_tpu._internal.device import _device_module as tpu_device
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.distributed.spmd_util import spmd_safe
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.benchmarks import xprof_adapter
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


def linear_layer_forward_baseline(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
  return x + F.relu(F.linear(x, weight))


class LinearLayer(nn.Module):
  """Single Linear layer executing within a nested compile region."""

  def __init__(self, dim: int):
    super().__init__()
    self.linear = nn.Linear(dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return linear_layer_forward(x, self.linear.weight)


class BaselineLinearLayer(nn.Module):
  """Single Linear layer executing standard unrolled eager/compiled path."""

  def __init__(self, dim: int):
    super().__init__()
    self.linear = nn.Linear(dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return linear_layer_forward_baseline(x, self.linear.weight)


class MultiLayerNestedModel(nn.Module):
  """Multi-layer sequential model composed of nested compile regions."""

  def __init__(self, num_layers: int = 4, dim: int = 16):
    super().__init__()
    self.layers = nn.ModuleList([LinearLayer(dim) for _ in range(num_layers)])

  @spmd_safe
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return x


class MultiLayerBaselineModel(nn.Module):
  """Multi-layer sequential baseline model without nested compile regions."""

  def __init__(self, num_layers: int = 4, dim: int = 16):
    super().__init__()
    self.layers = nn.ModuleList(
        [BaselineLinearLayer(dim) for _ in range(num_layers)]
    )

  @spmd_safe
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return x


def run_comm_compute_scaling_sweep(
    dim: int,
    num_layers: int = 4,
    mode: str = "fully_shard",
) -> None:
  """Evaluates compute vs communication scaling and verifies loop-rolled while with embedded collectives."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  mesh = DeviceMesh("tpu", torch.arange(world_size))
  device = torch.device("tpu")
  torch.manual_seed(42 + rank)

  # 1. Instantiate nested model and identical baseline model
  nested_model = MultiLayerNestedModel(num_layers=num_layers, dim=dim).to(
      device
  )
  baseline_model = MultiLayerBaselineModel(num_layers=num_layers, dim=dim).to(
      device
  )
  baseline_model.load_state_dict(nested_model.state_dict())

  sharded_nested_model = simple_data_parallel(
      nested_model, mesh, mode=mode, shard_dim=0
  )
  sharded_baseline_model = simple_data_parallel(
      baseline_model, mesh, mode=mode, shard_dim=0
  )

  tpu_backend_nested = _backend.TpuBackend(debug=True)
  tpu_backend_baseline = _backend.TpuBackend(debug=True)

  compiled_nested = torch.compile(
      sharded_nested_model, backend=tpu_backend_nested
  )
  compiled_baseline = torch.compile(
      sharded_baseline_model, backend=tpu_backend_baseline
  )

  batch_size = 8
  x = torch.randn(batch_size, dim, device=device)

  # 2. Warmup step with xprof TraceMe
  with torch.no_grad(), torch._dynamo.config.patch(
      inline_single_use_invoke_subgraph=False
  ):
    with xprof_adapter.TraceMe("warmup_nested_compile_region"):
      out_nested = compiled_nested(x)
    tpu_device.get_device_module("tpu").synchronize()

    with xprof_adapter.TraceMe("warmup_baseline_model"):
      out_baseline = compiled_baseline(x)
    tpu_device.get_device_module("tpu").synchronize()

  assert out_nested is not None
  assert out_baseline is not None
  assert out_nested.shape == (batch_size, dim)
  assert not torch.isnan(out_nested).any()
  assert not torch.isnan(out_baseline).any()

  # Verify numerical equivalence between nested loop-rolled model and baseline model
  utils.assert_close(out_nested, out_baseline, rtol=1e-3, atol=1e-3)

  # 3. Verify MLIR structures
  assert len(tpu_backend_nested._compiled_executables) == 1, (
      "Expected 1 compiled executable for nested model, got"
      f" {len(tpu_backend_nested._compiled_executables)}"
  )
  mlir_text_nested = tpu_backend_nested._compiled_executables[0].mlir_texts[0]
  assert (
      "stablehlo.while" in mlir_text_nested
  ), "Expected stablehlo.while in generated MLIR for nested model"
  assert (
      "stablehlo.dot_general" in mlir_text_nested
      or "stablehlo.multiply" in mlir_text_nested
  ), "Expected compute ops in MLIR for nested model"
  if mode == "fully_shard":
    assert (
        "stablehlo.all_gather" in mlir_text_nested
    ), "Expected stablehlo.all_gather collective in MLIR for nested model"

  mlir_text_baseline = tpu_backend_baseline._compiled_executables[0].mlir_texts[
      0
  ]
  assert (
      "stablehlo.while" not in mlir_text_baseline
  ), "Did not expect stablehlo.while in generated MLIR for baseline model"

  # 4. Measure timed iterations for side-by-side xprof tracing and latency comparison
  num_timed_steps = 5

  session = xprof_adapter.XprofSession()
  try:
    session.start_session(host_trace_level=3, enable_python_tracer=True)
  except Exception:
    pass

  # Scoped execution for nested compile region with comm/compute overlap
  t0_nested = time.perf_counter()
  with torch.no_grad():
    with xprof_adapter.TraceMe("nested_compile_region_with_overlap"):
      for step_i in range(num_timed_steps):
        with xprof_adapter.TraceMe(f"nested_step_{step_i}"):
          out_nested = compiled_nested(x)
      tpu_device.get_device_module("tpu").synchronize()
  elapsed_nested = (time.perf_counter() - t0_nested) / num_timed_steps

  # Scoped execution for baseline model without nested compile region
  t0_baseline = time.perf_counter()
  with torch.no_grad():
    with xprof_adapter.TraceMe("baseline_without_nested_compile_region"):
      for step_i in range(num_timed_steps):
        with xprof_adapter.TraceMe(f"baseline_step_{step_i}"):
          out_baseline = compiled_baseline(x)
      tpu_device.get_device_module("tpu").synchronize()
  elapsed_baseline = (time.perf_counter() - t0_baseline) / num_timed_steps

  xprof_url = None
  try:
    xprof_url = session.end_session_and_get_url()
  except Exception:
    pass

  if rank == 0:
    print(
        f"[Compute/Comm Overlap Comparison] dim={dim}, num_layers={num_layers},"
        f" mode={mode}:\n  - Nested Compile Region (Loop Rolled + Sunk Comms):"
        f" avg_step_time={elapsed_nested * 1000.0:.2f}ms\n  - Baseline"
        " (Unrolled / No Nested Region):         "
        f" avg_step_time={elapsed_baseline * 1000.0:.2f}ms\n  - XProf URL:"
        f" {xprof_url}"
    )


def run_test_comm_compute_dim_16():
  run_comm_compute_scaling_sweep(dim=16, num_layers=4, mode="fully_shard")


def run_test_comm_compute_dim_64():
  run_comm_compute_scaling_sweep(dim=64, num_layers=4, mode="fully_shard")


def run_test_comm_compute_dim_128():
  run_comm_compute_scaling_sweep(dim=128, num_layers=4, mode="fully_shard")


class NestedCompileRegionCommComputeOverlapTest(
    seed_test_utils.MultiProcessRepeatableTest
):
  """Tests verifying compute and collective communication overlap in loop-rolled regions."""

  def test_comm_compute_overlap_dim_16(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_test_comm_compute_dim_16,
            world_size=8,
        ),
    )

  def test_comm_compute_overlap_dim_64(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_test_comm_compute_dim_64,
            world_size=8,
        ),
    )

  def test_comm_compute_overlap_dim_128(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_test_comm_compute_dim_128,
            world_size=8,
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
