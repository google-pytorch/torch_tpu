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

"""Tests standalone SimpleFSDP on TPU."""

from collections.abc import Callable
import copy
import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.distributed import fsdp
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from examples.fsdp import simple_fsdp
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

BATCH_SIZE = 4
DIM = 32
LR = 0.05
NUM_STEPS = 2
RANDOM_SEED = 42


class ToyMLP(nn.Module):
  """Simple two-layer MLP for testing SimpleFSDP."""

  def __init__(self, dim: int):
    super().__init__()
    self.fc1 = nn.Linear(dim, dim, bias=False)
    self.act = nn.ReLU()
    self.fc2 = nn.Linear(dim, dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc2(self.act(self.fc1(x)))


def _build_fsdp2_model(
    base_model: nn.Module,
    mesh: dt.DeviceMesh,
) -> nn.Module:
  """Shards a copy of `base_model` using PyTorch FSDPv2 (`fsdp.fully_shard`)."""
  model = copy.deepcopy(base_model)
  mp_policy = fsdp.MixedPrecisionPolicy(
      param_dtype=torch.float32,
      reduce_dtype=torch.bfloat16,
  )
  # FSDPv2 requires wrapping each child module individually before the root.
  fsdp.fully_shard(model.fc1, mesh=mesh, mp_policy=mp_policy)
  fsdp.fully_shard(model.fc2, mesh=mesh, mp_policy=mp_policy)
  fsdp.fully_shard(model, mesh=mesh, mp_policy=mp_policy)
  return model


def _build_simple_fsdp_model(
    base_model: nn.Module,
    mesh: dt.DeviceMesh,
) -> nn.Module:
  """Shards a copy of `base_model` using SimpleFSDP (`simple_fsdp.data_parallel`)."""
  model = copy.deepcopy(base_model)
  mp_policy = simple_fsdp.MixedPrecisionPolicy(
      param_dtype=torch.float32,
      reduce_dtype=torch.bfloat16,
  )
  # SimpleFSDP wraps all parameterized submodules in a single call and installs
  # DTensor redistribution parametrizations on parameter access.
  return simple_fsdp.data_parallel(
      model,
      device_mesh=mesh,
      mode="fully_shard",
      mp_policy=mp_policy,
  )


def _fsdp2_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Runs a single forward + loss + backward step for FSDPv2."""
  y_pred = model(x)
  loss = nn.MSELoss()(y_pred.float(), y.float())
  # FSDPv2 defaults to AVG gradient reduction across data-parallel ranks.
  loss.backward()
  return loss.detach(), y_pred.detach()


def _simple_fsdp_step(
    model: nn.Module | Callable[..., torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Runs a single forward + loss + backward step for SimpleFSDP."""
  y_pred = model(x)
  loss = nn.MSELoss()(y_pred.float(), y.float())
  # SimpleFSDP uses Partial(reduce_op="sum") for gradient reduction, so we scale
  # the loss by 1 / world_size to match FSDPv2's average gradient reduction.
  (loss / world_size).backward()
  return loss.detach(), y_pred.detach()


def run_simple_fsdp_fully_shard_parity() -> None:
  """Verifies compiled SimpleFSDP matches FSDPv2 across training steps."""
  # 1. Initialize distributed process group and 1D FSDP device mesh across TPUs.
  dist.init_process_group(backend="tpu_dist")
  try:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = dt.init_device_mesh("tpu", (world_size,), mesh_dim_names=("fsdp",))

    # 2. Create deterministic base model weights and per-rank training data.
    torch.manual_seed(RANDOM_SEED)
    base_model = ToyMLP(DIM).to("tpu")
    x_all = torch.randn(world_size, BATCH_SIZE, DIM)
    y_all = torch.randn(world_size, BATCH_SIZE, DIM)
    x = x_all[rank].to("tpu")
    y = y_all[rank].to("tpu")

    # 3. Build FSDPv2 reference model and optimizer from the base model.
    fsdp2_model = _build_fsdp2_model(base_model, mesh)
    fsdp2_opt = optim.SGD(fsdp2_model.parameters(), lr=LR)

    # 4. Build SimpleFSDP model and optimizer from the same base model weights,
    #    and compile the model for TPU execution.
    simple_model = _build_simple_fsdp_model(base_model, mesh)
    simple_opt = optim.SGD(simple_model.parameters(), lr=LR)
    compiled_simple_model = torch.compile(simple_model, backend="tpu")

    # 5. Run training steps on both models and verify numerical parity of losses
    #    and predictions at every step.
    for _ in range(NUM_STEPS):
      fsdp2_opt.zero_grad()
      fsdp2_loss, fsdp2_pred = _fsdp2_step(fsdp2_model, x, y)
      fsdp2_opt.step()

      simple_opt.zero_grad()
      simple_loss, simple_pred = _simple_fsdp_step(
          compiled_simple_model, x, y, world_size
      )
      simple_opt.step()

      utils.assert_close(
          simple_loss.to("cpu"),
          fsdp2_loss.to("cpu"),
          rtol=1e-2,
          atol=1e-2,
      )
      utils.assert_close(
          simple_pred.to("cpu"),
          fsdp2_pred.to("cpu"),
          rtol=1e-2,
          atol=1e-2,
      )
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def run_simple_fsdp_options() -> None:
  """Verifies shard_dim, ignored_params, small-param replication, and context manager."""
  # 1. Initialize distributed process group and 1D FSDP device mesh.
  dist.init_process_group(backend="tpu_dist")
  try:
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = dt.init_device_mesh("tpu", (world_size,), mesh_dim_names=("fsdp",))

    # 2. Define a module exercising custom sharding options:
    #    - `weight`: sharded along dimension 1 via per-parameter `shard_dim` dict.
    #    - `small_bias`: dimension 0 size (2) < world_size (8), so SimpleFSDP
    #      automatically replicates it instead of failing to shard.
    #    - `frozen`: passed in `ignored_params`, so it remains an unsharded Tensor.
    class CustomModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(DIM, DIM * 2, device="tpu"))
        self.small_bias = nn.Parameter(torch.randn(2, device="tpu"))
        self.frozen = nn.Parameter(torch.randn(DIM, device="tpu"))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight[:, :DIM] + self.small_bias[0] + self.frozen

    mod = CustomModule()
    mod = simple_fsdp.data_parallel(
        mod,
        device_mesh=mesh,
        mode="fully_shard",
        shard_dim={"weight": 1, "small_bias": 0},
        ignored_params={mod.frozen},
    )

    # 3. Verify `ignored_params` leaves `frozen` as a plain Tensor (not a DTensor).
    assert not isinstance(mod.frozen, dt.DTensor)

    # 4. Verify `disable_active_parametrization()` bypasses the all-gather
    #    parametrization and exposes the underlying sharded/replicated DTensors.
    with simple_fsdp.disable_active_parametrization():
      assert isinstance(mod.weight, dt.DTensor)
      assert mod.weight.placements == (dt.Shard(1),)
      assert isinstance(mod.small_bias, dt.DTensor)
      assert mod.small_bias.placements == (dt.Replicate(),)

    # 5. Verify normal forward/backward execution triggers the parametrization and
    #    populates gradients on the underlying DTensor parameter.
    x = torch.randn(BATCH_SIZE, DIM, device="tpu")
    out = mod(x)
    out.sum().backward()
    with simple_fsdp.disable_active_parametrization():
      assert mod.weight.grad is not None
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


class SimpleFSDPTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_fully_shard_parity(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_simple_fsdp_fully_shard_parity,
    )

  def test_options(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=run_simple_fsdp_options,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
