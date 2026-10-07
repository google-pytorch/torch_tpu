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

"""Tutorial: Migrating from PyTorch FSDPv2 (fully_shard) to SimpleFSDP on TPU.

================================================================================
OVERVIEW: WHY MIGRATE TO SIMPLEFSDP?
================================================================================
SimpleFSDP (`simple_fsdp.data_parallel`) is a compile-first Fully Sharded Data
Parallel implementation that maximizes performance with `torch.compile` and XLA
on TPUs:

1. Compile-First Architecture:
   Instead of dynamic runtime pre-forward/post-backward Python hooks, SimpleFSDP
   expresses parameter all-gathers and gradient reduce-scatters directly as
   `DTensor` redistributions (`DTensor.redistribute`) inside parameter access
   parametrizations.

2. Single Whole-Graph Compilation:
   `torch.compile(..., backend="tpu")` traces the entire forward pass, loss
   computation, backward pass, and all collective operations into a single XLA
   graph without graph breaks. This allows the XLA compiler to globally schedule
   and overlap communication and computation across the entire step.

Under most cases, we expect SimpleFSDP to provide significant speedups over
FSDPv2 for TPU-based training.

================================================================================
HOW TO RUN THIS TUTORIAL
================================================================================

Via CLI shell launcher:
  ./run_fsdp_to_simple_fsdp.sh
"""

from collections.abc import Callable
import copy
import functools
import time
from absl import logging
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.distributed import fsdp
from torch.distributed import tensor as dt
from torch.utils import checkpoint as checkpoint_utils
from examples.fsdp import simple_fsdp
from examples.fsdp.fsdp_to_simple_fsdp import model as model_lib

BATCH_SIZE = 4  # Per-device batch size.
HIDDEN_DIM = 128
LR = 0.05
NUM_STEPS = 5
RANDOM_SEED = 42


def make_data(
    world_size: int, in_features: int, out_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Creates reproducible training inputs and targets for all ranks."""
  torch.manual_seed(RANDOM_SEED)
  x = torch.randn(world_size, BATCH_SIZE, in_features)
  y = torch.randn(world_size, BATCH_SIZE, out_features)
  return x, y


# ==============================================================================
# STEP 1: MODEL SHARDING & ACTIVATION CHECKPOINTING (BEFORE vs. AFTER)
# ==============================================================================


def build_fsdp2_model(
    base_model: nn.Module,
    mesh: dt.DeviceMesh,
) -> nn.Module:
  """BEFORE: Shards a simple layered model using PyTorch FSDPv2 (`fully_shard`).

  In FSDPv2, you must:
    1. Configure `fsdp.MixedPrecisionPolicy`.
    2. Apply activation checkpointing (`checkpoint_utils.checkpoint`) to each
       submodule before FSDP wrapping to bound activation memory and discard
       unsharded weights after forward.
    3. Explicitly call `fsdp.fully_shard` on every individual submodule
       followed by the root module (`m`).
    4. Manually configure forward and backward prefetching across layers
       (`set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`)
       so that runtime hooks can overlap communication with computation.
  """
  m = copy.deepcopy(base_model)

  # 1. Configure MixedPrecisionPolicy from torch.distributed.fsdp
  mp_policy = fsdp.MixedPrecisionPolicy(
      param_dtype=torch.float32,
      reduce_dtype=torch.bfloat16,
  )

  # 2. Apply activation checkpointing and FSDP sharding to each parameterized
  #    submodule, then shard the root module.
  layers = [layer for layer in m.children() if list(layer.parameters())]
  for layer in layers:
    layer.forward = functools.partial(
        checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
    )
    fsdp.fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
  fsdp.fully_shard(m, mesh=mesh, mp_policy=mp_policy)

  # 3. Manually configure FSDPv2 forward & backward prefetching for overlap.
  # This makes FSDPv2 more efficient, but requires manual configuration which
  # may require significant tuning.
  num_to_prefetch = 2
  for i, layer in enumerate(layers):
    if i < len(layers) - num_to_prefetch:
      layer.set_modules_to_forward_prefetch(
          [layers[i + j] for j in range(1, num_to_prefetch + 1)]
      )
    if i >= num_to_prefetch:
      layer.set_modules_to_backward_prefetch(
          [layers[i - j] for j in range(1, num_to_prefetch + 1)]
      )

  return m


def build_simple_fsdp_model(
    base_model: nn.Module,
    mesh: dt.DeviceMesh,
) -> nn.Module:
  """AFTER: Shards a simple layered model using SimpleFSDP (`data_parallel`).

  In SimpleFSDP, you:
    1. Configure `simple_fsdp.MixedPrecisionPolicy`.
    2. Apply activation checkpointing (`checkpoint_utils.checkpoint`) to each
       parameterized submodule. Activation checkpointing is essential for
       SimpleFSDP: because parameter all-gathers are traced as regular DTensor
       operations in the forward graph, checkpointing each layer ensures the
       compiler frees (reshards) the unsharded weights after each layer's
       forward pass and re-all-gathers them during backward recomputation.
    3. Wrap the entire model in a single call to `simple_fsdp.data_parallel`.
       No manual per-layer FSDP wrapping or prefetch bucket tuning is required
       because XLA schedules and overlaps the DTensor collectives automatically
       across the full compiled graph.
  """
  m = copy.deepcopy(base_model)

  # 1. Configure MixedPrecisionPolicy from simple_fsdp
  mp_policy = simple_fsdp.MixedPrecisionPolicy(
      param_dtype=torch.float32,
      reduce_dtype=torch.bfloat16,
  )

  # 2. Apply activation checkpointing to each parameterized submodule so
  #    unsharded weights are freed after forward and re-all-gathered in backward.
  for layer in m.children():
    if list(layer.parameters()):
      layer.forward = functools.partial(
          checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
      )

  # 3. Wrap the entire model in a single call (no manual prefetch tuning needed!)
  m = simple_fsdp.data_parallel(
      m,
      device_mesh=mesh,
      mode="fully_shard",
      mp_policy=mp_policy,
  )

  return m


# ==============================================================================
# STEP 2: COMPILING THE TRAINING STEP & GRADIENT REDUCTION SCALING
# ==============================================================================
#
# With SimpleFSDP, the training step is compiled using
# `torch.compile(..., backend="tpu")`. The parameter redistributions
# (`all-gather` in forward, `reduce-scatter` in backward) are captured directly
# into the XLA graph.
#
# Important Migration Note on Gradient Reduction:
#   PyTorch FSDPv2 (`fully_shard`) defaults to `AVG` gradient reduction across
#   data-parallel ranks, whereas SimpleFSDP uses `Partial(reduce_op="sum")`
#   (sum reduction). When migrating from FSDPv2 to SimpleFSDP, scale the loss by
#   `1.0 / world_size` before calling `.backward()` (or scale your learning rate
#   by `1.0 / world_size`) to maintain identical gradient magnitude.


def simple_fsdp_train_step(
    model: nn.Module | Callable[..., torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Single compiled forward + loss + backward step for SimpleFSDP."""
  y_pred = model(x)
  loss = nn.MSELoss()(y_pred.float(), y.float())
  # Scale by 1/world_size to match FSDPv2's AVG gradient reduction
  (loss / world_size).backward()
  return loss.detach(), y_pred.detach()


# ==============================================================================
# STEP 3: DISTRIBUTED EXECUTION AND NUMERICAL PARITY VERIFICATION
# ==============================================================================
#
# Expected Output on 8 TPU cores (v6e-8):
#   Step 0: FSDPv2 loss=0.952937 (10663.96 ms) | SimpleFSDP loss=0.952937 (8205.92 ms)
#   Step 1: FSDPv2 loss=0.952836 (194.41 ms)   | SimpleFSDP loss=0.952836 (138.65 ms)
#   Step 2: FSDPv2 loss=0.952740 (156.00 ms)   | SimpleFSDP loss=0.952740 (36.90 ms)
#   Step 3: FSDPv2 loss=0.952636 (154.55 ms)   | SimpleFSDP loss=0.952636 (52.44 ms)
#   Step 4: FSDPv2 loss=0.952552 (157.09 ms)   | SimpleFSDP loss=0.952553 (51.82 ms)
#   ✅ FSDPv2 to SimpleFSDP migration tutorial verified successfully!
#
# Even when FSDPv2 is compiled with `torch.compile(backend="tpu")`, compiled
# SimpleFSDP achieves a ~3.3x steady-state speedup because FSDPv2's runtime
# hooks introduce graph breaks, whereas SimpleFSDP compiles the full forward +
# backward pass into a single whole-step XLA program.


def worker_fn() -> None:
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  logging.info("Worker started: rank=%d, world_size=%d", rank, world_size)

  mesh = dt.DeviceMesh(
      "tpu", torch.arange(world_size), mesh_dim_names=("fsdp",)
  )

  in_features = HIDDEN_DIM
  hidden_features = HIDDEN_DIM
  out_features = HIDDEN_DIM

  # 1. Initialize unsharded base model with deterministic seed, then shard
  #    identical copies using FSDPv2 vs SimpleFSDP.
  torch.manual_seed(RANDOM_SEED)
  base_model = model_lib.FeedForwardBlock(
      in_features=in_features,
      hidden_features=hidden_features,
      out_features=out_features,
  ).to("tpu")

  fsdp2_model = build_fsdp2_model(base_model, mesh)
  simple_fsdp_model = build_simple_fsdp_model(base_model, mesh)

  # 2. Load per-rank training batch on TPU.
  x_all, y_all = make_data(world_size, in_features, out_features)
  x, y = x_all[rank].to("tpu"), y_all[rank].to("tpu")

  # 3. Run compiled FSDPv2 training loop.
  compiled_fsdp2_model = torch.compile(fsdp2_model, backend="tpu")
  fsdp2_opt = optim.SGD(fsdp2_model.parameters(), lr=LR)
  fsdp2_losses = []
  fsdp2_preds = []
  fsdp2_times_ms = []

  for _ in range(NUM_STEPS):
    t0 = time.perf_counter()
    fsdp2_opt.zero_grad()
    y_pred = compiled_fsdp2_model(x)
    loss = nn.MSELoss()(y_pred.float(), y.float())
    loss.backward()
    fsdp2_opt.step()
    fsdp2_losses.append(loss.detach().to("cpu"))
    fsdp2_preds.append(y_pred.detach().to("cpu"))
    fsdp2_times_ms.append((time.perf_counter() - t0) * 1000.0)

  # 4. Run compiled SimpleFSDP training loop.
  compiled_simple_fsdp_model = torch.compile(simple_fsdp_model, backend="tpu")
  simple_fsdp_opt = optim.SGD(simple_fsdp_model.parameters(), lr=LR)
  simple_losses = []
  simple_preds = []
  simple_times_ms = []

  for _ in range(NUM_STEPS):
    t0 = time.perf_counter()
    simple_fsdp_opt.zero_grad()
    loss, y_pred = simple_fsdp_train_step(
        compiled_simple_fsdp_model, x, y, world_size
    )
    simple_fsdp_opt.step()
    simple_losses.append(loss.to("cpu"))
    simple_preds.append(y_pred.to("cpu"))
    simple_times_ms.append((time.perf_counter() - t0) * 1000.0)

  # 5. Verify numerical parity between FSDPv2 and SimpleFSDP across all steps.
  for step_idx in range(NUM_STEPS):
    torch.testing.assert_close(
        simple_losses[step_idx],
        fsdp2_losses[step_idx],
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        simple_preds[step_idx],
        fsdp2_preds[step_idx],
        rtol=1e-2,
        atol=1e-2,
    )

    if rank == 0:
      print(
          f"Step {step_idx}: "
          f"FSDPv2 loss={fsdp2_losses[step_idx].item():.6f} "
          f"({fsdp2_times_ms[step_idx]:.2f} ms) | "
          f"SimpleFSDP loss={simple_losses[step_idx].item():.6f} "
          f"({simple_times_ms[step_idx]:.2f} ms)"
      )

  dist.barrier()
  if rank == 0:
    print("✅ FSDPv2 to SimpleFSDP migration tutorial verified successfully!")
  dist.destroy_process_group()


if __name__ == "__main__":
  worker_fn()
