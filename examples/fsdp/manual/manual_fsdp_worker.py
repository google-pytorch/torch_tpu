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

"""Example of manual Fully Sharded Data Parallel (FSDP).

Model parameters and data are sharded across devices.
At each layer, each rank performs an all_gather() to collect the parameters
for that layer. During the backward pass, local gradients are reduce-scattered
into sharded gradients.
"""

from absl import logging
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
import torch.distributed._functional_collectives as _func_col
from examples.fsdp.manual import model
from examples.fsdp.manual import simple_model as simple_model_lib

logging.set_verbosity(logging.INFO)

BATCH_SIZE = 16  # Per-device batch size.
LR = 0.01
NUM_STEPS = 10
RANDOM_SEED = 0


def make_data(
    world_size: int, in_features: int, out_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Makes reproducible training data for all ranks."""
  torch.manual_seed(RANDOM_SEED)
  x = torch.randn(world_size, BATCH_SIZE, in_features)
  y = torch.randn(world_size, BATCH_SIZE, out_features)
  return x, y


def worker_fn() -> None:
  dist.init_process_group(backend="tpu_dist")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  torch.set_default_dtype(torch.bfloat16)

  in_features = 4096
  out_features = 8192

  # Initializes the sharded model directly on TPU device.
  torch.manual_seed(RANDOM_SEED + rank)
  with torch.device("tpu"):
    sharded_model = model.ManualSimpleNN(
        in_features=in_features, out_features=out_features
    )

  torch.accelerator.synchronize()

  # Creates a CPU copy of the initial model on rank 0. This is used for
  # verifying the results of the FSDP model. It is not necessary for FSDP.
  single_device_model = None
  weight1 = sharded_model.linear1.get_full_weight().to("cpu")
  bias1 = sharded_model.linear1.bias.to("cpu")
  weight2 = sharded_model.linear2.get_full_weight().to("cpu")
  bias2 = sharded_model.linear2.bias.to("cpu")
  if rank == 0:
    single_device_model = simple_model_lib.SimpleNN(
        in_features=in_features, out_features=out_features
    )
    single_device_model.linear1.weight = nn.Parameter(weight1)
    single_device_model.linear1.bias = nn.Parameter(bias1)
    single_device_model.linear2.weight = nn.Parameter(weight2)
    single_device_model.linear2.bias = nn.Parameter(bias2)

  optimizer = optim.SGD(sharded_model.parameters(), lr=LR)

  def fwd_and_bwd(
      model_to_run: nn.Module, data: torch.Tensor, y_true: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.fx.traceback.annotate({"spmd_safe": True}):
      y_pred = model_to_run(data)
      loss = nn.functional.mse_loss(y_pred, y_true) / world_size
      loss.backward()
      # Unsharded parameters (biases) are replicated across ranks, so their
      # gradients are summed across ranks via all_reduce.
      for m in model_to_run.modules():
        if isinstance(m, model.ManualFSDPLinear) and m.bias.grad is not None:
          m.bias.grad = _func_col.all_reduce(m.bias.grad, "sum", group=m.group)
      return loss.detach(), y_pred.detach()

  # Compiles the fused forward and backward step with torch.compile.
  compiled_fwd_and_bwd = torch.compile(
      fwd_and_bwd, backend="tpu", fullgraph=True
  )

  # Runs the training loop on the sharded model.
  x, y = make_data(world_size, in_features, out_features)
  x, y = x[rank].to("tpu"), y[rank].to("tpu")
  y_pred_history = []
  grad_history = []

  for _ in range(NUM_STEPS):
    optimizer.zero_grad()
    loss, y_pred = compiled_fwd_and_bwd(sharded_model, x, y)
    del loss
    grad_history.append([
        p.grad.to("cpu")
        for p in sharded_model.parameters()
        if p.grad is not None
    ])
    optimizer.step()
    y_pred_history.append(y_pred.to("cpu"))

  # Runs an identical training loop for the unsharded CPU model on rank 0.
  if rank == 0:
    assert single_device_model is not None
    single_device_model = single_device_model.to(torch.float32)
    x_ref, y_ref = make_data(world_size, in_features, out_features)
    x_ref = x_ref.reshape(-1, in_features).to(torch.float32)
    y_ref = y_ref.reshape(-1, out_features).to(torch.float32)
    optimizer = optim.SGD(single_device_model.parameters(), lr=LR)

    # Verifies that the FSDP model produces the same output and gradients as
    # the unsharded CPU model.
    for i in range(NUM_STEPS):
      optimizer.zero_grad()
      y_pred_ref = single_device_model(x_ref)
      loss_ref = nn.MSELoss()(y_pred_ref, y_ref)
      loss_ref.backward()
      for p_ref, g_local in zip(
          single_device_model.parameters(), grad_history[i]
      ):
        assert p_ref.grad is not None
        torch.testing.assert_close(
            actual=p_ref.grad[: g_local.shape[0]],
            expected=g_local.to(torch.float32),
            rtol=1e-2,
            atol=2e-2,
            msg=f"step {i} grad",
        )
      optimizer.step()
      torch.testing.assert_close(
          actual=y_pred_ref[:BATCH_SIZE],
          expected=y_pred_history[i].to(torch.float32),
          rtol=1e-2,
          atol=2e-2,
          msg=f"step {i} y_pred",
      )
    logging.info("rank: %d, tpu_tp_output: %s", rank, y_pred_ref)
  dist.destroy_process_group()


if __name__ == "__main__":
  worker_fn()
