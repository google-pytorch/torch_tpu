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

"""Example of Fully Sharded Data Parallel (FSDP2).

Model parameters and data are sharded across devices.
At each layer, each rank performs an all_gather() to collect the parameters
for that layer. During the backward pass, local gradients are reduce-scattered
into sharded gradients.
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""

from absl import logging
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.distributed import fsdp
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.distributed.fsdp import model


log_utils.log_to_stderr()

CheckValueMode = utils.CheckValueMode

BATCH_SIZE = 3  # Per-device batch size.
LR = 0.1
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
  dist.init_process_group(backend='tpu_dist')
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  logging.info('Worker function started, rank: %d', rank)

  in_features = world_size * 8
  out_features = world_size

  # Initializes the sharded model.
  with torch.device('meta'):
    simple_model = model.SimpleNN(
        in_features=in_features, out_features=out_features
    )

  fsdp.fully_shard(simple_model.linear1)
  fsdp.fully_shard(simple_model.linear2)
  fsdp.fully_shard(simple_model)
  simple_model.to_empty(device='tpu')
  simple_model.linear1.reset_parameters()
  simple_model.linear2.reset_parameters()

  # Creates a CPU copy of the initial model on rank 0.
  single_device_model = None
  weight1 = simple_model.linear1.weight.full_tensor().to('cpu')
  bias1 = simple_model.linear1.bias.full_tensor().to('cpu')
  weight2 = simple_model.linear2.weight.full_tensor().to('cpu')
  bias2 = simple_model.linear2.bias.full_tensor().to('cpu')
  if rank == 0:
    single_device_model = model.SimpleNN(
        in_features=in_features, out_features=out_features
    )
    single_device_model.linear1.weight = nn.Parameter(weight1)
    single_device_model.linear1.bias = nn.Parameter(bias1)
    single_device_model.linear2.weight = nn.Parameter(weight2)
    single_device_model.linear2.bias = nn.Parameter(bias2)

  # Runs the training loop on the sharded model.
  x, y = make_data(world_size, in_features, out_features)
  x, y = x[rank].to('tpu'), y[rank].to('tpu')
  optimizer = optim.SGD(simple_model.parameters(), lr=LR)
  y_pred_history = []

  for _ in range(NUM_STEPS):
    optimizer.zero_grad()
    y_pred = simple_model(x)
    loss = nn.MSELoss()(y_pred, y)
    loss.backward()
    optimizer.step()
    y_pred_history.append(y_pred.to('cpu'))

  # Runs an identical training loop for the unsharded CPU model on rank 0.
  if rank == 0:
    assert single_device_model is not None
    x_ref, y_ref = make_data(world_size, in_features, out_features)
    x_ref = x_ref.reshape(-1, in_features)
    y_ref = y_ref.reshape(-1, out_features)
    optimizer = optim.SGD(single_device_model.parameters(), lr=LR)

    # Verifies that the FSDP model produces the same output as the unsharded
    # CPU model.
    for i in range(NUM_STEPS):
      optimizer.zero_grad()
      y_pred_ref = single_device_model(x_ref)
      loss_ref = nn.MSELoss()(y_pred_ref, y_ref)
      loss_ref.backward()
      optimizer.step()
      utils.assert_close(
          actual=y_pred_ref[:BATCH_SIZE],
          expected=y_pred_history[i],
          check_value=CheckValueMode.LOOSE,
          rtol=1e-3,
          atol=3e-3,
          preamble=f'step {i}',
      )
    logging.info('rank: %d, tpu_tp_output: %s', rank, y_pred_ref)
  dist.destroy_process_group()


if __name__ == '__main__':
  worker_fn()
