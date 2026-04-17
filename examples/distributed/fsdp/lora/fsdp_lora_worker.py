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

"""Example of Fully Sharded Data Parallel (FSDP2) with PEFT LoRA.

Model parameters and data are sharded across devices.
At each layer, each rank performs an all_gather() to collect the parameters
for that layer. During the backward pass, local gradients are reduce-scattered
into sharded gradients.
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""

import os
import tempfile

from absl import logging
import peft
from peft.utils.constants import WEIGHTS_NAME
from peft.utils.save_and_load import get_peft_model_state_dict
import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.distributed import fsdp
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.distributed.fsdp import fsdp_worker as fsdp_tp_worker
from examples.distributed.fsdp import model


log_utils.log_to_stderr()


def worker_fn() -> None:
  dist.init_process_group(backend='tpu_dist')

  rank = dist.get_rank()
  world_size = dist.get_world_size()

  in_features = world_size * 8
  out_features = world_size

  # Initializes the sharded model.
  with torch.device('meta'):
    simple_model = model.SimpleNN(
        in_features=in_features,
        out_features=out_features,
    )

  peft_config = peft.LoraConfig(
      r=4,
      lora_alpha=8,
      target_modules=['linear1', 'linear2'],
      lora_dropout=0.0,
      bias='none',
  )
  simple_model = peft.get_peft_model(simple_model, peft_config)

  fsdp.fully_shard(simple_model.linear1)
  fsdp.fully_shard(simple_model.linear2)
  fsdp.fully_shard(simple_model)
  simple_model.to_empty(device='tpu')
  simple_model.linear1.base_layer.reset_parameters()
  simple_model.linear2.base_layer.reset_parameters()
  simple_model.linear1.reset_lora_parameters('default', init_lora_weights=True)
  simple_model.linear2.reset_lora_parameters('default', init_lora_weights=True)

  # Check sharding and trainable parameters.
  for name, param in simple_model.named_parameters():
    is_sharded = param.numel() != param.to_local().numel()
    if world_size > 1:
      assert is_sharded, f'Parameter {name} should be sharded.'
    else:
      assert not is_sharded, f'Parameter {name} should NOT be sharded.'

    if 'lora_' in name:
      assert param.requires_grad, f'LoRA parameter {name} should be trainable.'
    elif 'base_layer' in name:
      assert (
          not param.requires_grad
      ), f'Base parameter {name} should NOT be trainable.'

  # Creates a CPU copy of the initial model on rank 0.
  single_device_model = None
  weight1 = simple_model.linear1.base_layer.weight.full_tensor().to('cpu')
  bias1 = simple_model.linear1.base_layer.bias.full_tensor().to('cpu')
  lora_a1 = (
      simple_model.linear1.lora_A['default'].weight.full_tensor().to('cpu')
  )
  lora_b1 = (
      simple_model.linear1.lora_B['default'].weight.full_tensor().to('cpu')
  )
  weight2 = simple_model.linear2.base_layer.weight.full_tensor().to('cpu')
  bias2 = simple_model.linear2.base_layer.bias.full_tensor().to('cpu')
  lora_a2 = (
      simple_model.linear2.lora_A['default'].weight.full_tensor().to('cpu')
  )
  lora_b2 = (
      simple_model.linear2.lora_B['default'].weight.full_tensor().to('cpu')
  )
  if rank == 0:
    base_model = model.SimpleNN(
        in_features=in_features,
        out_features=out_features,
    )
    base_model.linear1.weight = nn.Parameter(weight1)
    base_model.linear1.bias = nn.Parameter(bias1)
    base_model.linear2.weight = nn.Parameter(weight2)
    base_model.linear2.bias = nn.Parameter(bias2)
    single_device_model = peft.get_peft_model(base_model, peft_config)
    single_device_model.linear1.lora_A['default'].weight = nn.Parameter(lora_a1)
    single_device_model.linear1.lora_B['default'].weight = nn.Parameter(lora_b1)
    single_device_model.linear2.lora_A['default'].weight = nn.Parameter(lora_a2)
    single_device_model.linear2.lora_B['default'].weight = nn.Parameter(lora_b2)

  # Runs the training loop on the sharded model.
  x, y = fsdp_tp_worker.make_data(world_size, in_features, out_features)
  x, y = x[rank].to('tpu'), y[rank].to('tpu')
  optimizer = optim.SGD(simple_model.parameters(), lr=fsdp_tp_worker.LR)
  y_pred_history = []

  for _ in range(fsdp_tp_worker.NUM_STEPS):
    optimizer.zero_grad()
    y_pred = simple_model(x)
    loss = nn.MSELoss()(y_pred, y)
    loss.backward()
    optimizer.step()
    y_pred_history.append(y_pred.to('cpu'))

  # Saving LoRA checkpoint
  output_dir = os.path.join(
      os.getenv('TEST_TMPDIR', tempfile.gettempdir()), 'lora_output'
  )
  full_state_dict_config = FullStateDictConfig(
      offload_to_cpu=False, rank0_only=True
  )
  FSDP.set_state_dict_type(
      simple_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
  )
  state_dict = simple_model.state_dict()
  if dist.get_rank() == 0:
    # state_dict() gathers tensors to rank0 TPU since offload_to_cpu=False.
    peft_state_dict_tpu = get_peft_model_state_dict(
        simple_model, state_dict=state_dict
    )

    assert len(peft_state_dict_tpu) > 0, 'PEFT state dict is empty!'
    assert all(
        'lora' in k for k in peft_state_dict_tpu
    ), 'Non-LoRA keys found in PEFT state dict.'
    assert all(
        'base_layer' not in k for k in peft_state_dict_tpu
    ), 'Base layer keys found in PEFT state dict.'

    peft_state_dict_cpu = {
        k: v.to('cpu').clone().contiguous()
        for k, v in peft_state_dict_tpu.items()
    }

    simple_model.peft_config['default'].save_pretrained(output_dir)
    torch.save(peft_state_dict_cpu, os.path.join(output_dir, WEIGHTS_NAME))

  # Runs an identical training loop for the unsharded CPU model on rank 0.
  if rank == 0:
    assert (
        single_device_model is not None
    ), 'single_device_model should be initialized on rank 0.'
    x_ref, y_ref = fsdp_tp_worker.make_data(
        world_size, in_features, out_features
    )
    x_ref = x_ref.reshape(-1, in_features)
    y_ref = y_ref.reshape(-1, out_features)
    optimizer = optim.SGD(
        single_device_model.parameters(), lr=fsdp_tp_worker.LR
    )

    # Verifies that the FSDP model produces the same output as the unsharded
    # CPU model.
    for i in range(fsdp_tp_worker.NUM_STEPS):
      optimizer.zero_grad()
      y_pred_ref = single_device_model(x_ref)
      loss_ref = nn.MSELoss()(y_pred_ref, y_ref)
      loss_ref.backward()
      optimizer.step()
      utils.assert_close(
          actual=y_pred_ref[: fsdp_tp_worker.BATCH_SIZE],
          expected=y_pred_history[i],
          check_value=utils.CheckValueMode.LOOSE,
          rtol=1e-3,
          atol=3e-3,
          preamble=f'step {i}',
      )
      logging.info('rank: %d, tpu_tp_output: %s', rank, y_pred_ref)
    dist.destroy_process_group()


if __name__ == '__main__':
  worker_fn()
