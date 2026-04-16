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

"""Tests for Distributed Checkpointing (DCP) on TPU.

These are basic smoke tests to verify that torch_tpu works correctly with
PyTorch's upstream Distributed Checkpointing library. They are designed to
catch regressions early on by ensuring the happy path is functional.
"""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch import nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import state_dict
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_IN_FEATURES = 64
_OUT_FEATURES = 64
_BATCH_SIZE = 4
_LEARNING_RATE = 0.001


def _create_sharded_model(device_mesh, seed):
  torch.manual_seed(seed)
  model = nn.Linear(_IN_FEATURES, _OUT_FEATURES)

  sharded_weight = dt.distribute_tensor(
      model.weight.detach(), device_mesh=device_mesh, placements=[dt.Shard(0)]
  )
  model.weight = nn.Parameter(sharded_weight)

  sharded_bias = dt.distribute_tensor(
      model.bias.detach(), device_mesh=device_mesh, placements=[dt.Shard(0)]
  )
  model.bias = nn.Parameter(sharded_bias)

  return model.to("tpu")


def _init_test_env(seed):
  rank = int(os.environ["RANK"])
  dist.init_process_group(backend="tpu_dist")
  world_size = dist.get_world_size()
  device_mesh = dt.init_device_mesh("tpu", (world_size,))
  model = _create_sharded_model(device_mesh, seed)
  device = torch.device("tpu", rank)
  return model, device, device_mesh


def _warmup_model(model, device, device_mesh):
  local_x = torch.ones(
      _BATCH_SIZE, _IN_FEATURES, dtype=torch.float32, device=device
  )
  x = dt.DTensor.from_local(local_x, device_mesh, [dt.Replicate()])
  model(x)
  dist.barrier()


def _cleanup_test_env():
  dist.barrier()
  dist.destroy_process_group()


def run_dtensor_dcp_load_save(checkpoint_dir: str) -> None:
  """Tests DCP with DTensor using the save and load API."""
  model, device, device_mesh = _init_test_env(seed=42)
  _warmup_model(model, device, device_mesh)

  # Create a checkpoint
  writer = dcp.FileSystemWriter(checkpoint_dir)
  dcp.save({"model": model}, storage_writer=writer)

  # Create a new model with a different seed to ensure it is initialized
  # differently
  model_new = _create_sharded_model(device_mesh, seed=43)

  # Assert that the model parameters are different before load
  for param, param_new in zip(model.parameters(), model_new.parameters()):
    assert not torch.allclose(param.to_local(), param_new.to_local())

  # Restore the checkpoint
  reader = dcp.FileSystemReader(checkpoint_dir)
  dcp.load({"model": model_new}, storage_reader=reader)

  # Assert that the model parameters are the same
  for param, param_new in zip(model.parameters(), model_new.parameters()):
    utils.assert_close(param.to_local(), param_new.to_local())

  _cleanup_test_env()


def run_dtensor_dcp_state_dict_optimizer(checkpoint_dir: str) -> None:
  """Tests DCP with DTensor using the state_dict API for model and optimizer."""
  model, device, device_mesh = _init_test_env(seed=42)

  optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)

  # Run a step to populate optimizer state
  local_x = torch.ones(
      _BATCH_SIZE, _IN_FEATURES, dtype=torch.float32, device=device
  )
  x = dt.DTensor.from_local(local_x, device_mesh, [dt.Replicate()])
  out = model(x)
  loss = out.sum()
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  dist.barrier()

  # Create a checkpoint
  model_state, optim_state = state_dict.get_state_dict(model, optimizer)
  writer = dcp.FileSystemWriter(checkpoint_dir)
  dcp.save(
      {"model": model_state, "optimizer": optim_state}, storage_writer=writer
  )

  # Create a new model with a different seed to ensure it is initialized
  # differently
  model_new = _create_sharded_model(device_mesh, seed=43)
  optimizer_new = torch.optim.Adam(model_new.parameters(), lr=_LEARNING_RATE)

  # Assert that the model parameters are different before load
  for param, param_new in zip(model.parameters(), model_new.parameters()):
    assert not torch.allclose(param.to_local(), param_new.to_local())

  # Restore the checkpoint
  model_state_new, optim_state_new = state_dict.get_state_dict(
      model_new, optimizer_new
  )
  reader = dcp.FileSystemReader(checkpoint_dir)
  dcp.load(
      {"model": model_state_new, "optimizer": optim_state_new},
      storage_reader=reader,
  )

  state_dict.set_state_dict(
      model_new,
      optimizer_new,
      model_state_dict=model_state_new,
      optim_state_dict=optim_state_new,
  )

  # Assert that the model parameters are the same after load
  for param, param_new in zip(model.parameters(), model_new.parameters()):
    utils.assert_close(param.to_local(), param_new.to_local())

  # Take another step with both models using new input to verify that the
  # optimizer state was restored correctly.
  local_x_next = (
      torch.ones(_BATCH_SIZE, _IN_FEATURES, dtype=torch.float32, device=device)
      * 2
  )
  x_next = dt.DTensor.from_local(local_x_next, device_mesh, [dt.Replicate()])

  out_orig = model(x_next)
  loss_orig = out_orig.sum()
  loss_orig.backward()
  optimizer.step()
  optimizer.zero_grad()

  out_new = model_new(x_next)
  loss_new = out_new.sum()
  loss_new.backward()
  optimizer_new.step()
  optimizer_new.zero_grad()

  # Assert that the model parameters are still the same after another step
  for param, param_new in zip(model.parameters(), model_new.parameters()):
    utils.assert_close(param.to_local(), param_new.to_local())

  _cleanup_test_env()


class DCPTest(absltest.TestCase):
  _world_size = 8

  def test_dtensor_dcp_load_save(self):
    checkpoint_dir = self.create_tempdir().full_path
    distributed_utils.dist_run(
        self._world_size,
        singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_dcp_load_save, world_size=self._world_size
        ),
        checkpoint_dir=checkpoint_dir,
    )

  def test_dtensor_dcp_state_dict_optimizer(self):
    checkpoint_dir = self.create_tempdir().full_path
    distributed_utils.dist_run(
        self._world_size,
        singlehost_wrapper.tpu_env_wrapper(
            run_dtensor_dcp_state_dict_optimizer, world_size=self._world_size
        ),
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
