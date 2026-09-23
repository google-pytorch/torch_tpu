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
from torch.distributed.checkpoint import staging
from torch.distributed.checkpoint import state_dict
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

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


def _init_test_env(seed: int, enable_cpu_backend: bool = False):
  rank = int(os.environ["RANK"])
  backend = "cpu:gloo,tpu:tpu_dist" if enable_cpu_backend else "tpu_dist"
  dist.init_process_group(backend=backend)
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


def run_dtensor_dcp_save_load(checkpoint_dir: str) -> None:
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


def run_dtensor_dcp_async_save_load(checkpoint_dir: str) -> None:
  """Tests DCP with DTensor using the async_save and load API."""
  model, device, device_mesh = _init_test_env(seed=42, enable_cpu_backend=True)
  _warmup_model(model, device, device_mesh)

  # Capture the original parameter values
  with torch.no_grad():
    original_weights = [
        param.to_local().clone() for param in model.parameters()
    ]

  # Start an async checkpoint, which will consists of two phases:
  # 1. Staging: The model parameters are copied to CPU memory
  # 2. Upload: The staged parameters are copied to the final storage
  stager = staging.DefaultStager(staging.StagingOptions(use_async_staging=True))
  writer = dcp.FileSystemWriter(checkpoint_dir)
  save_response = dcp.async_save(
      {"model": model},
      storage_writer=writer,
      async_stager=stager,
  )

  # Wait for the staging phase to complete by waiting on the future
  save_response.staging_completion.result()

  # Mutate model parameters
  with torch.no_grad():
    for param in model.parameters():
      param.to_local().add_(1.0)

  # Wait for the upload phase to complete by waiting on the future
  save_response.upload_completion.result()
  stager.close()

  # Create a new model with a different seed to ensure it is initialized
  # differently
  model_new = _create_sharded_model(device_mesh, seed=43)

  # Assert that the model parameters are different before load
  for orig_w, param_new in zip(original_weights, model_new.parameters()):
    assert not torch.allclose(orig_w, param_new.to_local())

  # Restore the checkpoint
  reader = dcp.FileSystemReader(checkpoint_dir)
  dcp.load({"model": model_new}, storage_reader=reader)

  # Assert that the model parameters are the same as original weights. The
  # intermediate mutation is expected to be discarded.
  for orig_w, param_new in zip(original_weights, model_new.parameters()):
    utils.assert_close(orig_w, param_new.to_local())

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


class DCPTest(seed_test_utils.MultiProcessRepeatableTest):

  _world_size = 8

  def test_dtensor_dcp_save_load(self):
    checkpoint_dir = self.create_tempdir().full_path
    distributed_utils.dist_run(
        self._world_size,
        run_dtensor_dcp_save_load,
        checkpoint_dir=checkpoint_dir,
    )

  def test_dtensor_dcp_async_save_load(self):
    checkpoint_dir = self.create_tempdir().full_path
    distributed_utils.dist_run(
        self._world_size,
        run_dtensor_dcp_async_save_load,
        checkpoint_dir=checkpoint_dir,
    )

  def test_dtensor_dcp_state_dict_optimizer(self):
    checkpoint_dir = self.create_tempdir().full_path
    distributed_utils.dist_run(
        self._world_size,
        run_dtensor_dcp_state_dict_optimizer,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
