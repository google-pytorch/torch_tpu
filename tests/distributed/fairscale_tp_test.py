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

"""Tests fairscale tensor parallelism.

Uses torch_tpu/examples/distributed/tensor_parallel/model.py as a guide.
"""

import os
from absl import logging
from absl.testing import absltest
from fairscale.nn.model_parallel import initialize
from fairscale.nn.model_parallel import layers
import torch
from torch import distributed as dist
from torch import nn
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils
from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

RowParallelLinear = layers.RowParallelLinear
ColumnParallelLinear = layers.ColumnParallelLinear
RANDOM_SEED = 42
WORLD_SIZE = 8
BATCH_SIZE = 64
MODEL_DIM = 128


def fake_dataloader_read():
  return torch.randn(BATCH_SIZE, MODEL_DIM)


class MyModel(nn.Module):
  """Simple example model with optional fairscale tensor parallelism."""

  def __init__(self, dmodel, use_tp=False):
    super().__init__()
    self.ff1 = FeedForwardTP(dmodel, use_tp)
    self.ff2 = FeedForwardTP(dmodel, use_tp)
    self.ff3 = FeedForwardTP(dmodel, use_tp)

  def forward(self, x):
    x = self.ff1(x)
    x = self.ff2(x)
    x = self.ff3(x)
    return x


class FeedForwardTP(nn.Module):
  """Simple example model with fairscale tensor parallelism."""

  def __init__(self, dmodel, use_tp=False):
    super().__init__()
    if use_tp:
      self.w1 = ColumnParallelLinear(
          dmodel, 4 * dmodel, gather_output=False
      )  # Keep sharded
      self.w2 = RowParallelLinear(4 * dmodel, dmodel, input_is_parallel=True)
    else:
      self.w1 = torch.nn.Linear(dmodel, 4 * dmodel)
      self.w2 = torch.nn.Linear(4 * dmodel, dmodel)
    self.act = nn.SiLU()

  def forward(self, x):
    x = self.w1(x)
    x = self.act(x)
    x = self.w2(x)
    return x


def _copy_weights_to_tp_model(
    reference_model: nn.Module,
    tp_model: nn.Module,
    world_size: int,
    rank: int,
) -> None:
  """Copies weights from a regular model to a fairscale tensor-parallel model."""
  for (_, ref_module), (_, tp_module) in zip(
      reference_model.named_modules(), tp_model.named_modules()
  ):
    if isinstance(tp_module, ColumnParallelLinear):
      # Sharded along dimension 0 for weights and bias.
      chunk_size = ref_module.weight.shape[0] // world_size
      start = rank * chunk_size
      end = start + chunk_size
      tp_module.weight.data = ref_module.weight.data[start:end, :].to(
          tp_module.weight.data.device
      )
      if ref_module.bias is not None:
        tp_module.bias.data = ref_module.bias.data[start:end].to(
            tp_module.weight.data.device
        )
    elif isinstance(tp_module, RowParallelLinear):
      # Sharded along dimension 1 for weights. Bias is not sharded.
      chunk_size = ref_module.weight.shape[1] // world_size
      start = rank * chunk_size
      end = start + chunk_size
      tp_module.weight.data = ref_module.weight.data[:, start:end].to(
          tp_module.weight.data.device
      )
      if ref_module.bias is not None:
        tp_module.bias.data = ref_module.bias.data.to(
            tp_module.weight.data.device
        )
  logging.info("Successfully copied weights to TP model on rank %d.", rank)


def run_forward_tp(device="tpu") -> None:
  """Runs a non-distributed and an equivalent tensor-parallel versions.

  A fixed random seed ensures an identical dataset. However, model weights must
  be copied manually because the Tensor Parallel (TP) version using FairScale.

  Args:
    device: The device to run the tensor parallel model on (e.g., "tpu",
      "cuda").
  """
  dist.init_process_group(backend="tpu_dist")

  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  # Initialize non-TP model on CPU for reference.
  torch.manual_seed(RANDOM_SEED)
  model = MyModel(dmodel=MODEL_DIM, use_tp=False)
  model.eval()

  # Run the non-distributed variant on a single CPU, for reference output.
  reference_output = None
  if rank == 0:
    with torch.no_grad():
      torch.manual_seed(RANDOM_SEED)
      data = fake_dataloader_read()
      reference_output = model(data.clone())

  # Run on multiple devices using Tensor Parallelism.
  if not initialize.model_parallel_is_initialized():
    initialize.initialize_model_parallel(world_size)

  torch.manual_seed(RANDOM_SEED)
  mymodel_tp = MyModel(dmodel=MODEL_DIM, use_tp=True)
  mymodel_tp.to(device=device)

  # Copy weights from reference_model to mymodel_tp, handling sharding.
  _copy_weights_to_tp_model(model, mymodel_tp, world_size, rank)

  mymodel_tp.eval()
  with torch.no_grad():
    torch.manual_seed(RANDOM_SEED)
    data_tp = fake_dataloader_read().to(device=device)
    tp_output = mymodel_tp(data_tp)
    tp_output = tp_output.cpu()

  # Assert TPU outputs on rank=0 match the reference outputs.
  if rank == 0:
    utils.assert_close(tp_output, reference_output, atol=1e-4, rtol=1e-4)


class FairscaleTest(absltest.TestCase):

  def test_forward_fairscale_tp_against_non_tp(self):
    logging.info("Running test on %d TPUs.", WORLD_SIZE)
    distributed_utils.dist_run(
        nproc_per_node=WORLD_SIZE,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_forward_tp, world_size=WORLD_SIZE
        ),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
