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

"""Example of tensor parallelism using PyTorch's native `parallelize_module` wrappers.

This file leverages the `MyModel` definition from the parent directory
(which is a standard sequential PyTorch model) and wraps its linear
layers automatically across TPU chunks without manually rewriting the
model code.
"""

import os

from absl import logging
import torch
from torch import distributed as dist
from torch.distributed import device_mesh
from torch.distributed import tensor
from torch.distributed.tensor import parallel
# Direct all logs to stdout so kubectl logs can see them
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.distributed.tensor_parallel.dtensor_wrapped import model


log_utils.log_to_stderr()

BATCH_SIZE = 64
MODEL_DIM = 128

RANDOM_SEED = 4242


def fake_dataloader_read():
  return torch.randn(BATCH_SIZE, MODEL_DIM)


def worker_fn() -> None:
  """Runs a non-distributed and an equivalent wrapper-based tensor-parallel version.

  We rely on torch.manual_seed() to seed the random dataloader and model
  initializations, so that we get the same outputs in both cases.
  """
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  logging.info("Worker function started, rank: %d", rank)

  dist.init_process_group(backend="tpu_dist")

  # Run the non-distributed variant on a single CPU, for reference output.
  reference_output = None
  if rank == 0:
    torch.manual_seed(RANDOM_SEED)
    mymodel = model.MyModel(dmodel=MODEL_DIM)
    mymodel.eval()
    with torch.no_grad():
      data = fake_dataloader_read()
      reference_output = mymodel(data)
    logging.info("CPU rank: %d, reference_output: %s", rank, reference_output)

  # Run on multiple TPUs using Tensor Parallelism via Wrappers.
  torch.manual_seed(RANDOM_SEED)

  # Notice that we use the PyTorch-native model.
  mymodel_tp = model.MyModel(dmodel=MODEL_DIM)
  mymodel_tp.to(device="tpu")
  mymodel_tp.eval()

  device_mesh_obj = device_mesh.init_device_mesh("tpu", (world_size,))

  # Applying TP plan using PyTorch wrappers
  tp_plan = {
      # Showcasing explicit input staging: Convert local tensor -> Replicated DTensor natively
      "ff1": parallel.PrepareModuleInput(
          input_layouts=(tensor.Replicate(),),
          desired_input_layouts=(tensor.Replicate(),),
      ),
      "ff1.w1": parallel.ColwiseParallel(),
      "ff1.w2": parallel.RowwiseParallel(),
      "ff2.w1": parallel.ColwiseParallel(),
      "ff2.w2": parallel.RowwiseParallel(),
      "ff3.w1": parallel.ColwiseParallel(),
      "ff3.w2": parallel.RowwiseParallel(),
      # Showcasing explicit output staging: Convert DTensor -> Local Tensor natively
      "ff3": parallel.PrepareModuleOutput(
          output_layouts=tensor.Replicate(),
          desired_output_layouts=tensor.Replicate(),
          use_local_output=True,
      ),
  }

  mymodel_tp = parallel.parallelize_module(mymodel_tp, device_mesh_obj, tp_plan)

  with torch.no_grad():
    data = fake_dataloader_read().to(device="tpu")
    tpu_tp_output = mymodel_tp(data)

    # We resolve the local tensor in case parallelize_module or `+` ops carried the DTensor status up the stack
    if isinstance(tpu_tp_output, torch.Tensor) and not hasattr(
        tpu_tp_output, "_local_tensor"
    ):
      tpu_tp_output = tpu_tp_output.cpu()
    else:
      # PyTorch DTensors use the to_local() method to drop the proxy and get the inner tensor.
      tpu_tp_output = tpu_tp_output.to_local().cpu()

  logging.info("TPU rank: %d, tpu_tp_output: %s", rank, tpu_tp_output)

  # Assert TPU outputs on rank=0 match the reference outputs.
  if rank == 0:
    utils.assert_close(
        actual=tpu_tp_output,
        expected=reference_output,
        rtol=1e-4,
        atol=7e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )

  dist.destroy_process_group()


if __name__ == "__main__":
  worker_fn()
