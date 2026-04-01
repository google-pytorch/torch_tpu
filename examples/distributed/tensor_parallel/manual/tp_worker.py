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

"""Example of "manual" tensor parallelism.

This file contains model initializations, inference loop, and result validation.
For the model itself, including the tensor-parallelism logic, see model.py.
"""

import logging
import os
import sys

import torch
from torch import distributed as dist
from torch_tpu import api
from torch_tpu._internal.utils import utils
from examples.distributed.tensor_parallel.manual import model

# Direct all logs to stdout so kubectl logs can see them
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 64
MODEL_DIM = 256

RANDOM_SEED = 4242


def fake_dataloader_read():
  return torch.randn(BATCH_SIZE, MODEL_DIM)


def worker_fn() -> None:
  """Runs a non-distributed and an equivalent tensor-parallel versions.

  We rely on torch.manual_seed() to seed the random dataloader and model
  initializations, so that we get the same outputs in both cases.
  """
  rank = int(os.environ.get("RANK", -1))
  local_rank = int(os.environ.get("LOCAL_RANK", -1))

  _ = api.tpu_device()

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

  # Run on multiple TPUs using Tensor Parallelism.
  torch.manual_seed(RANDOM_SEED)
  mymodel_tp = model.MyModel(dmodel=MODEL_DIM, use_tp=True)
  mymodel_tp.to(device="tpu")
  mymodel_tp.eval()

  import time

  with torch.no_grad():
    data = fake_dataloader_read().to(device="tpu")

    start_time = time.time()
    tpu_tp_output = mymodel_tp(data)
    tpu_tp_output = tpu_tp_output.cpu()
    step_time = time.time() - start_time

  logger.info(
      "rank: %d, step_time: %f s, tpu_tp_output: %s",
      rank,
      step_time,
      tpu_tp_output,
  )

  # Assert TPU outputs on rank=0 match the reference outputs.
  if rank == 0:
    logger.info("Comparing TPU output with reference output...")
    utils.assert_close(
        actual=tpu_tp_output,
        expected=reference_output,
        rtol=1e-4,
        atol=7e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )
    logger.info("Comparison successful.")

  logger.info("Destroying process group...")
  dist.destroy_process_group()
  logger.info("Worker function finished.")


if __name__ == "__main__":
  worker_fn()
