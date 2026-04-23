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

"""Example of "manual" data parallelism.

This file contains model initializations, training loop, and result validation.
For the model itself, see simple_model.py
"""

import os

from absl import logging
from torch import distributed as dist
from torch import nn
from torch import optim
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.distributed.data_parallel import dp_utils


log_utils.log_to_stderr()

IN_FEATURES = 10
OUT_FEATURES = 3
LR = 0.1
RANDOM_SEED = 3435
PER_RANK_BATCH_SIZE = 16


def worker_fn() -> None:
  """Run one round of backprop and parameter update for a basic model.

  This is first done on process 0, then repeated with the data distributed
  across all processes, to confirm that the resulting weights are the same.
  """
  # Get rank and world size from environment variables.
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  batch_size = world_size * PER_RANK_BATCH_SIZE

  dist.init_process_group(
      backend="tpu_dist",
      rank=rank,
      world_size=world_size,
  )

  # Runs one forward and backward pass on CPU in process 0. The resulting
  # parameters will be used to verify the correctness of data parallel.
  cpu_final_weight = None
  if rank == 0:
    cpu_model = dp_utils.run_cpu_model_training_step(
        batch_size=batch_size,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        lr=LR,
        random_seed=RANDOM_SEED,
    )
    cpu_final_weight = cpu_model.linear1_weight()
    logging.info(
        "CPU rank: %d, final weight 1 (reference): %s",
        rank,
        cpu_final_weight,
    )

  #  Re-creates the model and data with the same initial values.
  model, data, target = dp_utils.make_model_and_data(
      batch_size=batch_size,
      in_features=IN_FEATURES,
      out_features=OUT_FEATURES,
      random_seed=RANDOM_SEED,
  )
  # Put the model on TPU.
  model.to("tpu")

  #  Takes the relevant slice of the training data.
  local_batch_size = batch_size // world_size
  start_idx, stop_idx = local_batch_size * rank, local_batch_size * (rank + 1)
  data = data[start_idx:stop_idx, :].to("tpu")
  target = target[start_idx:stop_idx, :].to("tpu")

  # Runs one forward and backward pass on TPU for sliced data.
  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=LR)
  optimizer.zero_grad()
  output = model(data)

  # Compute the loss and gradients.
  loss = loss_fn(output, target)
  loss.backward()

  # Manually averages the gradients across all processes.
  for param in model.parameters():
    if param.grad is not None:
      dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
      param.grad /= world_size

  optimizer.step()

  tpu_final_weight = model.linear1_weight()
  logging.info(
      "TPU rank: %d, final weight 1 (distributed): %s",
      rank,
      tpu_final_weight,
  )

  # Assert TPU final weight on rank=0 match the reference CPU weight.
  if rank == 0:
    utils.assert_close(
        actual=tpu_final_weight,
        expected=cpu_final_weight,
        rtol=1e-4,
        atol=7e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )

  dist.destroy_process_group()


if __name__ == "__main__":
  worker_fn()
