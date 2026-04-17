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

"""Example of data parallelism using DDP wrapper.

This file contains model initializations, training loop, and result validation.
For the model itself, see simple_model.py
"""

import sys

from absl import flags
from absl import logging
from torch import distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP  # pylint: disable=g-importing-member
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.distributed.data_parallel import dp_utils


FLAGS = flags.FLAGS

log_utils.log_to_stderr()

flags.DEFINE_bool(
    "broadcast_buffers",
    True,
    "Flag that enables syncing (broadcasting) buffers of the module at "
    "beginning of the forward function.",
)
flags.DEFINE_bool(
    "init_sync",
    True,
    "Flag that controls whether or not to sync parameters during "
    "initialization. If False, user is responsible for ensuring that all "
    "parameters have the same values.",
)
# TODO: b/440076976 - this fails when True.
flags.DEFINE_bool(
    "find_unused_parameters",
    False,
    "Traverse the autograd graph from all tensors contained in the return "
    "value of the wrapped module's forward function. Parameters that don't "
    "receive gradients as part of this graph are preemptively marked as being "
    "ready to be reduced.",
)
flags.DEFINE_bool(
    "gradient_as_bucket_view",
    False,
    "When set to True, gradients will be views pointing to different offsets "
    "of allreduce communication buckets. This can reduce peak memory usage.",
)
# TODO: b/440076976 - this fails when True.
flags.DEFINE_bool(
    "static_graph",
    False,
    "When set to True, DDP knows the trained graph is static, which can "
    "enable some optimizations.",
)

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
  # Initialize TPU and distributed backend.
  dist.init_process_group(backend="tpu_dist")

  # Get rank and world size for torch.distributed.
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  batch_size = world_size * PER_RANK_BATCH_SIZE

  # Runs one forward and backward pass on CPU in process 0. The resulting
  # parameters will be used to verify the correctness of DDP.
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
        "CPU: rank: %d, final weight 1 (reference): %s",
        rank,
        cpu_final_weight,
    )

  model, data, target = dp_utils.make_model_and_data(
      batch_size=batch_size,
      in_features=IN_FEATURES,
      out_features=OUT_FEATURES,
      random_seed=RANDOM_SEED,
  )

  # Put the model on TPU and wrap it with DDP.
  model.to("tpu")
  ddp_model = DDP(
      model,
      broadcast_buffers=FLAGS.broadcast_buffers,
      init_sync=FLAGS.init_sync,
      find_unused_parameters=FLAGS.find_unused_parameters,
      gradient_as_bucket_view=FLAGS.gradient_as_bucket_view,
      static_graph=FLAGS.static_graph,
  )

  #  Take the relevant slice of the training data.
  local_batch_size = batch_size // world_size
  start_idx, stop_idx = local_batch_size * rank, local_batch_size * (rank + 1)
  data = data[start_idx:stop_idx, :].to("tpu")
  target = target[start_idx:stop_idx, :].to("tpu")

  # Runs one forward and backward pass on TPU for sliced data.
  optimizer = optim.SGD(ddp_model.parameters(), lr=LR)
  dp_utils.run_training_step(
      model=ddp_model, data=data, target=target, optimizer=optimizer
  )

  tpu_final_weight = model.linear1_weight()
  logging.info(
      "TPU: rank: %d, final weight 1 (distributed): %s",
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
  FLAGS(sys.argv)
  worker_fn()
