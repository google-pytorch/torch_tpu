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

import os

from absl import logging
import torch
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.distributed.tensor_parallel_multihost import model

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


log_utils.log_to_stderr()


BATCH_SIZE = 64
MODEL_DIM = 128
RANDOM_SEED = 4242


def fake_dataloader_read():
  return torch.randn(BATCH_SIZE, MODEL_DIM)


def run_worker(rank: int, world_size: int) -> None:
  """Runs a non-distributed and an equivalent tensor-parallel version."""
  device = torch.device('tpu')
  logging.info('tpu_device initialized')

  torch.distributed.init_process_group(
      backend='tpu_dist',
      init_method='env://',
      rank=rank,
      world_size=world_size,
  )
  logging.info('init_process_group done')

  reference_output = None
  if rank == 0:
    torch.manual_seed(RANDOM_SEED)
    mymodel = model.MyModel(dmodel=MODEL_DIM)
    mymodel.eval()
    with torch.no_grad():
      data = fake_dataloader_read()
      reference_output = mymodel(data)
    logging.info('rank: %d, reference_output: %s', rank, reference_output)

  torch.manual_seed(RANDOM_SEED)
  mymodel_tp = model.MyModel(dmodel=MODEL_DIM, use_tp=True)
  mymodel_tp.to(device)
  mymodel_tp.eval()
  with torch.no_grad():
    data = fake_dataloader_read().to(device)
    tpu_tp_output = mymodel_tp(data)
    tpu_tp_output = tpu_tp_output.cpu()
  logging.info('rank: %d, tpu_tp_output: %s', rank, tpu_tp_output)

  # Assert TPU outputs on rank=0 match the reference outputs.
  if rank == 0:
    utils.assert_close(
        actual=tpu_tp_output,
        expected=reference_output,
        rtol=1e-4,
        atol=7e-2,
        check_value=utils.CheckValueMode.LOOSE,
    )
  torch.distributed.destroy_process_group()


def main(_):
  """Runs a multi-host tensor-parallel version on Borg."""
  run_worker(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']))


if __name__ == '__main__':
  g3_multiprocessing.handle_main(main)
