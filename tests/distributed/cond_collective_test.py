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

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def _compiled_mlir_texts(backend):
  texts = []
  for executable in backend._compiled_executables:
    text = getattr(executable, "mlir_text", None)
    if text:
      texts.append(text)
    split_gm = getattr(executable, "_split_gm", None)
    if split_gm is None:
      continue
    for child in split_gm.children():
      text = getattr(getattr(child, "submod", None), "mlir_text", None)
      if text:
        texts.append(text)
  return texts


def _test_wrapper(test_fn, *args, **kwargs):
  dist.init_process_group(backend="tpu_dist")
  try:
    test_fn(*args, **kwargs)
  finally:
    if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()


def run_uniform_predicate_branch_collective() -> None:
  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x, pred):

    def true_fn(v):
      y = v + 1.0
      dist.all_reduce(y)
      return y

    def false_fn(v):
      return v - 100.0

    y = torch.cond(pred, true_fn, false_fn, (x,))
    dist.all_reduce(y)
    return y

  backend = _backend.TpuBackend(debug=True)
  compiled = torch.compile(func, backend=backend, dynamic=False)

  for pred_value in (True, False):
    x = torch.tensor([float(rank + 1)], device="tpu")
    pred = torch.tensor(pred_value, device="tpu")
    actual = compiled(x, pred)

    if pred_value:
      branch_sum = sum(float(r + 2) for r in range(world_size))
      expected = torch.tensor([float(world_size) * branch_sum])
    else:
      false_sum = sum(float(r + 1 - 100) for r in range(world_size))
      expected = torch.tensor([false_sum])
    utils.assert_close(actual.cpu(), expected)

  hlo = "\n".join(_compiled_mlir_texts(backend))
  assert "stablehlo.if" in hlo, hlo
  assert "stablehlo.all_reduce" in hlo, hlo


def run_per_rank_predicate_without_branch_collectives() -> None:
  rank = int(os.environ["RANK"])
  world_size = dist.get_world_size()

  def func(x, pred):

    def true_fn(v):
      return v + 10.0

    def false_fn(v):
      return v - 10.0

    y = torch.cond(pred, true_fn, false_fn, (x,))
    dist.all_reduce(y)
    return y

  backend = _backend.TpuBackend(debug=True)
  compiled = torch.compile(func, backend=backend, dynamic=False)

  x = torch.tensor([float(rank)], device="tpu")
  pred = torch.tensor(rank % 2 == 0, device="tpu")
  actual = compiled(x, pred)

  expected_sum = sum(
      float(r + 10 if r % 2 == 0 else r - 10) for r in range(world_size)
  )
  expected = torch.tensor([expected_sum])
  utils.assert_close(actual.cpu(), expected)

  hlo = "\n".join(_compiled_mlir_texts(backend))
  assert "stablehlo.if" in hlo, hlo
  assert "stablehlo.all_reduce" in hlo, hlo


class CondCollectiveTest(absltest.TestCase):

  def test_uniform_predicate_branch_collective(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=8
        ),
        test_fn=run_uniform_predicate_branch_collective,
    )

  def test_per_rank_predicate_without_branch_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(
            _test_wrapper, world_size=8
        ),
        test_fn=run_per_rank_predicate_without_branch_collectives,
    )


if __name__ == "__main__":
  absltest.main()
