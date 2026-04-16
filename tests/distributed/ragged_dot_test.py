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

"""Tests for ragged_dot in a distributed environment on TPU."""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal import compile as tt_compile
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import utils
from tests.distributed import distributed_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


def manual_ragged_dot(x, w, gs):
  """Reference implementation of ragged_dot."""
  out = torch.zeros(x.shape[0], w.shape[-1])
  idx = 0
  for i, g in enumerate(gs):
    if g == 0:
      continue
    out[idx : (idx + g), :] = x[idx : (idx + g), :] @ w[i]
    idx += g
  return out


def run_ragged_dot_local_test(compile_test: bool = False) -> None:
  """Runs ragged_dot independently on each rank."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])

  # Different shapes/inputs for each rank based on rank.
  b_fused = (rank + 1) * 8
  d_m = 16
  d_f = 32
  num_experts = 4

  torch.manual_seed(42 + rank)
  h = torch.randn(b_fused, d_m, device="tpu")
  weights = torch.randn(num_experts, d_m, d_f, device="tpu")

  # Distribute h across experts.
  # Simple case: each expert gets 1/num_experts of the tokens.
  gs_list = [b_fused // num_experts] * num_experts
  group_sizes = torch.tensor(gs_list, dtype=torch.int32, device="tpu")

  def op_fn(x, w, gs):
    return torch.ops.torch_tpu.ragged_dot(x, w, gs)

  if compile_test:
    backend = tt_compile.TpuBackend()
    compiled_op = torch.compile(op_fn, backend=backend)
    out = compiled_op(h, weights, group_sizes)
  else:
    out = op_fn(h, weights, group_sizes)

  # Compare against CPU reference
  expected = manual_ragged_dot(
      h.cpu(), weights.cpu(), group_sizes.cpu().tolist()
  )
  utils.assert_close(
      out.cpu(),
      expected,
      atol=5e-2,
      rtol=1e-1,
      check_value=utils.CheckValueMode.LOOSE,
  )

  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


class RaggedDotDistributedTest(absltest.TestCase):
  """Tests for ragged_dot in a distributed environment on TPU.

  Note that this test is not testing the distributed version of ragged_dot.
  Instead it is testing that the local version of ragged_dot works correctly in
  a distributed environment.
  """

  def test_ragged_dot_independent_eager(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_ragged_dot_local_test),
        compile_test=False,
    )

  def test_ragged_dot_independent_compiled(self):
    distributed_utils.dist_run(
        nproc_per_node=8,
        fn=singlehost_wrapper.tpu_env_wrapper(run_ragged_dot_local_test),
        compile_test=True,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
