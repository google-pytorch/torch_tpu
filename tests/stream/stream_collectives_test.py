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

"""Stream integration tests that need a process group: collectives and P2P.

One class per expected-failure test: a failing rank poisons its whole class.
"""

import os
import threading
import time

from absl.testing import absltest

if __name__ == '__main__':  # We are in the parent process.
  # A per-run compilation cache root, picked before torch is imported.
  os.environ['TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT'] += (
      f'/{time.strftime("%Y-%m-%d", time.localtime())}'
      f'.{os.environ.get("TEST_RANDOM_SEED", "0")}'
  )
# pylint: disable=g-code-after-main

import torch  # pylint: disable=g-import-not-at-top
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils

_SKIP_P2P_ON_TPU_V7_REASON = (
    'P2P send/recv aborts in libtpu on TPU v7 (b/540414751).'
)
try:
  _IS_TPU_V7 = torch.tpu.get_device_properties().name == 'TPU v7'
except RuntimeError:  # No TPU generation that TorchTPU supports is attached.
  _IS_TPU_V7 = False

# float32 elements per peer in the all_to_all: 4 MiB each, so it is in flight.
_ALL_TO_ALL_CHUNK = 1024 * 1024


def _all_to_all_inputs(
    rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns (input, output buffer, expected output) of an all_to_all_single."""
  # Chunk j of rank r carries r * world_size + j, so output chunk i on rank r
  # must hold i * world_size + r.
  moe_in = torch.arange(world_size, dtype=torch.float32) + rank * world_size
  moe_in = moe_in.repeat_interleave(_ALL_TO_ALL_CHUNK).to('tpu')
  moe_out = torch.empty_like(moe_in)
  expected = torch.arange(world_size, dtype=torch.float32) * world_size + rank
  return moe_in, moe_out, expected.repeat_interleave(_ALL_TO_ALL_CHUNK)


class StreamCollectivesTest(continuous_test_utils.TpuMultiProcContinuousTest):
  world_size = distributed_utils.get_tpu_device_count()

  def test_collectives_on_side_stream_and_async_op(self) -> None:
    """Collectives on a side stream and async_op, same order on every rank."""
    rank, w = self.rank, self.world_size
    ag_stream = torch.tpu.Stream()
    shard_next = torch.full((256, 256), float(rank), device='tpu')
    out_next = torch.empty((w * 256, 256), device='tpu')
    moe_in, moe_out, expected_moe = _all_to_all_inputs(rank, w)
    x = torch.full((256, 256), 1 / 256, device='tpu')

    ag_stream.wait_stream(torch.accelerator.current_stream())
    with torch.tpu.stream(ag_stream):
      dist.all_gather_into_tensor(out_next, shard_next, group=None)
    work = dist.all_to_all_single(moe_out, moe_in, group=None, async_op=True)
    y = x @ x  # shared_expert(x)
    work.wait()
    torch.accelerator.current_stream().wait_stream(ag_stream)

    expected_gather = torch.arange(w, dtype=torch.float32).repeat_interleave(
        256
    )
    utils.assert_close(out_next.cpu(), expected_gather[:, None].expand(-1, 256))
    utils.assert_close(moe_out.cpu(), expected_moe)
    utils.assert_close(y.cpu(), torch.full((256, 256), 1 / 256))


class AsyncCollectiveWorkTest(continuous_test_utils.TpuMultiProcContinuousTest):
  world_size = distributed_utils.get_tpu_device_count()

  @absltest.expectedFailure
  def test_async_op_work_tracks_completion(self) -> None:
    # Fails until a collective's Work tracks the collective's completion.
    moe_in, moe_out, expected = _all_to_all_inputs(self.rank, self.world_size)

    work = dist.all_to_all_single(moe_out, moe_in, group=None, async_op=True)
    completed_at_issue = work.is_completed()
    work.wait()
    deadline = time.monotonic() + 60
    while not work.is_completed() and time.monotonic() < deadline:
      time.sleep(0.001)

    self.assertFalse(
        completed_at_issue, 'is_completed() was True right after issue'
    )
    self.assertTrue(
        work.is_completed(), 'is_completed() stayed False after wait()'
    )
    utils.assert_close(moe_out.cpu(), expected)


class PipelineP2PTest(continuous_test_utils.TpuMultiProcContinuousTest):
  world_size = distributed_utils.get_tpu_device_count()

  @absltest.skipIf(_IS_TPU_V7, _SKIP_P2P_ON_TPU_V7_REASON)
  @absltest.expectedFailure
  def test_send_first_pipeline_ring(self) -> None:
    # Fails until isend returns before the peer posts its receive.
    rank, w = self.rank, self.world_size
    nxt, prv = (rank + 1) % w, (rank - 1) % w
    act = torch.full((1024, 1024), float(rank), device='tpu')
    buf = torch.empty_like(act)
    torch.accelerator.synchronize()

    # Rank 0 receives first so the ring can unwind; the others delay theirs.
    peer_delay_s = 3.0
    recv_work = dist.irecv(buf, src=prv, tag=0) if rank == 0 else None
    send_work, send_elapsed_s = [], []

    def send() -> None:
      start = time.perf_counter()
      send_work.append(dist.isend(act, dst=nxt, tag=0))
      send_elapsed_s.append(time.perf_counter() - start)

    sender = threading.Thread(target=send)
    sender.start()
    if recv_work is None:
      time.sleep(peer_delay_s)
      recv_work = dist.irecv(buf, src=prv, tag=0)
    sender.join()
    send_work[0].wait()
    recv_work.wait()
    utils.assert_close(buf.cpu(), torch.full((1024, 1024), float(prv)))

    # Every rank must take the same branch, so decide on the slowest send.
    all_elapsed = torch.empty(w, device='tpu')
    dist.all_gather_into_tensor(
        all_elapsed, torch.tensor(send_elapsed_s, device='tpu')
    )
    self.assertLess(
        all_elapsed.max().item(),
        0.5,
        'isend blocked until the peer posted its receive',
    )

    # The batch as written for CUDA: sends listed first on every rank.
    buf2 = torch.empty_like(act)
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, act * 2, nxt),
        dist.P2POp(dist.irecv, buf2, prv),
    ])
    stage_out = act @ torch.eye(1024, device='tpu')  # microbatch_compute(cur)
    for r in reqs:
      r.wait()
    utils.assert_close(buf2.cpu(), torch.full((1024, 1024), 2.0 * prv))
    utils.assert_close(stage_out.cpu(), torch.full((1024, 1024), float(rank)))


if __name__ == '__main__':  # We are in the parent process.
  mp.set_start_method('spawn')
  multiprocessing.handle_test_main(absltest.main)
