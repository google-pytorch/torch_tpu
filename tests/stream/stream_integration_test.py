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

"""Stream integration tests: common stream usage patterns as users write them.

Each test asserts the behavior users expect of the stream and event API. An
`expectedFailure` marks a gap at HEAD; the BUILD comment names the change that
removes it. The multi-rank patterns live in `stream_collectives_test.py`.
"""

from collections.abc import Callable
import concurrent.futures
import time
from typing import Any

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

# 256 MiB of float32: a copy of this size stays in flight for tens of ms.
_LARGE_SHAPE = (8192, 8192)


def _pinned_like(tensor: torch.Tensor) -> torch.Tensor:
  return torch.empty(
      tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True
  )


# 8192x8192 float32 matmuls, 5.5 ms each on v6e and 12.5 ms on v5e: an event
# recorded after the chain stays pending far longer than the tests' host calls.
_BUSY_CHAIN = 80


def _keep_busy(stream: torch.Stream) -> torch.Tensor:
  """Queues a long matmul chain on `stream`; `x @ x == x` so the result is `x`.

  The compile is warmed on the current stream so only device time is pending.
  """
  x = torch.full((8192, 8192), 1 / 8192, device='tpu')
  (x @ x).to('cpu')
  y = x
  with torch.tpu.stream(stream):
    for _ in range(_BUSY_CHAIN):
      y = y @ x
  return y


class StreamIntegrationTest(seed_test_utils.RepeatableTest):

  def test_non_blocking_h2d_sources_can_be_dropped(self):
    """Pinned and pageable sources are safe to drop; dtype casts on device."""
    n = 1024 * 1024

    def stage(
        make_source: Callable[[], torch.Tensor], **to_kwargs: Any
    ) -> torch.Tensor:
      # The CPU tensor goes out of scope right after the call, as in a loader.
      return make_source().to('tpu', non_blocking=True, **to_kwargs)

    batch = {
        'pinned': stage(
            lambda: torch.arange(n, dtype=torch.float32).pin_memory()
        ),
        'pageable': stage(lambda: torch.arange(n, dtype=torch.float32)),
        'cast': stage(
            lambda: torch.arange(n, dtype=torch.float32) / 3,
            dtype=torch.bfloat16,
        ),
    }
    loss = sum(v.float().sum() for v in batch.values())

    expected = torch.arange(n, dtype=torch.float32)
    utils.assert_close(batch['pinned'].cpu(), expected)
    utils.assert_close(batch['pageable'].cpu(), expected)
    utils.assert_close(batch['cast'].cpu(), (expected / 3).to(torch.bfloat16))
    utils.assert_close(
        loss.cpu(),
        expected.sum() * 2 + (expected / 3).to(torch.bfloat16).float().sum(),
    )

  @absltest.expectedFailure
  def test_side_stream_prefetch_with_record_stream(self):
    # Fails until aten::record_stream has a TPU kernel.
    copy_stream = torch.tpu.Stream()
    pinned_batch = torch.arange(1024 * 1024, dtype=torch.float32).pin_memory()

    with torch.tpu.stream(copy_stream):
      buf = pinned_batch.to('tpu', non_blocking=True)
    torch.accelerator.current_stream().wait_stream(copy_stream)
    buf.record_stream(torch.accelerator.current_stream())
    out = buf * 2 + 1
    del buf

    utils.assert_close(out.cpu(), pinned_batch * 2 + 1)

  def test_side_stream_prefetch(self):
    """The prefetch on the copy stream is consumed after a wait_stream()."""
    copy_stream = torch.tpu.Stream()
    pinned_batch = torch.arange(1024 * 1024, dtype=torch.float32).pin_memory()

    with torch.tpu.stream(copy_stream):
      buf = pinned_batch.to('tpu', non_blocking=True)
    torch.accelerator.current_stream().wait_stream(copy_stream)
    out = buf * 2 + 1
    del buf

    utils.assert_close(out.cpu(), pinned_batch * 2 + 1)

  @absltest.expectedFailure
  def test_side_stream_offload_with_record_stream(self):
    # Fails until aten::record_stream has a TPU kernel.
    offload_stream = torch.tpu.Stream()
    act = torch.full(_LARGE_SHAPE, 1.5, device='tpu') * 2
    pinned_cpu = _pinned_like(act)

    offload_stream.wait_stream(torch.accelerator.current_stream())
    act.record_stream(offload_stream)
    with torch.tpu.stream(offload_stream):
      pinned_cpu.copy_(act, non_blocking=True)
      done = offload_stream.record_event()
    act.untyped_storage().resize_(0)

    done.synchronize()
    utils.assert_close(pinned_cpu, torch.full(_LARGE_SHAPE, 3.0))

  def test_side_stream_offload_and_resize(self):
    """resize_(0) under an in-flight copy keeps the source alive."""
    offload_stream = torch.tpu.Stream()
    act = torch.full(_LARGE_SHAPE, 1.5, device='tpu') * 2
    pinned_cpu = _pinned_like(act)

    offload_stream.wait_stream(torch.accelerator.current_stream())
    with torch.tpu.stream(offload_stream):
      pinned_cpu.copy_(act, non_blocking=True)
      done = offload_stream.record_event()
    act.untyped_storage().resize_(0)

    done.synchronize()
    self.assertEqual(act.untyped_storage().size(), 0)
    utils.assert_close(pinned_cpu, torch.full(_LARGE_SHAPE, 3.0))

  @absltest.expectedFailure
  def test_wait_event_holds_the_called_stream(self):
    # Fails until TpuStream.wait_event() waits on self.
    a, b = torch.tpu.Stream(), torch.tpu.Stream()
    y = _keep_busy(a)
    ev = a.record_event()

    # The default stream is current here: only `b` may be held behind `ev`.
    b.wait_event(ev)
    ev_b = b.record_event()
    ev_default = torch.tpu.default_stream().record_event()
    b_held = not ev_b.query()
    default_held = not ev_default.query()
    ev.synchronize()
    ev_b.synchronize()

    self.assertTrue(b_held, 'b.wait_event(ev) must hold later work on b')
    self.assertFalse(default_held, 'the current stream must not be held')
    utils.assert_close(y[0, :2].to('cpu'), torch.full((2,), 1 / 8192))

  @absltest.expectedFailure
  def test_non_blocking_d2h_into_pageable_memory_is_synchronous(self):
    # Fails until a non_blocking copy into pageable memory is synchronous.
    src = torch.full((128, 1024, 1024), 3.0, device='tpu')
    pageable = torch.zeros(src.shape)

    pageable.copy_(src, non_blocking=True)
    try:
      # The last element lands last; CUDA has it in place when copy_() returns.
      self.assertEqual(pageable[-1, -1, -1].item(), 3.0)
    finally:
      # An in-flight copy must land before `pageable` is freed.
      torch.accelerator.synchronize()
    utils.assert_close(pageable, torch.full(src.shape, 3.0))

  def test_event_polling_and_timing(self):
    """query() makes progress visible without a synchronize()."""
    x = torch.full((2048, 2048), 1 / 2048, device='tpu')
    t0 = torch.tpu.Event(enable_timing=True)
    t1 = torch.tpu.Event(enable_timing=True)
    host_start = time.perf_counter()

    t0.record()
    out = x
    for _ in range(10):
      out = out @ x
    t1.record()
    polls = 0
    while not t1.query():
      polls += 1  # host bookkeeping
    elapsed_ms = t0.elapsed_time(t1)
    host_elapsed_ms = (time.perf_counter() - host_start) * 1000

    self.assertTrue(t1.query())
    self.assertBetween(elapsed_ms, 0.0, host_elapsed_ms)
    self.assertTrue(sync.is_materialized(out))
    utils.assert_close(out.cpu(), torch.full((2048, 2048), 1 / 2048))

  @absltest.expectedFailure
  def test_stream_query_reports_pending_work(self):
    # Fails until torch.tpu.Stream.query() is implemented.
    s = torch.tpu.Stream()
    y = _keep_busy(s)
    busy = s.query()
    s.synchronize()

    self.assertFalse(busy, 'query() must report pending work on the stream')
    self.assertTrue(s.query())
    utils.assert_close(y[0, :2].to('cpu'), torch.full((2,), 1 / 8192))

  def test_elapsed_time_requires_enable_timing(self):
    """elapsed_time() enforces enable_timing=True, as on CUDA."""
    timed = torch.tpu.Event(enable_timing=True)
    untimed = torch.tpu.Event()
    timed.record()
    untimed.record()
    untimed.synchronize()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests the timing contract
        ValueError, 'enable_timing=True'
    ):
      timed.elapsed_time(untimed)

  def test_compiled_step_on_side_stream(self):
    """A compiled step launched on a side stream behind a prefetch wait."""
    copy_stream, side = torch.tpu.Stream(), torch.tpu.Stream()
    pinned_batch = torch.arange(1024 * 1024, dtype=torch.float32).pin_memory()
    step = torch.compile(lambda x: x * 2 + 1, backend='tpu', fullgraph=True)

    with torch.tpu.stream(copy_stream):
      buf = pinned_batch.to('tpu', non_blocking=True)
    side.wait_stream(copy_stream)
    with torch.tpu.stream(side):
      out = step(buf)
    torch.accelerator.current_stream().wait_stream(side)

    utils.assert_close(out.cpu(), pinned_batch * 2 + 1)

  @absltest.expectedFailure
  def test_stream_switch_inside_torch_compile(self):
    # Fails until Dynamo can trace a torch.tpu.stream() switch.
    side = torch.tpu.Stream()

    def step(x: torch.Tensor) -> torch.Tensor:
      side.wait_stream(torch.accelerator.current_stream())
      with torch.tpu.stream(side):
        y = x + 1
      torch.accelerator.current_stream().wait_stream(side)
      return y * 2

    x = torch.arange(1024, dtype=torch.float32, device='tpu')
    eager = step(x)
    compiled = torch.compile(step, backend='tpu', fullgraph=True)(x)

    utils.assert_close(compiled.cpu(), eager.cpu())
    utils.assert_close(compiled.cpu(), (x.cpu() + 1) * 2)

  def test_worker_thread_offload_on_shared_stream(self):
    """A worker thread waits for the producer's event, then offloads."""

    def dcp_worker(
        shard: torch.Tensor, ready_ev: torch.Event, shared_stream: torch.Stream
    ) -> torch.Tensor:
      shared_stream.wait_event(ready_ev)  # recorded on the producing stream
      with torch.tpu.stream(shared_stream):
        cpu_copy = torch.empty_like(shard, device='cpu', pin_memory=True)
        cpu_copy.copy_(shard, non_blocking=True)
      shared_stream.synchronize()
      return cpu_copy

    x = torch.full((2048, 2048), 1 / 2048, device='tpu')
    shard = x @ x
    ready_ev = torch.accelerator.current_stream().record_event()
    shared_stream = torch.tpu.Stream()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      cpu_copy = executor.submit(
          dcp_worker, shard, ready_ev, shared_stream
      ).result()

    self.assertTrue(cpu_copy.is_pinned())
    utils.assert_close(cpu_copy, torch.full((2048, 2048), 1 / 2048))

  def test_autograd_on_side_stream(self):
    """Backward of an op issued on a side stream produces the right grads."""
    side = torch.tpu.Stream()
    # Small integers: exact in bfloat16 passes, so the matmul matches the CPU.
    x_cpu = (torch.arange(64 * 64, dtype=torch.float32) % 5 - 2).reshape(64, 64)
    w_cpu = (torch.arange(64 * 64, dtype=torch.float32) % 7 - 3).reshape(64, 64)
    x = x_cpu.to('tpu').requires_grad_()
    w = w_cpu.to('tpu')

    with torch.tpu.stream(side):
      loss = (x @ w).relu().sum()
    loss.backward()

    x_ref = x_cpu.clone().requires_grad_()
    (x_ref @ w_cpu).relu().sum().backward()
    utils.assert_close(loss.cpu(), (x_cpu @ w_cpu).relu().sum())
    utils.assert_close(x.grad.cpu(), x_ref.grad)

  def test_independent_streams_on_one_device(self):
    """Two models on two streams give the single-stream results."""
    ctx = torch.arange(256 * 256, dtype=torch.float32, device='tpu')
    ctx = ctx.reshape(256, 256) / 65536
    w_draft = (torch.arange(256 * 256, device='tpu') % 5 - 2).float()
    w_main = (torch.arange(256 * 256, device='tpu') % 3 - 1).float()

    def draft_model(c: torch.Tensor) -> torch.Tensor:
      return torch.tanh(c @ w_draft.reshape(256, 256))

    def target_model(c: torch.Tensor) -> torch.Tensor:
      return torch.tanh(c @ w_main.reshape(256, 256)) * 2

    def verify(d: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
      return d.argmax(dim=-1) == m.argmax(dim=-1)

    s_draft, s_main = torch.tpu.Stream(), torch.tpu.Stream()
    with torch.tpu.stream(s_draft):
      draft_logits = draft_model(ctx)
    with torch.tpu.stream(s_main):
      main_logits = target_model(ctx)
    torch.accelerator.current_stream().wait_stream(s_draft)
    torch.accelerator.current_stream().wait_stream(s_main)
    accepted = verify(draft_logits, main_logits)

    # The same models on the default stream alone.
    single_draft, single_main = draft_model(ctx), target_model(ctx)
    utils.assert_close(draft_logits.cpu(), single_draft.cpu())
    utils.assert_close(main_logits.cpu(), single_main.cpu())
    utils.assert_close(accepted.cpu(), verify(single_draft, single_main).cpu())

  def test_first_sync_reports_and_clears_failure(self):
    """The first synchronize() reports the failure, the second is clean."""
    a = torch.tpu.Stream()
    with torch.tpu.stream(a):
      # Fails to compile (compile-time OOM: 4 PiB of HBM).
      x = torch.ones((1024, 1024, 1024, 1024, 1024), device='tpu')

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests error reporting
        RuntimeError, 'out of memory'
    ):
      a.synchronize()
    a.synchronize()

    # Using the failed tensor raises again.
    y = (x + 1)[0, 0, 0, 0, 0]
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests error reporting
        RuntimeError, 'out of memory'
    ):
      y.item()

  def test_waiting_stream_is_unaffected_by_failure(self):
    """A stream that waited on the failed stream runs and reports no error."""
    a, b = torch.tpu.Stream(), torch.tpu.Stream()
    # Deferred, so the event is recorded before the failure is known.
    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_AND_FUSE):
      with torch.tpu.stream(a):
        # Fails to compile (compile-time OOM: 4 PiB of HBM).
        x = torch.ones((1024, 1024, 1024, 1024, 1024), device='tpu')
        failing = a.record_event()
    failing.wait(b)
    with torch.tpu.stream(b):
      y = torch.ones(8, device='tpu') + 1

    b.synchronize()
    utils.assert_close(y.cpu(), torch.full((8,), 2.0))
    # The event recorded over the failing work is what reports the failure.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=tests error reporting
        RuntimeError, 'out of memory'
    ):
      failing.synchronize()
    del x


if __name__ == '__main__':
  absltest.main()
