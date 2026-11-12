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

"""Unit tests for torch.accelerator.Graph on TPU."""

import gc
import os
import threading
import time
from unittest import mock

from absl.testing import absltest
import torch
from torch._higher_order_ops import associative_scan
from torch_tpu._internal import dynamism
from torch_tpu._internal import execution_mode
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


@absltest.skipIf(
    utils.get_torch_version() < (2, 12),
    "torch.accelerator.Graph requires PyTorch >= 2.12.",
)
class GraphTest(seed_test_utils.RepeatableTest):
  """Tests for torch.accelerator.Graph capture and replay on TPU."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu", torch.accelerator.current_device_index())
    # Like CUDA graphs, graphs are captured on a non-default stream.
    self.stream = torch.Stream(device=self.device)

  def _run_on_other_thread(self, fn, *args):
    """Runs fn(*args) on another thread, returning the RuntimeError raised."""
    errors = []
    unexpected = []

    def run():
      try:
        fn(*args)
      except RuntimeError as e:
        errors.append(e)
      except BaseException as e:  # pylint: disable=broad-except
        unexpected.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=60)
    self.assertFalse(thread.is_alive(), "the other thread hangs")
    if unexpected:
      raise unexpected[0]
    return errors[0] if errors else None

  def test_context_manager_capture_and_multiple_replays(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")
    y = torch.zeros_like(x)

    g = torch.accelerator.Graph()
    with self.stream, g:
      z = (x + 2.0) * 3.0
      y.copy_(z)

    # First replay with initial x.
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([9.0, 12.0, 15.0, 18.0]))
    utils.assert_close(z.cpu(), torch.tensor([9.0, 12.0, 15.0, 18.0]))

    # Mutate x in-place and replay again.
    x.fill_(10.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([36.0, 36.0, 36.0, 36.0]))
    utils.assert_close(z.cpu(), torch.tensor([36.0, 36.0, 36.0, 36.0]))

    # Mutate x via copy_ from CPU and replay again.
    x.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([3.0, 6.0, 9.0, 12.0]))

  def test_manual_capture_begin_end_and_inplace_ops(self):
    x = torch.ones((4,), dtype=torch.float32, device="tpu")
    out = torch.zeros((4,), dtype=torch.float32, device="tpu")

    g = torch.accelerator.Graph()
    with self.stream:
      g.capture_begin()
      x.add_(2.0)
      torch.mul(x, 4.0, out=out)
      g.capture_end()

    # Before replay, x should still have its pre-capture value (1.0).
    utils.assert_close(x.cpu(), torch.ones((4,)))

    # First replay: x becomes 1 + 2 = 3, out becomes 3 * 4 = 12.
    g.replay()
    utils.assert_close(x.cpu(), torch.full((4,), 3.0))
    utils.assert_close(out.cpu(), torch.full((4,), 12.0))

    # Second replay: x becomes 3 + 2 = 5, out becomes 5 * 4 = 20.
    g.replay()
    utils.assert_close(x.cpu(), torch.full((4,), 5.0))
    utils.assert_close(out.cpu(), torch.full((4,), 20.0))

  def test_direct_slot_copy_without_compute_ops(self):
    src = torch.tensor([5.0, 6.0, 7.0], device="tpu")
    dst = torch.zeros_like(src)

    g = torch.accelerator.Graph()
    with self.stream, g:
      dst.copy_(src)

    g.replay()
    utils.assert_close(dst.cpu(), torch.tensor([5.0, 6.0, 7.0]))

    src.fill_(42.0)
    g.replay()
    utils.assert_close(dst.cpu(), torch.full((3,), 42.0))

  def test_write_after_read_and_tensors_created_in_capture(self):
    x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
    z = torch.tensor([10.0, 20.0, 30.0], device="tpu")

    g = torch.accelerator.Graph()
    with self.stream, g:
      a = torch.full((3,), 4.0, device="tpu")
      y = x + a
      x.copy_(z)

    # First replay: y reads initial x ([1, 2, 3]) + 4 = [5, 6, 7], while x is
    # overwritten with z ([10, 20, 30]).
    g.replay()
    utils.assert_close(a.cpu(), torch.full((3,), 4.0))
    utils.assert_close(y.cpu(), torch.tensor([5.0, 6.0, 7.0]))
    utils.assert_close(x.cpu(), torch.tensor([10.0, 20.0, 30.0]))

    # Second replay: y reads updated x ([10, 20, 30]) + 4 = [14, 24, 34].
    z.fill_(100.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([14.0, 24.0, 34.0]))
    utils.assert_close(x.cpu(), torch.full((3,), 100.0))

  def test_intermediate_tensors_freed_during_and_after_capture(self):
    x = torch.tensor([1.0, 2.0, 3.0], device="tpu")

    def run_body(inp):
      tmp1 = inp + 1.0
      tmp2 = tmp1 * 2.0
      return tmp2 - 3.0

    g = torch.accelerator.Graph()
    with self.stream, g:
      out = run_body(x)

    g.replay()
    utils.assert_close(out.cpu(), torch.tensor([1.0, 3.0, 5.0]))

    # Free one of two outputs after capture and verify replay still succeeds.
    g2 = torch.accelerator.Graph()
    with self.stream, g2:
      out_keep = x * 2.0
      out_drop = x * 3.0
    del out_drop
    x.fill_(4.0)
    g2.replay()
    utils.assert_close(out_keep.cpu(), torch.full((3,), 8.0))

  def test_replay_after_inputs_and_outputs_are_freed(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    keep = torch.zeros_like(x)
    g = torch.accelerator.Graph()
    with self.stream, g:
      y = x + 1.0
      keep.copy_(y * 2.0)

    # Like a CUDA graph, the graph keeps reading and writing the storages of
    # its inputs and outputs after the user frees them.
    del x, y
    g.replay()
    utils.assert_close(keep.cpu(), torch.tensor([4.0, 6.0]))

  def test_resize_storage_created_during_capture(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      out = torch.empty(0, device="tpu")
      torch.add(x, 1.0, out=out)

    g.replay()
    utils.assert_close(out.cpu(), torch.tensor([2.0, 3.0]))

    x.fill_(5.0)
    g.replay()
    utils.assert_close(out.cpu(), torch.tensor([6.0, 6.0]))

  def test_replay_after_input_written_through_reinterpreting_view(self):
    x = torch.zeros(4, device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      y = x * 2.0

    # Writing through a view of a different shape or dtype can change the shape
    # or dtype of the buffer that backs the storage of x.
    x.view(2, 2).copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 4.0, 6.0, 8.0]))

    x.view(torch.int32).copy_(
        torch.tensor([5.0, 6.0, 7.0, 8.0]).view(torch.int32)
    )
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([10.0, 12.0, 14.0, 16.0]))

  def test_small_and_large_multi_output_graphs(self):
    def run_body(inp):
      tmp = inp + 1.0
      y = tmp * 2.0
      return y, tmp + y

    # Verify both small and large (>64 MiB per tensor) multi-output graphs
    # replay correctly when an intermediate (`tmp`) is shared across outputs
    # (`y` and `z`).
    for num_elements in (4, 16 * 1024 * 1024 + 1):
      x = torch.ones(num_elements, device="tpu")
      g = torch.accelerator.Graph()
      with self.stream, g:
        y, z = run_body(x)

      g.replay()
      self.assertEqual(y[-1].item(), 4.0)
      self.assertEqual(z[-1].item(), 6.0)

      x.fill_(2.0)
      g.replay()
      self.assertEqual(y[-1].item(), 6.0)
      self.assertEqual(z[-1].item(), 9.0)

  def test_reset_and_pool_sharing(self):
    x = torch.tensor([2.0, 3.0], device="tpu")

    g1 = torch.accelerator.Graph()
    with self.stream, g1:
      y1 = x + 1.0

    pool_id = g1.pool()
    self.assertNotEqual(pool_id, (0, 0))

    g2 = torch.accelerator.Graph(pool=pool_id)
    with self.stream, g2:
      y2 = x * 5.0
    self.assertEqual(g2.pool(), pool_id)

    g1.replay()
    g2.replay()
    utils.assert_close(y1.cpu(), torch.tensor([3.0, 4.0]))
    utils.assert_close(y2.cpu(), torch.tensor([10.0, 15.0]))

    # Reset g1 and recapture a new graph on the same instance.
    g1.reset()
    with self.stream, g1:
      y1_new = x - 1.0
    g1.replay()
    utils.assert_close(y1_new.cpu(), torch.tensor([1.0, 2.0]))

  def test_backward_and_optimizer_step_in_captured_graph(self):
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn(4, generator=gen)
    xs = [torch.randn(4, generator=gen) for _ in range(3)]
    ts = [torch.randn((), generator=gen) for _ in range(3)]

    w_ref = w0.clone().requires_grad_()
    opt_ref = torch.optim.SGD([w_ref], lr=0.1)
    for x, t in zip(xs, ts, strict=True):
      opt_ref.zero_grad(set_to_none=True)
      (((w_ref * x).sum() - t) ** 2).backward()
      opt_ref.step()

    w = w0.clone().to("tpu").requires_grad_()
    opt = torch.optim.SGD([w], lr=0.1)
    static_x = torch.zeros(4, device="tpu")
    static_t = torch.zeros((), device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      (((w * static_x).sum() - static_t) ** 2).backward()
      opt.step()

    self.assertTrue(torch._C._is_multithreading_enabled())
    utils.assert_close(w.detach().cpu(), w0)
    for x, t in zip(xs, ts, strict=True):
      static_x.copy_(x)
      static_t.copy_(t)
      g.replay()
    utils.assert_close(w.detach().cpu(), w_ref.detach(), rtol=1e-4, atol=1e-4)

  def test_capture_errors(self):
    x = torch.tensor([1.0, 2.0], device="tpu")

    # 1. Allocating an untyped storage and resizing a storage that existed
    # before capture raise RuntimeError without invalidating the capture.
    g_forbidden = torch.accelerator.Graph()
    with self.stream, g_forbidden:
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
          RuntimeError, "cannot allocate an untyped storage while capturing"
      ):
        torch.UntypedStorage(16, device=x.device)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
          RuntimeError, "cannot resize a tensor storage created before graph"
      ):
        x.untyped_storage().resize_(16)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
          RuntimeError, "cannot resize a tensor storage created before graph"
      ):
        x.resize_(4)
      y = x + 1.0
    g_forbidden.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))

    # 2. Ending capture on a different stream raises RuntimeError, and capture
    # continues.
    other_stream = torch.Stream(device=x.device)
    g_stream = torch.accelerator.Graph()
    with self.stream:
      g_stream.capture_begin()
      y = x + 1.0
      with other_stream:
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
            RuntimeError, "same device and stream"
        ):
          g_stream.capture_end()
      z = y * 2.0
      g_stream.capture_end()
    g_stream.replay()
    utils.assert_close(z.cpu(), torch.tensor([4.0, 6.0]))

    # 3. Running an operation on a stream other than the capture stream raises
    # RuntimeError, as capture does not record it.
    g_other_stream = torch.accelerator.Graph()
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
        RuntimeError, "must run on the stream the capture began on"
    ):
      with self.stream, g_other_stream:
        with other_stream:
          _ = x * 2.0
    utils.assert_close((x * 2.0).cpu(), torch.tensor([2.0, 4.0]))

    # 4. Ending capture on a thread other than the capturing thread raises
    # RuntimeError and leaves the capture open.
    g_thread = torch.accelerator.Graph()
    with self.stream:
      g_thread.capture_begin()
      try:
        error = self._run_on_other_thread(g_thread.capture_end)
      finally:
        g_thread.capture_end()
    self.assertRegex(str(error), "capture must end on the thread it began on")

    # 5. Resetting a graph whose capturing thread exited during capture ends
    # the capture: the tensors that it read are usable again, and its stream
    # can capture again.
    g_abandoned = torch.accelerator.Graph()
    abandoned_stream = torch.Stream(device=x.device)

    def begin_capture_and_read_x(graph):
      with abandoned_stream:
        graph.capture_begin()
        _ = x * 2.0

    self.assertIsNone(
        self._run_on_other_thread(begin_capture_and_read_x, g_abandoned)
    )
    self._reset_graph_of_exited_thread(g_abandoned)
    self.assertFalse(abandoned_stream.is_capturing())
    utils.assert_close((x + 1.0).cpu(), torch.tensor([2.0, 3.0]))
    with abandoned_stream, g_abandoned:
      y = x * 3.0
    g_abandoned.replay()
    utils.assert_close(y.cpu(), torch.tensor([3.0, 6.0]))

  def test_capture_on_default_stream_raises(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    g = torch.accelerator.Graph()
    # Like a CUDA graph, a graph cannot be captured on the default stream...
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks capture after error
        RuntimeError, "graphs must be captured on a non-default stream"
    ):
      g.capture_begin()

    # ...but can be replayed on it.
    with self.stream, g:
      y = x + 1.0
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))

  def test_waiting_for_device_invalidates_capture(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    pinned = torch.zeros(2).pin_memory()
    event = torch.Event(device="tpu")
    event.record()
    other_stream = torch.Stream(device=self.device)
    other_graph = torch.accelerator.Graph()
    with self.stream, other_graph:
      _ = x * 3.0

    # Like on CUDA, these calls wait for the device, query it, or make the
    # capture stream wait for work that capture did not record, so they raise
    # and invalidate the capture.
    sync_message = "cannot materialize tensors or synchronize streams"
    cases = (
        ("synchronize device", torch.accelerator.synchronize, sync_message),
        ("synchronize capture stream", self.stream.synchronize, sync_message),
        ("synchronize other stream", other_stream.synchronize, sync_message),
        ("query capture stream", self.stream.query, "cannot query streams"),
        ("query event", event.query, "cannot query events"),
        ("synchronize event", event.synchronize, "cannot synchronize events"),
        (
            "wait on event",
            lambda: self.stream.wait_event(event),
            "cannot make a stream wait on an event",
        ),
        ("item", lambda: x.sum().item(), "cannot read the value of a tensor"),
        ("replay", other_graph.replay, "cannot replay a graph"),
        (
            "blocking copy from pinned tensor",
            lambda: x.copy_(pinned),
            "cannot copy a CPU tensor to a TPU tensor synchronously",
        ),
        (
            "blocking copy to pinned tensor",
            lambda: pinned.copy_(x),
            "cannot copy a TPU tensor to a CPU tensor synchronously",
        ),
    )
    for name, wait, message in cases:
      with self.subTest(name):
        g = torch.accelerator.Graph()
        with self.stream:
          g.capture_begin()
          _ = x + 1.0
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
              RuntimeError, message
          ):
            wait()
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
              RuntimeError,
              "graph capture failed because of an earlier error during"
              " capture: .*"
              + message,
          ):
            g.capture_end()
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
            RuntimeError, "without a preceding successful capture"
        ):
          g.replay()

        # Like a CUDA graph, reset() is required before capturing again.
        with self.stream:
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
              RuntimeError, "already owns a captured graph"
          ):
            g.capture_begin()
        g.reset()
        with self.stream, g:
          y = x + 2.0
        g.replay()
        utils.assert_close(y.cpu(), torch.tensor([3.0, 4.0]))

  def test_streams_report_graph_capture(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    default_stream = torch.accelerator.current_stream()
    other_stream = torch.Stream(device=self.device)
    g = torch.accelerator.Graph()
    self.assertFalse(self.stream.is_capturing())

    with self.stream:
      g.capture_begin()
      try:
        self.assertTrue(self.stream.is_capturing())
        self.assertTrue(torch.accelerator.current_stream().is_capturing())
        self.assertFalse(default_stream.is_capturing())
        self.assertFalse(other_stream.is_capturing())
        with other_stream:
          self.assertFalse(torch.accelerator.current_stream().is_capturing())
        _ = x + 1.0

        # Like on CUDA, a capture that an error invalidated is in progress
        # until it ends.
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
            RuntimeError, "cannot materialize tensors or synchronize streams"
        ):
          torch.accelerator.synchronize()
        self.assertTrue(self.stream.is_capturing())
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
            RuntimeError, "graph capture failed because of an earlier error"
        ):
          g.capture_end()
      finally:
        g.reset()

    self.assertFalse(self.stream.is_capturing())

  def test_optimizer_step_capture_requires_capturable(self):
    w = torch.ones(4, device="tpu", requires_grad=True)
    w.sum().backward()
    opt = torch.optim.Adam([w], lr=0.1, capturable=False)
    g = torch.accelerator.Graph()
    with self.stream:
      g.capture_begin()
      try:
        # PyTorch optimizers find out about the capture from
        # `torch.Stream.is_capturing()`.
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=raised by PyTorch
            RuntimeError, "capturable is False"
        ):
          opt.step()
      finally:
        g.reset()

  def _check_captured_adam_steps(self, fused: bool, alias_during_capture: bool):
    """Checks replays of a captured Adam step against eager CPU steps.

    Also checks that the tensors that share the buffer of the parameter when a
    replay reads it keep the value that the parameter had before the step.

    Args:
      fused: Whether the TPU optimizer uses the fused implementation, which
        donates the buffers of its arguments.
      alias_during_capture: Whether tensors also share the buffer of the
        parameter during capture.
    """
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn(4, generator=gen)
    xs = [torch.randn(4, generator=gen) for _ in range(4)]

    def loss_fn(w, x):
      return ((w * x).sum() - 1.0) ** 2

    w_ref = w0.clone().requires_grad_()
    opt_ref = torch.optim.Adam([w_ref], lr=0.1)

    def step_ref(x):
      opt_ref.zero_grad(set_to_none=True)
      loss_fn(w_ref, x).backward()
      opt_ref.step()

    w = w0.to("tpu").requires_grad_()
    opt = torch.optim.Adam([w], lr=0.1, fused=fused, capturable=True)
    static_x = xs[0].to("tpu")
    snapshot_of_w = torch.zeros(4, device="tpu")
    # Like for CUDA graphs, warm up on the capture stream, so that the
    # optimizer state exists before capture.
    with self.stream:
      loss_fn(w, static_x).backward()
      opt.step()
    step_ref(xs[0])

    g = torch.accelerator.Graph()
    with self.stream, g:
      if alias_during_capture:
        snapshot_of_w.copy_(w.detach())
        clone_of_w = w.detach().clone()
      opt.zero_grad(set_to_none=True)
      loss_fn(w, static_x).backward()
      opt.step()

    for x in xs[1:]:
      w_before_step = w_ref.detach().clone()
      step_ref(x)
      static_x.copy_(x)
      # Tensors that share the buffer of `w` when the replay reads it.
      aliases = [
          w.detach().clone(),
          torch.zeros(4, device="tpu").copy_(w.detach()),
      ]
      g.replay()
      if alias_during_capture:
        aliases.extend((snapshot_of_w, clone_of_w))
      for alias in aliases:
        utils.assert_close(alias.cpu(), w_before_step, rtol=1e-4, atol=1e-4)
      utils.assert_close(w.detach().cpu(), w_ref.detach(), rtol=1e-4, atol=1e-4)

  def test_capturable_adam_step_replays(self):
    self._check_captured_adam_steps(fused=False, alias_during_capture=True)

  def test_fused_adam_step_replays_without_donating_graph_inputs(self):
    # The fused step donates the buffers of its arguments, which every replay
    # reads from tensors that other tensors can share. The graph must not
    # donate them.
    self._check_captured_adam_steps(fused=True, alias_during_capture=False)

  def test_capture_and_replay_in_every_eager_mode(self):
    modes = (
        execution_mode.EagerMode.DEFER_NEVER,
        execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING,
        execution_mode.EagerMode.DEFER_AND_FUSE,
    )
    for mode in modes:
      with self.subTest(mode.name), execution_mode.set_eager_mode(mode):
        x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
        y = torch.zeros_like(x)
        g = torch.accelerator.Graph()
        with self.stream, g:
          y.copy_(x * 2.0 + 1.0)

        utils.assert_close(y.cpu(), torch.zeros(3))
        g.replay()
        utils.assert_close(y.cpu(), torch.tensor([3.0, 5.0, 7.0]))
        x.fill_(-1.0)
        g.replay()
        utils.assert_close(y.cpu(), torch.full((3,), -1.0))

  def test_async_assertions_are_not_checked_during_capture(self):
    x = torch.tensor([1.0, -1.0], device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      # Only the replays compute the condition, which is false, and replays do
      # not check assertions.
      torch._assert_async((x > 0).all())
      y = x + 1.0

    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 0.0]))
    utils.assert_close((x * 2.0).cpu(), torch.tensor([2.0, -2.0]))

  def test_capture_that_begins_and_ends_in_different_contexts(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    g = torch.accelerator.Graph()
    with self.stream:
      with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
        g.capture_begin()
        y = x * 3.0
      z = y + 1.0
      g.capture_end()

    g.replay()
    utils.assert_close(z.cpu(), torch.tensor([4.0, 7.0]))

  def test_associative_scan_during_capture(self):
    # Tracing the combine function of the scan runs an FX trace during
    # capture, which capture does not record.
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      y = associative_scan(torch.add, x, dim=0, combine_mode="pointwise")

    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([1.0, 3.0, 6.0, 10.0]))
    x.fill_(1.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0]))

  def test_tpu_to_cpu_during_fx_trace_during_capture(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    untracked = torch.tensor([5.0, 6.0], device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      y = x + 1.0
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      ):
        copied = torch.tensor([3.0, 4.0], device="tpu")
        copied_cpu = copied.cpu()
        untracked_cpu = untracked.cpu()
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=capture continues
            RuntimeError,
            "cannot use a tensor accessed by a graph capture in progress",
        ):
          x.cpu()
    utils.assert_close(copied_cpu, torch.tensor([3.0, 4.0]))
    utils.assert_close(untracked_cpu, torch.tensor([5.0, 6.0]))
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))

  def test_dynamism_during_capture(self):
    x = torch.ones(4, device="tpu")
    dynamism.mark_dynamic(x, 0, 2, 8)
    unused_dyn = torch.ones(4, device="tpu")
    dynamism.mark_dynamic(unused_dyn, 0, 2, 8)
    static = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")

    g = torch.accelerator.Graph()
    with self.stream, g:
      self.assertEmpty(dynamism.get_dynamism_info(static))
      # Inspecting dynamism before a tensor is read in capture reports its
      # pre-capture dynamic bounds without installing a placeholder.
      unused_info = dynamism.get_dynamism_info(unused_dyn)
      self.assertLen(unused_info, 1)
      self.assertEqual(unused_info[0].upper_bound, 8)
      self.assertIsNone(self._run_on_other_thread(lambda: unused_dyn.cpu()))
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      ):
        unused_dyn_cpu = unused_dyn.cpu()

      y = x + static
      # Once `x` has been read by a captured op, its slot holds a static
      # capture placeholder until capture ends.
      self.assertEmpty(dynamism.get_dynamism_info(x))
      self.assertEmpty(dynamism.get_dynamism_info(y))
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=capture continues
          RuntimeError,
          "cannot mark a tensor dynamic while capturing a graph",
      ):
        dynamism.mark_dynamic(static, 0, 2, 8)

    utils.assert_close(unused_dyn_cpu, torch.ones(4))

    # Pre-capture dynamic bounds on `x` are restored after capture, and replay
    # runs at the runtime shape of `x` without padded-shape mismatch.
    info = dynamism.get_dynamism_info(x)
    self.assertLen(info, 1)
    self.assertEqual(info[0].dimension, 0)
    self.assertEqual(info[0].lower_bound, 2)
    self.assertEqual(info[0].upper_bound, 8)

    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    x.fill_(10.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([11.0, 12.0, 13.0, 14.0]))

  def test_copies_during_fx_trace_during_capture(self):
    # An FX trace during capture, such as the trace of the combine function of
    # a scan, records nothing into the graph, so its copies between CPU and TPU
    # tensors take the FX path, as outside capture.
    x = torch.tensor([1.0, 2.0], device="tpu")
    g = torch.accelerator.Graph()
    pinned = torch.tensor([5.0, 6.0], pin_memory=True)
    with self.stream, g:
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      ):
        copied = torch.tensor([3.0, 4.0], device="tpu")
        copied_pinned = pinned.to("tpu", non_blocking=True)
      y = x + 1.0
    pinned.fill_(0.0)
    utils.assert_close(copied.cpu(), torch.tensor([3.0, 4.0]))
    utils.assert_close(copied_pinned.cpu(), torch.tensor([5.0, 6.0]))
    g.replay()
    utils.assert_close(copied_pinned.cpu(), torch.tensor([5.0, 6.0]))
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))

  @absltest.skipIf(
      not hasattr(torch.autograd.graph, "set_override_stale_capture_stream"),
      "set_override_stale_capture_stream() requires PyTorch >= 2.13.",
  )
  def test_backward_with_accumulate_grad_node_from_warmup(self):
    w = torch.ones(4, device="tpu", requires_grad=True)
    x = torch.full((4,), 2.0, device="tpu")
    # Keeping the autograd graph of the warmup alive keeps the AccumulateGrad
    # node of `w`, which runs on the default stream.
    loss = (w * x).sum()
    loss.backward()
    w.grad = None

    g = torch.accelerator.Graph()
    with self.stream:
      g.capture_begin()
      try:
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=raised by PyTorch
            RuntimeError, "stale reference to the default stream"
        ):
          (w * x).sum().backward()
      finally:
        g.reset()

    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
      g = torch.accelerator.Graph()
      with self.stream, g:
        (w * x).sum().backward()
    finally:
      torch.autograd.graph.set_override_stale_capture_stream(False)
    g.replay()
    utils.assert_close(w.grad.cpu(), torch.full((4,), 2.0))
    del loss

  def test_autograd_multithreading_restored_after_error_reset_and_del(self):
    x = torch.tensor([1.0, 2.0], device="tpu")

    # 1. After invalidating error + capture_end()
    g1 = torch.accelerator.Graph()
    with self.stream:
      g1.capture_begin()
      self.assertFalse(torch._C._is_multithreading_enabled())
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
          RuntimeError, "cannot materialize tensors"
      ):
        torch.accelerator.synchronize()
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
          RuntimeError, "graph capture failed because of an earlier error"
      ):
        g1.capture_end()
    self.assertTrue(torch._C._is_multithreading_enabled())

    # 2. After reset()
    g2 = torch.accelerator.Graph()
    with self.stream:
      g2.capture_begin()
      _ = x + 1.0
      self.assertFalse(torch._C._is_multithreading_enabled())
      g2.reset()
    self.assertTrue(torch._C._is_multithreading_enabled())

    # 3. After graph destruction on owner thread
    g3 = torch.accelerator.Graph()
    with self.stream:
      g3.capture_begin()
      _ = x + 1.0
      self.assertFalse(torch._C._is_multithreading_enabled())
      del g3
      gc.collect()
    self.assertTrue(torch._C._is_multithreading_enabled())

  def test_recording_event_during_capture_raises(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      # Unlike a CUDA graph, a captured graph has no nodes that events could
      # mark, so recording an event raises without invalidating the capture.
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=capture continues
          RuntimeError,
          "recording events while capturing a graph is not supported",
      ):
        torch.Event(device="tpu").record()
      y = x + 1.0
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))

  def test_non_blocking_copy_from_pinned_cpu_tensor_during_capture(self):
    host = torch.ones(4).pin_memory()
    int_host = torch.ones(4, dtype=torch.int32).pin_memory()
    scalar_host = torch.tensor(1.0).pin_memory()
    x = torch.zeros(4, device="tpu")
    converted = torch.zeros(4, device="tpu")

    g = torch.accelerator.Graph()
    with self.stream, g:
      x.copy_(host, non_blocking=True)
      y = x * 2.0
      # Like on CUDA, the device converts the dtype of non-blocking copies.
      converted.copy_(int_host, non_blocking=True)
      z = scalar_host.to("tpu", non_blocking=True) + 1.0
    utils.assert_close(x.cpu(), torch.zeros(4))

    # Like on CUDA, every replay copies the contents that the pinned CPU
    # tensors have at replay time.
    for value in (7.0, 3.0):
      host.fill_(value)
      int_host.fill_(int(value))
      scalar_host.fill_(value)
      g.replay()
      utils.assert_close(x.cpu(), torch.full((4,), value))
      utils.assert_close(y.cpu(), torch.full((4,), 2 * value))
      utils.assert_close(converted.cpu(), torch.full((4,), value))
      utils.assert_close(z.cpu(), torch.tensor(value + 1.0))

    # Replaying on a stream waiting on pending device work defers the pinned
    # host-to-device transfer and retains the pinned host allocation until done.
    work_stream = torch.Stream(device=self.device)
    replay_stream = torch.Stream(device=self.device)
    event = torch.Event(device="tpu")
    with work_stream:
      pending = torch.ones(1024, device="tpu") + 1.0
      event.record()
    host.fill_(9.0)
    with replay_stream:
      replay_stream.wait_event(event)
      g.replay()
    torch.accelerator.current_stream().wait_stream(replay_stream)
    self.assertEqual(pending[0].item(), 2.0)
    utils.assert_close(y.cpu(), torch.full((4,), 18.0))

    # Sub-byte dtypes such as float4_e2m1fn_x2 pack two elements per byte on
    # the host, whereas their device layout is unpacked.
    fp4_host = torch.zeros(2, 4, dtype=torch.uint8, pin_memory=True).view(
        torch.float4_e2m1fn_x2
    )
    fp4_x = torch.empty(2, 4, dtype=torch.float4_e2m1fn_x2, device="tpu")
    g_fp4 = torch.accelerator.Graph()
    with self.stream, g_fp4:
      fp4_x.copy_(fp4_host, non_blocking=True)
    fp4_host.view(torch.uint8).fill_(0x21)
    g_fp4.replay()
    utils.assert_close(fp4_x.cpu(), fp4_host)

  def test_unsupported_copies_between_cpu_and_tpu_during_capture_raise(self):
    x = torch.ones(4, device="tpu")
    pinned = torch.zeros(4).pin_memory()
    g = torch.accelerator.Graph()
    with self.stream, g:
      # Like on CUDA, copies that would transfer CPU memory that is not pinned
      # raise without invalidating the capture.
      unpinned_copies = (
          (
              "pageable source",
              lambda: x.copy_(torch.zeros(4), non_blocking=True),
          ),
          ("tensor literal", lambda: torch.tensor([1.0, 2.0], device="tpu")),
          (
              "broadcast source",
              lambda: x.copy_(torch.zeros(1).pin_memory(), non_blocking=True),
          ),
          (
              "non-contiguous source",
              lambda: x.copy_(
                  torch.zeros(8).pin_memory()[::2], non_blocking=True
              ),
          ),
          (
              "blocking dtype conversion",
              lambda: x.copy_(torch.zeros(4, dtype=torch.int32).pin_memory()),
          ),
          ("cpu", x.cpu),
          ("tolist", x.tolist),
      )
      for name, copy in unpinned_copies:
        with self.subTest(name):
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=capture continues
              RuntimeError,
              "cannot copy between CPU and TPU tensors while capturing a graph"
              " unless the CPU tensor is pinned",
          ):
            copy()
      # Unlike on CUDA, non-blocking copies to pinned CPU tensors are not
      # supported.
      non_blocking_copies_to_pinned = (
          ("copy_", lambda: pinned.copy_(x, non_blocking=True)),
          ("to", lambda: x.to("cpu", non_blocking=True)),
      )
      for name, copy in non_blocking_copies_to_pinned:
        with self.subTest(name):
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=capture continues
              RuntimeError,
              "copying a TPU tensor to a pinned CPU tensor without blocking"
              " while capturing a graph is not supported",
          ):
            copy()
      # Like on CUDA, copying no elements works.
      torch.empty(0, device="tpu").copy_(torch.empty(0))
      torch.empty(0).copy_(torch.empty(0, device="tpu"))
      y = x + 1.0
    g.replay()
    utils.assert_close(y.cpu(), torch.full((4,), 2.0))

  def test_other_threads_during_capture_raise(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    w = torch.tensor([3.0, 4.0], device="tpu")
    capturing = []

    def read_captured_tensor():
      (x * 2.0).cpu()

    def write_captured_tensor():
      x.fill_(7.0)

    def use_capture_stream():
      with self.stream:
        (w * 2.0).cpu()

    def capture_on_capture_stream():
      with self.stream:
        torch.accelerator.Graph().capture_begin()

    g = torch.accelerator.Graph()
    with self.stream, g:
      y = x + 1.0
      # Other threads raise instead of waiting for capture to end. Unlike on
      # CUDA, capture does not record the operations of other threads on the
      # capture stream, and the errors do not invalidate the capture.
      read_error = self._run_on_other_thread(read_captured_tensor)
      write_error = self._run_on_other_thread(write_captured_tensor)
      stream_error = self._run_on_other_thread(use_capture_stream)
      capture_error = self._run_on_other_thread(capture_on_capture_stream)
      # Synchronizing the device skips the capture stream, whose operations
      # belong to the graph, so it neither raises nor invalidates the capture.
      self.assertIsNone(
          self._run_on_other_thread(torch.accelerator.synchronize)
      )
      # Other threads see that the capture stream is capturing.
      self.assertIsNone(
          self._run_on_other_thread(
              lambda: capturing.append(self.stream.is_capturing())
          )
      )
    self.assertEqual(capturing, [True])
    for error in (read_error, write_error):
      self.assertRegex(
          str(error),
          "cannot use a tensor accessed by a graph capture in progress",
      )
    self.assertRegex(
        str(stream_error),
        "cannot run operations on a stream that another thread is capturing",
    )
    self.assertRegex(
        str(capture_error), "another thread is capturing a graph on"
    )

    # The write of the other thread did not happen, rather than being lost
    # when capture ended.
    utils.assert_close(x.cpu(), torch.tensor([1.0, 2.0]))
    x.fill_(5.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([6.0, 6.0]))
    # Other threads can use the capture stream and the tensors that capture
    # accessed once capture ends, and their writes persist.
    self.assertIsNone(self._run_on_other_thread(read_captured_tensor))
    self.assertIsNone(self._run_on_other_thread(use_capture_stream))
    self.assertIsNone(self._run_on_other_thread(write_captured_tensor))
    utils.assert_close(x.cpu(), torch.full((2,), 7.0))

  def test_concurrent_captures_cannot_share_tensors(self):
    w = torch.tensor([1.0, 2.0], device="tpu")
    other_stream = torch.Stream(device=self.device)

    def capture_reading_w():
      g_other = torch.accelerator.Graph()
      with other_stream:
        g_other.capture_begin()
        try:
          _ = w + 1.0
        finally:
          g_other.capture_end()

    g = torch.accelerator.Graph()
    with self.stream, g:
      y = w * 2.0
      # The graph that another thread captures concurrently cannot read the
      # tensors that this capture accessed, which would make each graph read
      # the placeholders of the other.
      error = self._run_on_other_thread(capture_reading_w)
    self.assertRegex(
        str(error),
        "cannot use a tensor accessed by a graph capture in progress",
    )

    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 4.0]))
    utils.assert_close(w.cpu(), torch.tensor([1.0, 2.0]))
    # Once capture ends, other threads can capture graphs that read `w`.
    self.assertIsNone(self._run_on_other_thread(capture_reading_w))

  def _reset_graph_of_exited_thread(self, graph):
    """Resets `graph`, whose capturing thread exited during capture.

    The thread records that it exits when it destroys its thread-local
    variables, which the Python versions of some platforms do after
    `Thread.join()` returns, so this retries until the reset succeeds.

    Args:
      graph: The graph to reset.
    """
    deadline = time.monotonic() + 30.0
    while True:
      try:
        graph.reset()
        return
      except RuntimeError as e:
        if "another thread is capturing" not in str(e):
          raise
        if time.monotonic() > deadline:
          raise
        time.sleep(0.01)

  def test_capture_whose_graph_another_thread_destroys(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    graphs = [torch.accelerator.Graph()]
    with self.stream:
      graphs[0].capture_begin()
      try:
        _ = x * 2.0
        # Destroying the graph on another thread cannot end the capture, which
        # this thread owns.
        self.assertIsNone(self._run_on_other_thread(graphs.clear))
        self.assertTrue(self.stream.is_capturing())

        # The next capture of this thread ends the abandoned capture first.
        g = torch.accelerator.Graph()
        g.capture_begin()
        try:
          y = x + 1.0
        finally:
          g.capture_end()
      finally:
        if self.stream.is_capturing():
          # Clean up if abandoned capture stayed open on failure.
          cleanup_g = torch.accelerator.Graph()
          cleanup_g.capture_begin()
          cleanup_g.reset()

    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))
    utils.assert_close(x.cpu(), torch.tensor([1.0, 2.0]))

  def test_capture_whose_graph_is_destroyed_ends_once_its_thread_exits(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    graphs = [torch.accelerator.Graph()]
    captured = threading.Event()
    graph_destroyed = threading.Event()

    def capture_until_graph_destroyed():
      with self.stream:
        graphs[0].capture_begin()
        try:
          _ = x * 2.0
        finally:
          captured.set()
          graph_destroyed.wait(timeout=60)

    thread = threading.Thread(target=capture_until_graph_destroyed, daemon=True)
    thread.start()
    try:
      self.assertTrue(captured.wait(timeout=60))
      # Destroying the graph cannot end the capture while the other thread, which
      # owns it, is alive.
      graphs.clear()
      self.assertTrue(self.stream.is_capturing())
    finally:
      graph_destroyed.set()
      thread.join(timeout=60)
    self.assertFalse(thread.is_alive(), "the other thread hangs")

    # Once the other thread exited, the next operation of this thread ends the
    # capture, without capturing another graph. The other thread records that
    # it exits when it destroys its thread-local variables, which the Python
    # versions of some platforms do after Thread.join() returns.
    deadline = time.monotonic() + 30.0
    while self.stream.is_capturing():
      self.assertLess(time.monotonic(), deadline)
      time.sleep(0.01)
    utils.assert_close(x.cpu(), torch.tensor([1.0, 2.0]))
    with self.stream:
      y = x + 1.0
    utils.assert_close(y.cpu(), torch.tensor([2.0, 3.0]))

  def test_orphan_capture_ends_directly_on_stream_and_tensor_entries(self):
    # Exercises direct entry points (wait_event / query / tensor read) without
    # calling stream.is_capturing() first.
    for entry_name in ("wait_event", "query_stream", "tensor_read"):
      with self.subTest(entry_name):
        x = torch.tensor([1.0, 2.0], device="tpu")
        graphs = [torch.accelerator.Graph()]
        captured = threading.Event()
        release_owner = threading.Event()

        def owner(
            g_list=graphs,
            x_ref=x,
            ready=captured,
            done=release_owner,
        ):
          with self.stream:
            g_list[0].capture_begin()
            try:
              _ = x_ref * 2.0
            finally:
              ready.set()
              done.wait(timeout=60)

        thread = threading.Thread(target=owner, daemon=True)
        thread.start()
        try:
          self.assertTrue(captured.wait(timeout=60))
          graphs.clear()
        finally:
          release_owner.set()
          thread.join(timeout=60)
        self.assertFalse(thread.is_alive(), "the other thread hangs")

        event = torch.Event(device=self.device)
        event.record()
        deadline = time.monotonic() + 30.0
        while True:
          try:
            if entry_name == "wait_event":
              self.stream.wait_event(event)
            elif entry_name == "query_stream":
              self.stream.query()
            else:
              utils.assert_close(x.cpu(), torch.tensor([1.0, 2.0]))
            break
          except RuntimeError as e:
            msg = str(e)
            if (
                "another thread is capturing" not in msg
                and "in progress" not in msg
            ):
              raise
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertFalse(self.stream.is_capturing())
        utils.assert_close(x.cpu(), torch.tensor([1.0, 2.0]))

  def test_debug_check_ops_during_capture(self):
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")
    idx = torch.tensor([0, 2, 1], device="tpu", dtype=torch.int64)

    g = torch.accelerator.Graph()
    with mock.patch.dict(os.environ, {"TORCH_TPU_ENABLE_DEBUG_CHECKS": "1"}):
      with self.stream, g:
        taken = torch.take(vals, idx)
        hist = torch.histc(vals, bins=4, min=0.0, max=0.0)

    g.replay()
    utils.assert_close(taken.cpu(), torch.tensor([1.0, 3.0, 2.0]))
    utils.assert_close(hist.cpu(), torch.tensor([1.0, 1.0, 1.0, 1.0]))

  def test_tpu_stream_query_during_and_outside_capture(self):
    s = torch.tpu.Stream(device=self.device)
    other_s = torch.tpu.Stream(device=self.device)
    with self.assertRaises(  # ASSERT_RAISES_OK=unimplemented
        NotImplementedError
    ):
      s.query()

    for query_fn in (s.query, other_s.query):
      g = torch.accelerator.Graph()
      with s:
        g.capture_begin()
        try:
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
              RuntimeError, "cannot query streams while capturing a graph"
          ):
            query_fn()
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
              RuntimeError, "cannot query streams while capturing a graph"
          ):
            g.capture_end()
        finally:
          g.reset()

  def test_assert_async_passing_at_capture_false_at_replay_and_created_resize(
      self,
  ):
    x = torch.ones(2, device="tpu")
    a = torch.ones(2, device="tpu")
    b = torch.ones(3, device="tpu")
    g = torch.accelerator.Graph()
    with self.stream, g:
      torch._assert_async((x > 0).all())
      tmp = torch.empty(2, device="tpu")
      tmp.untyped_storage().resize_(16)
      tmp.fill_(3.0)
      y = x + tmp
      # Shape-mismatched torch.equal does not invalidate capture.
      self.assertFalse(torch.equal(a, b))

    # Flipping x to negative would fail an eager assertion, but replays skip
    # async assertions cleanly.
    x.fill_(-1.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.full((2,), 2.0))

  def test_failed_replay_validation_skips_earlier_host_inputs(self):
    host = torch.tensor([1.0, 2.0]).pin_memory()
    x_host = torch.zeros(2, device="tpu")
    x_out = torch.tensor([1.0, 2.0], device="tpu")
    y = torch.zeros(2, device="tpu")

    g = torch.accelerator.Graph()
    with self.stream, g:
      x_host.copy_(host, non_blocking=True)
      torch.add(x_out, 1.0, out=y)

    # Mutate `host` and resize `x_out` so replay validation fails after
    # `host` would otherwise have scheduled a host-to-device copy.
    host.fill_(42.0)
    x_out.resize_(4)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=verifies atomicity
        RuntimeError, "storage was resized"
    ):
      g.replay()
    utils.assert_close(x_host.cpu(), torch.zeros(2))
    utils.assert_close(y.cpu(), torch.zeros(2))

  def test_host_input_slice_lifetime_conj_neg_and_fp4_view(self):
    # Pinned slice with storage_offset > 0 and dropped Python host handle.
    base = torch.arange(8, dtype=torch.float32).pin_memory()
    sliced = base[2:6]
    out_slice = torch.zeros(4, device="tpu")
    out_dropped = torch.zeros(2, device="tpu")
    pinned_temp = torch.tensor([9.0, 10.0]).pin_memory()

    # Lazy conj and neg views of pinned CPU tensors.
    pinned_complex = torch.tensor(
        [1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64
    ).pin_memory()
    out_complex = torch.zeros(2, dtype=torch.complex64, device="tpu")
    pinned_neg = torch.tensor([2.0, -5.0]).pin_memory()
    out_neg = torch.zeros(2, device="tpu")

    # FP4 tensor written through uint8 view between replays.
    fp4 = torch.tensor([0x21, 0x43], dtype=torch.uint8, device="tpu").view(
        torch.float4_e2m1fn_x2
    )

    g = torch.accelerator.Graph()
    with self.stream, g:
      out_slice.copy_(sliced, non_blocking=True)
      out_dropped.copy_(pinned_temp, non_blocking=True)
      out_complex.copy_(pinned_complex.conj(), non_blocking=True)
      out_neg.copy_(torch._neg_view(pinned_neg), non_blocking=True)
      fp4_byte = fp4.view(torch.uint8) + 1

    del pinned_temp, sliced
    gc.collect()
    base[2:6].copy_(torch.tensor([11.0, 12.0, 13.0, 14.0]))
    pinned_complex.copy_(torch.tensor([5.0 + 6.0j, -7.0 + 8.0j]))
    pinned_neg.copy_(torch.tensor([-3.0, 8.0]))
    fp4.view(torch.uint8).copy_(torch.tensor([0x10, 0x32], dtype=torch.uint8))

    g.replay()
    utils.assert_close(out_slice.cpu(), torch.tensor([11.0, 12.0, 13.0, 14.0]))
    utils.assert_close(out_dropped.cpu(), torch.tensor([9.0, 10.0]))
    utils.assert_close(
        out_complex.cpu(),
        torch.tensor([5.0 - 6.0j, -7.0 - 8.0j], dtype=torch.complex64),
    )
    utils.assert_close(out_neg.cpu(), torch.tensor([3.0, -8.0]))
    self.assertEqual(fp4_byte.cpu().tolist(), [0x11, 0x33])


if __name__ == "__main__":
  absltest.main()
