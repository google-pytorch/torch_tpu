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
import threading

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

    # 5. Resetting a graph that another thread is capturing raises
    # RuntimeError, and leaves the current thread usable.
    g_abandoned = torch.accelerator.Graph()

    def begin_capture(graph):
      with torch.Stream(device=x.device):
        graph.capture_begin()

    self.assertIsNone(self._run_on_other_thread(begin_capture, g_abandoned))
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
        RuntimeError, "cannot reset a graph that another thread is capturing"
    ):
      g_abandoned.reset()
    utils.assert_close((x + 1.0).cpu(), torch.tensor([2.0, 3.0]))

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
    other_stream = torch.Stream(device=self.device)
    other_graph = torch.accelerator.Graph()
    with self.stream, other_graph:
      _ = x * 3.0

    # Like on CUDA, these calls wait for the device, so they raise and
    # invalidate the capture.
    sync_message = "cannot materialize tensors or synchronize streams"
    cases = (
        ("synchronize device", torch.accelerator.synchronize, sync_message),
        ("synchronize capture stream", self.stream.synchronize, sync_message),
        ("synchronize other stream", other_stream.synchronize, sync_message),
        ("replay", other_graph.replay, "cannot replay a graph"),
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
    with self.stream, g:
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
      ):
        copied = torch.tensor([3.0, 4.0], device="tpu")
      y = x + 1.0
    utils.assert_close(copied.cpu(), torch.tensor([3.0, 4.0]))
    g.replay()
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


if __name__ == "__main__":
  absltest.main()
