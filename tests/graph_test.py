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

import threading

from absl.testing import absltest
import torch
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

    def run():
      try:
        fn(*args)
      except RuntimeError as e:
        errors.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=60)
    self.assertFalse(thread.is_alive(), "the other thread hangs")
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

  def test_rng_ops_in_captured_graph(self):
    torch.tpu.manual_seed(1234)
    out = torch.empty((16,), dtype=torch.float32, device="tpu")

    g = torch.accelerator.Graph()
    with self.stream, g:
      out.uniform_(0.0, 1.0)

    g.replay()
    sample1 = out.cpu().clone()

    # Second replay without resetting seed should advance RNG state and produce
    # a different sample.
    g.replay()
    sample2 = out.cpu().clone()
    self.assertFalse(torch.allclose(sample1, sample2))

    # Resetting the seed before replay should reproduce sample1.
    torch.tpu.manual_seed(1234)
    g.replay()
    sample1_repeat = out.cpu().clone()
    utils.assert_close(sample1, sample1_repeat)

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

    # 5. Resetting a graph that another thread is capturing warns, and leaves
    # the current thread usable.
    g_abandoned = torch.accelerator.Graph()

    def begin_capture(graph):
      with torch.Stream(device=x.device):
        graph.capture_begin()

    self.assertIsNone(self._run_on_other_thread(begin_capture, g_abandoned))
    with self.assertWarnsRegex(UserWarning, "remains in capture mode"):
      g_abandoned.reset()
    del g_abandoned
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

  def test_generator_during_capture(self):
    torch.manual_seed(1234)
    generator = torch.tpu.default_generators[
        torch.accelerator.current_device_index()
    ]
    state = generator.get_state()
    out = torch.empty((16,), dtype=torch.float32, device="tpu")

    g = torch.accelerator.Graph()
    with self.stream, g:
      # Like on CUDA, setting the seed that generators already have keeps their
      # offsets, and reading the seed works.
      torch.manual_seed(1234)
      generator.manual_seed(1234)
      self.assertEqual(generator.initial_seed(), 1234)
      # Like on CUDA, changing the seed or the offset and reading the offset
      # raise without invalidating the capture. Unlike on CUDA, so do
      # constructing a generator and reading or setting the state, as the RNG
      # state lives on the device.
      cases = (
          (
              "new generator",
              lambda: torch.Generator(device="tpu"),
              "cannot construct a new generator while capturing a graph",
          ),
          (
              "manual_seed",
              lambda: generator.manual_seed(4321),
              (
                  "cannot set the seed of a generator while capturing a graph,"
                  " except to its current seed"
              ),
          ),
          ("seed", generator.seed, "cannot seed a generator randomly"),
          (
              "get_offset",
              generator.get_offset,
              "cannot read the offset of a generator",
          ),
          (
              "set_offset",
              lambda: generator.set_offset(0),
              "cannot set the offset of a generator",
          ),
          (
              "get_state",
              generator.get_state,
              "cannot read the state of a generator",
          ),
          (
              "set_state",
              lambda: generator.set_state(state),
              "cannot set the state of a generator",
          ),
      )
      for name, call, message in cases:
        with self.subTest(name):
          with self.assertRaisesRegex(  # ASSERT_RAISES_OK=capture continues
              RuntimeError, message
          ):
            call()
      out.uniform_(0.0, 1.0)

    # Every replay advances the offset.
    g.replay()
    sample1 = out.cpu().clone()
    g.replay()
    sample2 = out.cpu().clone()
    self.assertFalse(torch.allclose(sample1, sample2))
    self.assertEqual(generator.initial_seed(), 1234)

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

    def read_captured_tensor():
      (x * 2.0).cpu()

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
      stream_error = self._run_on_other_thread(use_capture_stream)
      capture_error = self._run_on_other_thread(capture_on_capture_stream)
    self.assertRegex(
        str(read_error),
        "cannot use a tensor accessed by a graph capture in progress",
    )
    self.assertRegex(
        str(stream_error),
        "cannot run operations on a stream that another thread is capturing",
    )
    self.assertRegex(
        str(capture_error), "another thread is capturing a graph on"
    )

    x.fill_(5.0)
    g.replay()
    utils.assert_close(y.cpu(), torch.tensor([6.0, 6.0]))
    # Other threads can use the capture stream and the tensors that capture
    # accessed once capture ends.
    self.assertIsNone(self._run_on_other_thread(read_captured_tensor))
    self.assertIsNone(self._run_on_other_thread(use_capture_stream))


if __name__ == "__main__":
  absltest.main()
