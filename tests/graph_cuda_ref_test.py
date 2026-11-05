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

"""Checks that graph capture and replay on TPU behave like on CUDA.

The tests run on TPU with torch.accelerator.Graph and TorchTPU, and on CUDA
with torch.cuda.CUDAGraph and PyTorch alone, since loading TorchTPU makes TPU
the accelerator. Error messages differ between backends, so the tests only
check that errors are raised.

graph_test covers where TPU differs from CUDA on purpose: recording events,
constructing generators, reading or setting generator states, copying TPU
tensors to pinned CPU tensors without blocking, and running operations on other
streams or threads during capture.
"""

import contextlib
import gc

from absl import flags
from absl.testing import absltest
import torch

_BACKEND = flags.DEFINE_enum(
    "backend", "tpu", ["tpu", "gpu"], "The backend to test."
)


_TORCH_VERSION = tuple(
    int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
)


@absltest.skipIf(
    _TORCH_VERSION < (2, 12),
    "torch.accelerator.Graph requires PyTorch >= 2.12.",
)
class GraphCudaRefTest(absltest.TestCase):  # ABSLTEST_OK=no TorchTPU on GPU
  """Tests graph capture and replay on the current device."""

  def setUp(self):
    super().setUp()
    device_type = "cuda" if _BACKEND.value == "gpu" else "tpu"
    self.assertEqual(torch.accelerator.current_accelerator().type, device_type)
    self.device = torch.device(
        device_type, torch.accelerator.current_device_index()
    )
    self.device_module = torch.get_device_module(device_type)
    torch.manual_seed(1234)
    # Graphs must be captured on a non-default stream.
    self.stream = torch.Stream(device=self.device)

  def tearDown(self):
    torch.accelerator.synchronize()
    super().tearDown()

  def _new_graph(self, keep_graph=False, pool=None):
    # CUDA does not implement torch.accelerator.Graph.
    if self.device.type == "cuda":
      return torch.cuda.CUDAGraph(keep_graph=keep_graph)
    return torch.accelerator.Graph(keep_graph=keep_graph, pool=pool)

  @contextlib.contextmanager
  def _capture(self, graph, pool=None):
    """Captures the body of the with statement on the side stream."""
    with self.stream:
      if self.device.type == "cuda":
        graph.capture_begin(pool=pool)
      else:
        if pool is not None:
          graph.graph_pool = pool
        graph.capture_begin()
      try:
        yield
      except BaseException:
        try:
          graph.capture_end()
        except RuntimeError:
          pass
        raise
      else:
        graph.capture_end()

  def _warm_up(self, fn):
    """Calls fn on the side stream, as CUDA requires before capture.

    CUDA initializes some state on first use, such as cuBLAS handles, which
    capture does not allow.

    Args:
      fn: The function to call.
    """
    current_stream = torch.accelerator.current_stream()
    self.stream.wait_stream(current_stream)
    with self.stream:
      fn()
    current_stream.wait_stream(self.stream)

  def _assert_values(self, tensor, expected):
    self.assertEqual(tensor.cpu().tolist(), expected)

  def _assert_raises_without_invalidating_capture(self, cases):
    """Checks that every call raises, and that capture continues without it."""
    for name, call in cases:
      with self.subTest(name):
        x = torch.ones(4, device=self.device)
        g = self._new_graph()
        with self._capture(g):
          with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
            call(x)
          y = x + 1.0
        g.replay()
        self._assert_values(y, [2.0] * 4)

  def test_capture_on_default_stream_raises(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    g = self._new_graph()
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.capture_begin()

    # The graph can still capture on another stream, and replay on the default
    # stream.
    with self._capture(g):
      y = x + 1.0
    g.replay()
    self._assert_values(y, [2.0, 3.0])

  def test_replay_requires_capture_and_capture_requires_reset(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    g = self._new_graph()
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.replay()

    with self._capture(g):
      _ = x + 1.0
    with self.stream, self.assertRaises(  # ASSERT_RAISES_OK=CUDA parity
        RuntimeError
    ):
      g.capture_begin()

    g.reset()
    with self._capture(g):
      y = x + 2.0
    g.replay()
    self._assert_values(y, [3.0, 4.0])

  def test_other_methods_raise_without_captured_graph(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    g = self._new_graph()
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.instantiate()
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.pool()
    with self.stream, self.assertRaises(  # ASSERT_RAISES_OK=CUDA parity
        RuntimeError
    ):
      g.capture_end()

    with self._capture(g):
      _ = x + 1.0
    # Only graphs that keep the captured graph can be instantiated again.
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.instantiate()
    g.reset()
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.replay()
    with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
      g.pool()

    # The graph can still capture and replay.
    with self._capture(g):
      y = x + 2.0
    g.replay()
    self._assert_values(y, [3.0, 4.0])

  def test_capture_continues_after_capture_end_on_another_stream(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    other_stream = torch.Stream(device=self.device)
    g = self._new_graph()
    with self.stream:
      g.capture_begin()
      y = x + 1.0
      with other_stream, self.assertRaises(  # ASSERT_RAISES_OK=CUDA parity
          RuntimeError
      ):
        g.capture_end()
      z = y * 2.0
      g.capture_end()
    g.replay()
    self._assert_values(z, [4.0, 6.0])

  def test_capture_stream_reports_capture(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    other_stream = torch.Stream(device=self.device)
    g = self._new_graph()
    self.assertFalse(self.stream.is_capturing())
    with self._capture(g):
      self.assertTrue(self.stream.is_capturing())
      self.assertTrue(torch.accelerator.current_stream().is_capturing())
      self.assertFalse(other_stream.is_capturing())
      y = x + 1.0
    self.assertFalse(self.stream.is_capturing())
    g.replay()
    self._assert_values(y, [2.0, 3.0])

  def test_kept_graphs_instantiate_on_first_replay_or_when_asked(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    kept = self._new_graph(keep_graph=True)
    debug = self._new_graph()
    # Debug mode keeps the captured graph, like keep_graph=True.
    debug.enable_debug_mode()
    with self._capture(kept):
      y = x + 1.0
    with self._capture(debug):
      z = x * 2.0

    kept.replay()
    debug.instantiate()
    debug.replay()
    self._assert_values(y, [2.0, 3.0])
    self._assert_values(z, [2.0, 4.0])

    # Graphs that keep the captured graph can be instantiated again.
    x.fill_(5.0)
    for graph in (kept, debug):
      graph.instantiate()
      graph.replay()
    self._assert_values(y, [6.0, 6.0])
    self._assert_values(z, [10.0, 10.0])

  def test_replays_rerun_captured_operations_on_current_values(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    z = torch.tensor([10.0, 20.0], device=self.device)
    acc = torch.zeros(2, device=self.device)
    g = self._new_graph()
    with self._capture(g):
      y = x + 1.0
      acc.add_(y)
      x.copy_(z)

    # Capture runs nothing.
    self._assert_values(x, [1.0, 2.0])
    self._assert_values(acc, [0.0, 0.0])

    # Every replay reads the values that tensors have when it runs, before it
    # overwrites them.
    g.replay()
    self._assert_values(y, [2.0, 3.0])
    self._assert_values(acc, [2.0, 3.0])
    self._assert_values(x, [10.0, 20.0])
    g.replay()
    self._assert_values(y, [11.0, 21.0])
    self._assert_values(acc, [13.0, 24.0])
    z.fill_(100.0)
    g.replay()
    self._assert_values(x, [100.0, 100.0])

  def test_replays_write_to_tensors_from_capture(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    out = torch.zeros(2, device=self.device)
    acc = torch.zeros(2, device=self.device)
    acc_version = acc._version
    g = self._new_graph()
    with self._capture(g):
      y = x * 2.0
      torch.add(x, 1.0, out=out)
      acc.add_(x)
    # Capture increments version counters, but replays do not.
    self.assertGreater(acc._version, acc_version)
    acc_version = acc._version
    y_data_ptr = y.data_ptr()

    out.fill_(-1.0)
    for _ in range(3):
      g.replay()
    self.assertEqual(acc._version, acc_version)
    self.assertEqual(y.data_ptr(), y_data_ptr)
    self._assert_values(y, [2.0, 4.0])
    self._assert_values(out, [2.0, 3.0])
    self._assert_values(acc, [3.0, 6.0])

    # Outputs keep the values of the last replay after the graph is deleted.
    del g
    gc.collect()
    y.add_(1.0)
    self._assert_values(y, [3.0, 5.0])

  def test_replays_views_reductions_matmuls_and_casts(self):
    x = torch.arange(6, dtype=torch.float32, device=self.device)
    ones = torch.ones(3, 2, device=self.device)
    self._warm_up(lambda: x.view(2, 3) @ ones)
    g = self._new_graph()
    with self._capture(g):
      x[3:].mul_(2.0)
      transposed = x.view(2, 3).t() + 0.5
      total = x.sum()
      product = x.view(2, 3) @ ones
      truncated = (x + 0.75).to(torch.int32)

    g.replay()
    self._assert_values(x, [0.0, 1.0, 2.0, 6.0, 8.0, 10.0])
    self._assert_values(transposed, [[0.5, 6.5], [1.5, 8.5], [2.5, 10.5]])
    self._assert_values(total, 27.0)
    self._assert_values(product, [[3.0, 3.0], [24.0, 24.0]])
    self._assert_values(truncated, [0, 1, 2, 6, 8, 10])
    g.replay()
    self._assert_values(x, [0.0, 1.0, 2.0, 12.0, 16.0, 20.0])
    self._assert_values(transposed, [[0.5, 12.5], [1.5, 16.5], [2.5, 20.5]])
    self._assert_values(total, 51.0)
    self._assert_values(product, [[3.0, 3.0], [48.0, 48.0]])
    self._assert_values(truncated, [0, 1, 2, 12, 16, 20])

  def test_replays_repeat_interleave_with_output_size(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    repeats = torch.tensor([1, 2], device=self.device)
    self._warm_up(lambda: x.repeat_interleave(repeats, output_size=3))
    g = self._new_graph()
    with self._capture(g):
      # Unlike without output_size, the output shape does not depend on the
      # values of repeats, so capture records the operation.
      y = x.repeat_interleave(repeats, output_size=3)

    g.replay()
    self._assert_values(y, [1.0, 2.0, 2.0])
    repeats.copy_(torch.tensor([2, 1]))
    g.replay()
    self._assert_values(y, [1.0, 1.0, 2.0])

  def test_graphs_sharing_pool_replay_in_capture_order(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    g1 = self._new_graph()
    with self._capture(g1):
      y1 = x + 1.0
    g2 = self._new_graph(pool=g1.pool())
    with self._capture(g2, pool=g1.pool()):
      y2 = y1 * 2.0
    self.assertEqual(g2.pool(), g1.pool())

    g1.replay()
    g2.replay()
    self._assert_values(y2, [4.0, 6.0])
    x.fill_(5.0)
    g1.replay()
    g2.replay()
    self._assert_values(y2, [12.0, 12.0])

  def test_replaying_empty_graph(self):
    g = self._new_graph()
    with self._capture(g):
      pass
    g.replay()

  def test_replay_on_another_stream(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    g = self._new_graph()
    with self._capture(g):
      y = x + 1.0
    other_stream = torch.Stream(device=self.device)
    current_stream = torch.accelerator.current_stream()
    other_stream.wait_stream(current_stream)
    with other_stream:
      g.replay()
    current_stream.wait_stream(other_stream)
    self._assert_values(y, [2.0, 3.0])

  def test_cpu_scalars_keep_value_from_capture(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    scale = torch.tensor(2.0)
    g = self._new_graph()
    with self._capture(g):
      y = x * scale
    scale.fill_(5.0)
    g.replay()
    self._assert_values(y, [2.0, 4.0])

  def test_replay_after_inputs_and_outputs_are_freed(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    keep = torch.zeros(2, device=self.device)
    g = self._new_graph()
    with self._capture(g):
      y = x + 1.0
      keep.copy_(y * 2.0)

    # The graph keeps using the memory of the tensors it captured.
    del x, y
    g.replay()
    self._assert_values(keep, [4.0, 6.0])

  def test_backward_and_optimizer_step(self):
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

    w = w0.to(self.device).requires_grad_()
    opt = torch.optim.SGD([w], lr=0.1)
    static_x = torch.zeros(4, device=self.device)
    static_t = torch.zeros((), device=self.device)

    def train_step():
      (((w * static_x).sum() - static_t) ** 2).backward()
      opt.step()

    # The gradients of the warm-up step are zero, so it does not change w.
    self._warm_up(train_step)
    opt.zero_grad(set_to_none=True)
    g = self._new_graph()
    with self._capture(g):
      train_step()
    torch.testing.assert_close(  # TORCH_ASSERT_CLOSE_OK=no TorchTPU on CUDA
        w.detach().cpu(), w0
    )

    for x, t in zip(xs, ts, strict=True):
      static_x.copy_(x)
      static_t.copy_(t)
      g.replay()
    torch.testing.assert_close(  # TORCH_ASSERT_CLOSE_OK=no TorchTPU on CUDA
        w.detach().cpu(), w_ref.detach(), rtol=1e-4, atol=1e-4
    )

  def test_backward_accumulates_into_existing_gradients(self):
    w = torch.ones(4, device=self.device, requires_grad=True)
    x = torch.full((4,), 2.0, device=self.device)
    self._warm_up(lambda: (w * x).sum().backward())
    with torch.no_grad():
      w.grad.zero_()
    grad = w.grad
    g = self._new_graph()
    with self._capture(g):
      (w * x).sum().backward()
    self._assert_values(grad, [0.0] * 4)

    g.replay()
    self._assert_values(grad, [2.0] * 4)
    x.fill_(3.0)
    g.replay()
    self._assert_values(grad, [5.0] * 4)
    self.assertIs(w.grad, grad)

  def test_replays_advance_rng_offset(self):
    out = torch.empty(16, device=self.device)
    g = self._new_graph()
    with self._capture(g):
      out.uniform_()

    g.replay()
    sample1 = out.cpu()
    g.replay()
    self.assertFalse(torch.equal(out.cpu(), sample1))

    # Setting the seed resets the offset that the next replay starts from.
    torch.manual_seed(1234)
    g.replay()
    self.assertTrue(torch.equal(out.cpu(), sample1))

  def test_replays_draw_same_numbers_as_eager_operations(self):
    out = torch.empty(16, device=self.device)
    g = self._new_graph()
    with self._capture(g):
      out.uniform_()

    torch.manual_seed(7)
    first = torch.empty(16, device=self.device).uniform_()
    second = torch.empty(16, device=self.device).uniform_()
    # A replay draws the numbers that the captured operation would draw if it
    # ran eagerly instead, and advances the offset as much.
    torch.manual_seed(7)
    g.replay()
    after_replay = torch.empty(16, device=self.device).uniform_()
    self.assertTrue(torch.equal(out.cpu(), first.cpu()))
    self.assertTrue(torch.equal(after_replay.cpu(), second.cpu()))

  def test_waiting_for_device_invalidates_capture(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    repeats = torch.tensor([1, 2], device=self.device)
    one_repeat = torch.tensor([2], device=self.device)
    pinned = torch.zeros(2).pin_memory()
    event = torch.Event(device=self.device)
    event.record()
    other_stream = torch.Stream(device=self.device)
    other_graph = self._new_graph()
    with self._capture(other_graph):
      _ = x * 3.0

    cases = (
        ("synchronize device", torch.accelerator.synchronize),
        ("synchronize capture stream", self.stream.synchronize),
        ("synchronize other stream", other_stream.synchronize),
        ("query capture stream", self.stream.query),
        ("query other stream", other_stream.query),
        ("query event", event.query),
        ("synchronize event", event.synchronize),
        ("wait on event", lambda: self.stream.wait_event(event)),
        ("item", lambda: x.sum().item()),
        ("replay", other_graph.replay),
        ("blocking copy from pinned tensor", lambda: x.copy_(pinned)),
        ("blocking copy to pinned tensor", lambda: pinned.copy_(x)),
        # The shapes of these outputs depend on the values of tensors.
        ("nonzero", x.nonzero),
        ("boolean mask indexing", lambda: x[x > 1.0]),
        ("unique", x.unique),
        ("unique dim", lambda: x.unsqueeze(0).unique(dim=0)),
        ("repeat_interleave", lambda: x.repeat_interleave(repeats)),
        (
            "repeat_interleave with one repeat count",
            lambda: x.repeat_interleave(one_repeat),
        ),
    )
    for name, wait in cases:
      with self.subTest(name):
        g = self._new_graph()
        with self.stream:
          g.capture_begin()
          _ = x + 1.0
          with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
            wait()
          # CUDA cannot allocate memory on a stream whose capture is
          # invalidated, so capture ends right away.
          with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
            g.capture_end()
        with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
          g.replay()

        # Like after a successful capture, reset() is required to capture again.
        with self.stream, self.assertRaises(  # ASSERT_RAISES_OK=CUDA parity
            RuntimeError
        ):
          g.capture_begin()
        g.reset()
        with self._capture(g):
          y = x + 2.0
        g.replay()
        self._assert_values(y, [3.0, 4.0])

  def test_reseeding_or_accessing_rng_offset_during_capture_raises(self):
    generator = self.device_module.default_generators[self.device.index]
    self._assert_raises_without_invalidating_capture((
        ("manual_seed", lambda _: generator.manual_seed(4321)),
        ("seed", lambda _: generator.seed()),
        ("get_offset", lambda _: generator.get_offset()),
        ("set_offset", lambda _: generator.set_offset(0)),
    ))
    self.assertEqual(generator.initial_seed(), 1234)

  def test_setting_current_seed_during_capture(self):
    generator = self.device_module.default_generators[self.device.index]
    out = torch.empty(16, device=self.device)
    g = self._new_graph()
    with self._capture(g):
      torch.manual_seed(1234)
      self.device_module.manual_seed(1234)
      generator.manual_seed(1234)
      self.assertEqual(generator.initial_seed(), 1234)
      out.uniform_()

    # Setting the current seed keeps the offset, so replays still advance it.
    g.replay()
    sample1 = out.cpu()
    g.replay()
    self.assertFalse(torch.equal(out.cpu(), sample1))

  def test_invalid_arguments_during_capture_raise(self):
    self._assert_raises_without_invalidating_capture((
        ("mismatched shapes", lambda x: x + x[:3]),
        ("invalid view", lambda x: x.view(3)),
        (
            "mismatched matrix shapes",
            lambda x: x.view(2, 2) @ x.view(4, 1),
        ),
    ))

  def test_copies_of_unpinned_cpu_memory_during_capture_raise(self):
    # Pinned memory is allocated before capture.
    pinned_scalar = torch.zeros(1).pin_memory()
    pinned_strided = torch.zeros(8).pin_memory()[::2]
    pinned_int = torch.zeros(4, dtype=torch.int32).pin_memory()
    self._assert_raises_without_invalidating_capture((
        ("blocking copy", lambda x: x.copy_(torch.zeros(4))),
        (
            "non-blocking copy",
            lambda x: x.copy_(torch.zeros(4), non_blocking=True),
        ),
        ("0-dim copy", lambda x: x.copy_(torch.tensor(2.0))),
        (
            "tensor literal",
            lambda _: torch.tensor([1.0, 2.0], device=self.device),
        ),
        (
            "0-dim tensor literal",
            lambda _: torch.tensor(2.0, device=self.device),
        ),
        # These copies go through temporary CPU tensors, which are not pinned.
        (
            "broadcasting copy from pinned tensor",
            lambda x: x.copy_(pinned_scalar, non_blocking=True),
        ),
        (
            "non-contiguous copy from pinned tensor",
            lambda x: x.copy_(pinned_strided, non_blocking=True),
        ),
        (
            "blocking dtype conversion from pinned tensor",
            lambda x: x.copy_(pinned_int),
        ),
        ("cpu", lambda x: x.cpu()),
        ("tolist", lambda x: x.tolist()),
        # bincount copies a reduction of its input to a CPU tensor, which is
        # not pinned.
        ("bincount", lambda x: x.long().bincount()),
    ))

  def test_non_blocking_copies_from_pinned_memory_read_it_at_replay(self):
    host = torch.ones(4).pin_memory()
    int_host = torch.ones(4, dtype=torch.int32).pin_memory()
    scalar_host = torch.tensor(1.0).pin_memory()
    x = torch.zeros(4, device=self.device)
    converted = torch.zeros(4, device=self.device)
    g = self._new_graph()
    with self._capture(g):
      x.copy_(host, non_blocking=True)
      y = x * 2.0
      converted.copy_(int_host, non_blocking=True)
      z = scalar_host.to(self.device, non_blocking=True) + 1.0
    self._assert_values(x, [0.0] * 4)

    # Every replay copies the values that the pinned tensors have when it runs.
    for value in (7.0, 3.0):
      host.fill_(value)
      int_host.fill_(int(value))
      scalar_host.fill_(value)
      g.replay()
      self._assert_values(x, [value] * 4)
      self._assert_values(y, [2 * value] * 4)
      self._assert_values(converted, [value] * 4)
      self._assert_values(z, value + 1.0)

  def test_copying_no_elements_during_capture(self):
    x = torch.tensor([1.0, 2.0], device=self.device)
    g = self._new_graph()
    with self._capture(g):
      torch.empty(0, device=self.device).copy_(torch.empty(0))
      torch.empty(0).copy_(torch.empty(0, device=self.device))
      torch.empty(0, 2, device=self.device).copy_(torch.empty(1, 2))
      with self.assertRaises(RuntimeError):  # ASSERT_RAISES_OK=CUDA parity
        torch.empty(0, 2, device=self.device).copy_(torch.empty(0, 3))
      y = x + 1.0
    g.replay()
    self._assert_values(y, [2.0, 3.0])

  def test_pinned_slice_and_normal_during_capture(self):
    base = torch.arange(8, dtype=torch.float32).pin_memory()
    out_slice = torch.zeros(4, device=self.device)
    std = torch.full((4,), 0.5, device=self.device)

    g = self._new_graph()
    with self._capture(g):
      out_slice.copy_(base[2:6], non_blocking=True)
      sample = torch.normal(0.0, std)

    base[2:6].copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    g.replay()
    self._assert_values(out_slice, [10.0, 20.0, 30.0, 40.0])
    self.assertEqual(sample.shape, (4,))


if __name__ == "__main__":
  absltest.main()
