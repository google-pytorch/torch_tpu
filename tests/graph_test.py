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

  def test_replay_during_capture_invalidates_capture(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    other_graph = torch.accelerator.Graph()
    with self.stream, other_graph:
      _ = x * 3.0

    g = torch.accelerator.Graph()
    with self.stream:
      g.capture_begin()
      _ = x + 1.0
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
          RuntimeError, "cannot replay a graph"
      ):
        other_graph.replay()
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=checks invalidation
          RuntimeError,
          "graph capture failed because of an earlier error during"
          " capture: .*cannot replay a graph",
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
    other_graph = torch.accelerator.Graph()
    with self.stream, other_graph:
      _ = x * 2.0

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
            RuntimeError, "cannot replay a graph while capturing a graph"
        ):
          other_graph.replay()
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

  def test_autograd_multithreading_restored_after_error_reset_and_del(self):
    x = torch.tensor([1.0, 2.0], device="tpu")
    other_graph = torch.accelerator.Graph()
    with self.stream, other_graph:
      _ = x * 2.0

    # 1. After invalidating error + capture_end()
    g1 = torch.accelerator.Graph()
    with self.stream:
      g1.capture_begin()
      self.assertFalse(torch._C._is_multithreading_enabled())
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=state after error
          RuntimeError, "cannot replay a graph while capturing a graph"
      ):
        other_graph.replay()
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
