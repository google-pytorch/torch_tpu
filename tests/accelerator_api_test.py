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

"""Systematic tests for torch.accelerator.* APIs supported by TorchTPU."""

from collections import OrderedDict
import concurrent.futures
import gc
import time
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.utils import utils as tt_utils
from tests import oss_utils
from tests import seed_test_utils

EagerMode: TypeAlias = execution_mode.EagerMode


def setUpModule():
  # Compare torch.accelerator APIs with the internal torch.tpu APIs they use.
  annotations._internal_apis_allowed_for_testing = True  # pylint: disable=protected-access


class AcceleratorDeviceTest(seed_test_utils.RepeatableTest):
  """Tests device and availability APIs in torch.accelerator."""

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self._orig_device_index = torch.accelerator.current_device_index()

  def tearDown(self):
    torch.accelerator.set_device_index(self._orig_device_index)
    super().tearDown()

  def test_current_accelerator(self):
    acc_default = torch.accelerator.current_accelerator()
    self.assertIsInstance(acc_default, torch.device)
    self.assertEqual(acc_default, torch.device("tpu"))
    self.assertIsNone(acc_default.index)

    acc_unchecked = torch.accelerator.current_accelerator(check_available=False)
    self.assertEqual(acc_unchecked, torch.device("tpu"))

    acc_checked = torch.accelerator.current_accelerator(check_available=True)
    self.assertEqual(acc_checked, torch.device("tpu"))

  def test_is_available(self):
    avail = torch.accelerator.is_available()
    self.assertIsInstance(avail, bool)
    self.assertTrue(avail)
    self.assertEqual(avail, torch.tpu.is_available())
    self.assertEqual(avail, torch.accelerator.device_count() > 0)

  def test_device_count(self):
    count = torch.accelerator.device_count()
    self.assertIsInstance(count, int)
    self.assertGreater(count, 0)
    self.assertEqual(count, torch.tpu.device_count())

  def test_current_device_index(self):
    idx = torch.accelerator.current_device_index()
    self.assertIsInstance(idx, int)
    self.assertGreaterEqual(idx, 0)
    self.assertLess(idx, torch.accelerator.device_count())
    self.assertEqual(idx, torch.tpu.current_device())

  def test_current_device_idx_deprecated(self):
    with self.assertWarnsRegex(
        FutureWarning, "Use `current_device_index` instead"
    ):
      idx = torch.accelerator.current_device_idx()
    self.assertIsInstance(idx, int)
    self.assertEqual(idx, torch.accelerator.current_device_index())

  def test_set_device_index(self):
    for valid_device in (0, "tpu:0", torch.device("tpu", 0)):
      with self.subTest(valid_device=valid_device):
        torch.accelerator.set_device_index(valid_device)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    if torch.accelerator.device_count() > 1:
      for device_one in (1, "tpu:1", torch.device("tpu", 1)):
        with self.subTest(device_one=device_one):
          torch.accelerator.set_device_index(device_one)
          self.assertEqual(torch.accelerator.current_device_index(), 1)
          torch.accelerator.set_device_index(0)
          self.assertEqual(torch.accelerator.current_device_index(), 0)

    # Negative index is a documented no-op in C++ at::accelerator::setDeviceIndex.
    before_idx = torch.accelerator.current_device_index()
    torch.accelerator.set_device_index(-1)
    self.assertEqual(torch.accelerator.current_device_index(), before_idx)

    # Positional-only parameter enforcement.
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      torch.accelerator.set_device_index(device=0)  # pytype: disable=wrong-keyword-args

    # Unindexed device raises ValueError.
    for unindexed in ("tpu", torch.device("tpu")):
      with self.subTest(unindexed=unindexed):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
            ValueError,
            "Expected a torch.device with a specified index or an integer",
        ):
          torch.accelerator.set_device_index(unindexed)

    # Mismatched device type raises ValueError.
    for wrong_type in ("cpu", "cpu:0", torch.device("cpu", 0)):
      with self.subTest(wrong_type=wrong_type):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
            ValueError, "doesn't match the current accelerator"
        ):
          torch.accelerator.set_device_index(wrong_type)

    # Out-of-range device index raises RuntimeError.
    out_of_range = torch.accelerator.device_count()
    with self.assertRaises(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        RuntimeError
    ):
      torch.accelerator.set_device_index(out_of_range)

  def test_set_device_idx_deprecated(self):
    for valid_device in (0, "tpu:0", torch.device("tpu", 0)):
      with self.subTest(valid_device=valid_device):
        with self.assertWarnsRegex(
            FutureWarning, "Use `set_device_index` instead"
        ):
          torch.accelerator.set_device_idx(valid_device)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      with self.assertWarnsRegex(
          FutureWarning, "Use `set_device_index` instead"
      ):
        torch.accelerator.set_device_idx(device=0)  # pytype: disable=wrong-keyword-args

  def test_device_index_context_manager(self):
    initial_idx = torch.accelerator.current_device_index()

    # None is a no-op context manager.
    with torch.accelerator.device_index(None):
      self.assertEqual(torch.accelerator.current_device_index(), initial_idx)
    self.assertEqual(torch.accelerator.current_device_index(), initial_idx)

    # Valid device index 0 enters and restores cleanly.
    with torch.accelerator.device_index(0):
      self.assertEqual(torch.accelerator.current_device_index(), 0)
    self.assertEqual(torch.accelerator.current_device_index(), initial_idx)

    # If multiple devices are available, verify switching and restoration
    # including when an exception occurs inside the block.
    if torch.accelerator.device_count() > 1:
      with torch.accelerator.device_index(0):
        self.assertEqual(torch.accelerator.current_device_index(), 0)
        with torch.accelerator.device_index(1):
          self.assertEqual(torch.accelerator.current_device_index(), 1)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

      with self.assertRaises(  # ASSERT_RAISES_OK=Context manager cleanup test
          ZeroDivisionError
      ):
        with torch.accelerator.device_index(1):
          self.assertEqual(torch.accelerator.current_device_index(), 1)
          _ = 1 / 0
      self.assertEqual(torch.accelerator.current_device_index(), initial_idx)

    # Positional-only parameter enforcement.
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      torch.accelerator.device_index(device=0)  # pytype: disable=wrong-keyword-args

    # Negative or out-of-range device index raises RuntimeError on context entry.
    for bad_index in (-1, torch.accelerator.device_count()):
      with self.subTest(bad_index=bad_index):
        with self.assertRaises(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
            RuntimeError
        ):
          with torch.accelerator.device_index(bad_index):
            pass

  def test_device_index_context_manager_torch_compile(self):
    # Dynamo fullgraph support for torch.accelerator.device_index was added in
    # PyTorch 2.14 (AcceleratorDeviceIndexVariable); older PyTorch versions
    # graph-break on device_index.__init__.
    supports_fullgraph = tt_utils.parse_version(torch.__version__) >= (
        2,
        14,
        0,
    )

    @torch.compile(backend="tpu", fullgraph=supports_fullgraph)
    def fn(x):
      with torch.accelerator.device_index(0):
        return x + 1

    x = torch.ones(4, device="tpu")
    y = fn(x)
    utils.assert_close(y.cpu(), torch.full((4,), 2.0))


class AcceleratorStreamAndSyncTest(seed_test_utils.RepeatableTest):
  """Tests stream and synchronization APIs in torch.accelerator."""

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.old_eager_mode = execution_mode.eager_mode
    execution_mode.eager_mode = EagerMode.DEFER_AND_FUSE
    self._orig_stream = torch.accelerator.current_stream()

  def tearDown(self):
    torch.accelerator.set_stream(self._orig_stream)
    execution_mode.eager_mode = self.old_eager_mode
    super().tearDown()

  def test_current_stream(self):
    for device_arg in (
        None,
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        s = (
            torch.accelerator.current_stream()
            if device_arg is None
            else torch.accelerator.current_stream(device_arg)
        )
        self.assertIsInstance(s, torch.Stream)
        self.assertEqual(s.device, torch.device("tpu", 0))
        self.assertEqual(s.device_index, 0)
        self.assertEqual(s, torch.tpu.current_stream())

    # Positional-only parameter enforcement.
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      torch.accelerator.current_stream(device=0)  # pytype: disable=wrong-keyword-args

    # Mismatched device type raises ValueError.
    for wrong_type in ("cpu", torch.device("cpu"), torch.device("cpu", 0)):
      with self.subTest(wrong_type=wrong_type):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
            ValueError, "doesn't match the current accelerator"
        ):
          torch.accelerator.current_stream(wrong_type)

    # Out-of-range or negative index raises RuntimeError.
    for bad_index in (-1, torch.accelerator.device_count(), 100):
      with self.subTest(bad_index=bad_index):
        with self.assertRaises(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
            RuntimeError
        ):
          torch.accelerator.current_stream(bad_index)

  def test_set_stream(self):
    default_stream = torch.accelerator.current_stream()
    new_tpu_stream = torch.tpu.Stream()
    dummy = torch.empty(1, device="tpu")
    new_generic_stream = torch.Stream(device=dummy.device)
    del dummy

    # Setting a TpuStream updates current_stream.
    torch.accelerator.set_stream(new_tpu_stream)
    self.assertEqual(torch.accelerator.current_stream(), new_tpu_stream)
    self.assertEqual(torch.tpu.current_stream(), new_tpu_stream)

    # Setting a generic torch.Stream updates current_stream.
    torch.accelerator.set_stream(new_generic_stream)
    self.assertEqual(torch.accelerator.current_stream(), new_generic_stream)

    # Restoring default stream.
    torch.accelerator.set_stream(default_stream)
    self.assertEqual(torch.accelerator.current_stream(), default_stream)

    # Switching away from a stream via set_stream materializes pending lazy ops
    # on that stream.
    x = torch.ones(4, device="tpu")
    self.assertFalse(sync.is_materializing(x))
    torch.accelerator.set_stream(new_tpu_stream)
    self.assertTrue(sync.is_materializing(x))
    torch.accelerator.set_stream(default_stream)

  def test_synchronize_device_args_and_errors(self):
    for device_arg in (
        None,
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        x = torch.ones(8, device="tpu")
        self.assertFalse(sync.is_materialized(x))
        if device_arg is None:
          torch.accelerator.synchronize()
        else:
          torch.accelerator.synchronize(device_arg)
        self.assertTrue(sync.is_materialized(x))

    # Positional-only parameter enforcement.
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      torch.accelerator.synchronize(device=0)  # pytype: disable=wrong-keyword-args

    # Mismatched device type raises ValueError.
    for wrong_type in ("cpu", torch.device("cpu")):
      with self.subTest(wrong_type=wrong_type):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
            ValueError, "doesn't match the current accelerator"
        ):
          torch.accelerator.synchronize(wrong_type)

  def test_accelerator_synchronize_device_materialized(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(11, device=torch.device("tpu"))

    self.assertFalse(sync.is_materializing(x))
    self.assertFalse(sync.is_materializing(y))

    torch.accelerator.synchronize()

    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materializing(y))

  def test_accelerator_synchronize_device_ready(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(11, device=torch.device("tpu"))

    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))

    torch.accelerator.synchronize()

    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materialized(y))

  def test_accelerator_synchronize_from_multiple_threads(self):
    size = 8192 * 8192
    t_tpu = torch.ones(size, device="tpu", dtype=torch.float32)
    t_cpu = torch.empty(size, device="cpu", pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)

    def sync_and_check():
      torch.accelerator.synchronize()
      utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      futures = [
          executor.submit(sync_and_check),
          executor.submit(sync_and_check),
      ]
      for future in concurrent.futures.as_completed(futures):
        future.result()

  def test_accelerator_synchronize_compiled_mode(self):
    def func(lhs, rhs):
      return lhs.mm(rhs)

    compiled = torch.compile(func, backend="tpu")

    x = torch.ones(256, 256, dtype=torch.float32, device="tpu")
    y = compiled(x, x)
    torch.accelerator.synchronize()
    self.assertTrue(sync.is_materialized(y))
    expected = torch.full((256, 256), 256.0, dtype=torch.float32, device="cpu")
    utils.assert_close(y.cpu(), expected)

  def test_non_blocking_returns_immediately_accelerator(self):
    device = torch.device("tpu")
    size = 8192  # 8192x8192 int32 = 256MB

    # 1. Warmup: compile the 'a + b' graph and drive materialization.
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = a + b
    cpu_dst = torch.empty_like(c, device="cpu", pin_memory=True)
    cpu_dst.copy_(c, non_blocking=True)
    torch.accelerator.synchronize()

    # 2. Main test: measure immediate return of a TPU computation copy.
    a2_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b2_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a2 = a2_cpu.to(device)
    b2 = b2_cpu.to(device)
    c2 = a2 + b2

    t0 = time.time()
    cpu_dst.copy_(c2, non_blocking=True)
    async_duration = time.time() - t0

    torch.accelerator.synchronize()

    # 3. Measure blocking return of the same size.
    a3_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b3_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    a3 = a3_cpu.to(device)
    b3 = b3_cpu.to(device)
    c3 = a3 + b3

    t1 = time.time()
    cpu_dst.copy_(c3, non_blocking=False)
    sync_duration = time.time() - t1

    self.assertLess(async_duration, 0.01)  # 10ms threshold
    self.assertLess(async_duration, sync_duration)

  def test_accelerator_synchronize_default(self):
    """Tests that torch.accelerator.synchronize() correctly waits for async ops."""
    size = 1024 * 1024
    t_tpu = torch.ones(size, device="tpu", dtype=torch.float32)
    t_cpu = torch.empty(size, device="cpu", pin_memory=True)

    t_cpu.copy_(t_tpu, non_blocking=True)
    torch.accelerator.synchronize()

    utils.assert_close(t_cpu, torch.ones(size, dtype=torch.float32))

  def test_to_cpu_non_blocking_accelerator(self):
    device = torch.device("tpu")

    size = 2048
    a_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)
    b_cpu = torch.randint(1, 10, (size, size), dtype=torch.int32)

    a = a_cpu.to(device)
    b = b_cpu.to(device)

    c = a * 2 + b
    cpu_tensor = c.to("cpu", non_blocking=True)

    self.assertEqual(cpu_tensor.device.type, "cpu")
    self.assertTrue(cpu_tensor.is_pinned())

    torch.accelerator.synchronize()

    expected_cpu_tensor = a_cpu * 2 + b_cpu
    utils.assert_close(cpu_tensor, expected_cpu_tensor)

  def test_h2d_accelerator_sync(self):
    """Verifies that torch.accelerator.synchronize() waits for H2D transfers."""
    size = 4096

    cpu_tensor = torch.ones(
        size, size, dtype=torch.float32, device="cpu"
    ).pin_memory()

    # Warmup
    _ = cpu_tensor.to("tpu", non_blocking=True)
    torch.accelerator.synchronize()

    cpu_tensor.fill_(1.0)

    tpu_tensor = cpu_tensor.to("tpu", non_blocking=True)
    torch.accelerator.synchronize()

    # Modify the host tensor immediately after synchronize
    cpu_tensor.fill_(2.0)

    result_tensor = tpu_tensor.cpu()
    mean_val = result_tensor.mean().item()
    self.assertEqual(
        mean_val,
        1.0,
        "torch.accelerator.synchronize() failed to wait for H2D transfer.",
    )


class AcceleratorMemoryTest(seed_test_utils.RepeatableTest):
  """Tests memory management APIs in torch.accelerator and torch.accelerator.memory."""

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_memory_module_aliases(self):
    self.assertIs(
        torch.accelerator.empty_cache, torch.accelerator.memory.empty_cache
    )
    self.assertIs(
        torch.accelerator.get_memory_info,
        torch.accelerator.memory.get_memory_info,
    )
    self.assertIs(
        torch.accelerator.max_memory_allocated,
        torch.accelerator.memory.max_memory_allocated,
    )
    self.assertIs(
        torch.accelerator.max_memory_reserved,
        torch.accelerator.memory.max_memory_reserved,
    )
    self.assertIs(
        torch.accelerator.memory_allocated,
        torch.accelerator.memory.memory_allocated,
    )
    self.assertIs(
        torch.accelerator.memory_reserved,
        torch.accelerator.memory.memory_reserved,
    )
    self.assertIs(
        torch.accelerator.memory_stats, torch.accelerator.memory.memory_stats
    )
    self.assertIs(
        torch.accelerator.reset_peak_memory_stats,
        torch.accelerator.memory.reset_peak_memory_stats,
    )

  def test_empty_cache(self):
    self.assertIsNone(torch.accelerator.empty_cache())
    self.assertIsNone(torch.accelerator.memory.empty_cache())

    t = torch.ones(1024, device="tpu")
    torch.accelerator.synchronize()
    del t
    gc.collect()

    self.assertIsNone(torch.accelerator.empty_cache())
    self.assertIsNone(torch.accelerator.memory.empty_cache())

  @parameterized.named_parameters(
      ("top_level", torch.accelerator.get_memory_info),
      ("memory_submodule", torch.accelerator.memory.get_memory_info),
  )
  def test_get_memory_info(self, get_memory_info_fn):
    for device_arg in (
        None,
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        info = (
            get_memory_info_fn()
            if device_arg is None
            else get_memory_info_fn(device_arg)
        )
        self.assertIsInstance(info, tuple)
        self.assertLen(info, 2)
        free_bytes, total_bytes = info
        self.assertIsInstance(free_bytes, int)
        self.assertIsInstance(total_bytes, int)
        self.assertGreater(total_bytes, 0)
        self.assertGreaterEqual(free_bytes, 0)
        self.assertLessEqual(free_bytes, total_bytes)

    # Allocating a materialized tensor decreases free memory while total remains
    # constant.
    free_before, total_before = get_memory_info_fn()
    size_in_bytes = 4 * 1024 * 1024
    t = torch.randn(size_in_bytes // 4, device="cpu").to(
        torch.accelerator.current_accelerator()
    )
    free_during, total_during = get_memory_info_fn()
    self.assertEqual(total_during, total_before)
    self.assertLessEqual(free_during, free_before - size_in_bytes)

    del t
    gc.collect()
    free_after, total_after = get_memory_info_fn()
    self.assertEqual(total_after, total_before)
    self.assertGreater(free_after, free_during)

    # Positional-only parameter enforcement.
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      get_memory_info_fn(device_index=0)  # pytype: disable=wrong-keyword-args

    # Invalid device checks.
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        ValueError, "doesn't match the current accelerator"
    ):
      get_memory_info_fn("cpu")

  @parameterized.named_parameters(
      ("top_level", torch.accelerator.memory_stats),
      ("memory_submodule", torch.accelerator.memory.memory_stats),
  )
  def test_memory_stats(self, memory_stats_fn):
    for device_arg in (
        None,
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        stats = (
            memory_stats_fn()
            if device_arg is None
            else memory_stats_fn(device_arg)
        )
        self.assertIsInstance(stats, OrderedDict)
        self.assertIn("active_bytes.all.current", stats)
        self.assertIn("active_bytes.all.peak", stats)
        self.assertIn("allocated_bytes.all.current", stats)
        self.assertIn("allocated_bytes.all.peak", stats)
        self.assertIn("reserved_bytes.all.current", stats)
        self.assertIn("reserved_bytes.all.peak", stats)
        self.assertIn("allocation.all.current", stats)
        self.assertIn("num_ooms", stats)

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      memory_stats_fn(device_index=0)  # pytype: disable=wrong-keyword-args

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        ValueError, "doesn't match the current accelerator"
    ):
      memory_stats_fn("cpu")

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 42),
      "TODO(b/565506249): Requires libtpu >= 0.0.42 for"
      " reset_peak_memory_stats.",
  )
  @parameterized.named_parameters(
      (
          "top_level",
          torch.accelerator.reset_peak_memory_stats,
          torch.accelerator.memory_allocated,
          torch.accelerator.max_memory_allocated,
      ),
      (
          "memory_submodule",
          torch.accelerator.memory.reset_peak_memory_stats,
          torch.accelerator.memory.memory_allocated,
          torch.accelerator.memory.max_memory_allocated,
      ),
  )
  def test_reset_peak_memory_stats(
      self, reset_peak_fn, memory_allocated_fn, max_memory_allocated_fn
  ):
    """Tests that resetting memory stats works correctly."""
    # 1. Reset stats to establish a clean baseline where peak == current
    reset_peak_fn()

    allocated_before = memory_allocated_fn()
    max_allocated_before = max_memory_allocated_fn()
    self.assertEqual(max_allocated_before, allocated_before)

    # 2. Allocate some memory (e.g., 4MB)
    size_in_bytes = 4 * 1024 * 1024
    num_elements = size_in_bytes // 4
    t = torch.randn(num_elements, device="cpu").to(
        torch.accelerator.current_accelerator()
    )
    # 3. Verify peak memory increased
    allocated_after = memory_allocated_fn()
    max_allocated_after = max_memory_allocated_fn()
    self.assertGreaterEqual(allocated_after, allocated_before + size_in_bytes)
    self.assertGreaterEqual(max_allocated_after, allocated_after)
    # 4. Free the allocated tensor
    del t
    gc.collect()
    allocated_after_free = memory_allocated_fn()
    self.assertLess(allocated_after_free, allocated_after)

    # 5. Reset peak memory stats again
    reset_peak_fn()

    max_allocated_after_reset = max_memory_allocated_fn()
    self.assertEqual(max_allocated_after_reset, allocated_after_free)

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      reset_peak_fn(device_index=0)  # pytype: disable=wrong-keyword-args

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 42),
      "TODO(b/565506249): Requires libtpu >= 0.0.42 for"
      " reset_peak_memory_stats.",
  )
  @parameterized.named_parameters(
      (
          "top_level",
          torch.accelerator.reset_peak_memory_stats,
          torch.accelerator.memory_allocated,
          torch.accelerator.max_memory_allocated,
      ),
      (
          "memory_submodule",
          torch.accelerator.memory.reset_peak_memory_stats,
          torch.accelerator.memory.memory_allocated,
          torch.accelerator.memory.max_memory_allocated,
      ),
  )
  def test_memory_allocated_and_max_memory_allocated(
      self, reset_peak_fn, memory_allocated_fn, max_memory_allocated_fn
  ):
    """Tests that memory allocation works with the device."""
    reset_peak_fn()
    allocated_before = memory_allocated_fn()
    max_allocated_before = max_memory_allocated_fn()
    self.assertGreaterEqual(allocated_before, 0)
    self.assertEqual(max_allocated_before, allocated_before)

    # Allocate a known amount of memory (4MB).
    size_in_bytes = 4 * 1024 * 1024
    num_elements = size_in_bytes // 4
    t = torch.randn(
        num_elements, device=torch.accelerator.current_accelerator()
    )
    t = t + 1
    _ = t.cpu()

    allocated_after = memory_allocated_fn()
    max_allocated_after = max_memory_allocated_fn()

    self.assertGreater(allocated_after, allocated_before)
    self.assertGreaterEqual(allocated_after, allocated_before + size_in_bytes)

    self.assertGreater(max_allocated_after, 0)
    self.assertGreater(max_allocated_after, max_allocated_before)
    self.assertGreaterEqual(max_allocated_after, allocated_after)

    for device_arg in (0, "tpu:0", torch.device("tpu", 0)):
      with self.subTest(device_arg=device_arg):
        self.assertEqual(memory_allocated_fn(device_arg), allocated_after)
        self.assertEqual(
            max_memory_allocated_fn(device_arg), max_allocated_after
        )

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      memory_allocated_fn(device_index=0)  # pytype: disable=wrong-keyword-args
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      max_memory_allocated_fn(device_index=0)  # pytype: disable=wrong-keyword-args

  @parameterized.named_parameters(
      (
          "top_level",
          torch.accelerator.memory_reserved,
          torch.accelerator.max_memory_reserved,
          torch.accelerator.memory_stats,
      ),
      (
          "memory_submodule",
          torch.accelerator.memory.memory_reserved,
          torch.accelerator.memory.max_memory_reserved,
          torch.accelerator.memory.memory_stats,
      ),
  )
  def test_memory_reserved_and_max_memory_reserved(
      self, memory_reserved_fn, max_memory_reserved_fn, memory_stats_fn
  ):
    t = torch.ones(1024, device="tpu")
    torch.accelerator.synchronize()
    del t

    for device_arg in (
        None,
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        reserved = (
            memory_reserved_fn()
            if device_arg is None
            else memory_reserved_fn(device_arg)
        )
        max_reserved = (
            max_memory_reserved_fn()
            if device_arg is None
            else max_memory_reserved_fn(device_arg)
        )
        stats = (
            memory_stats_fn()
            if device_arg is None
            else memory_stats_fn(device_arg)
        )
        self.assertIsInstance(reserved, int)
        self.assertIsInstance(max_reserved, int)
        self.assertGreaterEqual(reserved, 0)
        self.assertGreaterEqual(max_reserved, reserved)
        self.assertEqual(reserved, stats.get("reserved_bytes.all.current", 0))
        self.assertEqual(max_reserved, stats.get("reserved_bytes.all.peak", 0))

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      memory_reserved_fn(device_index=0)  # pytype: disable=wrong-keyword-args
    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      max_memory_reserved_fn(device_index=0)  # pytype: disable=wrong-keyword-args


class AcceleratorRandomTest(seed_test_utils.RepeatableTest):
  """Tests RNG APIs in torch.accelerator.random."""

  def setUp(self):
    super().setUp()
    if not hasattr(torch.accelerator, "random"):
      self.skipTest(
          "torch.accelerator.random is not available in PyTorch "
          f"{torch.__version__} (requires PyTorch >= 2.14)."
      )
    tt_testing.reset_eager_state()

  def test_initial_seed(self):
    torch.manual_seed(987654)
    seed = torch.accelerator.random.initial_seed()
    self.assertIsInstance(seed, int)
    self.assertEqual(seed, 987654)
    self.assertEqual(seed, torch.tpu.initial_seed())

    torch.tpu.manual_seed(123456)
    for device_arg in (
        None,
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        actual = (
            torch.accelerator.random.initial_seed()
            if device_arg is None
            else torch.accelerator.random.initial_seed(device_arg)
        )
        self.assertEqual(actual, 123456)

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      torch.accelerator.random.initial_seed(device=0)  # pytype: disable=wrong-keyword-args

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        ValueError, "doesn't match the current accelerator"
    ):
      torch.accelerator.random.initial_seed("cpu")

    with self.assertRaises(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        RuntimeError
    ):
      torch.accelerator.random.initial_seed(torch.accelerator.device_count())

  def test_get_rng_state(self):
    torch.manual_seed(42)
    state_before = torch.accelerator.random.get_rng_state()
    self.assertIsInstance(state_before, torch.Tensor)
    self.assertEqual(state_before.dtype, torch.uint8)
    self.assertEqual(state_before.device.type, "cpu")
    self.assertEqual(state_before.ndim, 1)
    utils.assert_close(state_before, torch.tpu.get_rng_state())

    for device_arg in (
        0,
        "tpu",
        "tpu:0",
        torch.device("tpu"),
        torch.device("tpu", 0),
    ):
      with self.subTest(device_arg=device_arg):
        utils.assert_close(
            torch.accelerator.random.get_rng_state(device_arg), state_before
        )

    # Running a random op on TPU advances the RNG state.
    _ = torch.rand(16, device="tpu").cpu()
    state_after = torch.accelerator.random.get_rng_state()
    self.assertFalse(torch.equal(state_before, state_after))

    # Re-seeding with the same seed restores the initial state.
    torch.manual_seed(42)
    state_reseeded = torch.accelerator.random.get_rng_state()
    utils.assert_close(state_reseeded, state_before)

    with self.assertRaises(  # ASSERT_RAISES_OK=Positional-only API check
        TypeError
    ):
      torch.accelerator.random.get_rng_state(device=0)  # pytype: disable=wrong-keyword-args

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        ValueError, "doesn't match the current accelerator"
    ):
      torch.accelerator.random.get_rng_state("cpu")

    with self.assertRaises(  # ASSERT_RAISES_OK=PyTorch accelerator API contract check
        RuntimeError
    ):
      torch.accelerator.random.get_rng_state(torch.accelerator.device_count())

  def test_get_rng_state_all(self):
    torch.manual_seed(2026)
    all_states = torch.accelerator.random.get_rng_state_all()
    self.assertIsInstance(all_states, list)
    self.assertLen(all_states, torch.accelerator.device_count())
    for idx, state in enumerate(all_states):
      self.assertIsInstance(state, torch.Tensor)
      self.assertEqual(state.dtype, torch.uint8)
      self.assertEqual(state.device.type, "cpu")
      if idx == torch.accelerator.current_device_index():
        utils.assert_close(state, torch.accelerator.random.get_rng_state(idx))


if __name__ == "__main__":
  absltest.main()
