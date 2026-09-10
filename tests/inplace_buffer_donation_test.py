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

"""Unit tests for in-place buffer donation in DeferNever eager mode."""

import gc
import unittest
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu import _loader
from torch_tpu._internal import execution_mode
from torch_tpu._internal.utils import test_utils
from tests import oss_utils
from tests import seed_test_utils

_loader._init_device("tpu")

EagerMode = execution_mode.EagerMode


class InplaceBufferDonationTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    gc.collect()
    torch.accelerator.synchronize()

  def tearDown(self):
    super().tearDown()
    gc.collect()
    torch.accelerator.synchronize()

  def test_peak_memory_tests_skip_on_cloud_libtpu_older_than_0_0_42(self):
    """Verifies that all peak memory tests skip on Cloud TPU when libtpu < 0.0.42.

    Why we check for libtpu version:
    - In OSS Cloud TPU CI environments (e.g. linux-x86-ct5lp-224-8tpu on v5e),
      libtpu is pinned to 0.0.41.
    - PJRT_Device_ClearMemoryStats (which powers reset_peak_memory_stats()) was
      only added in PJRT C API version 0.106 and libtpu 0.0.42 (see CL
      940717234).
    - On libtpu < 0.0.42, reset_peak_memory_stats() silently no-ops with
      absl::UnimplementedError, so max_memory_allocated() never resets and
      retains the process lifetime peak.
    - Any subsequent peak memory check (`peak_mem - base_mem < 4 MB`) then
      fails whenever prior tests have allocated device memory.
    - All peak memory tests must therefore be decorated with
      @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42") to prevent
      flakiness in OSS CI on older libtpu versions, while still executing
      in Google3 and on newer libtpu versions.
    """
    peak_memory_test_names = [
        "test_multi_input_aliasing_peak_memory_no_spike_in_defer_never",
        "test_inplace_peak_memory_no_spike_in_defer_never",
        "test_inplace_peak_memory_spikes_without_donation_in_defer_and_fuse",
        "test_non_zero_offset_inplace_does_not_donate_in_defer_never",
        "test_non_contiguous_inplace_does_not_donate_in_defer_never",
        "test_binary_inplace_donates_second_input_peak_memory_in_defer_never",
        "test_inplace_buffer_donation_disabled_via_python_api",
    ]
    for test_name in peak_memory_test_names:
      method = getattr(self, test_name, None)
      self.assertIsNotNone(method, f"Method {test_name} not found")
      with mock.patch.object(oss_utils, "running_in_cloud", return_value=True):
        with mock.patch.object(
            oss_utils, "libtpu_version", return_value="0.0.41"
        ):
          skipped = False
          try:
            method()
          except unittest.SkipTest:
            skipped = True
          self.assertTrue(
              skipped,
              f"{test_name} did not skip on cloud with libtpu 0.0.41",
          )

  def test_unary_inplace_donates_buffer_in_defer_never(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      x = torch.tensor(
          [-2.0, -1.0, 3.0], dtype=torch.float32, device=self.device
      )
      old_x_view = x.view_as(x)

      x.relu_()

      # Correct mutated values: negative numbers are clamped to 0.0 by relu_
      expected = torch.tensor([0.0, 0.0, 3.0])
      self.assertTrue(torch.equal(x.cpu(), expected))
      # In eager mode, views share the updated underlying storage
      self.assertTrue(torch.equal(old_x_view.cpu(), expected))

  def test_binary_inplace_donates_buffer_in_defer_never(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
      y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=self.device)
      old_x_view = x.view_as(x)

      x.add_(y)

      # Correct mutated values
      expected = torch.tensor([5.0, 7.0, 9.0])
      self.assertTrue(torch.equal(x.cpu(), expected))
      # In eager mode, views share the updated underlying storage
      self.assertTrue(torch.equal(old_x_view.cpu(), expected))

  def test_multi_input_aliasing_same_tensor_donates_buffer_in_defer_never(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      # Case 1: x.add_(x) where self and other are the exact same tensor
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
      x.add_(x)
      self.assertTrue(torch.equal(x.cpu(), torch.tensor([2.0, 4.0, 6.0])))

      # Case 2: torch.add(a, a, out=a)
      a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
      torch.add(a, a, out=a)
      self.assertTrue(torch.equal(a.cpu(), torch.tensor([2.0, 4.0, 6.0])))

      # Case 3: baddbmm where self, batch1, batch2 are the same tensor
      m = torch.ones(2, 2, 2, dtype=torch.float32, device=self.device)
      torch.baddbmm(m, m, m, out=m)
      # m + (m @ m) = 1 + 2 = 3
      self.assertTrue(torch.equal(m.cpu(), torch.full((2, 2, 2), 3.0)))

      # Case 4: clamp where self, min, max are the same tensor
      c = torch.tensor([5.0, 5.0], dtype=torch.float32, device=self.device)
      torch.clamp(c, min=c, max=c, out=c)
      self.assertTrue(torch.equal(c.cpu(), torch.tensor([5.0, 5.0])))

      # Case 5: lerp where self, end, weight are the same tensor
      l = torch.tensor([2.0, 4.0], dtype=torch.float32, device=self.device)
      # self + weight * (end - self) = 2 + 2 * (2 - 2) = 2
      torch.lerp(l, l, l, out=l)
      self.assertTrue(torch.equal(l.cpu(), torch.tensor([2.0, 4.0])))

      # Case 6: views that alias the output tensor
      v1 = torch.tensor(
          [1.0, 2.0, 3.0], dtype=torch.float32, device=self.device
      )
      v2 = v1.view(3)
      torch.add(v1, v2, out=v1)
      self.assertTrue(torch.equal(v1.cpu(), torch.tensor([2.0, 4.0, 6.0])))
      self.assertTrue(torch.equal(v2.cpu(), torch.tensor([2.0, 4.0, 6.0])))

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_multi_input_aliasing_peak_memory_no_spike_in_defer_never(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      # 16 MB tensor
      num_elements = 4 * 1024 * 1024
      x = torch.ones(num_elements, dtype=torch.float32, device=self.device)
      _ = x.cpu()

      # Warmup to compile and initialize runtime buffers
      x.add_(x)
      torch.accelerator.synchronize()
      gc.collect()

      torch.accelerator.reset_peak_memory_stats()
      base_mem = torch.accelerator.memory_allocated()

      # Both inputs are the same tensor and alias the output
      x.add_(x)
      torch.accelerator.synchronize()

      peak_mem = torch.accelerator.max_memory_allocated()
      extra_mem = peak_mem - base_mem
      # Buffer is donated in-place without allocating another 16 MB.
      # Allow up to 4 MB for allocator page alignment/runtime scratchpad.
      self.assertLess(extra_mem, 4 * 1024 * 1024)
      self.assertTrue(torch.equal(x.cpu(), torch.full((num_elements,), 4.0)))

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_inplace_peak_memory_no_spike_in_defer_never(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      # 16 MB tensor (4M float32 elements = 16,777,216 bytes)
      num_elements = 4 * 1024 * 1024
      x = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
      _ = x.cpu()

      # Warmup to compile and initialize runtime buffers
      x.add_(1.0)
      torch.accelerator.synchronize()
      gc.collect()

      torch.accelerator.reset_peak_memory_stats()
      base_mem = torch.accelerator.memory_allocated()

      x.add_(1.0)
      torch.accelerator.synchronize()

      peak_mem = torch.accelerator.max_memory_allocated()
      extra_mem = peak_mem - base_mem
      # With buffer donation, the 16MB buffer is reused in-place.
      # Allow up to 4 MB for allocator page alignment/runtime scratchpad.
      self.assertLess(extra_mem, 4 * 1024 * 1024)
      self.assertTrue(torch.equal(x.cpu(), torch.full((num_elements,), 2.0)))

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_inplace_peak_memory_spikes_without_donation_in_defer_and_fuse(self):
    with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
      # 16 MB tensor (4M float32 elements)
      num_elements = 4 * 1024 * 1024
      x = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
      _ = x.cpu()

      # Warmup to compile and initialize runtime buffers
      x.add_(1.0)
      _ = x.cpu()
      torch.accelerator.synchronize()
      gc.collect()

      torch.accelerator.reset_peak_memory_stats()
      base_mem = torch.accelerator.memory_allocated()

      x.add_(1.0)
      _ = x.cpu()  # Force execution in DeferAndFuse
      torch.accelerator.synchronize()

      peak_mem = torch.accelerator.max_memory_allocated()
      extra_mem = peak_mem - base_mem
      # Without buffer donation, a separate 16 MB buffer is allocated for
      # the output.
      self.assertGreaterEqual(extra_mem, 12 * 1024 * 1024)

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_non_zero_offset_inplace_does_not_donate_in_defer_never(self):
    """Verifies that non-zero offset slices fall back safely to allocating in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      num_elements = 4 * 1024 * 1024  # 16 MB
      base = torch.zeros(
          num_elements + 1, dtype=torch.float32, device=self.device
      )
      _ = base.cpu()
      # Non-zero storage offset: ShouldDonateInPlaceBuffer returns false
      x_offset = base[1:]

      # Warmup to compile and initialize runtime buffers
      x_offset.add_(1.0)
      torch.accelerator.synchronize()
      gc.collect()

      torch.accelerator.reset_peak_memory_stats()
      base_mem = torch.accelerator.memory_allocated()
      x_offset.add_(1.0)
      torch.accelerator.synchronize()
      peak_mem = torch.accelerator.max_memory_allocated()
      delta = peak_mem - base_mem
      self.assertGreaterEqual(delta, 12 * 1024 * 1024)

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_non_contiguous_inplace_does_not_donate_in_defer_never(self):
    """Verifies that non-contiguous tensors fall back safely to allocating in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      # 2048 x 2048 float32 = 16 MB
      base = torch.zeros((2048, 2048), dtype=torch.float32, device=self.device)
      _ = base.cpu()
      # Non-contiguous transposed view: ShouldDonateInPlaceBuffer returns false
      t = base.t()
      self.assertFalse(t.is_contiguous())

      # Warmup to compile and initialize runtime buffers
      t.add_(1.0)
      torch.accelerator.synchronize()
      gc.collect()

      torch.accelerator.reset_peak_memory_stats()
      base_mem = torch.accelerator.memory_allocated()
      t.add_(1.0)
      torch.accelerator.synchronize()
      peak_mem = torch.accelerator.max_memory_allocated()
      delta = peak_mem - base_mem
      self.assertGreaterEqual(delta, 12 * 1024 * 1024)

  def test_binary_inplace_donates_second_input_in_defer_never(self):
    """Verifies that operations with out=rhs/other donate the second input buffer."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      # 1. add with out=other
      with self.subTest(op="add_out_other"):
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device=self.device
        )
        y = torch.tensor(
            [4.0, 5.0, 6.0], dtype=torch.float32, device=self.device
        )
        v = y.view_as(y)
        torch.add(x, y, out=y)
        expected = torch.tensor([5.0, 7.0, 9.0])
        self.assertTrue(torch.equal(y.cpu(), expected))
        self.assertTrue(torch.equal(v.cpu(), expected))
        del x, y, v
        gc.collect()
        torch.accelerator.synchronize()

      # 2. mm with out=rhs
      with self.subTest(op="mm_out_rhs"):
        a = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=self.device
        )
        b = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device=self.device
        )
        v = b.view_as(b)
        torch.mm(a, b, out=b)
        expected = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        test_utils.assert_close(b.cpu(), expected)
        test_utils.assert_close(v.cpu(), expected)
        del a, b, v
        gc.collect()
        torch.accelerator.synchronize()

      # 3. where with out=other
      with self.subTest(op="where_out_other"):
        cond = torch.tensor(
            [True, False, True], dtype=torch.bool, device=self.device
        )
        x = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device=self.device
        )
        y = torch.tensor(
            [10.0, 20.0, 30.0], dtype=torch.float32, device=self.device
        )
        v = y.view_as(y)
        torch.where(cond, x, y, out=y)
        expected = torch.tensor([1.0, 20.0, 3.0])
        self.assertTrue(torch.equal(y.cpu(), expected))
        self.assertTrue(torch.equal(v.cpu(), expected))
        del cond, x, y, v
        gc.collect()
        torch.accelerator.synchronize()

      # 4. xlogy with out=other
      with self.subTest(op="xlogy_out_other"):
        x = torch.tensor([2.0, 3.0], dtype=torch.float32, device=self.device)
        y = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        v = y.view_as(y)
        torch.xlogy(x, y, out=y)
        expected = torch.tensor([0.0, 2.0794415])
        test_utils.assert_close(y.cpu(), expected, atol=1e-4, rtol=1e-4)
        test_utils.assert_close(v.cpu(), expected, atol=1e-4, rtol=1e-4)
        del x, y, v
        gc.collect()
        torch.accelerator.synchronize()

      # 5. bmm with out=mat2
      with self.subTest(op="bmm_out_mat2"):
        a = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32, device=self.device
        )
        b = torch.tensor(
            [[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32, device=self.device
        )
        v = b.view_as(b)
        torch.bmm(a, b, out=b)
        expected = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        self.assertTrue(torch.equal(b.cpu(), expected))
        self.assertTrue(torch.equal(v.cpu(), expected))
        del a, b, v
        gc.collect()
        torch.accelerator.synchronize()

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_binary_inplace_donates_second_input_peak_memory_in_defer_never(self):
    """Verifies that operations with out=rhs/other donate the second input buffer without memory spike."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      # 1. add with out=other (16 MB peak memory verification)
      with self.subTest(op="add_out_other"):
        num_elements = 4 * 1024 * 1024
        x = torch.ones(num_elements, dtype=torch.float32, device=self.device)
        y = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
        _ = x.cpu()
        _ = y.cpu()

        # Warmup to compile and initialize runtime buffers
        torch.add(x, y, out=y)
        torch.accelerator.synchronize()
        gc.collect()

        torch.accelerator.reset_peak_memory_stats()
        base_mem = torch.accelerator.memory_allocated()
        torch.add(x, y, out=y)
        torch.accelerator.synchronize()

        peak_mem = torch.accelerator.max_memory_allocated()
        extra_mem = peak_mem - base_mem
        self.assertLess(extra_mem, 4 * 1024 * 1024)
        self.assertTrue(torch.equal(y.cpu(), torch.full((num_elements,), 2.0)))
        del x, y
        gc.collect()
        torch.accelerator.synchronize()

      # 2. mm with out=rhs (16 MB peak memory verification)
      with self.subTest(op="mm_out_rhs"):
        # 2048 x 2048 float32 = 16 MB
        a = torch.eye(2048, dtype=torch.float32, device=self.device)
        b = torch.ones((2048, 2048), dtype=torch.float32, device=self.device)
        _ = a.cpu()
        _ = b.cpu()

        # Warmup to compile and initialize runtime buffers
        torch.mm(a, b, out=b)
        torch.accelerator.synchronize()
        gc.collect()

        torch.accelerator.reset_peak_memory_stats()
        base_mem = torch.accelerator.memory_allocated()
        torch.mm(a, b, out=b)
        torch.accelerator.synchronize()

        peak_mem = torch.accelerator.max_memory_allocated()
        extra_mem = peak_mem - base_mem
        self.assertLess(extra_mem, 4 * 1024 * 1024)
        test_utils.assert_close(b.cpu(), torch.ones((2048, 2048)))
        del a, b
        gc.collect()
        torch.accelerator.synchronize()

      # 3. where with out=other (16 MB peak memory verification)
      with self.subTest(op="where_out_other"):
        num_elements = 4 * 1024 * 1024
        cond = torch.zeros(num_elements, dtype=torch.bool, device=self.device)
        x = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
        y = torch.ones(num_elements, dtype=torch.float32, device=self.device)
        _ = cond.cpu()
        _ = x.cpu()
        _ = y.cpu()

        # Warmup to compile and initialize runtime buffers
        torch.where(cond, x, y, out=y)
        torch.accelerator.synchronize()
        gc.collect()

        torch.accelerator.reset_peak_memory_stats()
        base_mem = torch.accelerator.memory_allocated()
        torch.where(cond, x, y, out=y)
        torch.accelerator.synchronize()

        peak_mem = torch.accelerator.max_memory_allocated()
        extra_mem = peak_mem - base_mem
        self.assertLess(extra_mem, 4 * 1024 * 1024)
        self.assertTrue(torch.equal(y.cpu(), torch.ones(num_elements)))
        del cond, x, y
        gc.collect()
        torch.accelerator.synchronize()

  @oss_utils.skip_if_cloud_and_libtpu_older_than("0.0.42")
  def test_inplace_buffer_donation_disabled_via_python_api(self):
    """Verifies that disabling donation via Python API causes allocation in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      num_elements = 4 * 1024 * 1024  # 16 MB
      x = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
      _ = x.cpu()

      # Warmup to compile and initialize runtime buffers
      x.add_(1.0)
      torch.accelerator.synchronize()
      gc.collect()

      # Disabling via module property: disable_inplace_buffer_donation = True
      execution_mode.disable_inplace_buffer_donation = True
      try:
        torch.accelerator.reset_peak_memory_stats()
        base_mem = torch.accelerator.memory_allocated()
        x.add_(1.0)
        torch.accelerator.synchronize()
        peak_mem = torch.accelerator.max_memory_allocated()
        extra_mem = peak_mem - base_mem
        # Without donation, a separate 16 MB buffer is allocated
        self.assertGreaterEqual(extra_mem, 12 * 1024 * 1024)
      finally:
        execution_mode.disable_inplace_buffer_donation = False

      # When re-enabled, donation reuses the buffer
      gc.collect()
      torch.accelerator.synchronize()
      torch.accelerator.reset_peak_memory_stats()
      base_mem = torch.accelerator.memory_allocated()
      x.add_(1.0)
      torch.accelerator.synchronize()
      peak_mem = torch.accelerator.max_memory_allocated()
      extra_mem = peak_mem - base_mem
      self.assertLess(extra_mem, 4 * 1024 * 1024)

  def test_activation_and_loss_ops_donate_in_defer_never(self):
    """Verifies that activation and loss in-place/out-variant ops work correctly with buffer donation in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      with self.subTest(op="clamp_"):
        x = torch.tensor(
            [-1.0, 0.5, 2.0], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        x.clamp_(0.0, 1.0)
        expected = torch.tensor([0.0, 0.5, 1.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="lerp_"):
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        end = torch.tensor([3.0, 6.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        x.lerp_(end, 0.5)
        expected = torch.tensor([2.0, 4.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="threshold"):
        x = torch.tensor([-1.0, 2.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.threshold(x, 0.0, 10.0, out=x)
        expected = torch.tensor([10.0, 2.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="hardtanh_"):
        x = torch.tensor(
            [-2.0, 0.0, 2.0], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        torch.nn.functional.hardtanh_(x, -1.0, 1.0)
        expected = torch.tensor([-1.0, 0.0, 1.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="elu_"):
        x = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.nn.functional.elu_(x)
        expected = torch.tensor([0.0, 1.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="logit_"):
        x = torch.tensor([0.5, 0.75], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        x.logit_()
        expected_logit = torch.tensor([0.0, 1.0986123])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected_logit, atol=1e-4, rtol=1e-4)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="digamma_out_self"):
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.digamma(x, out=x)
        expected = torch.digamma(torch.tensor([1.0, 2.0]))
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="softplus_out_self"):
        x = torch.tensor([-1.0, 2.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.nn.functional.softplus(x, out=x)
        expected = torch.nn.functional.softplus(torch.tensor([-1.0, 2.0]))
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="leaky_relu_out_self"):
        x = torch.tensor([-2.0, 3.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.leaky_relu.out(x, 0.1, out=x)
        expected = torch.tensor([-0.2, 3.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="masked_softmax_out_self"):
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        mask = torch.tensor(
            [[True, False], [False, True]], dtype=torch.bool, device=self.device
        )
        v = x.view_as(x)
        torch.ops.aten._masked_softmax.out(x, mask, dim=-1, out=x)
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="log_sigmoid_forward_out_self"):
        x = torch.tensor([-1.0, 2.0], dtype=torch.float32, device=self.device)
        buf = torch.empty_like(x)
        v = x.view_as(x)
        torch.ops.aten.log_sigmoid_forward.output(x, output=x, buffer=buf)
        expected = torch.nn.functional.logsigmoid(torch.tensor([-1.0, 2.0]))
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="clamp_out_min"):
        x = torch.tensor([1.0, 5.0], dtype=torch.float32, device=self.device)
        min_t = torch.tensor(
            [2.0, 2.0], dtype=torch.float32, device=self.device
        )
        max_t = torch.tensor(
            [4.0, 4.0], dtype=torch.float32, device=self.device
        )
        v = min_t.view_as(min_t)
        torch.clamp(x, min=min_t, max=max_t, out=min_t)
        expected = torch.tensor([2.0, 4.0])
        min_t_cpu = min_t.cpu()

        test_utils.assert_close(min_t_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), min_t.cpu()))

      with self.subTest(op="lerp_out_end"):
        start = torch.tensor(
            [0.0, 0.0], dtype=torch.float32, device=self.device
        )
        end = torch.tensor(
            [10.0, 20.0], dtype=torch.float32, device=self.device
        )
        weight = torch.tensor(
            [0.5, 0.5], dtype=torch.float32, device=self.device
        )
        v = end.view_as(end)
        torch.lerp(start, end, weight, out=end)
        expected = torch.tensor([5.0, 10.0])
        end_cpu = end.cpu()

        test_utils.assert_close(end_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), end.cpu()))

      with self.subTest(op="binary_cross_entropy"):
        x = torch.tensor([0.25, 0.75], dtype=torch.float32, device=self.device)
        target = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        torch.ops.aten.binary_cross_entropy(x, target, None, 0, out=x)
        expected = torch.tensor([0.287682, 0.287682])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected, atol=1e-4, rtol=1e-4)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="binary_cross_entropy_backward_grad_input_self"):
        x = torch.tensor([0.25, 0.75], dtype=torch.float32, device=self.device)
        target = torch.tensor(
            [0.0, 1.0], dtype=torch.float32, device=self.device
        )
        grad_output = torch.tensor(
            [1.0, 1.0], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        torch.ops.aten.binary_cross_entropy_backward(
            grad_output, x, target, None, 0, grad_input=x
        )
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="nll_loss_backward_grad_input_self"):
        x = torch.tensor(
            [[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32, device=self.device
        )
        target = torch.tensor([1, 0], dtype=torch.int64, device=self.device)
        total_weight = torch.tensor(
            2.0, dtype=torch.float32, device=self.device
        )
        grad_output = torch.tensor(
            [1.0, 1.0], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        torch.ops.aten.nll_loss_backward(
            grad_output, x, target, None, 0, -100, total_weight, grad_input=x
        )
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="tanh_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [0.5, 0.0], dtype=torch.float32, device=self.device
        )
        v = grad_output.view_as(grad_output)
        torch.ops.aten.tanh_backward(
            grad_output, output, grad_input=grad_output
        )
        expected = torch.tensor([0.75, 2.0])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="sigmoid_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [0.5, 0.25], dtype=torch.float32, device=self.device
        )
        v = grad_output.view_as(grad_output)
        torch.ops.aten.sigmoid_backward(
            grad_output, output, grad_input=grad_output
        )
        expected = torch.tensor([0.25, 0.375])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="threshold_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, -0.5], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.threshold_backward(
            grad_output, x, 0.0, grad_input=grad_output
        )
        expected = torch.tensor([1.0, 0.0])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="hardtanh_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, 2.0], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.hardtanh_backward(
            grad_output, x, -1.0, 1.0, grad_input=grad_output
        )
        expected = torch.tensor([1.0, 0.0])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="elu_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, -1.0], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.elu_backward(
            grad_output, 1.0, 1.0, 1.0, False, x, grad_input=grad_output
        )
        expected = torch.tensor([1.0, 2.0 * 0.36787944])
        test_utils.assert_close(
            grad_output.cpu(), expected, atol=1e-4, rtol=1e-4
        )
        test_utils.assert_close(v.cpu(), expected, atol=1e-4, rtol=1e-4)

      with self.subTest(op="gelu_backward"):
        grad_output = torch.ones(2, dtype=torch.float32, device=self.device)
        x = torch.zeros(2, dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.gelu_backward(
            grad_output, x, approximate="none", grad_input=grad_output
        )
        expected = torch.tensor([0.5, 0.5])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="hardsigmoid_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.0, 4.0], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.hardsigmoid_backward(
            grad_output, x, grad_input=grad_output
        )
        expected = torch.tensor([1.0 / 6.0, 0.0])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="leaky_relu_backward"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([1.0, -1.0], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.leaky_relu_backward(
            grad_output, x, 0.01, False, grad_input=grad_output
        )
        expected = torch.tensor([1.0, 0.02])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="logit_backward"):
        grad_output = torch.tensor(
            [1.0, 1.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, 0.25], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.logit_backward(
            grad_output, x, eps=-1.0, grad_input=grad_output
        )
        expected = torch.tensor([4.0, 16.0 / 3.0])
        test_utils.assert_close(
            grad_output.cpu(), expected, atol=1e-4, rtol=1e-4
        )
        test_utils.assert_close(v.cpu(), expected, atol=1e-4, rtol=1e-4)

      with self.subTest(op="softplus_backward"):
        grad_output = torch.tensor(
            [1.0, 1.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.0, 25.0], dtype=torch.float32, device=self.device)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.softplus_backward(
            grad_output, x, 1.0, 20.0, grad_input=grad_output
        )
        expected = torch.tensor([0.5, 1.0])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="_softmax_backward_data"):
        grad_output = torch.tensor(
            [1.0, 0.0], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [0.5, 0.5], dtype=torch.float32, device=self.device
        )
        v = grad_output.view_as(grad_output)
        torch.ops.aten._softmax_backward_data(
            grad_output, output, -1, grad_output.dtype, grad_input=grad_output
        )
        expected = torch.tensor([0.25, -0.25])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="_log_softmax_backward_data"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [-0.693147, -0.693147], dtype=torch.float32, device=self.device
        )
        v = grad_output.view_as(grad_output)
        torch.ops.aten._log_softmax_backward_data(
            grad_output, output, -1, grad_output.dtype, out=grad_output
        )
        expected = torch.tensor([-0.5, 0.5])
        test_utils.assert_close(
            grad_output.cpu(), expected, atol=1e-4, rtol=1e-4
        )
        test_utils.assert_close(v.cpu(), expected, atol=1e-4, rtol=1e-4)

      with self.subTest(op="_masked_softmax_backward"):
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=self.device)
        mask = torch.tensor(
            [[True, True]], dtype=torch.bool, device=self.device
        )
        output = torch.ops.aten._masked_softmax(x, mask, dim=-1, mask_type=2)
        grad = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32, device=self.device
        )
        v = grad.view_as(grad)
        torch.ops.aten._masked_softmax_backward.out(
            grad, output, mask, dim=-1, out=grad
        )
        _ = grad.cpu()
        self.assertTrue(torch.equal(v.cpu(), grad.cpu()))

      with self.subTest(op="tanh_backward_grad_input_output"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [0.5, 0.0], dtype=torch.float32, device=self.device
        )
        v = output.view_as(output)
        torch.ops.aten.tanh_backward(grad_output, output, grad_input=output)
        expected = torch.tensor([1.0 * (1.0 - 0.25), 2.0 * 1.0])
        output_cpu = output.cpu()

        test_utils.assert_close(output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), output.cpu()))

      with self.subTest(op="gelu_backward_grad_input_self"):
        grad_output = torch.ones(2, dtype=torch.float32, device=self.device)
        x = torch.zeros(2, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.gelu_backward(
            grad_output, x, approximate="none", grad_input=x
        )
        expected = torch.tensor([0.5, 0.5])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="hardsigmoid_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.0, 5.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.hardsigmoid_backward(grad_output, x, grad_input=x)
        expected = torch.tensor([1.0 / 6.0, 0.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="sigmoid_backward_grad_input_output"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [0.5, 0.2], dtype=torch.float32, device=self.device
        )
        v = output.view_as(output)
        torch.ops.aten.sigmoid_backward(grad_output, output, grad_input=output)
        expected = torch.tensor([1.0 * 0.5 * 0.5, 2.0 * 0.2 * 0.8])
        output_cpu = output.cpu()

        test_utils.assert_close(output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), output.cpu()))

      with self.subTest(op="log_sigmoid_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, -1.0], dtype=torch.float32, device=self.device)
        buf = torch.exp(-x)
        v = x.view_as(x)
        torch.ops.aten.log_sigmoid_backward.grad_input(
            grad_output, x, buf, grad_input=x
        )
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="elu_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, -1.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.elu_backward(
            grad_output, 1.0, 1.0, 1.0, False, x, grad_input=x
        )
        expected = torch.tensor([1.0, 2.0 * 0.36787944])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected, atol=1e-4, rtol=1e-4)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="threshold_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, -1.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.threshold_backward(grad_output, x, 0.0, grad_input=x)
        expected = torch.tensor([1.0, 0.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="hardtanh_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, 2.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.hardtanh_backward(
            grad_output, x, -1.0, 1.0, grad_input=x
        )
        expected = torch.tensor([1.0, 0.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="logit_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, 0.25], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.logit_backward(grad_output, x, grad_input=x)
        expected = torch.tensor([1.0 / (0.5 * 0.5), 2.0 / (0.25 * 0.75)])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="leaky_relu_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.5, -1.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.leaky_relu_backward(
            grad_output, x, 0.1, False, grad_input=x
        )
        expected = torch.tensor([1.0, 0.2])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="softplus_backward_grad_input_self"):
        grad_output = torch.tensor(
            [1.0, 2.0], dtype=torch.float32, device=self.device
        )
        x = torch.tensor([0.0, 10.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.softplus_backward(
            grad_output, x, 1.0, 20.0, grad_input=x
        )
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="masked_softmax_backward_grad_input_output"):
        grad_output = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [[0.5, 0.5], [0.1, 0.9]], dtype=torch.float32, device=self.device
        )
        mask = torch.tensor(
            [[False, False], [False, False]],
            dtype=torch.bool,
            device=self.device,
        )
        v = output.view_as(output)
        torch.ops.aten._masked_softmax_backward.out(
            grad_output, output, mask, dim=-1, out=output
        )
        _ = output.cpu()
        self.assertTrue(torch.equal(v.cpu(), output.cpu()))

      with self.subTest(op="softmax_backward_data_grad_input_output"):
        grad_output = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        output = torch.tensor(
            [[0.5, 0.5], [0.1, 0.9]], dtype=torch.float32, device=self.device
        )
        v = output.view_as(output)
        torch.ops.aten._softmax_backward_data.out(
            grad_output,
            output,
            dim=-1,
            input_dtype=output.dtype,
            grad_input=output,
        )
        _ = output.cpu()
        self.assertTrue(torch.equal(v.cpu(), output.cpu()))

  def test_indexing_scatter_and_search_ops_donate_in_defer_never(self):
    """Verifies that indexing, scatter, fill, and search in-place/out-variant ops work correctly with buffer donation in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      with self.subTest(op="put_"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        idx = torch.tensor([1, 3], dtype=torch.int64, device=self.device)
        src = torch.tensor([5.0, 7.0], dtype=torch.float32, device=self.device)
        x.put_(idx, src)
        expected = torch.tensor([0.0, 5.0, 0.0, 7.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="index_put_"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        idx = torch.tensor([0, 2], dtype=torch.int64, device=self.device)
        vals = torch.tensor(
            [11.0, 22.0], dtype=torch.float32, device=self.device
        )
        x.index_put_((idx,), vals)
        expected = torch.tensor([11.0, 0.0, 22.0, 0.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="scatter_"):
        x = torch.zeros(2, 4, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        idx = torch.tensor([[2], [0]], dtype=torch.int64, device=self.device)
        src = torch.tensor(
            [[1.5], [2.5]], dtype=torch.float32, device=self.device
        )
        x.scatter_(1, idx, src)
        expected = torch.tensor([[0.0, 0.0, 1.5, 0.0], [2.5, 0.0, 0.0, 0.0]])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="where"):
        cond = torch.tensor([True, False], dtype=torch.bool, device=self.device)
        x = torch.tensor([10.0, 20.0], dtype=torch.float32, device=self.device)
        y = torch.tensor([99.0, 88.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.where(cond, x, y, out=x)
        expected = torch.tensor([10.0, 88.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="index_copy_"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        idx = torch.tensor([1, 3], dtype=torch.int64, device=self.device)
        src = torch.tensor([5.0, 7.0], dtype=torch.float32, device=self.device)
        x.index_copy_(0, idx, src)
        expected = torch.tensor([0.0, 5.0, 0.0, 7.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="index_add_"):
        x = torch.ones(4, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        idx = torch.tensor([0, 2], dtype=torch.int64, device=self.device)
        src = torch.tensor(
            [10.0, 20.0], dtype=torch.float32, device=self.device
        )
        x.index_add_(0, idx, src)
        expected = torch.tensor([11.0, 1.0, 21.0, 1.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="index_reduce_"):
        x = torch.tensor(
            [2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        idx = torch.tensor([0, 2], dtype=torch.int64, device=self.device)
        src = torch.tensor(
            [10.0, 20.0], dtype=torch.float32, device=self.device
        )
        x.index_reduce_(0, idx, src, reduce="prod")
        expected = torch.tensor([20.0, 3.0, 80.0, 5.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="index_fill_scalar"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        idx = torch.tensor([1, 3], dtype=torch.int64, device=self.device)
        v = x.view_as(x)
        x.index_fill_(0, idx, 99.0)
        expected = torch.tensor([0.0, 99.0, 0.0, 99.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="index_fill_tensor"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        idx = torch.tensor([0, 2], dtype=torch.int64, device=self.device)
        val = torch.tensor(42.0, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        x.index_fill_(0, idx, val)
        expected = torch.tensor([42.0, 0.0, 42.0, 0.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="masked_fill_scalar"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        mask = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device=self.device
        )
        v = x.view_as(x)
        x.masked_fill_(mask, 7.0)
        expected = torch.tensor([7.0, 0.0, 7.0, 0.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="masked_fill_tensor"):
        x = torch.zeros(4, dtype=torch.float32, device=self.device)
        mask = torch.tensor(
            [False, True, False, True], dtype=torch.bool, device=self.device
        )
        val = torch.tensor(13.0, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        x.masked_fill_(mask, val)
        expected = torch.tensor([0.0, 13.0, 0.0, 13.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="masked_scatter_"):
        self_t = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device=self.device
        )
        mask = torch.tensor(
            [True, False, True], dtype=torch.bool, device=self.device
        )
        source = torch.tensor(
            [10.0, 20.0], dtype=torch.float32, device=self.device
        )
        v = self_t.view_as(self_t)
        self_t.masked_scatter_(mask, source)
        expected = torch.tensor([10.0, 2.0, 20.0])
        self_t_cpu = self_t.cpu()

        test_utils.assert_close(self_t_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), self_t.cpu()))

      with self.subTest(op="bucketize_out_self"):
        values = torch.tensor(
            [1, 3, 5, 7, 9], dtype=torch.int64, device=self.device
        )
        boundaries = torch.tensor(
            [2, 4, 6, 8], dtype=torch.int64, device=self.device
        )
        v = values.view_as(values)
        torch.bucketize(values, boundaries, out=values)
        expected = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        values_cpu = values.cpu()

        self.assertTrue(torch.equal(values_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), values.cpu()))

      with self.subTest(op="searchsorted_out_values"):
        sorted_seq = torch.tensor(
            [2, 4, 6, 8], dtype=torch.int64, device=self.device
        )
        values = torch.tensor(
            [1, 3, 5, 7, 9], dtype=torch.int64, device=self.device
        )
        v = values.view_as(values)
        torch.searchsorted(sorted_seq, values, out=values)
        expected = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        values_cpu = values.cpu()

        self.assertTrue(torch.equal(values_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), values.cpu()))

      with self.subTest(op="sort_out_values_self"):
        x = torch.tensor(
            [3.0, 1.0, 2.0], dtype=torch.float32, device=self.device
        )
        indices = torch.empty(3, dtype=torch.int64, device=self.device)
        v = x.view_as(x)
        torch.sort(x, out=(x, indices))
        expected_vals = torch.tensor([1.0, 2.0, 3.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected_vals)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="topk_out_values_self"):
        x = torch.tensor(
            [3.0, 1.0, 2.0], dtype=torch.float32, device=self.device
        )
        indices = torch.empty(3, dtype=torch.int64, device=self.device)
        v = x.view_as(x)
        torch.topk(x, k=3, out=(x, indices))
        expected_vals = torch.tensor([3.0, 2.0, 1.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected_vals)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="cummin_out_values_self"):
        x = torch.tensor(
            [3.0, 1.0, 4.0, 0.5], dtype=torch.float32, device=self.device
        )
        indices = torch.empty(4, dtype=torch.int64, device=self.device)
        v = x.view_as(x)
        torch.cummin(x, dim=0, out=(x, indices))
        expected_vals = torch.tensor([3.0, 1.0, 1.0, 0.5])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected_vals)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="cummax_out_values_self"):
        x = torch.tensor(
            [1.0, 3.0, 2.0, 4.0], dtype=torch.float32, device=self.device
        )
        indices = torch.empty(4, dtype=torch.int64, device=self.device)
        v = x.view_as(x)
        torch.cummax(x, dim=0, out=(x, indices))
        expected_vals = torch.tensor([1.0, 3.0, 3.0, 4.0])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected_vals)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="xlogy"):
        x = torch.tensor([2.0, 3.0], dtype=torch.float32, device=self.device)
        y = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.xlogy(x, y, out=x)
        expected_xlogy = torch.tensor([0.0, 2.0794415])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected_xlogy, atol=1e-4, rtol=1e-4)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

  def test_linear_algebra_and_nn_ops_donate_in_defer_never(self):
    """Verifies that linear algebra, matrix multiplication, pooling, and convolution ops work correctly with buffer donation in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      with self.subTest(op="addmm"):
        x = torch.zeros(2, 2, dtype=torch.float32, device=self.device)
        m1 = torch.ones(2, 2, dtype=torch.float32, device=self.device)
        m2 = torch.ones(2, 2, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.addmm(x, m1, m2, out=x)
        expected = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="baddbmm"):
        x = torch.zeros(1, 2, 2, dtype=torch.float32, device=self.device)
        b1 = torch.ones(1, 2, 2, dtype=torch.float32, device=self.device)
        b2 = torch.ones(1, 2, 2, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.baddbmm(x, b1, b2, out=x)
        expected = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="addmv"):
        x = torch.ones(2, dtype=torch.float32, device=self.device)
        mat = torch.ones(2, 3, dtype=torch.float32, device=self.device)
        vec = torch.ones(3, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.addmv(x, mat, vec, beta=2.0, alpha=3.0, out=x)
        expected = torch.tensor([11.0, 11.0])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="mm"):
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        y = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        torch.mm(x, y, out=x)
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="bmm"):
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32, device=self.device
        )
        y = torch.tensor(
            [[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32, device=self.device
        )
        v = x.view_as(x)
        torch.bmm(x, y, out=x)
        expected = torch.tensor([[[19.0, 22.0], [43.0, 50.0]]])
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="solve_triangular"):
        a = torch.tensor(
            [[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32, device=self.device
        )
        b = torch.tensor(
            [[4.0], [5.0]], dtype=torch.float32, device=self.device
        )
        v = b.view_as(b)
        torch.linalg.solve_triangular(a, b, upper=False, out=b)
        expected = torch.tensor([[2.0], [1.0]])
        b_cpu = b.cpu()

        test_utils.assert_close(b_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), b.cpu()))

      with self.subTest(op="lu_solve"):
        a_mat = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        lu, pivots = torch.linalg.lu_factor(a_mat)
        b = torch.tensor(
            [[5.0], [11.0]], dtype=torch.float32, device=self.device
        )
        v = b.view_as(b)
        torch.linalg.lu_solve(lu, pivots, b, out=b)
        expected = torch.tensor([[1.0], [2.0]])
        b_cpu = b.cpu()

        test_utils.assert_close(b_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), b.cpu()))

      with self.subTest(op="geqrf_out_a"):
        x = torch.randn(8, 4, dtype=torch.float32, device=self.device)
        tau = torch.empty(4, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.geqrf(x, out=(x, tau))
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="lu_factor_out_a"):
        a = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        pivots = torch.empty(4, dtype=torch.int32, device=self.device)
        v = a.view_as(a)
        torch.linalg.lu_factor(a, out=(a, pivots))
        _ = a.cpu()
        self.assertTrue(torch.equal(v.cpu(), a.cpu()))

      with self.subTest(op="linalg_qr_out_q_self"):
        a = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        r = torch.empty_like(a)
        v = a.view_as(a)
        torch.linalg.qr(a, mode="reduced", out=(a, r))
        _ = a.cpu()
        self.assertTrue(torch.equal(v.cpu(), a.cpu()))

      with self.subTest(op="addcdiv_out_self"):
        s = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        t1 = torch.tensor([6.0, 8.0], dtype=torch.float32, device=self.device)
        t2 = torch.tensor([2.0, 4.0], dtype=torch.float32, device=self.device)
        v = s.view_as(s)
        torch.addcdiv(s, t1, t2, value=2.0, out=s)
        expected = torch.tensor([7.0, 6.0])
        s_cpu = s.cpu()

        self.assertTrue(torch.equal(s_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), s.cpu()))

      with self.subTest(op="addcdiv_out_t1"):
        s = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        t1 = torch.tensor([6.0, 8.0], dtype=torch.float32, device=self.device)
        t2 = torch.tensor([2.0, 4.0], dtype=torch.float32, device=self.device)
        v = t1.view_as(t1)
        torch.addcdiv(s, t1, t2, value=2.0, out=t1)
        expected = torch.tensor([7.0, 6.0])
        t1_cpu = t1.cpu()

        self.assertTrue(torch.equal(t1_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), t1.cpu()))

      with self.subTest(op="addcmul_out_self"):
        s = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        t1 = torch.tensor([3.0, 4.0], dtype=torch.float32, device=self.device)
        t2 = torch.tensor([2.0, 5.0], dtype=torch.float32, device=self.device)
        v = s.view_as(s)
        torch.addcmul(s, t1, t2, value=2.0, out=s)
        expected = torch.tensor([13.0, 42.0])
        s_cpu = s.cpu()

        self.assertTrue(torch.equal(s_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), s.cpu()))

      with self.subTest(op="addcmul_out_t1"):
        s = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
        t1 = torch.tensor([3.0, 4.0], dtype=torch.float32, device=self.device)
        t2 = torch.tensor([2.0, 5.0], dtype=torch.float32, device=self.device)
        v = t1.view_as(t1)
        torch.addcmul(s, t1, t2, value=2.0, out=t1)
        expected = torch.tensor([13.0, 42.0])
        t1_cpu = t1.cpu()

        self.assertTrue(torch.equal(t1_cpu, expected))

        self.assertTrue(torch.equal(v.cpu(), t1.cpu()))

      with self.subTest(op="addmv_out_vec"):
        mat = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        vec = torch.tensor([1.0, 1.0], dtype=torch.float32, device=self.device)
        self_t = torch.tensor(
            [0.5, 0.5], dtype=torch.float32, device=self.device
        )
        v = vec.view_as(vec)
        torch.addmv(self_t, mat, vec, out=vec)
        expected = torch.tensor([3.5, 7.5])
        vec_cpu = vec.cpu()

        test_utils.assert_close(vec_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), vec.cpu()))

      with self.subTest(op="addmm_out_mat1"):
        self_t = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=self.device
        )
        m1 = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        m2 = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=self.device
        )
        v = m1.view_as(m1)
        torch.addmm(self_t, m1, m2, out=m1)
        expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        m1_cpu = m1.cpu()

        test_utils.assert_close(m1_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), m1.cpu()))

      with self.subTest(op="addmm_out_mat2"):
        self_t = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=self.device
        )
        m1 = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=self.device
        )
        m2 = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device
        )
        v = m2.view_as(m2)
        torch.addmm(self_t, m1, m2, out=m2)
        expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        m2_cpu = m2.cpu()

        test_utils.assert_close(m2_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), m2.cpu()))

      with self.subTest(op="baddbmm_out_batch1"):
        b = torch.zeros(2, 2, 2, dtype=torch.float32, device=self.device)
        m1 = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        m2 = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        v = m1.view_as(m1)
        torch.baddbmm(b, m1, m2, out=m1)
        expected = torch.bmm(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]),
        )
        m1_cpu = m1.cpu()

        test_utils.assert_close(m1_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), m1.cpu()))

      with self.subTest(op="baddbmm_out_batch2"):
        b = torch.zeros(2, 2, 2, dtype=torch.float32, device=self.device)
        m1 = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        m2 = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        v = m2.view_as(m2)
        torch.baddbmm(b, m1, m2, out=m2)
        expected = torch.bmm(
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]]),
        )
        m2_cpu = m2.cpu()

        test_utils.assert_close(m2_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), m2.cpu()))

      with self.subTest(op="avg_pool2d_out"):
        x = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        v = x.view_as(x)
        torch.ops.aten.avg_pool2d.out(
            x, [1, 1], [1, 1], [0, 0], False, True, None, out=x
        )
        expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="avg_pool2d_backward_grad_output"):
        grad_output = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        self_t = torch.zeros_like(grad_output)
        v = grad_output.view_as(grad_output)
        torch.ops.aten.avg_pool2d_backward(
            grad_output,
            self_t,
            [1, 1],
            [1, 1],
            [0, 0],
            False,
            True,
            None,
            grad_input=grad_output,
        )
        expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        grad_output_cpu = grad_output.cpu()

        test_utils.assert_close(grad_output_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), grad_output.cpu()))

      with self.subTest(op="avg_pool2d_backward_self"):
        grad_output = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        self_t = torch.zeros_like(grad_output)
        v = self_t.view_as(self_t)
        torch.ops.aten.avg_pool2d_backward(
            grad_output,
            self_t,
            [1, 1],
            [1, 1],
            [0, 0],
            False,
            True,
            None,
            grad_input=self_t,
        )
        expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self_t_cpu = self_t.cpu()

        test_utils.assert_close(self_t_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), self_t.cpu()))

      with self.subTest(op="adaptive_avg_pool2d_out"):
        x = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        v = x.view_as(x)
        torch.ops.aten.adaptive_avg_pool2d.out(x, [2, 2], out=x)
        expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="adaptive_avg_pool3d_backward_self"):
        grad_output = torch.ones(
            1, 1, 2, 2, 2, dtype=torch.float32, device=self.device
        )
        self_t = torch.zeros(
            1, 1, 2, 2, 2, dtype=torch.float32, device=self.device
        )
        v = self_t.view_as(self_t)
        torch.ops.aten.adaptive_avg_pool3d_backward(
            grad_output, self_t, grad_input=self_t
        )
        expected = torch.ones(1, 1, 2, 2, 2)
        self_t_cpu = self_t.cpu()

        test_utils.assert_close(self_t_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), self_t.cpu()))

      with self.subTest(op="max_pool2d_with_indices_backward_grad_input_self"):
        x = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        grad_output = torch.tensor(
            [[[[1.0]]]], dtype=torch.float32, device=self.device
        )
        indices = torch.tensor([[[[3]]]], dtype=torch.int64, device=self.device)
        v = x.view_as(x)
        torch.ops.aten.max_pool2d_with_indices_backward(
            grad_output,
            x,
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            False,
            indices,
            grad_input=x,
        )
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="convolution_out"):
        inp = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        weight = torch.tensor(
            [[[[2.0]]]], dtype=torch.float32, device=self.device
        )
        v = inp.view_as(inp)
        torch.ops.aten.convolution.out(
            inp,
            weight,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
            out=inp,
        )
        expected = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
        inp_cpu = inp.cpu()

        test_utils.assert_close(inp_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), inp.cpu()))

      with self.subTest(op="native_batch_norm_out"):
        inp = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        save_mean = torch.empty(1, dtype=torch.float32, device=self.device)
        save_invstd = torch.empty(1, dtype=torch.float32, device=self.device)
        v = inp.view_as(inp)
        torch.ops.aten.native_batch_norm.out(
            inp,
            None,
            None,
            None,
            None,
            False,
            0.1,
            1e-5,
            out=inp,
            save_mean=save_mean,
            save_invstd=save_invstd,
        )
        self.assertTrue(torch.equal(inp.cpu(), v.cpu()))

      with self.subTest(op="glu_backward_grad_input_self"):
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
            device=self.device,
        )
        grad_output = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        v = x.view_as(x)
        torch.ops.aten.glu_backward.grad_input(grad_output, x, 0, grad_input=x)
        _ = x.cpu()
        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="fft_c2c_out_self"):
        x = torch.tensor(
            [1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex64, device=self.device
        )
        v = x.view_as(x)
        torch.fft.fft(x, out=x)
        expected = torch.fft.fft(torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]))
        x_cpu = x.cpu()

        test_utils.assert_close(x_cpu, expected)

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

  def test_fused_cells_and_distributions_donate_in_defer_never(self):
    """Verifies that fused RNN cells, random distributions, and fused optimizers work correctly with buffer donation in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      with self.subTest(op="_thnn_fused_gru_cell_out_hx"):
        batch, hidden = 2, 4
        ig = torch.randn(
            batch, 3 * hidden, dtype=torch.float32, device=self.device
        )
        hg = torch.randn(
            batch, 3 * hidden, dtype=torch.float32, device=self.device
        )
        hx = torch.randn(batch, hidden, dtype=torch.float32, device=self.device)
        ws = torch.empty(
            batch, 5 * hidden, dtype=torch.float32, device=self.device
        )
        v = hx.view_as(hx)
        torch.ops.aten._thnn_fused_gru_cell.out(
            ig, hg, hx, None, None, out0=hx, out1=ws
        )
        _ = hx.cpu()
        self.assertTrue(torch.equal(v.cpu(), hx.cpu()))

      with self.subTest(op="_thnn_fused_lstm_cell_out_cx"):
        batch, hidden = 2, 4
        ig = torch.randn(
            batch, 4 * hidden, dtype=torch.float32, device=self.device
        )
        hg = torch.randn(
            batch, 4 * hidden, dtype=torch.float32, device=self.device
        )
        cx = torch.randn(batch, hidden, dtype=torch.float32, device=self.device)
        hy = torch.empty(batch, hidden, dtype=torch.float32, device=self.device)
        ws = torch.empty(
            batch, 4 * hidden, dtype=torch.float32, device=self.device
        )
        v = cx.view_as(cx)
        torch.ops.aten._thnn_fused_lstm_cell.out(
            ig, hg, cx, None, None, out0=hy, out1=cx, out2=ws
        )
        _ = cx.cpu()
        self.assertTrue(torch.equal(v.cpu(), cx.cpu()))

      with self.subTest(op="bernoulli_out_self"):
        x = torch.ones(10, dtype=torch.float32, device=self.device)
        v = x.view_as(x)
        torch.bernoulli(x, out=x)
        _ = x.cpu()
        x_cpu = x.cpu()

        self.assertTrue(torch.equal(x_cpu, torch.ones(10)))

        self.assertTrue(torch.equal(v.cpu(), x.cpu()))

      with self.subTest(op="bernoulli_tensor_p"):
        p = torch.ones(10, dtype=torch.float32, device=self.device)
        v = p.view_as(p)
        p.bernoulli_(p)
        _ = p.cpu()
        p_cpu = p.cpu()

        self.assertTrue(torch.equal(p_cpu, torch.ones(10)))

        self.assertTrue(torch.equal(v.cpu(), p.cpu()))

      with self.subTest(op="normal_out_float_tensor"):
        std = torch.ones(10, dtype=torch.float32, device=self.device)
        v = std.view_as(std)
        torch.normal(0.0, std, out=std)
        _ = std.cpu()
        self.assertTrue(torch.equal(v.cpu(), std.cpu()))

      with self.subTest(op="normal_out_tensor_float"):
        mean = torch.zeros(10, dtype=torch.float32, device=self.device)
        v = mean.view_as(mean)
        torch.normal(mean, 1.0, out=mean)
        _ = mean.cpu()
        self.assertTrue(torch.equal(v.cpu(), mean.cpu()))

      with self.subTest(op="normal_out_tensor_tensor_mean"):
        mean = torch.zeros(10, dtype=torch.float32, device=self.device)
        std = torch.ones(10, dtype=torch.float32, device=self.device)
        v = mean.view_as(mean)
        torch.normal(mean, std, out=mean)
        _ = mean.cpu()
        self.assertTrue(torch.equal(v.cpu(), mean.cpu()))

      with self.subTest(op="normal_out_tensor_tensor_std"):
        mean = torch.zeros(10, dtype=torch.float32, device=self.device)
        std = torch.ones(10, dtype=torch.float32, device=self.device)
        v = std.view_as(std)
        torch.normal(mean, std, out=std)
        _ = std.cpu()
        self.assertTrue(torch.equal(v.cpu(), std.cpu()))

      with self.subTest(op="_fused_moving_avg_obs_fq_helper"):
        x = torch.tensor([-2.0, 3.0], dtype=torch.float32, device=self.device)
        obs_on = torch.tensor(1, dtype=torch.int32, device=self.device)
        fq_on = torch.tensor(1, dtype=torch.int32, device=self.device)
        rmin = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        rmax = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        sc = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        zp = torch.tensor([0], dtype=torch.int32, device=self.device)
        v_rmin = rmin.view_as(rmin)
        v_rmax = rmax.view_as(rmax)
        v_sc = sc.view_as(sc)
        v_zp = zp.view_as(zp)
        torch.ops.aten._fused_moving_avg_obs_fq_helper(
            x,
            obs_on,
            fq_on,
            rmin,
            rmax,
            sc,
            zp,
            0.5,
            -128,
            127,
            0,
            False,
            False,
        )
        _ = rmin.cpu()
        self.assertTrue(torch.equal(v_rmin.cpu(), rmin.cpu()))
        self.assertTrue(torch.equal(v_rmax.cpu(), rmax.cpu()))
        self.assertTrue(torch.equal(v_sc.cpu(), sc.cpu()))
        self.assertTrue(torch.equal(v_zp.cpu(), zp.cpu()))

      with self.subTest(op="_fused_adamw__grad_scale"):
        p = torch.tensor([1.0, -2.0], dtype=torch.float32, device=self.device)
        g = torch.tensor([0.4, 0.8], dtype=torch.float32, device=self.device)
        ea = torch.tensor([0.1, 0.2], dtype=torch.float32, device=self.device)
        eas = torch.tensor(
            [0.01, 0.02], dtype=torch.float32, device=self.device
        )
        step = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        grad_scale = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        v_g = g.view_as(g)
        v_p = p.view_as(p)
        torch.ops.aten._fused_adamw_.default(
            [p],
            [g],
            [ea],
            [eas],
            [],
            [step],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
            grad_scale=grad_scale,
        )
        _ = g.cpu()
        self.assertTrue(torch.equal(v_g.cpu(), g.cpu()))
        self.assertTrue(torch.equal(v_p.cpu(), p.cpu()))

  def test_foreach_inplace_ops_donate_in_defer_never(self):
    """Verifies that in-place foreach ops donate buffers correctly in DeferNever."""
    with execution_mode.set_eager_mode(EagerMode.DEFER_NEVER):
      t1 = torch.tensor([1.0, 2.0], dtype=torch.float32, device=self.device)
      t2 = torch.tensor([3.0, 4.0], dtype=torch.float32, device=self.device)
      v1 = t1.view_as(t1)
      v2 = t2.view_as(t2)

      o1 = torch.tensor([10.0, 20.0], dtype=torch.float32, device=self.device)
      o2 = torch.tensor([30.0, 40.0], dtype=torch.float32, device=self.device)

      torch._foreach_add_([t1, t2], [o1, o2])

      self.assertTrue(torch.equal(t1.cpu(), torch.tensor([11.0, 22.0])))
      self.assertTrue(torch.equal(t2.cpu(), torch.tensor([33.0, 44.0])))
      self.assertTrue(torch.equal(v1.cpu(), torch.tensor([11.0, 22.0])))
      self.assertTrue(torch.equal(v2.cpu(), torch.tensor([33.0, 44.0])))

      # Test foreach unary inplace: _foreach_neg_
      torch._foreach_neg_([t1, t2])
      self.assertTrue(torch.equal(t1.cpu(), torch.tensor([-11.0, -22.0])))
      self.assertTrue(torch.equal(t2.cpu(), torch.tensor([-33.0, -44.0])))
      self.assertTrue(torch.equal(v1.cpu(), torch.tensor([-11.0, -22.0])))
      self.assertTrue(torch.equal(v2.cpu(), torch.tensor([-33.0, -44.0])))


if __name__ == "__main__":
  absltest.main()
