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

"""Tests for multi-bucket bounded dynamism compilation and lookahead prefetching."""

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class MultiBucketCompilationTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_multi_bucket_lookahead_async_compilation(self):
    def fn(x):
      return x + 1.0

    orig_static_call = _backend.compiler.StaticCompiler.__call__
    orig_resolve = torch_tpu_compiled_executable.AsyncCompiledArtifact.resolve
    compiled_upper_bounds = []
    bounds_at_first_resolve = []

    def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
      for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.dim() == 1:
          compiled_upper_bounds.append(inp.shape[0])
          break
      return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

    def wrapped_resolve(artifact_self):
      if not bounds_at_first_resolve:
        bounds_at_first_resolve.extend(compiled_upper_bounds)
      return orig_resolve(artifact_self)

    with (
        mock.patch.object(
            _backend.dynamic_compiler,
            "DynamicCompiler",
            wraps=_backend.dynamic_compiler.DynamicCompiler,
        ) as mock_dyn_compiler,
        mock.patch.object(
            _backend.compiler.StaticCompiler,
            "__call__",
            autospec=True,
            side_effect=wrapped_static_call,
        ) as mock_static_call,
        mock.patch.object(
            torch_tpu_compiled_executable.AsyncCompiledArtifact,
            "resolve",
            autospec=True,
            side_effect=wrapped_resolve,
        ),
    ):
      compiled = torch.compile(
          fn,
          backend="tpu",
          options={
              "num_dynamic_buckets": 4,
          },
      )

      # Step 1: Execute in Bucket 0 ([2, 128]).
      # By default num_lookahead_buckets=num_dynamic_buckets=4 compiles
      # Bucket 0 (128) and prefetches Buckets 1 (256), 2 (384), and 3 (512)
      # BEFORE resolving Bucket 0.
      x0 = torch.ones(32, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=512)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 4)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384, 512])
      self.assertEqual(bounds_at_first_resolve, [128, 256, 384, 512])

      # Step 2: Another shape in Bucket 0 ([2, 128]) -> no new compilations.
      x0_again = torch.ones(64, device=self.device)
      out0_again = compiled(x0_again)
      utils.assert_close(out0_again, fn(x0_again))
      self.assertEqual(mock_static_call.call_count, 4)

      # Step 3: Execute in Buckets 1 ([129, 256]), 2 ([257, 384]), and 3 ([385, 512]).
      # Reuses prefetched buckets; no further compilations needed.
      for length in (160, 300, 420):
        x_next = torch.ones(length, device=self.device)
        out_next = compiled(x_next)
        utils.assert_close(out_next, fn(x_next))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 4)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384, 512])

  def test_multi_bucket_custom_bucket_size(self):
    def fn(x):
      return x + 1.0

    orig_static_call = _backend.compiler.StaticCompiler.__call__
    compiled_upper_bounds = []

    def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
      for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.dim() == 1:
          compiled_upper_bounds.append(inp.shape[0])
          break
      return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

    with mock.patch.object(
        _backend.compiler.StaticCompiler,
        "__call__",
        autospec=True,
        side_effect=wrapped_static_call,
    ) as mock_static_call:
      compiled = torch.compile(
          fn,
          backend="tpu",
          options={
              "num_dynamic_buckets": 3,
              "bucket_size": 64,
          },
      )

      x0 = torch.ones(16, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=192)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_static_call.call_count, 3)
      self.assertEqual(compiled_upper_bounds, [64, 128, 192])

      for length in (96, 160):
        x_next = torch.ones(length, device=self.device)
        out_next = compiled(x_next)
        utils.assert_close(out_next, fn(x_next))
      self.assertEqual(mock_static_call.call_count, 3)

  def test_bucket_switch_keeps_capacity_alignment_of_clamped_symbol(self):
    # Slicing `rope` at `past_len : past_len + 1` refines `past_len`'s ShapeEnv
    # upper bound to `max_len - 1 = 199` (unaligned), while `torch.cat` gives
    # the updated KV cache output shape `past_len + 1` (offset c = 1).
    # Bucket 0 covers [2, 127] and bucket 1 must align its capacity to
    # `256 - 1 = 255` (instead of clamping to 199) so the KV cache output
    # keeps the same TPU layout across the bucket switch and does not force a
    # recompile on the step after switching to bucket 1.
    max_len = 200
    rope = torch.randn(
        1, 2, max_len, 128, dtype=torch.bfloat16, device=self.device
    )

    def decode_step(q, k_new, v_new, k_cache, v_cache):
      past_len = k_cache.shape[2]
      # Clamps `past_len` to <= max_len - 1 (199) in ShapeEnv.
      k_new = k_new + rope[:, :, past_len : past_len + 1, :]
      # Output cache length is `past_len + 1`, fed back as next step's input.
      k_cache = torch.cat([k_cache, k_new], dim=2)
      v_cache = torch.cat([v_cache, v_new], dim=2)
      out = torch.nn.functional.scaled_dot_product_attention(
          q, k_cache, v_cache, enable_gqa=True
      )
      return out, k_cache, v_cache

    compiled_upper_bounds: list[int] = []
    orig_static_call = _backend.compiler.StaticCompiler.__call__

    def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
      for inp, bound in zip(example_inputs, kwargs.get("bounds") or ()):
        if bound is not None:
          compiled_upper_bounds.append(inp.shape[2])
          break
      return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

    with mock.patch.object(
        _backend.compiler.StaticCompiler,
        "__call__",
        autospec=True,
        side_effect=wrapped_static_call,
    ) as mock_static_call:
      compiled = torch.compile(
          decode_step, backend="tpu", options={"num_dynamic_buckets": 2}
      )
      q = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16, device=self.device)
      # Start at 126 so steps 0..1 run in bucket 0 ([2, 127]) and steps 2..3
      # cross into bucket 1 ([128, 255]).
      k_cache = torch.randn(
          1, 2, 126, 128, dtype=torch.bfloat16, device=self.device
      )
      v_cache = torch.randn(
          1, 2, 126, 128, dtype=torch.bfloat16, device=self.device
      )
      torch._dynamo.mark_dynamic(k_cache, 2)
      torch._dynamo.mark_dynamic(v_cache, 2)

      for step in range(4):
        k_new = torch.randn(
            1, 2, 1, 128, dtype=torch.bfloat16, device=self.device
        )
        v_new = torch.randn(
            1, 2, 1, 128, dtype=torch.bfloat16, device=self.device
        )
        expected_out, k_ref, v_ref = decode_step(
            q, k_new, v_new, k_cache.clone(), v_cache.clone()
        )
        actual_out, k_cache, v_cache = compiled(
            q, k_new, v_new, k_cache, v_cache
        )
        utils.assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
        utils.assert_close(k_cache, k_ref)
        if step == 1:
          # Step 0 compiles [127, 255] for the initial eager input layout and
          # step 1 compiles [127, 255] for the recurrent KV output layout.
          self.assertEqual(compiled_upper_bounds, [127, 255, 127, 255])

      # Steps 2..3 in bucket 1 ([128, 255]) reuse the lookahead executable with
      # no layout-mismatch recompile after switching buckets.
      self.assertEqual(compiled_upper_bounds, [127, 255, 127, 255])
      self.assertEqual(mock_static_call.call_count, 4)

  def test_bucket_switch_with_unaligned_mark_dynamic_max(self):
    # `mark_dynamic(min=2, max=200)` is a StrictMinMaxConstraint: the second
    # bucket extends to the aligned capacity 255 (>= 200) instead of 200, and
    # the bound checks on that capacity must not add a guard (Dynamo would
    # raise a ConstraintViolationError for a guard narrowing the range).
    def fn(x, y):
      return (
          torch.cat([x, y]) + 1.0
      )  # Output length s + 1 -> capacities 128k - 1.

    compiled_upper_bounds: list[int] = []
    orig_static_call = _backend.compiler.StaticCompiler.__call__

    def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
      for inp, bound in zip(example_inputs, kwargs.get("bounds") or ()):
        if bound is not None:
          compiled_upper_bounds.append(inp.shape[0])
          break
      return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

    with mock.patch.object(
        _backend.compiler.StaticCompiler,
        "__call__",
        autospec=True,
        side_effect=wrapped_static_call,
    ) as mock_static_call:
      compiled = torch.compile(
          fn, backend="tpu", options={"num_dynamic_buckets": 2}
      )
      for length in (100, 127, 128, 129, 200):
        x = torch.ones(length, device=self.device)
        y = torch.ones(1, device=self.device)
        if length == 100:
          torch._dynamo.mark_dynamic(x, 0, min=2, max=200)
        utils.assert_close(compiled(x, y), fn(x, y))

      self.assertEqual(compiled_upper_bounds, [127, 255])
      self.assertEqual(mock_static_call.call_count, 2)

  def test_static_parameter_sliced_by_dynamic_end_backward(self):
    # Slicing a static parameter `pos_emb[:s]` refines `s` to <= max_len (200),
    # while bucket 1's aligned capacity (256) overshoots `max_len`.
    # `slice_backward` must bound the gradient window by the ShapeEnv range
    # and zero padded rows past `s` so the static parameter's gradient stays
    # static and uncorrupted across bucket 0 ([2, 128]) and bucket 1 ([129, 256]).
    max_len, dim = 200, 8
    pos_emb = torch.randn(max_len, dim, device=self.device, requires_grad=True)

    def fn(x, emb):
      return (x + emb[: x.shape[1]]).sum()

    compiled = torch.compile(
        fn, backend="tpu", options={"num_dynamic_buckets": 2}
    )
    # 100 runs in bucket 0 ([2, 128]); 150 and 200 run in bucket 1 ([129, 256]).
    for length in (100, 150, max_len):
      x = torch.randn(2, length, dim, device=self.device)
      torch._dynamo.mark_dynamic(x, 1)
      pos_emb.grad = None
      compiled(x, pos_emb).backward()
      expected_grad = torch.zeros(max_len, dim, device=self.device)
      expected_grad[:length] = 2.0
      utils.assert_close(pos_emb.grad, expected_grad)


if __name__ == "__main__":
  absltest.main()
