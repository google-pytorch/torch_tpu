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
    orig_resolve = _backend.AsyncCompiledArtifact.resolve
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
            _backend.AsyncCompiledArtifact,
            "resolve",
            autospec=True,
            side_effect=wrapped_resolve,
        ),
    ):
      compiled = torch.compile(
          fn,
          backend="tpu",
          options={
              "bounded_dynamism": True,
              "num_dynamic_buckets": 4,
              "dynamic_async_compile": True,
          },
      )

      # Step 1: Execute in Bucket 0 ([2, 128]).
      # Default num_lookahead_buckets=1 compiles Bucket 0 (128) and prefetches
      # Bucket 1 (256) BEFORE resolving Bucket 0.
      x0 = torch.ones(32, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=512)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 2)
      self.assertEqual(compiled_upper_bounds, [128, 256])
      self.assertEqual(bounds_at_first_resolve, [128, 256])

      # Step 2: Another shape in Bucket 0 ([2, 128]) -> no new compilations.
      x0_again = torch.ones(64, device=self.device)
      out0_again = compiled(x0_again)
      utils.assert_close(out0_again, fn(x0_again))
      self.assertEqual(mock_static_call.call_count, 2)

      # Step 3: Execute in Bucket 1 ([129, 256]).
      # Reuses prefetched Bucket 1 (256) and prefetches Bucket 2 (384).
      x1 = torch.ones(160, device=self.device)
      out1 = compiled(x1)
      utils.assert_close(out1, fn(x1))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 3)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384])

      # Step 4: Execute in Bucket 2 ([257, 384]) and prefetch Bucket 3 ([385, 512]).
      # Reuses prefetched buckets; no further buckets to prefetch.
      for length in (300, 420):
        x_next = torch.ones(length, device=self.device)
        out_next = compiled(x_next)
        utils.assert_close(out_next, fn(x_next))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 4)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384, 512])

  def test_multi_bucket_custom_num_lookahead_buckets(self):
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
              "bounded_dynamism": True,
              "num_dynamic_buckets": 3,
              "dynamic_async_compile": True,
              "num_lookahead_buckets": 1,
          },
      )

      x0 = torch.ones(32, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=384)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_static_call.call_count, 2)
      self.assertEqual(compiled_upper_bounds, [128, 256])

      x1 = torch.ones(160, device=self.device)
      out1 = compiled(x1)
      utils.assert_close(out1, fn(x1))
      self.assertEqual(mock_static_call.call_count, 3)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384])

  def test_multi_bucket_sync_compile_no_lookahead(self):
    def fn(x):
      return x * 2.0

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
              "bounded_dynamism": True,
              "num_dynamic_buckets": 3,
              "dynamic_async_compile": False,
          },
      )

      # With dynamic_async_compile=False, Bucket 0 compiles only Bucket 0 (128)
      # without prefetching Bucket 1.
      x0 = torch.ones(32, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=384)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_static_call.call_count, 1)
      self.assertEqual(compiled_upper_bounds, [128])

      # Executing Bucket 1 compiles only Bucket 1 (256) without prefetching Bucket 2.
      x1 = torch.ones(160, device=self.device)
      out1 = compiled(x1)
      utils.assert_close(out1, fn(x1))
      self.assertEqual(mock_static_call.call_count, 2)
      self.assertEqual(compiled_upper_bounds, [128, 256])


if __name__ == "__main__":
  absltest.main()
