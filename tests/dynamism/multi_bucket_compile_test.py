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


if __name__ == "__main__":
  absltest.main()
