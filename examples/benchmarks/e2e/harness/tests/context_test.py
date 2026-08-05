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

"""Tests for harness/context.py."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import target as target_lib
from tests import seed_test_utils


class RunScopeTest(parameterized.TestCase):

  @parameterized.parameters(
      ("full", context_lib.RunScope.FULL),
      ("presubmit", context_lib.RunScope.PRESUBMIT),
  )
  def test_run_scope_values(
      self, value: str, expected_scope: context_lib.RunScope
  ):
    self.assertIs(context_lib.RunScope(value), expected_scope)
    self.assertEqual(expected_scope.value, value)

  def test_invalid_run_scope_raises_value_error(self):
    with self.assertRaises(ValueError) as cm:
      context_lib.RunScope("invalid_scope")
    self.assertIn("invalid_scope", str(cm.exception))


class ContextTest(parameterized.TestCase):

  @parameterized.product(
      platform=[
          target_lib.Platform.CPU,
          target_lib.Platform.B200_1,
          target_lib.Platform.V6E_1X1,
      ],
      dtype=[target_lib.DType.BF16, target_lib.DType.FP32],
      run_scope=[context_lib.RunScope.FULL, context_lib.RunScope.PRESUBMIT],
  )
  def test_context_properties_passthrough(
      self,
      platform: target_lib.Platform,
      dtype: target_lib.DType,
      run_scope: context_lib.RunScope,
  ):
    target = target_lib.make_target(platform=platform, dtype=dtype)
    context = context_lib.Context(target=target, run_scope=run_scope)
    self.assertEqual(context.target, target)
    self.assertEqual(context.run_scope, run_scope)
    self.assertIs(context.device_kind, target.device_kind)
    self.assertIs(context.dtype, target.dtype)

  def test_context_is_frozen(self):
    target = target_lib.make_target(target_lib.Platform.CPU)
    context = context_lib.Context(
        target=target, run_scope=context_lib.RunScope.FULL
    )
    with self.assertRaises(dataclasses.FrozenInstanceError):
      context.run_scope = context_lib.RunScope.PRESUBMIT
    with self.assertRaises(dataclasses.FrozenInstanceError):
      context.target = target_lib.make_target(target_lib.Platform.V6E_1X1)

  def test_context_excludes_mode_and_framework(self):
    """Verifies invariant that mode and framework are excluded from Context.

    This prevents workload factories from generating divergent models based on
    execution mode or framework, preserving eager-vs-compiled comparability.
    """
    target = target_lib.make_target(target_lib.Platform.CPU)
    context = context_lib.Context(
        target=target, run_scope=context_lib.RunScope.FULL
    )
    self.assertFalse(hasattr(context, "mode"))
    self.assertFalse(hasattr(context, "framework"))


class ContextIsFrameworkAgnosticTest(seed_test_utils.RepeatableTest):

  def test_module_imports_no_framework(self):
    """Importing harness.context must not bind torch or jax into its namespace."""
    self.assertFalse(hasattr(context_lib, "torch"))
    self.assertFalse(hasattr(context_lib, "jax"))


if __name__ == "__main__":
  absltest.main()
