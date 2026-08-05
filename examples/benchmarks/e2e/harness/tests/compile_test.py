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

"""Tests for `harness/compile.py`.

Mainly checks for two things:

  - Validating config at construction: A bad scope must raise where the
    benchmark is declared.
  - compile_config.apply forwards the fields to torch.compile; if any field is
    dropped, it silently changes what is being measured.

Note: torch.compile is mocked; Only the config plumbing is tested.
"""

from unittest import mock

from absl.testing import absltest
import torch.nn as nn
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import target as target_lib
from tests import seed_test_utils


class CompileConfigTest(seed_test_utils.RepeatableTest):

  def test_default_is_step_scope(self):
    cfg = compile_lib.CompileConfig()
    self.assertEqual(cfg.scope, compile_lib.Scope.STEP)
    self.assertIsNone(cfg.dynamic)
    self.assertFalse(cfg.fullgraph)

  def test_scope_enum_values(self):
    self.assertEqual(compile_lib.Scope.MODEL.value, "model")
    self.assertEqual(compile_lib.Scope.STEP.value, "step")

  def test_frozen(self):
    with self.assertRaises(Exception):
      compile_lib.CompileConfig().scope = compile_lib.Scope.STEP  # type: ignore[misc]

  def test_bad_scope_raises(self):
    with self.assertRaises(ValueError):
      compile_lib.Scope("custom")

  def test_both_scopes_accepted(self):
    self.assertEqual(
        compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL).scope,
        compile_lib.Scope.MODEL,
    )
    self.assertEqual(
        compile_lib.CompileConfig(scope=compile_lib.Scope.STEP).scope,
        compile_lib.Scope.STEP,
    )

  def test_apply_wraps_the_object(self):
    model = nn.Linear(4, 4)
    target = target_lib.make_target(target_lib.Platform.CPU)
    with mock.patch("torch.compile", return_value="WRAPPED") as tc:
      out = compile_lib.CompileConfig().apply(model, target)
    self.assertEqual(out, "WRAPPED")
    self.assertIs(tc.call_args.args[0], model)

  def test_apply_forwards_all_fields(self):
    cfg = compile_lib.CompileConfig(dynamic=False, fullgraph=True)
    target = target_lib.make_target(target_lib.Platform.CPU)
    with mock.patch("torch.compile", return_value="X") as tc:
      cfg.apply(nn.Linear(2, 2), target)
    kw = tc.call_args.kwargs
    self.assertEqual(kw["backend"], "inductor")
    self.assertIs(kw["dynamic"], False)
    self.assertIs(kw["fullgraph"], True)

  def test_apply_forwards_default_values(self):
    target = target_lib.make_target(target_lib.Platform.CPU)
    with mock.patch("torch.compile", return_value="X") as tc:
      compile_lib.CompileConfig().apply(nn.Linear(2, 2), target)
    kw = tc.call_args.kwargs
    self.assertIsNone(kw["dynamic"])
    self.assertFalse(kw["fullgraph"])

  def test_tpu_backend_selected(self):
    target = target_lib.make_target(target_lib.Platform.V5E_1X1)
    with mock.patch("torch.compile", return_value="X") as tc, mock.patch(
        "torch_tpu._internal.compile.TpuBackend", return_value="TPU_BACKEND"
    ):
      compile_lib.CompileConfig().apply(nn.Linear(2, 2), target)
    self.assertEqual(tc.call_args.kwargs["backend"], "TPU_BACKEND")

  def test_config_applies_to_a_callable(self):
    def step(model, args, kwargs, optim):  # pylint: disable=unused-argument
      return None

    target = target_lib.make_target(target_lib.Platform.CPU)
    with mock.patch("torch.compile", return_value="COMPILED_STEP") as tc:
      out = compile_lib.CompileConfig(scope=compile_lib.Scope.STEP).apply(
          step, target
      )
    self.assertEqual(out, "COMPILED_STEP")
    self.assertIs(tc.call_args.args[0], step)

  def test_scope_does_not_affect_apply(self):
    obj = nn.Linear(2, 2)
    target = target_lib.make_target(target_lib.Platform.CPU)
    with mock.patch("torch.compile", return_value="X") as tc:
      compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL).apply(
          obj, target
      )
      model_kwargs = tc.call_args.kwargs
      compile_lib.CompileConfig(scope=compile_lib.Scope.STEP).apply(obj, target)
      step_kwargs = tc.call_args.kwargs
    self.assertEqual(model_kwargs, step_kwargs)


if __name__ == "__main__":
  absltest.main()
