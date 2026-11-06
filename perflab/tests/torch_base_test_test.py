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

"""Tests for perflab/torch_base_test.py."""

import os
from typing import Sequence
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import torch
from torch import nn
from perflab import common
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import export as export_lib
from perflab import flags as flags_lib
from perflab import measure as measure_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_base_test
from tests import seed_test_utils

# Registers the steppers that `make_run_step` resolves.
discovery_lib.import_submodules(steps)

_IN_FEATURES = 4
_OUT_FEATURES = 3
_BATCH_SIZE = 2


def _forward_factory(ctx: context_lib.Context):
  del ctx  # Unused.
  return (
      nn.Linear(_IN_FEATURES, _OUT_FEATURES),
      (torch.randn(_BATCH_SIZE, _IN_FEATURES),),
      {},
  )


def _training_factory(ctx: context_lib.Context):
  del ctx  # Unused.
  model = nn.Linear(_IN_FEATURES, _OUT_FEATURES)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  return model, (torch.randn(_BATCH_SIZE, _IN_FEATURES),), {}, optimizer


def _failing_factory(ctx: context_lib.Context):
  del ctx  # Unused.
  raise RuntimeError("simulated factory failure")


def _spec(
    name: str,
    factory: registry_lib.Factory,
    stepper: step_lib.StepperType,
    skipped_run_modes: frozenset[str] = frozenset(),
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=factory,
      stepper=stepper,
      presubmit=registry_lib.PresubmitPolicy.RUN,
      dtype=target_lib.DType.FP32,
      skipped_run_modes=skipped_run_modes,
  )


_FORWARD_SPEC = _spec(
    "toy_forward", _forward_factory, step_lib.StepperType.FORWARD
)
_TRAINING_SPEC = _spec(
    "toy_training", _training_factory, step_lib.StepperType.TRAINING
)
_FAILING_SPEC = _spec(
    "toy_failing", _failing_factory, step_lib.StepperType.FORWARD
)
_SKIPPED_SPEC = _spec(
    "toy_skipped",
    _forward_factory,
    step_lib.StepperType.FORWARD,
    skipped_run_modes=frozenset({common.RunMode.EAGER_DEFAULT.value}),
)


def _cpu_context() -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(
          target_lib.Platform.CPU, dtype=target_lib.DType.FP32
      ),
      run_scope=context_lib.RunScope.PRESUBMIT,
  )


class MakeRunStepTest(seed_test_utils.RepeatableTest):

  def test_forward_step_runs_the_model(self):
    stepper, _ = torch_base_test.make_run_step(
        _FORWARD_SPEC, _cpu_context(), common.RunMode.EAGER_DEFAULT
    )

    out = stepper.get_step_fn()()

    self.assertEqual(out.shape, (_BATCH_SIZE, _OUT_FEATURES))
    self.assertFalse(out.requires_grad)

  def test_training_step_updates_params(self):
    stepper, _ = torch_base_test.make_run_step(
        _TRAINING_SPEC, _cpu_context(), common.RunMode.EAGER_DEFAULT
    )
    params = stepper.get_params()
    before = {name: p.detach().clone() for name, p in params.items()}

    loss = stepper.get_step_fn()()

    self.assertEqual(loss.ndim, 0)
    for name, p in params.items():
      self.assertFalse(torch.equal(before[name], p), name)

  def test_seeds_rngs_before_building(self):
    def build_weight():
      stepper, _ = torch_base_test.make_run_step(
          _TRAINING_SPEC, _cpu_context(), common.RunMode.EAGER_DEFAULT
      )
      return stepper.get_params()["weight"].detach().clone()

    first = build_weight()
    torch.manual_seed(1234)
    second = build_weight()

    self.assertTrue(torch.equal(first, second))


def _fake_test_class(
    failures: Sequence[discovery_lib.ImportFailure] = (),
) -> type[torch_base_test.BaseTorchBenchmarkTest]:
  """Returns a test class that runs the toy cases below on CPU.

  The class is built here rather than at module level so that the test loader
  does not run the `test_benchmark_imports` method it inherits.

  Args:
    failures: The import failures that the class reports.
  """

  class FakeTorchBenchmarkTest(torch_base_test.BaseTorchBenchmarkTest):
    framework = mode_lib.Framework.TORCH
    platform = target_lib.Platform.CPU
    import_failures = tuple(failures)

    def forward_case(self):
      self.run_case(_FORWARD_SPEC, common.RunMode.EAGER_DEFAULT)

    def failing_case(self):
      self.run_case(_FAILING_SPEC, common.RunMode.EAGER_DEFAULT)

    def skipped_case(self):
      self.run_case(_SKIPPED_SPEC, common.RunMode.EAGER_DEFAULT)

  return FakeTorchBenchmarkTest


class RunCaseTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(
            (flags_lib.RUN_IN_SUBPROCESS, False),
            (flags_lib.SKIP_BEHAVIOR, "skip"),
            (flags_lib.DRY_RUN, False),
            (flags_lib.COUNT_FLOPS, False),
            (measure_lib.MIN_WARMUP_STEPS, 1),
            (measure_lib.MAX_WARMUP_STEPS, 1),
            (measure_lib.POST_WARMUP_STEPS, 0),
        )
    )
    self.exported = []
    self.enter_context(
        mock.patch.object(export_lib, "export", new=self.exported.append)
    )
    self.mock_export_skip = self.enter_context(
        mock.patch.object(export_lib, "export_skip", autospec=True)
    )

  def _run(
      self,
      method_name: str,
      failures: Sequence[discovery_lib.ImportFailure] = (),
  ) -> unittest.TestResult:
    result = unittest.TestResult()  # UNITTEST_OK=result of a fake test case
    _fake_test_class(failures)(method_name).run(result)
    return result

  def test_measured_case_is_exported_as_succeeded(self):
    result = self._run("forward_case")

    self.assertEmpty(result.errors)
    self.assertEmpty(result.failures)
    self.assertEmpty(result.skipped)
    (data,) = self.exported
    self.assertEqual(data.spec_name, "toy_forward")
    self.assertEqual(data.platform, target_lib.Platform.CPU)
    self.assertEqual(data.framework, mode_lib.Framework.TORCH)
    self.assertEqual(data.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(data.succeeded)
    self.assertGreater(data.metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(data.metrics.first_step_time_seconds, 0.0)

  def test_failed_case_is_exported_as_failed(self):
    result = self._run("failing_case")

    self.assertLen(result.errors, 1)
    self.assertIn("simulated factory failure", result.errors[0][1])
    (data,) = self.exported
    self.assertEqual(data.spec_name, "toy_failing")
    self.assertFalse(data.succeeded)

  def test_skipped_case_is_not_run(self):
    result = self._run("skipped_case")

    self.assertLen(result.skipped, 1)
    self.assertEmpty(self.exported)
    self.mock_export_skip.assert_not_called()

  def test_dry_run_writes_the_test_method_name(self):
    output_dir = self.create_tempdir().full_path
    self.enter_context(
        mock.patch.dict(os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": output_dir})
    )

    with flagsaver.flagsaver((flags_lib.DRY_RUN, True)):
      result = self._run("forward_case")

    self.assertEmpty(result.errors)
    self.assertEmpty(self.exported)
    with open(os.path.join(output_dir, "test_targets.txt")) as f:
      self.assertEqual(f.read(), "forward_case\n")

  def test_import_failures_fail_the_imports_test(self):
    failure = discovery_lib.ImportFailure(
        module="perflab.models.broken", error=ImportError("no module")
    )

    result = self._run("test_benchmark_imports", failures=[failure])

    self.assertLen(result.failures, 1)
    self.assertIn("perflab.models.broken", result.failures[0][1])

  def test_imports_test_passes_without_import_failures(self):
    result = self._run("test_benchmark_imports")

    self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
  absltest.main()
