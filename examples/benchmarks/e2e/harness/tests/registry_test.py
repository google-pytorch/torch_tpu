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

"""Tests for registry."""

import dataclasses
from typing import Any, Mapping, Sequence, Tuple

from absl.testing import absltest
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import step_lib
from tests import seed_test_utils


def _factory() -> Tuple[str, Sequence[Any], Mapping[str, Any], Any | None]:
  return ("model", (), {}, None)


class RegisterTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self._saved = dict(registry_lib.REGISTRY)
    registry_lib.REGISTRY.clear()

  def tearDown(self):
    super().tearDown()
    registry_lib.REGISTRY.clear()
    registry_lib.REGISTRY.update(self._saved)

  def test_register_with_function_name(self):

    @registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
    def llama_1b_inference():
      return ("m", (), {}, None)

    self.assertIn("llama_1b_inference", registry_lib.REGISTRY)

  def test_decorator_returns_factory_unchanged(self):
    """Standalone tests import factories directly, so decorator must preserve them."""

    @registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)
    def sample_benchmark():
      return ("dummy_model", ("input_args",), {"kw": "input_kwargs"}, None)

    self.assertTrue(callable(sample_benchmark))
    self.assertEqual(sample_benchmark()[0], "dummy_model")
    self.assertEqual(sample_benchmark()[1], ("input_args",))
    self.assertEqual(sample_benchmark()[2], {"kw": "input_kwargs"})
    self.assertIs(
        registry_lib.REGISTRY["sample_benchmark"].factory, sample_benchmark
    )

  def test_duplicate_name_raises(self):
    registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)(
        _factory
    )
    with self.assertRaises(ValueError):
      registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)(
          _factory
      )

  def test_step_kwargs_not_aliased(self):
    """A shared mutable default would let one benchmark's kwargs leak into another."""
    kwargs = {"accum_steps": 4}
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.TRAINING, stepper_kwargs=kwargs
    )(_factory)
    kwargs["accum_steps"] = 999
    self.assertEqual(
        registry_lib.REGISTRY["_factory"].stepper_kwargs["accum_steps"], 4
    )

  def test_registered_specs_defaults(self):
    registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)(
        _factory
    )
    spec = registry_lib.REGISTRY["_factory"]
    self.assertEqual(spec.stepper_kwargs, {})
    self.assertIsNone(spec.compile_config)

  def test_register_with_compile_config(self):
    cfg = compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=True, fullgraph=True
    )
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD, compile_config=cfg
    )(_factory)
    spec = registry_lib.REGISTRY["_factory"]
    self.assertEqual(spec.compile_config, cfg)

  def test_registered_entry_frozen(self):
    registry_lib.register_benchmark(stepper=step_lib.StepperType.FORWARD)(
        _factory
    )
    with self.assertRaises(dataclasses.FrozenInstanceError):
      registry_lib.REGISTRY["_factory"].name = "other"


if __name__ == "__main__":
  absltest.main()
