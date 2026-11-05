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

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from absl.testing import absltest
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab import workload as workload_lib
from tests import seed_test_utils

_RUN = registry_lib.PresubmitPolicy.RUN


def _factory() -> tuple[str, Sequence[Any], Mapping[str, Any], Any | None]:
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

    @registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD, presubmit=_RUN, workload=None
    )
    def llama_1b_inference():
      return ("m", (), {}, None)

    self.assertIn("llama_1b_inference", registry_lib.REGISTRY)

  def test_decorator_returns_factory_unchanged(self):
    """Standalone tests import factories directly, so decorator must preserve them."""

    @registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD, presubmit=_RUN, workload=None
    )
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
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD, presubmit=_RUN, workload=None
    )(_factory)
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      registry_lib.register_benchmark(
          stepper=step_lib.StepperType.FORWARD, presubmit=_RUN, workload=None
      )(_factory)

  def test_step_kwargs_not_aliased(self):
    """A shared mutable default would let one benchmark's kwargs leak into another."""
    kwargs = {"accum_steps": 4}
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.TRAINING,
        stepper_kwargs=kwargs,
        presubmit=_RUN,
        workload=None,
    )(_factory)
    kwargs["accum_steps"] = 999
    self.assertEqual(
        registry_lib.REGISTRY["_factory"].stepper_kwargs["accum_steps"], 4
    )

  def test_registered_specs_defaults(self):
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD, presubmit=_RUN, workload=None
    )(_factory)
    spec = registry_lib.REGISTRY["_factory"]
    self.assertEqual(spec.stepper_kwargs, {})
    self.assertIsNone(spec.compile_config)
    self.assertIsNone(spec.microbenchmark_name)

  def test_register_with_microbenchmark_name(self):
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD,
        presubmit=_RUN,
        workload=None,
        microbenchmark_name="batch_size_1",
    )(_factory)
    spec = registry_lib.REGISTRY["_factory"]
    self.assertEqual(spec.microbenchmark_name, "batch_size_1")

  def test_add_benchmark_with_microbenchmark_name(self):
    registry_lib.add_benchmark(
        factory=_factory,
        stepper=step_lib.StepperType.FORWARD,
        name="linear_batch_size_1",
        presubmit=_RUN,
        workload=None,
        microbenchmark_name="batch_size_1",
    )
    spec = registry_lib.REGISTRY["linear_batch_size_1"]
    self.assertEqual(spec.microbenchmark_name, "batch_size_1")

  def test_register_with_compile_config(self):
    cfg = compile_lib.CompileConfig(
        scope=compile_lib.Scope.STEP, dynamic=True, fullgraph=True
    )
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD,
        compile_config=cfg,
        presubmit=_RUN,
        workload=None,
    )(_factory)
    spec = registry_lib.REGISTRY["_factory"]
    self.assertEqual(spec.compile_config, cfg)

  def test_registered_entry_frozen(self):
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD, presubmit=_RUN, workload=None
    )(_factory)
    with self.assertRaises(  # ASSERT_RAISES_OK=benchmark
        dataclasses.FrozenInstanceError
    ):
      registry_lib.REGISTRY["_factory"].name = "other"

  def test_presubmit_policy_is_required(self):
    with self.assertRaisesRegex(TypeError, "presubmit"):
      registry_lib.register_benchmark(  # pytype: disable=missing-parameter
          stepper=step_lib.StepperType.FORWARD
      )
    with self.assertRaisesRegex(TypeError, "presubmit"):
      registry_lib.add_benchmark(  # pytype: disable=missing-parameter
          factory=_factory, stepper=step_lib.StepperType.FORWARD
      )
    self.assertEmpty(registry_lib.REGISTRY)

  def test_presubmit_policy_must_be_enum(self):
    with self.assertRaisesRegex(TypeError, "must be a PresubmitPolicy"):
      registry_lib.register_benchmark(
          stepper=step_lib.StepperType.FORWARD,
          presubmit="run",
          workload=None,  # pytype: disable=wrong-arg-types
      )(_factory)

  def test_registers_presubmit_policy_and_reason(self):
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD,
        presubmit=registry_lib.PresubmitPolicy.SKIP,
        workload=None,
        presubmit_reason="too big",
    )(_factory)
    spec = registry_lib.REGISTRY["_factory"]
    self.assertIs(spec.presubmit, registry_lib.PresubmitPolicy.SKIP)
    self.assertEqual(spec.presubmit_reason, "too big")

  def test_presubmit_reason_required_for_policies_that_exclude(self):
    for policy in (
        registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
        registry_lib.PresubmitPolicy.SKIP,
    ):
      for reason in (None, ""):
        with self.subTest(policy=policy, reason=reason):
          with self.assertRaisesRegex(
              ValueError, "requires a presubmit_reason"
          ):
            registry_lib.add_benchmark(
                factory=_factory,
                stepper=step_lib.StepperType.FORWARD,
                presubmit=policy,
                workload=None,
                presubmit_reason=reason,
            )
    self.assertEmpty(registry_lib.REGISTRY)

  def test_workload_is_required(self):
    with self.assertRaisesRegex(TypeError, "workload"):
      registry_lib.add_benchmark(  # pytype: disable=missing-parameter
          factory=_factory, stepper=step_lib.StepperType.FORWARD, presubmit=_RUN
      )
    self.assertEmpty(registry_lib.REGISTRY)

  def test_registers_workload(self):
    workload_fn = registry_lib.scoped_workload(
        full=workload_lib.Workload(16, 2048)
    )
    registry_lib.register_benchmark(
        stepper=step_lib.StepperType.FORWARD,
        presubmit=_RUN,
        workload=workload_fn,
    )(_factory)
    self.assertIs(registry_lib.REGISTRY["_factory"].workload, workload_fn)

  def test_presubmit_reason_rejected_for_run_policy(self):
    with self.assertRaisesRegex(ValueError, "presubmit_reason is only allowed"):
      registry_lib.add_benchmark(
          factory=_factory,
          stepper=step_lib.StepperType.FORWARD,
          presubmit=_RUN,
          workload=None,
          presubmit_reason="unused",
      )
    self.assertEmpty(registry_lib.REGISTRY)


class BatchSweepTest(seed_test_utils.RepeatableTest):

  def test_defaults_to_sweeping_at_the_flag_batch_sizes(self):
    batch_sweep = registry_lib.BatchSweep()
    self.assertTrue(batch_sweep.enabled)
    self.assertIsNone(batch_sweep.batch_sizes)
    self.assertIsNone(batch_sweep.reason)

  def test_disabled_requires_a_reason(self):
    self.assertEqual(
        registry_lib.BatchSweep.disabled("packed").reason, "packed"
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        ValueError, "reason"
    ):
      registry_lib.BatchSweep(enabled=False)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        ValueError, "reason"
    ):
      registry_lib.BatchSweep(reason="not disabled")

  def test_batch_sizes_must_be_positive(self):
    for batch_sizes in ((), (0, 2), (-1,)):
      with self.subTest(batch_sizes=batch_sizes):
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
            ValueError, "batch_sizes"
        ):
          registry_lib.BatchSweep(batch_sizes=batch_sizes)


def _ctx(
    run_scope: context_lib.RunScope = context_lib.RunScope.FULL,
) -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(target_lib.Platform.CPU),
      run_scope=run_scope,
  )


def _spec(
    workload: registry_lib.WorkloadFn | None,
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name="spec",
      factory=_factory,
      stepper=step_lib.StepperType.FORWARD,
      presubmit=_RUN,
      workload=workload,
  )


class WorkloadTest(seed_test_utils.RepeatableTest):

  def test_scoped_workload(self):
    workload_fn = registry_lib.scoped_workload(
        full=workload_lib.Workload(16, 2048),
        presubmit=workload_lib.Workload(1, 128),
    )
    self.assertEqual(
        workload_fn(_ctx(context_lib.RunScope.FULL)),
        workload_lib.Workload(16, 2048),
    )
    self.assertEqual(
        workload_fn(_ctx(context_lib.RunScope.PRESUBMIT)),
        workload_lib.Workload(1, 128),
    )

  def test_scoped_workload_defaults_presubmit_to_full(self):
    workload_fn = registry_lib.scoped_workload(full=workload_lib.Workload(4))
    self.assertEqual(
        workload_fn(_ctx(context_lib.RunScope.PRESUBMIT)),
        workload_lib.Workload(4),
    )

  def test_resolve_context_sets_the_default_workload(self):
    spec = _spec(
        registry_lib.scoped_workload(full=workload_lib.Workload(8, 64))
    )
    ctx = registry_lib.resolve_context(spec, _ctx())
    self.assertEqual(ctx.workload, workload_lib.Workload(8, 64))
    self.assertEqual(ctx.require_workload(), workload_lib.Workload(8, 64))

  def test_resolve_context_at_another_batch_size(self):
    spec = _spec(
        registry_lib.scoped_workload(full=workload_lib.Workload(8, 64))
    )
    ctx = registry_lib.resolve_context(spec, _ctx(), batch_size=32)
    self.assertEqual(ctx.workload, workload_lib.Workload(32, 64))

  def test_resolve_context_without_workload(self):
    spec = _spec(None)
    self.assertIsNone(registry_lib.resolve_context(spec, _ctx()).workload)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        ValueError, "declares no workload"
    ):
      registry_lib.resolve_context(spec, _ctx(), batch_size=4)

  def test_workload_may_depend_on_the_target(self):
    spec = _spec(
        lambda ctx: workload_lib.Workload(
            2 if ctx.device_kind is target_lib.DeviceKind.CUDA else 1, 2048
        )
    )
    gpu_ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.B200_4),
        run_scope=context_lib.RunScope.FULL,
    )
    self.assertEqual(
        registry_lib.resolve_workload(spec, gpu_ctx),
        workload_lib.Workload(2, 2048),
    )
    self.assertEqual(
        registry_lib.resolve_workload(spec, _ctx()),
        workload_lib.Workload(1, 2048),
    )


if __name__ == "__main__":
  absltest.main()
