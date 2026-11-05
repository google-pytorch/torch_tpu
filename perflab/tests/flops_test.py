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

"""Tests for perflab/flops.py and the FLOPs metrics it feeds."""

from absl.testing import absltest
import torch
from torch._subclasses import fake_tensor
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import flops as flops_lib
from perflab import metrics as metrics_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_device_ops
from perflab import workload as workload_lib
from tests import seed_test_utils

# Registers the steppers that the benchmark specs below resolve.
discovery_lib.import_submodules(steps)

_IN = 256
_HIDDEN = 512
_OUT = 128
_FULL_BATCH = 64
_PRESUBMIT_BATCH = 2


def _batch_size(ctx: context_lib.Context) -> int:
  if ctx.workload is not None:
    return ctx.workload.batch_size
  if ctx.run_scope is context_lib.RunScope.PRESUBMIT:
    return _PRESUBMIT_BATCH
  return _FULL_BATCH


def _build_mlp(ctx: context_lib.Context) -> tuple[torch.nn.Module, tuple]:
  dtype = torch_device_ops.get_torch_dtype(ctx.dtype)
  device = ctx.device_kind.value
  model = torch.nn.Sequential(
      torch.nn.Linear(_IN, _HIDDEN, bias=False, dtype=dtype, device=device),
      torch.nn.ReLU(),
      torch.nn.Linear(_HIDDEN, _OUT, bias=False, dtype=dtype, device=device),
  )
  x = torch.randn(_batch_size(ctx), _IN, dtype=dtype, device=device)
  return model, (x,)


def _mlp_forward(ctx: context_lib.Context):
  model, args = _build_mlp(ctx)
  return model.eval(), args, {}


def _mlp_training(ctx: context_lib.Context):
  model, args = _build_mlp(ctx)
  opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
  return model.train(), args, {}, opt


def _spec(
    factory: registry_lib.Factory, stepper: step_lib.StepperType
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=factory.__name__,
      factory=factory,
      stepper=stepper,
      presubmit=registry_lib.PresubmitPolicy.RUN,
  )


def _forward_flops(batch: int) -> int:
  return 2 * batch * (_IN * _HIDDEN + _HIDDEN * _OUT)


class CountStepFlopsTest(seed_test_utils.RepeatableTest):

  def test_counts_forward_matmuls(self):
    flop_count = flops_lib.count_step_flops(
        _spec(_mlp_forward, step_lib.StepperType.FORWARD),
        context_lib.RunScope.FULL,
    )
    self.assertIsNotNone(flop_count)
    self.assertEqual(flop_count.total, _forward_flops(_FULL_BATCH))
    self.assertEqual(flop_count.by_op, {"aten.mm": _forward_flops(_FULL_BATCH)})

  def test_counts_backward_matmuls_of_training(self):
    flop_count = flops_lib.count_step_flops(
        _spec(_mlp_training, step_lib.StepperType.TRAINING),
        context_lib.RunScope.FULL,
    )
    self.assertIsNotNone(flop_count)
    # The input does not require grad, so the first layer computes only the
    # gradient of its weight: 2x its forward FLOPs, not 3x like the second.
    b = _FULL_BATCH
    expected = 3 * 2 * b * _IN * _HIDDEN - 2 * b * _IN * _HIDDEN
    expected += 3 * 2 * b * _HIDDEN * _OUT
    self.assertEqual(flop_count.total, expected)

  def test_builds_at_the_given_run_scope(self):
    flop_count = flops_lib.count_step_flops(
        _spec(_mlp_forward, step_lib.StepperType.FORWARD),
        context_lib.RunScope.PRESUBMIT,
    )
    self.assertIsNotNone(flop_count)
    self.assertEqual(flop_count.total, _forward_flops(_PRESUBMIT_BATCH))

  def test_counts_at_the_workload_of_the_measured_case(self):
    flop_count = flops_lib.count_step_flops(
        _spec(_mlp_forward, step_lib.StepperType.FORWARD),
        context_lib.RunScope.FULL,
        workload=workload_lib.Workload(8),
    )
    self.assertIsNotNone(flop_count)
    self.assertEqual(flop_count.total, _forward_flops(8))

  def test_builds_a_fake_cpu_copy(self):
    seen = {}

    def factory(ctx: context_lib.Context):
      seen["device_kind"] = ctx.device_kind
      model, args, kwargs = _mlp_forward(ctx)
      seen["is_fake"] = isinstance(args[0], fake_tensor.FakeTensor)
      return model, args, kwargs

    flops_lib.count_step_flops(
        _spec(factory, step_lib.StepperType.FORWARD),
        context_lib.RunScope.FULL,
    )
    self.assertEqual(seen["device_kind"], target_lib.DeviceKind.CPU)
    self.assertTrue(seen["is_fake"])

  def test_returns_none_if_the_benchmark_fails_to_build(self):
    def factory(ctx: context_lib.Context):
      del ctx
      raise RuntimeError("cannot build")

    self.assertIsNone(
        flops_lib.count_step_flops(
            _spec(factory, step_lib.StepperType.FORWARD),
            context_lib.RunScope.FULL,
        )
    )


class FlopsMetricsTest(seed_test_utils.RepeatableTest):

  def test_exports_achieved_tflops_and_mfu(self):
    metrics = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.5,
        average_post_warmup_device_time_seconds=0.25,
        model_flops_per_step=500e12,
        peak_flops_per_second=2000e12,
    )
    metric_map = metrics.metric_map()
    self.assertEqual(metric_map["model_flops_per_step"], 500e12)
    self.assertAlmostEqual(metric_map["achieved_tflops_per_second"], 1000.0)
    self.assertAlmostEqual(metric_map["mfu"], 0.5)
    self.assertAlmostEqual(
        metric_map["device_achieved_tflops_per_second"], 2000.0
    )
    self.assertAlmostEqual(metric_map["device_mfu"], 1.0)

  def test_exports_no_mfu_without_peak(self):
    metric_map = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.5,
        model_flops_per_step=500e12,
    ).metric_map()
    self.assertIn("achieved_tflops_per_second", metric_map)
    self.assertNotIn("mfu", metric_map)
    self.assertNotIn("device_achieved_tflops_per_second", metric_map)

  def test_exports_no_flops_metrics_if_not_counted(self):
    metric_map = metrics_lib.PerformanceMetrics(
        post_warmup_step_time_seconds=0.5,
        peak_flops_per_second=2000e12,
    ).metric_map()
    self.assertNotIn("model_flops_per_step", metric_map)
    self.assertNotIn("mfu", metric_map)


class CheckUtilizationTest(seed_test_utils.RepeatableTest):

  def test_plausible_metrics_have_no_problems(self):
    self.assertEmpty(
        flops_lib.check_utilization(
            "case",
            {"model_flops_per_step": 1e12, "mfu": 0.4, "device_mfu": 0.9},
        )
    )

  def test_uncounted_metrics_have_no_problems(self):
    self.assertEmpty(
        flops_lib.check_utilization("case", {"post_warmup_step_time_ms": 1.0})
    )

  def test_flags_mfu_above_one(self):
    problems = flops_lib.check_utilization(
        "case", {"model_flops_per_step": 1e12, "mfu": 0.5, "device_mfu": 1.2}
    )
    self.assertLen(problems, 1)
    self.assertIn("device_mfu", problems[0])

  def test_flags_zero_flops(self):
    problems = flops_lib.check_utilization(
        "case", {"model_flops_per_step": 0.0, "mfu": 0.0}
    )
    self.assertLen(problems, 1)
    self.assertIn("0 model FLOPs", problems[0])


class PeakFlopsTest(seed_test_utils.RepeatableTest):

  def test_v7_device_is_one_tensorcore(self):
    self.assertEqual(
        target_lib.peak_flops_per_device(
            target_lib.make_target(target_lib.Platform.V7_1X1X1)
        ),
        2307e12 / 2,
    )

  def test_b200(self):
    self.assertEqual(
        target_lib.peak_flops_per_device(
            target_lib.make_target(target_lib.Platform.B200_1)
        ),
        2250e12,
    )

  def test_unknown_for_cpu_and_fp32(self):
    self.assertIsNone(
        target_lib.peak_flops_per_device(
            target_lib.make_target(target_lib.Platform.CPU)
        )
    )
    self.assertIsNone(
        target_lib.peak_flops_per_device(
            target_lib.make_target(
                target_lib.Platform.V7_1X1X1, dtype=target_lib.DType.FP32
            )
        )
    )


if __name__ == "__main__":
  absltest.main()
