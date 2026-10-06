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

"""Unit tests for standalone distributed benchmark runner."""

import importlib.util
import os
import shutil
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.multiprocessing as mp
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from perflab import common
from perflab import discovery as discovery_lib
from perflab import distributed_runner
from perflab import export as export_lib
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_device_ops
from tests import seed_test_utils


def setUpModule():
  discovery_lib.import_submodules(steps)


class _FakeDistRun:
  """Stateful fake for `distributed_utils.dist_run`.

  Records the spec name of every dispatched target and simulates a worker rank
  raising (as surfaced by `torch.multiprocessing.spawn`) for specs listed in
  `failing_specs`.
  """

  def __init__(self, failing_specs: set[str]):
    self._failing_specs = failing_specs
    self.ran = []

  def __call__(self, nproc_per_node, fn, spec_name, mode_value, platform_name):
    del nproc_per_node, fn, mode_value, platform_name
    self.ran.append(spec_name)
    if spec_name in self._failing_specs:
      raise mp.ProcessRaisedException(
          f"RuntimeError: factory failed for {spec_name}",
          error_index=0,
          error_pid=1234,
      )


class DistributedRunnerTest(seed_test_utils.RepeatableTest):

  @mock.patch.dict(registry_lib.REGISTRY, {}, clear=True)
  def test_resolve_benchmark_targets_deterministic(self):
    spec1 = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    spec1.name = "llama_8b_inference"
    registry_lib.REGISTRY["llama_8b_inference"] = spec1

    # No filter returns all modes
    targets = distributed_runner.resolve_benchmark_targets(test_filter=None)
    self.assertLen(targets, len(common.RunMode))
    self.assertEqual(targets[0][0].name, "llama_8b_inference")

    # Exact target_name match
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter=(
            "llama_8b_inference_eager_default"
            " llama_8b_inference_eager_optimized"
        )
    )
    self.assertLen(targets, 2)
    self.assertEqual(targets[0][1], common.RunMode.EAGER_DEFAULT)
    self.assertEqual(targets[1][1], common.RunMode.EAGER_OPTIMIZED)

    # Duplicate targets preserved in order
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter=(
            "llama_8b_inference_eager_default llama_8b_inference_eager_default"
        )
    )
    self.assertLen(targets, 2)
    self.assertEqual(targets[0][1], common.RunMode.EAGER_DEFAULT)
    self.assertEqual(targets[1][1], common.RunMode.EAGER_DEFAULT)

    # pyformat: disable
    # Wildcard pattern throws ValueError (exact match only)
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="*llama_8b_inference*"
      )

    # Spec name without mode suffix throws ValueError
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="llama_8b_inference"
      )

    # Invalid target throws ValueError
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="invalid_target_xyz"
      )
    # pyformat: enable

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch_device_ops, "TorchDeviceOps")
  @mock.patch.object(measure_lib, "measure")
  @mock.patch.object(torch.distributed, "init_process_group")
  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_run_single_target_task_tpu(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
      mock_init_group,
      mock_measure,
      mock_device_ops,
      mock_make_target,
  ):
    del mock_is_initialized, mock_device_ops
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy"
    fake_spec.dtype = target_lib.DType.BF16
    fake_spec.factory = mock.MagicMock(
        return_value=(mock.MagicMock(), (), {}, None)
    )
    fake_spec.stepper = step_lib.StepperType.FORWARD
    fake_spec.stepper_kwargs = {}
    fake_spec.compile_config = None
    fake_spec.skipped_run_modes = set()

    registry_lib.REGISTRY["dummy"] = fake_spec

    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.TPU
    fake_target.platform = target_lib.Platform.V7_2X2X2
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    mock_metrics = mock.MagicMock(spec=metrics_lib.PerformanceMetrics)
    mock_metrics.e2e_wall_time_seconds = 1.0
    mock_metrics.first_step_time_seconds = 0.5
    mock_measure.return_value = mock_metrics

    with mock.patch.dict(os.environ, {"RANK": "0"}):
      distributed_runner._run_single_target_task(
          spec_name="dummy",
          mode_value="eager_default",
          platform_name="v7_2x2x2",
      )

    mock_init_group.assert_called_once_with(backend="tpu_dist")
    self.assertTrue(mock_barrier.called)
    mock_measure.assert_called_once()
    mock_compiler_reset.assert_called_once()
    mock_destroy_group.assert_called_once()

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch_device_ops, "TorchDeviceOps")
  @mock.patch.object(measure_lib, "measure")
  @mock.patch.object(torch.cuda, "set_device")
  @mock.patch.object(torch.distributed, "init_process_group")
  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_run_single_target_task_cuda(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
      mock_init_group,
      mock_set_device,
      mock_measure,
      mock_device_ops,
      mock_make_target,
  ):
    del mock_is_initialized, mock_device_ops
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy_gpu"
    fake_spec.dtype = target_lib.DType.BF16
    fake_spec.factory = mock.MagicMock(
        return_value=(mock.MagicMock(), (), {}, None)
    )
    fake_spec.stepper = step_lib.StepperType.FORWARD
    fake_spec.stepper_kwargs = {}
    fake_spec.compile_config = None
    fake_spec.skipped_run_modes = set()

    registry_lib.REGISTRY["dummy_gpu"] = fake_spec

    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.CUDA
    fake_target.platform = target_lib.Platform.B200_8
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    mock_metrics = mock.MagicMock(spec=metrics_lib.PerformanceMetrics)
    mock_metrics.e2e_wall_time_seconds = 2.0
    mock_metrics.first_step_time_seconds = 1.0
    mock_measure.return_value = mock_metrics

    with mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "3"}):
      distributed_runner._run_single_target_task(
          spec_name="dummy_gpu",
          mode_value="eager_default",
          platform_name="b200_8",
      )

    mock_set_device.assert_called_once_with(3)
    mock_init_group.assert_called_once_with(backend="nccl")
    self.assertTrue(mock_barrier.called)
    mock_measure.assert_called_once()
    mock_compiler_reset.assert_called_once()
    mock_destroy_group.assert_called_once()

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch.distributed, "init_process_group")
  def test_run_single_target_task_skips_unsupported_mode_on_cuda(
      self,
      mock_init_group,
      mock_make_target,
  ):
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy_gpu"
    fake_spec.dtype = target_lib.DType.BF16
    fake_spec.skipped_run_modes = set()
    registry_lib.REGISTRY["dummy_gpu"] = fake_spec

    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.CUDA
    fake_target.platform = target_lib.Platform.B200_8
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    distributed_runner._run_single_target_task(
        spec_name="dummy_gpu",
        mode_value="eager_optimized",
        platform_name="b200_8",
    )
    mock_init_group.assert_not_called()

  @mock.patch.dict(registry_lib.REGISTRY, {})
  @mock.patch.object(target_lib, "make_target")
  @mock.patch.object(torch_device_ops, "TorchDeviceOps")
  @mock.patch.object(
      measure_lib, "measure", side_effect=RuntimeError("Model OOM")
  )
  @mock.patch.object(torch.distributed, "init_process_group")
  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_run_single_target_task_exception_safety(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
      mock_init_group,
      mock_measure,
      mock_device_ops,
      mock_make_target,
  ):
    del mock_is_initialized, mock_init_group, mock_measure, mock_device_ops
    fake_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    fake_spec.name = "dummy"
    fake_spec.dtype = target_lib.DType.BF16
    fake_spec.factory = mock.MagicMock(
        return_value=(mock.MagicMock(), (), {}, None)
    )
    fake_spec.stepper = step_lib.StepperType.FORWARD
    fake_spec.stepper_kwargs = {}
    fake_spec.compile_config = None
    fake_spec.skipped_run_modes = set()

    registry_lib.REGISTRY["dummy"] = fake_spec

    fake_target = mock.MagicMock(spec=target_lib.Target)
    fake_target.device_kind = target_lib.DeviceKind.TPU
    fake_target.platform = target_lib.Platform.V7_2X2X2
    fake_target.dtype = target_lib.DType.BF16
    mock_make_target.return_value = fake_target

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Worker exception propagation and cleanup test.
        RuntimeError, "Model OOM"
    ):
      distributed_runner._run_single_target_task(
          spec_name="dummy",
          mode_value="eager_default",
          platform_name="v7_2x2x2",
      )

    mock_compiler_reset.assert_called_once()
    mock_destroy_group.assert_called_once()
    self.assertEqual(mock_barrier.call_count, 1)

  @mock.patch.dict(os.environ, {"BENCHMARK_PLATFORM": "v7_2x2x2"})
  @mock.patch("torch.tpu.distributed.set_tpu_launch_env")
  @mock.patch.object(distributed_runner.distributed_utils, "dist_run")
  @mock.patch.object(distributed_runner, "resolve_benchmark_targets")
  def test_main_successful_dispatch(
      self, mock_resolve_targets, mock_dist_run, mock_set_tpu_launch_env
  ):
    dummy_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    dummy_spec.name = "dummy"
    dummy_spec.skipped_run_modes = set()
    mock_resolve_targets.return_value = [
        (dummy_spec, common.RunMode.EAGER_DEFAULT)
    ]

    with mock.patch.dict(
        os.environ, {"TESTBRIDGE_TEST_ONLY": "dummy_eager_default"}
    ):
      distributed_runner.main(["distributed_runner"])

    mock_set_tpu_launch_env.assert_not_called()
    mock_dist_run.assert_called_once_with(
        8,  # V7_2X2X2 has 8 procs per node
        distributed_runner._run_single_target_task,
        "dummy",
        "eager_default",
        "v7_2x2x2",
    )

  @mock.patch.dict(os.environ, {"BENCHMARK_PLATFORM": "v7_2x2x1"})
  @mock.patch("torch.tpu.distributed.set_tpu_launch_env")
  @mock.patch.object(distributed_runner.distributed_utils, "dist_run")
  @mock.patch.object(distributed_runner, "resolve_benchmark_targets")
  def test_main_singlehost_tpu_prepares_environment(
      self, mock_resolve_targets, mock_dist_run, mock_set_tpu_launch_env
  ):
    dummy_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    dummy_spec.name = "dummy"
    dummy_spec.skipped_run_modes = set()
    mock_resolve_targets.return_value = [
        (dummy_spec, common.RunMode.EAGER_DEFAULT)
    ]

    with mock.patch.dict(
        os.environ, {"TESTBRIDGE_TEST_ONLY": "dummy_eager_default"}
    ):
      distributed_runner.main(["distributed_runner"])

    mock_set_tpu_launch_env.assert_called_once_with(nproc_per_node=8)
    mock_dist_run.assert_called_once_with(
        8,
        distributed_runner._run_single_target_task,
        "dummy",
        "eager_default",
        "v7_2x2x1",
    )

  def test_module_imports_without_torch_tpu_module(self):
    # Re-execute the module body as a GPU host with
    # TORCH_DEVICE_BACKEND_AUTOLOAD=0 would, where `torch.tpu` is not
    # registered. A module-level `torch.tpu` import would raise
    # ModuleNotFoundError here.
    spec = importlib.util.find_spec(distributed_runner.__name__)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"torch.tpu": None, "torch.tpu.distributed": None}
    ):
      spec.loader.exec_module(module)

  @mock.patch.dict(os.environ, {"BENCHMARK_PLATFORM": "b200_8"})
  @mock.patch.object(distributed_runner.distributed_utils, "dist_run")
  @mock.patch.object(distributed_runner, "resolve_benchmark_targets")
  def test_main_gpu_does_not_require_torch_tpu_module(
      self, mock_resolve_targets, mock_dist_run
  ):
    dummy_spec = mock.MagicMock(spec=registry_lib.BenchmarkSpec)
    dummy_spec.name = "dummy"
    dummy_spec.skipped_run_modes = set()
    mock_resolve_targets.return_value = [
        (dummy_spec, common.RunMode.EAGER_DEFAULT)
    ]

    # Simulate a GPU host with TORCH_DEVICE_BACKEND_AUTOLOAD=0, where
    # `torch.tpu` is not registered: any import of it raises ImportError.
    with mock.patch.dict(
        sys.modules, {"torch.tpu": None, "torch.tpu.distributed": None}
    ), mock.patch.dict(
        os.environ, {"TESTBRIDGE_TEST_ONLY": "dummy_eager_default"}
    ):
      distributed_runner.main(["distributed_runner"])

    mock_dist_run.assert_called_once_with(
        8,
        distributed_runner._run_single_target_task,
        "dummy",
        "eager_default",
        "b200_8",
    )

  def test_main_isolates_failed_target_and_exits_nonzero(self):
    fake_dist_run = _FakeDistRun(failing_specs={"broken"})
    exported = []
    registry = {
        "broken": types.SimpleNamespace(
            name="broken", skipped_run_modes=frozenset()
        ),
        "healthy": types.SimpleNamespace(
            name="healthy", skipped_run_modes=frozenset()
        ),
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "broken_eager_default healthy_eager_default"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        mock.patch.object(export_lib, "export", exported.append),
        self.assertLogs(level="INFO") as logs,
        self.assertRaises(
            SystemExit
        ) as exit_cm,  # ASSERT_RAISES_OK=Runner must exit non-zero when any target fails.
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertNotEqual(exit_cm.exception.code, 0)
    # The healthy target still runs after the broken one fails.
    self.assertEqual(fake_dist_run.ran, ["broken", "healthy"])
    # The failed target is reported once as FAILED to the export path.
    self.assertEqual(
        [(d.spec_name, d.run_mode, d.succeeded) for d in exported],
        [("broken", common.RunMode.EAGER_DEFAULT, False)],
    )
    summary = "\n".join(logs.output)
    self.assertRegex(summary, r"broken_eager_default\s+FAILED")
    self.assertRegex(summary, r"healthy_eager_default\s+PASSED")
    self.assertIn("1 passed, 1 failed, 0 skipped", summary)

  def test_main_reports_unsupported_mode_as_skipped(self):
    fake_dist_run = _FakeDistRun(failing_specs=set())
    registry = {
        "healthy": types.SimpleNamespace(
            name="healthy", skipped_run_modes=frozenset()
        ),
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                # eager_optimized is TPU-only, so it is skipped on CUDA.
                "TESTBRIDGE_TEST_ONLY": (
                    "healthy_eager_optimized healthy_eager_default"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(fake_dist_run.ran, ["healthy"])
    summary = "\n".join(logs.output)
    self.assertRegex(summary, r"healthy_eager_optimized\s+SKIPPED")
    self.assertIn("1 passed, 0 failed, 1 skipped", summary)

  @mock.patch.object(torch.distributed, "barrier")
  @mock.patch.object(torch.distributed, "destroy_process_group")
  @mock.patch.object(torch.distributed, "is_initialized", return_value=True)
  @mock.patch.object(torch.compiler, "reset")
  def test_cleanup_distributed_worker_state_skip_barrier_on_exception(
      self,
      mock_compiler_reset,
      mock_is_initialized,
      mock_destroy_group,
      mock_barrier,
  ):
    del mock_compiler_reset, mock_is_initialized
    distributed_runner._cleanup_distributed_worker_state(
        rank=0, barrier_policy=distributed_runner.BarrierPolicy.SKIP
    )

    mock_barrier.assert_not_called()
    mock_destroy_group.assert_called_once()

  @parameterized.named_parameters(
      ("tpu", target_lib.Platform.V7_2X2X1, 1),
      ("cuda", target_lib.Platform.B200_8, 0),
      ("cpu", target_lib.Platform.CPU, 0),
  )
  def test_cleanup_distributed_worker_state_resets_torch_tpu_only_on_tpu(
      self, platform, expected_tpu_cleanups
  ):
    eager_state_resets = []
    tpu_cache_clears = []
    with (
        mock.patch.object(
            tt_testing,
            "reset_eager_state",
            lambda: eager_state_resets.append(True),
        ),
        mock.patch.object(
            torch.tpu, "_clear_cache", lambda: tpu_cache_clears.append(True)
        ),
    ):
      distributed_runner._cleanup_distributed_worker_state(
          rank=0,
          target=target_lib.make_target(platform),
          barrier_policy=distributed_runner.BarrierPolicy.SKIP,
      )

    self.assertLen(eager_state_resets, expected_tpu_cleanups)
    self.assertLen(tpu_cache_clears, expected_tpu_cleanups)

  @mock.patch.object(shutil, "rmtree")
  def test_clear_persistent_compilation_caches(self, mock_rmtree):
    with mock.patch("glob.glob", return_value=["/tmp/torchinductor_test"]):
      distributed_runner._clear_persistent_compilation_caches()

    mock_rmtree.assert_any_call("/dev/shm/torch_tpu_cache", ignore_errors=True)
    mock_rmtree.assert_any_call("/tmp/torchinductor_test", ignore_errors=True)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
