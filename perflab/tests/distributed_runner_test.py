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

from collections.abc import Iterator
from collections.abc import Set
import contextlib
import importlib.util
import os
import sys
import types
from typing import Any
from unittest import mock
from xml.etree import ElementTree

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import torch
import torch.multiprocessing as mp
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from perflab import common
from perflab import discovery as discovery_lib
from perflab import distributed_runner
from perflab import export as export_lib
from perflab import known_failures
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import steps
from perflab import target as target_lib
from perflab import workload as workload_lib
from perflab.tests import fakes
from tests import seed_test_utils


def setUpModule():
  discovery_lib.import_submodules(steps)


@contextlib.contextmanager
def _single_rank_gloo_env(rank: int = 0) -> Iterator[None]:
  """Configures single-process Gloo rendezvous environment variables."""
  port = portpicker.pick_unused_port()
  with mock.patch.dict(
      os.environ,
      {
          "MASTER_ADDR": "127.0.0.1",
          "MASTER_PORT": str(port),
          "RANK": str(rank),
          "LOCAL_RANK": "0",
          "WORLD_SIZE": "1",
      },
  ):
    try:
      yield
    finally:
      if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _in_process_dist_run(nprocs: int, fn: Any, *args: Any) -> None:
  """In-process stand-in for `dist_run` that executes rank 0 over Gloo."""
  assert nprocs == 1
  with _single_rank_gloo_env(rank=0):
    fn(*args)


class _FakeDistRun:
  """Stateful fake for `distributed_utils.dist_run`.

  Records the spec name of every dispatched target, simulates rank 0 writing
  `METRICS` once every rank has measured for specs listed in `measured_specs`,
  simulates a worker rank raising (as surfaced by `torch.multiprocessing.spawn`)
  for specs listed in `failing_specs`, and simulates a factory raising
  `UnsupportedBenchmark` for specs listed in `unsupported_specs`. A spec in both
  `measured_specs` and `failing_specs` fails after measuring, e.g. in teardown.
  """

  METRICS = metrics_lib.PerformanceMetrics(
      e2e_wall_time_seconds=1.0, first_step_time_seconds=0.5
  )

  def __init__(
      self,
      failing_specs: Set[str],
      unsupported_specs: Set[str] = frozenset(),
      measured_specs: Set[str] = frozenset(),
  ):
    self._failing_specs = failing_specs
    self._unsupported_specs = unsupported_specs
    self._measured_specs = measured_specs
    self.ran = []

  def __call__(
      self,
      nproc_per_node,
      fn,
      spec_name,
      mode_value,
      platform_name,
      skip_reason_file,
      metrics_file,
  ):
    del nproc_per_node, fn, mode_value, platform_name
    self.ran.append(spec_name)
    if spec_name in self._measured_specs:
      distributed_runner._write_metrics(metrics_file, self.METRICS)
    if spec_name in self._unsupported_specs:
      # What rank 0 does when the factory raises UnsupportedBenchmark.
      with open(skip_reason_file, "w") as f:
        f.write(f"{spec_name} needs more chips")
    if spec_name in self._failing_specs:
      raise mp.ProcessRaisedException(
          f"RuntimeError: factory failed for {spec_name}",
          error_index=0,
          error_pid=1234,
      )


def _read_test_xml(path: str) -> dict[str, tuple[str, str]]:
  """Returns {test case name: (status, message)} from a JUnit XML report."""
  cases = {}
  for case in ElementTree.parse(path).getroot().iter("testcase"):
    status, message = "PASSED", ""
    for tag in ("failure", "skipped"):
      if (child := case.find(tag)) is not None:
        status, message = tag.upper(), child.get("message", "")
    cases[case.get("name")] = (status, message)
  return cases


class DistributedRunnerTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # Under blaze, XML_OUTPUT_FILE is this test's own report; main() must not
    # overwrite it.
    self.xml_path = os.path.join(self.create_tempdir().full_path, "test.xml")
    self.enter_context(
        mock.patch.dict(os.environ, {"XML_OUTPUT_FILE": self.xml_path})
    )
    self.exporter = fakes.RecordingExporter()
    self.enter_context(mock.patch.object(export_lib, "export", self.exporter))
    self.enter_context(mock.patch.dict(registry_lib.REGISTRY, {}, clear=True))

  def test_resolve_benchmark_targets_deterministic(self):
    registry_lib.REGISTRY["llama_8b_inference"] = fakes.tiny_spec(
        "llama_8b_inference"
    )

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

  def test_resolve_benchmark_targets_wildcards(self):
    # Registered out of name order to check that targets are sorted by name
    for name in ("llama_8b_inference", "gemma_2b_training"):
      registry_lib.REGISTRY[name] = fakes.tiny_spec(name)

    # Targets are in token order, and a wildcard's matches in name order
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter="llama_8b_inference_compiled *_eager_default"
    )
    self.assertEqual(
        [(spec.name, mode) for spec, mode in targets],
        [
            ("llama_8b_inference", common.RunMode.COMPILED),
            ("gemma_2b_training", common.RunMode.EAGER_DEFAULT),
            ("llama_8b_inference", common.RunMode.EAGER_DEFAULT),
        ],
    )

    # A wildcard can match every mode of a benchmark, in run mode order
    targets = distributed_runner.resolve_benchmark_targets(
        test_filter="gemma_2b_training_*"
    )
    self.assertEqual(
        [(spec.name, mode) for spec, mode in targets],
        [("gemma_2b_training", mode) for mode in common.RunMode],
    )

    # pyformat: disable
    # Wildcard pattern matching no target throws ValueError
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=Runner target resolution validation.
      distributed_runner.resolve_benchmark_targets(
          test_filter="*_no_such_mode"
      )
    # pyformat: enable

  def test_backend_by_device_kind(self):
    self.assertEqual(
        distributed_runner._BACKEND_BY_DEVICE_KIND,
        {
            target_lib.DeviceKind.CPU: "gloo",
            target_lib.DeviceKind.CUDA: "nccl",
            target_lib.DeviceKind.TPU: "tpu_dist",
        },
    )

  def test_run_single_target_task_end_to_end_writes_metrics(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    metrics_file = os.path.join(self.create_tempdir().full_path, "metrics")

    with _single_rank_gloo_env():
      distributed_runner._run_single_target_task(
          spec_name="tiny_linear",
          mode_value="eager_default",
          platform_name="cpu",
          metrics_file=metrics_file,
      )

    self.assertFalse(torch.distributed.is_initialized())
    # The launcher in main() exports the metrics, not the worker.
    self.assertEmpty(self.exporter.exported)
    metrics = distributed_runner._read_metrics(metrics_file)
    self.assertIsNotNone(metrics)
    self.assertGreater(metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(metrics.first_step_time_seconds, 0.0)
    self.assertGreater(metrics.post_warmup_step_time_seconds, 0.0)

  @parameterized.named_parameters(
      # The 4 ranks of a tensor-parallel replica all process its 2 x 128 tokens,
      # so each device exports 1/4 of them.
      (
          "tensor_parallel",
          {"devices_per_replica": lambda world_size: world_size},
          2 * 128 / 4,
      ),
      # By default each rank is a replica that processes its own batch, e.g.
      # with data parallelism (FSDP).
      ("one_device_per_replica", {}, 2 * 128),
  )
  def test_run_single_target_task_exports_per_device_throughput(
      self, spec_kwargs, expected_tokens_per_step
  ):
    registry_lib.REGISTRY["tiny_tp"] = fakes.tiny_spec(
        "tiny_tp",
        workload=lambda ctx: workload_lib.Workload(2, 128),
        **spec_kwargs,
    )
    metrics_file = os.path.join(self.create_tempdir().full_path, "metrics")

    with (
        _single_rank_gloo_env(),
        # The single Gloo rank stands in for one of 4 ranks.
        mock.patch.object(
            torch.distributed, "get_world_size", lambda group=None: 4
        ),
    ):
      distributed_runner._run_single_target_task(
          spec_name="tiny_tp",
          mode_value="eager_default",
          platform_name="cpu",
          metrics_file=metrics_file,
      )

    metrics = distributed_runner._read_metrics(metrics_file)
    self.assertIsNotNone(metrics)
    metric_map = metrics.metric_map()
    self.assertEqual(metric_map["batch_size"], 2.0)
    self.assertEqual(metric_map["seq_len"], 128.0)
    self.assertEqual(metric_map["tokens_per_step"], expected_tokens_per_step)
    self.assertAlmostEqual(
        metric_map["tokens_per_second"],
        expected_tokens_per_step / metrics.throughput_step_time_seconds,
    )

  @parameterized.named_parameters(
      ("not_dividing_world_size", 3),
      ("zero", 0),
  )
  def test_run_single_target_task_rejects_invalid_devices_per_replica(
      self, devices_per_replica
  ):
    registry_lib.REGISTRY["tiny_tp"] = fakes.tiny_spec(
        "tiny_tp",
        workload=lambda ctx: workload_lib.Workload(2, 128),
        devices_per_replica=lambda world_size: devices_per_replica,
    )

    with (
        _single_rank_gloo_env(),
        self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
            ValueError,
            f"{devices_per_replica} devices per replica, which must be >= 1",
        ),
    ):
      distributed_runner._run_single_target_task(
          spec_name="tiny_tp",
          mode_value="eager_default",
          platform_name="cpu",
      )

    self.assertFalse(torch.distributed.is_initialized())

  def test_run_single_target_task_skips_unsupported_mode(self):
    registry_lib.REGISTRY["tiny_gpu"] = fakes.tiny_spec("tiny_gpu")

    distributed_runner._run_single_target_task(
        spec_name="tiny_gpu",
        mode_value="eager_optimized",
        platform_name="b200_8",
    )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertEmpty(self.exporter.exported)

  def test_run_single_target_task_rejects_zero_metrics_on_rank_0(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    frozen_clock = fakes.FakeClock(start=10.0)
    orig_measure = measure_lib.measure
    metrics_file = os.path.join(self.create_tempdir().full_path, "metrics")

    with (
        _single_rank_gloo_env(),
        mock.patch.object(
            measure_lib,
            "measure",
            lambda *args, **kwargs: orig_measure(
                *args, clock=frozen_clock, **kwargs
            ),
        ),
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Rank 0 zero-metric guard.
          AssertionError, "Invalid metrics measured"
      ):
        distributed_runner._run_single_target_task(
            spec_name="tiny_linear",
            mode_value="eager_default",
            platform_name="cpu",
            metrics_file=metrics_file,
        )

    self.assertFalse(torch.distributed.is_initialized())
    # The launcher in main() exports the failure, not the worker.
    self.assertEmpty(self.exporter.exported)
    self.assertIsNone(distributed_runner._read_metrics(metrics_file))

  def test_run_single_target_task_exception_propagates_and_cleans_up(self):
    def failing_factory(unused_ctx):
      raise RuntimeError("Model OOM")

    registry_lib.REGISTRY["oom_spec"] = fakes.tiny_spec(
        "oom_spec", factory=failing_factory
    )

    with _single_rank_gloo_env():
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Worker exception propagation and cleanup test.
          RuntimeError, "Model OOM"
      ):
        distributed_runner._run_single_target_task(
            spec_name="oom_spec",
            mode_value="eager_default",
            platform_name="cpu",
        )

    self.assertFalse(torch.distributed.is_initialized())
    # The launcher in main() exports the failure, not the worker.
    self.assertEmpty(self.exporter.exported)

  def test_run_single_target_task_reports_unsupported_benchmark(self):
    def unsupported_factory(unused_ctx):
      raise target_lib.UnsupportedBenchmark("needs more chips")

    registry_lib.REGISTRY["too_big"] = fakes.tiny_spec(
        "too_big", factory=unsupported_factory
    )
    skip_reason_file = os.path.join(self.create_tempdir().full_path, "skip.txt")

    with _single_rank_gloo_env():
      distributed_runner._run_single_target_task(
          spec_name="too_big",
          mode_value="eager_default",
          platform_name="cpu",
          skip_reason_file=skip_reason_file,
      )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertEmpty(self.exporter.exported)
    with open(skip_reason_file) as f:
      self.assertEqual(f.read(), "needs more chips")

  def test_main_runs_filtered_target_end_to_end(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")

    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "cpu",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=_in_process_dist_run,
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertLen(self.exporter.exported, 1)
    record = self.exporter.exported[0]
    self.assertEqual(record.spec_name, "tiny_linear")
    self.assertEqual(record.platform, target_lib.Platform.CPU)
    self.assertEqual(record.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(record.succeeded)
    self.assertGreater(record.metrics.e2e_wall_time_seconds, 0.0)

  @mock.patch.object(mp, "spawn")
  def test_main_multihost_tpu_with_pinned_env_and_no_worker_hostnames(
      self, mock_mp_spawn
  ):
    registry_lib.REGISTRY["dummy"] = fakes.tiny_spec("dummy")
    pinned_env = {
        "BENCHMARK_PLATFORM": "v7_2x2x2",
        "TESTBRIDGE_TEST_ONLY": "dummy_eager_default",
        "JOB_COMPLETION_INDEX": "0",
        "NNODES": "2",
        "NODE_RANK": "0",
        "WORLD_SIZE": "16",
        "MASTER_ADDR": "host0",
        "MASTER_PORT": "29500",
        "TORCH_TPU_TOPOLOGY": "2,2,2,2",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": "host0:8070,host1:8070",
    }

    with (
        mock.patch.dict(os.environ, pinned_env, clear=True),
        mock.patch.object(
            distributed_runner.distributed_utils.env,
            "get_str_env_once",
            side_effect=lambda name, default_value=None: os.environ.get(
                name, default_value
            ),
        ),
        mock.patch.object(
            distributed_runner.distributed_utils.env,
            "get_int_env_once",
            side_effect=lambda name, default_value=None: (
                int(os.environ[name]) if name in os.environ else default_value
            ),
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    mock_mp_spawn.assert_called_once()
    self.assertEmpty(self.exporter.exported)

  def test_main_tpu_launch_env_singlehost_vs_multihost(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    dispatched: list[tuple[int, str, str, str]] = []

    def record_dispatch(
        nprocs,
        unused_fn,
        spec_name,
        mode_val,
        platform_val,
        unused_skip_file,
        unused_metrics_file,
    ):
      dispatched.append((nprocs, spec_name, mode_val, platform_val))

    # Single-host TPU (nnodes=1, nprocs_per_node=8) populates TPU launch env.
    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "v7_2x2x1",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
                "TORCH_TPU_TOPOLOGY": "2x2x1",
            },
            clear=False,
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=record_dispatch,
        ),
    ):
      os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
      distributed_runner.main(["distributed_runner"])
      slicebuilder_addrs = os.environ.get(
          "TORCH_TPU_SLICEBUILDER_ADDRESSES", ""
      )
      self.assertLen(slicebuilder_addrs.split(","), 8)

    # Multi-host TPU (nnodes=2) does not override single-host TPU launch env.
    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "v7_2x2x2",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
                "TORCH_TPU_TOPOLOGY": "2x2x2",
            },
            clear=False,
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=record_dispatch,
        ),
    ):
      os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
      distributed_runner.main(["distributed_runner"])
      self.assertNotIn("TORCH_TPU_SLICEBUILDER_ADDRESSES", os.environ)

    self.assertEqual(
        dispatched,
        [
            (8, "tiny_linear", "eager_default", "v7_2x2x1"),
            (8, "tiny_linear", "eager_default", "v7_2x2x2"),
        ],
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

  def test_main_gpu_does_not_require_torch_tpu_module(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    dispatched: list[tuple[int, str, str, str]] = []

    def record_dispatch(
        nprocs,
        unused_fn,
        spec_name,
        mode_val,
        platform_val,
        unused_skip_file,
        unused_metrics_file,
    ):
      dispatched.append((nprocs, spec_name, mode_val, platform_val))

    # Simulate a GPU host with TORCH_DEVICE_BACKEND_AUTOLOAD=0, where
    # `torch.tpu` is not registered: any import of it raises ImportError.
    with (
        mock.patch.dict(
            sys.modules, {"torch.tpu": None, "torch.tpu.distributed": None}
        ),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=record_dispatch,
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(
        dispatched,
        [(8, "tiny_linear", "eager_default", "b200_8")],
    )

  def test_main_isolates_failed_target_and_exits_nonzero(self):
    fake_dist_run = _FakeDistRun(failing_specs={"broken"})
    registry = {
        "broken": types.SimpleNamespace(
            name="broken",
            skipped_run_modes=frozenset(),
            microbenchmark_name="batch_size_1",
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
        [
            (d.spec_name, d.run_mode, d.succeeded, d.microbenchmark_name)
            for d in self.exporter.exported
        ],
        [("broken", common.RunMode.EAGER_DEFAULT, False, "batch_size_1")],
    )
    summary = "\n".join(logs.output)
    self.assertRegex(summary, r"broken_eager_default\s+FAILED")
    self.assertRegex(summary, r"healthy_eager_default\s+PASSED")
    self.assertIn("1 passed, 1 failed, 0 skipped", summary)

  def test_discovered_modules_import(self):
    self.assertEqual(distributed_runner._IMPORT_FAILURES, ())

  def _run_main_with_import_failure(
      self, error: Exception, test_filter: str
  ) -> tuple[_FakeDistRun, SystemExit]:
    """Runs main() with a model import failure and one healthy target."""
    failure = discovery_lib.ImportFailure(
        module="torch_tpu.perflab.distributed_models.broken", error=error
    )
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
                "TESTBRIDGE_TEST_ONLY": test_filter,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
    ):
      # pyformat: disable
      with self.assertRaises(SystemExit) as exit_cm:  # ASSERT_RAISES_OK=Runner must exit non-zero when a model fails to import.
        distributed_runner.main(
            ["distributed_runner"], import_failures=(failure,)
        )
      # pyformat: enable
    return fake_dist_run, exit_cm.exception

  @parameterized.named_parameters(
      ("import_error", ImportError("cannot import name 'Foo'")),
      (
          "missing_module",
          ModuleNotFoundError(
              "No module named 'transformers'", name="transformers"
          ),
      ),
      (
          "missing_torchtitan",
          ModuleNotFoundError(
              "No module named 'torchtitan.experiments'",
              name="torchtitan.experiments",
          ),
      ),
  )
  def test_main_fails_on_model_import_error(self, error):
    fake_dist_run, exit_exception = self._run_main_with_import_failure(
        error, test_filter="healthy_eager_default"
    )

    self.assertNotEqual(exit_exception.code, 0)
    # The import failure does not prevent the other benchmarks from running.
    self.assertEqual(fake_dist_run.ran, ["healthy"])
    cases = _read_test_xml(self.xml_path)
    self.assertEqual(cases["benchmark_imports"][0], "FAILURE")
    self.assertIn(
        "torch_tpu.perflab.distributed_models.broken",
        cases["benchmark_imports"][1],
    )
    self.assertEqual(cases["healthy_eager_default"], ("PASSED", ""))

  def test_main_reports_import_error_when_filter_matches_no_target(self):
    # The filter may only select targets of the module that failed to import.
    fake_dist_run, exit_exception = self._run_main_with_import_failure(
        ImportError("cannot import name 'Foo'"),
        test_filter="broken_eager_default",
    )

    self.assertNotEqual(exit_exception.code, 0)
    self.assertEqual(fake_dist_run.ran, [])
    cases = _read_test_xml(self.xml_path)
    self.assertEqual(list(cases), ["benchmark_imports"])
    self.assertEqual(cases["benchmark_imports"][0], "FAILURE")

  def test_main_exports_one_result_per_measured_target(self):
    fake_dist_run = _FakeDistRun(
        failing_specs={"teardown_broken"},
        measured_specs=frozenset({"teardown_broken", "healthy"}),
    )
    registry = {
        name: types.SimpleNamespace(
            name=name, skipped_run_modes=frozenset(), microbenchmark_name=None
        )
        for name in ("teardown_broken", "healthy")
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "teardown_broken_compiled healthy_compiled"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
        self.assertRaises(
            SystemExit
        ),  # ASSERT_RAISES_OK=Runner must exit non-zero when a rank fails.
    ):
      distributed_runner.main(["distributed_runner"])

    # Every rank measured both targets, so each has one record with its
    # metrics, even though teardown_broken failed afterwards.
    self.assertEqual(
        [(d.spec_name, d.succeeded, d.metrics) for d in self.exporter.exported],
        [
            ("teardown_broken", True, _FakeDistRun.METRICS),
            ("healthy", True, _FakeDistRun.METRICS),
        ],
    )
    summary = "\n".join(logs.output)
    self.assertRegex(
        summary,
        r"teardown_broken_compiled\s+FAILED\s+RuntimeError: factory failed",
    )
    self.assertRegex(summary, r"healthy_compiled\s+PASSED")

  @parameterized.named_parameters(
      ("all_ranks_measure", False, True),
      ("peer_fails_after_measuring", True, False),
  )
  def test_run_single_target_task_writes_metrics_once_all_ranks_measured(
      self, peer_fails, writes_metrics
  ):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    measured = []
    orig_measure = measure_lib.measure
    orig_barrier = torch.distributed.barrier

    def recording_measure(*args, **kwargs):
      measured.append(orig_measure(*args, **kwargs))
      return measured[-1]

    def barrier(*args, **kwargs):
      # A peer rank fails once rank 0 has measured, so every later barrier
      # raises.
      if measured and peer_fails:
        raise RuntimeError("peer rank failed")
      return orig_barrier(*args, **kwargs)

    metrics_file = os.path.join(self.create_tempdir().full_path, "metrics")
    expected_error = (
        self.assertRaisesRegex(  # ASSERT_RAISES_OK=A peer failure fails rank 0.
            RuntimeError, "peer rank failed"
        )
        if peer_fails
        else contextlib.nullcontext()
    )

    with (
        _single_rank_gloo_env(),
        mock.patch.object(measure_lib, "measure", recording_measure),
        mock.patch.object(torch.distributed, "barrier", barrier),
        expected_error,
    ):
      distributed_runner._run_single_target_task(
          spec_name="tiny_linear",
          mode_value="eager_default",
          platform_name="cpu",
          metrics_file=metrics_file,
      )

    # The launcher exports the metrics, so the worker never does.
    self.assertEmpty(self.exporter.exported)
    self.assertLen(measured, 1)
    self.assertEqual(
        distributed_runner._read_metrics(metrics_file),
        measured[0] if writes_metrics else None,
    )
    self.assertFalse(torch.distributed.is_initialized())

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
    self.assertRegex(
        summary,
        r"healthy_eager_optimized\s+SKIPPED\s+Run mode eager_optimized is not"
        r" supported on cuda",
    )
    self.assertIn("1 passed, 0 failed, 1 skipped", summary)

  def test_main_reports_known_failure_as_skipped(self):
    fake_dist_run = _FakeDistRun(failing_specs=set())
    registry = {
        "known_bad": types.SimpleNamespace(
            name="known_bad", skipped_run_modes=frozenset()
        ),
        "healthy": types.SimpleNamespace(
            name="healthy", skipped_run_modes=frozenset()
        ),
    }
    failures = (
        known_failures.KnownFailure(
            benchmark="known_bad",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.CUDA,
            run_modes=frozenset({common.RunMode.COMPILED}),
            reason="simulated CUDA compiled failure",
        ),
    )

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.object(known_failures, "KNOWN_FAILURES", failures),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "known_bad_compiled known_bad_eager_default"
                    " healthy_compiled"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(fake_dist_run.ran, ["known_bad", "healthy"])
    summary = "\n".join(logs.output)
    self.assertRegex(
        summary,
        r"known_bad_compiled\s+SKIPPED\s+Benchmark known_bad is known to fail:"
        r" simulated CUDA compiled failure",
    )
    self.assertRegex(summary, r"known_bad_eager_default\s+PASSED")
    self.assertRegex(summary, r"healthy_compiled\s+PASSED")
    self.assertIn("2 passed, 0 failed, 1 skipped", summary)

  @parameterized.named_parameters(
      ("first_shard_node_0", "0", "0", True),
      ("second_shard", "1", "0", False),
      ("node_1", "0", "1", False),
  )
  def test_main_exports_known_failure_skips_once(
      self, shard_index, node_rank, exports_skips
  ):
    registry = {
        "known_bad": types.SimpleNamespace(
            name="known_bad", skipped_run_modes=frozenset()
        ),
        "skips_eager": types.SimpleNamespace(
            name="skips_eager", skipped_run_modes=frozenset({"eager_default"})
        ),
    }
    # Matches every run mode, including eager_optimized, which CUDA does not
    # run.
    failures = (
        known_failures.KnownFailure(
            benchmark="known_bad",
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.CUDA,
            reason="simulated CUDA failure",
        ),
    )
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.object(known_failures, "KNOWN_FAILURES", failures),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "known_bad_compiled known_bad_eager_optimized"
                    " skips_eager_eager_default"
                ),
                "TEST_TOTAL_SHARDS": "2",
                "TEST_SHARD_INDEX": shard_index,
                "TEST_SHARD_STATUS_FILE": status_file,
                "NODE_RANK": node_rank,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            _FakeDistRun(failing_specs=set()),
        ),
        mock.patch.object(
            export_lib, "export_skip", autospec=True
        ) as mock_export_skip,
    ):
      distributed_runner.main(["distributed_runner"])

    if exports_skips:
      mock_export_skip.assert_called_once_with(
          spec_name="known_bad",
          platform=target_lib.Platform.B200_8,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )
    else:
      mock_export_skip.assert_not_called()

  def _run_main_sharded(
      self, total_shards: int, shard_index: int, status_file: str
  ) -> tuple[_FakeDistRun, str]:
    """Runs main() as one blaze test shard and returns (dist_run, summary)."""
    fake_dist_run = _FakeDistRun(failing_specs=set())
    registry = {
        name: types.SimpleNamespace(name=name, skipped_run_modes=frozenset())
        for name in ("a", "b", "c")
    }
    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                # eager_optimized is TPU-only, so it is skipped on CUDA and must
                # not count towards any shard.
                "TESTBRIDGE_TEST_ONLY": (
                    "a_eager_optimized a_compiled b_eager_optimized"
                    " b_compiled c_compiled"
                ),
                "TEST_TOTAL_SHARDS": str(total_shards),
                "TEST_SHARD_INDEX": str(shard_index),
                "TEST_SHARD_STATUS_FILE": status_file,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])
    return fake_dist_run, "\n".join(logs.output)

  @parameterized.named_parameters(
      ("shard_0", 0, ["a", "c"]),
      ("shard_1", 1, ["b"]),
  )
  def test_main_shards_runnable_targets_round_robin(
      self, shard_index, expected_ran
  ):
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    fake_dist_run, summary = self._run_main_sharded(
        total_shards=2, shard_index=shard_index, status_file=status_file
    )

    self.assertEqual(fake_dist_run.ran, expected_ran)
    # Blaze treats a sharded test that never touches the status file as not
    # supporting sharding.
    self.assertTrue(os.path.exists(status_file))
    # Every shard reports the targets skipped for the platform.
    self.assertRegex(summary, r"a_eager_optimized\s+SKIPPED")
    self.assertRegex(summary, r"b_eager_optimized\s+SKIPPED")
    self.assertIn(f"{len(expected_ran)} passed, 0 failed, 2 skipped", summary)

  def test_main_shard_without_targets_passes(self):
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    # 3 runnable targets across 4 shards leaves shard 3 empty.
    fake_dist_run, summary = self._run_main_sharded(
        total_shards=4, shard_index=3, status_file=status_file
    )

    self.assertEmpty(fake_dist_run.ran)
    self.assertTrue(os.path.exists(status_file))
    self.assertIn("0 passed, 0 failed, 2 skipped", summary)

  def test_main_writes_test_xml_per_target(self):
    fake_dist_run = _FakeDistRun(failing_specs={"broken"})
    registry = {
        name: types.SimpleNamespace(
            name=name, skipped_run_modes=frozenset(), microbenchmark_name=None
        )
        for name in ("broken", "healthy")
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                # eager_optimized is TPU-only, so it is skipped on CUDA.
                "TESTBRIDGE_TEST_ONLY": (
                    "broken_compiled healthy_compiled healthy_eager_optimized"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        self.assertRaises(
            SystemExit
        ),  # ASSERT_RAISES_OK=Runner must exit non-zero when any target fails.
    ):
      distributed_runner.main(["distributed_runner"])

    cases = _read_test_xml(self.xml_path)
    self.assertEqual(
        set(cases),
        {"broken_compiled", "healthy_compiled", "healthy_eager_optimized"},
    )
    self.assertEqual(cases["broken_compiled"][0], "FAILURE")
    self.assertIn("factory failed for broken", cases["broken_compiled"][1])
    self.assertEqual(cases["healthy_compiled"], ("PASSED", ""))
    self.assertEqual(
        cases["healthy_eager_optimized"],
        ("SKIPPED", "Run mode eager_optimized is not supported on cuda"),
    )
    suite = ElementTree.parse(self.xml_path).getroot().find("testsuite")
    self.assertEqual(suite.get("tests"), "3")
    self.assertEqual(suite.get("failures"), "1")
    self.assertEqual(suite.get("skipped"), "1")

  @parameterized.named_parameters(
      (
          "shard_0",
          0,
          {
              "a_compiled": "PASSED",
              "c_compiled": "PASSED",
              "a_eager_optimized": "SKIPPED",
              "b_eager_optimized": "SKIPPED",
          },
      ),
      ("shard_1", 1, {"b_compiled": "PASSED"}),
  )
  def test_main_test_xml_reports_skipped_targets_on_first_shard_only(
      self, shard_index, expected_cases
  ):
    status_file = os.path.join(self.create_tempdir().full_path, "status")

    self._run_main_sharded(
        total_shards=2, shard_index=shard_index, status_file=status_file
    )

    self.assertEqual(
        {
            name: status
            for name, (status, _) in _read_test_xml(self.xml_path).items()
        },
        expected_cases,
    )

  def test_shard_targets_returns_all_targets_without_sharding_env(self):
    targets = [
        (types.SimpleNamespace(name=name), common.RunMode.COMPILED)
        for name in ("a", "b", "c")
    ]
    self.enter_context(mock.patch.dict(os.environ))
    for k in (
        "TEST_TOTAL_SHARDS",
        "TEST_SHARD_INDEX",
        "TEST_SHARD_STATUS_FILE",
    ):
      os.environ.pop(k, None)

    self.assertEqual(distributed_runner._shard_targets(targets), targets)

  @parameterized.named_parameters(("negative", "-1"), ("too_large", "2"))
  def test_shard_env_rejects_out_of_range_index(self, shard_index):
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {"TEST_TOTAL_SHARDS": "2", "TEST_SHARD_INDEX": shard_index},
        )
    )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Shard env validation.
        ValueError, "out of range"
    ):
      distributed_runner._shard_env()

  @parameterized.named_parameters(
      ("node_0", "0", True),
      ("node_1", "1", False),
  )
  def test_main_reports_factory_unsupported_benchmark_as_skipped(
      self, node_rank, exports_skip
  ):
    fake_dist_run = _FakeDistRun(
        failing_specs=set(), unsupported_specs=frozenset({"too_big"})
    )
    registry = {
        name: types.SimpleNamespace(name=name, skipped_run_modes=frozenset())
        for name in ("too_big", "healthy")
    }

    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": "too_big_compiled healthy_compiled",
                # Runtime skips happen on the shard that ran the target, so
                # even a non-first shard reports them.
                "TEST_TOTAL_SHARDS": "2",
                "TEST_SHARD_INDEX": "1",
                "NODE_RANK": node_rank,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
        mock.patch.object(distributed_runner, "_shard_targets", lambda t: t),
        mock.patch.object(
            export_lib, "export_skip", autospec=True
        ) as mock_export_skip,
        self.assertLogs(level="INFO") as logs,
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertEqual(fake_dist_run.ran, ["too_big", "healthy"])
    if exports_skip:
      mock_export_skip.assert_called_once_with(
          spec_name="too_big",
          platform=target_lib.Platform.B200_8,
          framework=mode_lib.Framework.TORCH,
          run_mode=common.RunMode.COMPILED,
      )
    else:
      mock_export_skip.assert_not_called()
    summary = "\n".join(logs.output)
    self.assertRegex(
        summary, r"too_big_compiled\s+SKIPPED\s+too_big needs more chips"
    )
    self.assertIn("1 passed, 0 failed, 1 skipped", summary)
    self.assertEqual(
        _read_test_xml(self.xml_path),
        {
            "too_big_compiled": ("SKIPPED", "too_big needs more chips"),
            "healthy_compiled": ("PASSED", ""),
        },
    )

  def test_write_test_xml_strips_invalid_control_characters(self):
    distributed_runner._write_test_xml(
        self.xml_path,
        target_lib.Platform.B200_8,
        [
            distributed_runner._TargetResult(
                "broken_compiled",
                distributed_runner.TargetStatus.FAILED,
                "\x1b[31mRuntimeError\x1b[0m: bad\x00 value",
            )
        ],
    )

    self.assertEqual(
        _read_test_xml(self.xml_path),
        {"broken_compiled": ("FAILURE", "[31mRuntimeError[0m: bad value")},
    )

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

  def _enter_fake_tier2_cache_root(self) -> str:
    """Returns a temporary Tier-2 cache root.

    Also unsets the env vars that enable the Tier-2 cache, for the duration of
    the test.
    """
    self.enter_context(mock.patch.dict(os.environ))
    for name in (
        "TORCH_TPU_TIER2_COMPILATION_CACHE",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        "WORLD_SIZE",
    ):
      os.environ.pop(name, None)
    return self.create_tempdir().full_path

  def test_owned_compilation_caches_spares_shared_caches(self):
    tier2_root = self._enter_fake_tier2_cache_root()
    # Caches of other jobs on the host, which must survive the target.
    shared_tier2_cache = os.path.join(tier2_root, "default")
    os.makedirs(shared_tier2_cache)
    shared_inductor_cache = self.create_tempdir().full_path
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = shared_inductor_cache

    with distributed_runner._owned_compilation_caches(
        8, tier2_cache_root=tier2_root
    ):
      inductor_cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
      tier2_cache = os.path.join(
          tier2_root, os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"]
      )
      # What the workers' compilations leave behind.
      for cache in (inductor_cache, tier2_cache):
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, "entry.bin"), "w") as f:
          f.write("compiled")

    self.assertFalse(os.path.exists(inductor_cache))
    self.assertFalse(os.path.exists(tier2_cache))
    self.assertTrue(os.path.isdir(shared_tier2_cache))
    self.assertTrue(os.path.isdir(shared_inductor_cache))
    self.assertEqual(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"), shared_inductor_cache
    )
    self.assertNotIn("TORCH_TPU_TIER2_COMPILATION_CACHE", os.environ)

  def test_owned_compilation_caches_are_fresh_per_target(self):
    tier2_root = self._enter_fake_tier2_cache_root()
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "my_cache"

    caches = []
    for _ in range(2):
      # An explicitly named Tier-2 cache is enabled even for a single process.
      with distributed_runner._owned_compilation_caches(
          1, tier2_cache_root=tier2_root
      ):
        caches.append((
            os.environ["TORCHINDUCTOR_CACHE_DIR"],
            os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"],
        ))

    self.assertNotEqual(caches[0][0], caches[1][0])
    self.assertNotEqual(caches[0][1], caches[1][1])
    # A user-named Tier-2 cache is neither reused nor deleted.
    self.assertNotIn("my_cache", [name for _, name in caches])
    self.assertEqual(
        os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"], "my_cache"
    )

  def test_owned_compilation_caches_keep_tier2_cache_disabled(self):
    tier2_root = self._enter_fake_tier2_cache_root()
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"

    with distributed_runner._owned_compilation_caches(
        8, tier2_cache_root=tier2_root
    ):
      tier2_cache_name = os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"]

    self.assertEqual(tier2_cache_name, "disabled")

  def test_owned_compilation_caches_keep_tier2_cache_off_for_single_process(
      self,
  ):
    tier2_root = self._enter_fake_tier2_cache_root()
    os.makedirs(os.path.join(tier2_root, "default"))

    with distributed_runner._owned_compilation_caches(
        1, tier2_cache_root=tier2_root
    ):
      env_during_target = dict(os.environ)

    self.assertNotIn("TORCH_TPU_TIER2_COMPILATION_CACHE", env_during_target)
    self.assertEqual(os.listdir(tier2_root), ["default"])

  @parameterized.named_parameters(
      dict(testcase_name="multi_host", world_size="16", uses_tier2_cache=True),
      dict(testcase_name="single_rank", world_size="1", uses_tier2_cache=False),
  )
  def test_owned_compilation_caches_follow_worker_world_size(
      self, world_size, uses_tier2_cache
  ):
    tier2_root = self._enter_fake_tier2_cache_root()
    # The workers keep an inherited WORLD_SIZE over the per-node process count.
    os.environ["WORLD_SIZE"] = world_size

    with distributed_runner._owned_compilation_caches(
        1 if uses_tier2_cache else 8, tier2_cache_root=tier2_root
    ):
      env_during_target = dict(os.environ)

    self.assertEqual(
        "TORCH_TPU_TIER2_COMPILATION_CACHE" in env_during_target,
        uses_tier2_cache,
    )

  def test_owned_compilation_caches_own_tier2_cache_with_tier3_cache(self):
    tier2_root = self._enter_fake_tier2_cache_root()
    # The Tier-3 cache turns on the Tier-2 cache even for a single process.
    os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = (
        self.create_tempdir().full_path
    )

    with distributed_runner._owned_compilation_caches(
        1, tier2_cache_root=tier2_root
    ):
      tier2_cache_name = os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"]

    self.assertStartsWith(tier2_cache_name, "perflab_")

  def test_owned_compilation_caches_removed_when_target_fails(self):
    tier2_root = self._enter_fake_tier2_cache_root()

    with self.assertRaisesRegex(RuntimeError, "target failed"):
      with distributed_runner._owned_compilation_caches(
          8, tier2_cache_root=tier2_root
      ):
        inductor_cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        tier2_cache = os.path.join(
            tier2_root, os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"]
        )
        os.makedirs(tier2_cache)
        raise RuntimeError("target failed")

    self.assertFalse(os.path.exists(inductor_cache))
    self.assertFalse(os.path.exists(tier2_cache))

  def test_main_runs_each_target_with_its_own_compilation_caches(self):
    tier2_root = self._enter_fake_tier2_cache_root()
    # `main` takes no injected dependencies, so point it at the fake root.
    self.enter_context(
        mock.patch.object(distributed_runner, "_TIER2_CACHE_ROOT", tier2_root)
    )
    seen_caches = []

    def fake_dist_run(*args):
      del args
      inductor_cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
      tier2_cache = os.path.join(
          tier2_root, os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"]
      )
      # What the workers' compilations leave behind.
      os.makedirs(tier2_cache)
      seen_caches.append((inductor_cache, tier2_cache))

    registry = {
        name: types.SimpleNamespace(name=name, skipped_run_modes=frozenset())
        for name in ("first", "second")
    }
    with (
        mock.patch.dict(registry_lib.REGISTRY, registry, clear=True),
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "b200_8",
                "TESTBRIDGE_TEST_ONLY": (
                    "first_eager_default second_eager_default"
                ),
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils, "dist_run", fake_dist_run
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertLen(seen_caches, 2)
    self.assertLen(set(seen_caches), 2)
    for inductor_cache, tier2_cache in seen_caches:
      self.assertFalse(os.path.exists(inductor_cache))
      self.assertFalse(os.path.exists(tier2_cache))


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
