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
import contextlib
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
import portpicker
import torch
from torch_tpu._internal.distributed import multiprocessing
from perflab import common
from perflab import discovery as discovery_lib
from perflab import distributed_runner
from perflab import export as export_lib
from perflab import measure as measure_lib
from perflab import registry as registry_lib
from perflab import steps
from perflab import target as target_lib
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


class DistributedRunnerTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
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

  def test_backend_by_device_kind(self):
    self.assertEqual(
        distributed_runner._BACKEND_BY_DEVICE_KIND,
        {
            target_lib.DeviceKind.CPU: "gloo",
            target_lib.DeviceKind.CUDA: "nccl",
            target_lib.DeviceKind.TPU: "tpu_dist",
        },
    )

  def test_run_single_target_task_end_to_end_exports_metrics(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")

    with _single_rank_gloo_env():
      distributed_runner._run_single_target_task(
          spec_name="tiny_linear",
          mode_value="eager_default",
          platform_name="cpu",
      )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertLen(self.exporter.exported, 1)
    record = self.exporter.exported[0]
    self.assertEqual(record.spec_name, "tiny_linear")
    self.assertEqual(record.platform, target_lib.Platform.CPU)
    self.assertEqual(record.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(record.succeeded)
    self.assertGreater(record.metrics.e2e_wall_time_seconds, 0.0)
    self.assertGreater(record.metrics.first_step_time_seconds, 0.0)
    self.assertGreater(record.metrics.post_warmup_step_time_seconds, 0.0)

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
        )

    self.assertFalse(torch.distributed.is_initialized())
    self.assertLen(self.exporter.exported, 1)
    self.assertFalse(self.exporter.exported[0].succeeded)

  def test_run_single_target_task_exception_exports_failure_and_cleans_up(self):
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
    self.assertLen(self.exporter.exported, 1)
    record = self.exporter.exported[0]
    self.assertEqual(record.spec_name, "oom_spec")
    self.assertFalse(record.succeeded)

  def test_main_runs_filtered_target_and_clears_caches(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    custom_cache = os.path.join(
        self.create_tempdir().full_path, "inductor_cache"
    )
    os.makedirs(custom_cache)

    with (
        mock.patch.dict(
            os.environ,
            {
                "BENCHMARK_PLATFORM": "cpu",
                "TESTBRIDGE_TEST_ONLY": "tiny_linear_eager_default",
                "TORCHINDUCTOR_CACHE_DIR": custom_cache,
            },
        ),
        mock.patch.object(
            distributed_runner.distributed_utils,
            "dist_run",
            side_effect=_in_process_dist_run,
        ),
    ):
      distributed_runner.main(["distributed_runner"])

    self.assertFalse(os.path.exists(custom_cache))
    self.assertLen(self.exporter.exported, 1)
    record = self.exporter.exported[0]
    self.assertEqual(record.spec_name, "tiny_linear")
    self.assertEqual(record.platform, target_lib.Platform.CPU)
    self.assertEqual(record.run_mode, common.RunMode.EAGER_DEFAULT)
    self.assertTrue(record.succeeded)

  def test_main_tpu_launch_env_singlehost_vs_multihost(self):
    registry_lib.REGISTRY["tiny_linear"] = fakes.tiny_spec("tiny_linear")
    dispatched: list[tuple[int, str, str, str]] = []

    def record_dispatch(nprocs, unused_fn, spec_name, mode_val, platform_val):
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

  def test_clear_persistent_compilation_caches(self):
    root = self.create_tempdir().full_path
    shm_cache = os.path.join(root, "torch_tpu_cache")
    inductor_cache = os.path.join(root, "torchinductor_user")
    compile_cache = os.path.join(root, "torch_compile_debug")
    custom_cache = os.path.join(root, "custom_inductor")
    unrelated_dir = os.path.join(root, "keep_me")
    for path in (
        shm_cache,
        inductor_cache,
        compile_cache,
        custom_cache,
        unrelated_dir,
    ):
      os.makedirs(path)

    with mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": custom_cache}):
      distributed_runner._clear_persistent_compilation_caches(
          shm_cache_dir=shm_cache,
          cache_patterns=(
              os.path.join(root, "torchinductor*"),
              os.path.join(root, "torch_compile*"),
          ),
      )

    self.assertFalse(os.path.exists(shm_cache))
    self.assertFalse(os.path.exists(inductor_cache))
    self.assertFalse(os.path.exists(compile_cache))
    self.assertFalse(os.path.exists(custom_cache))
    self.assertTrue(os.path.exists(unrelated_dir))


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
