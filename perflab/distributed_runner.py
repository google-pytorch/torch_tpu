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

"""Standalone production worker runner for distributed performance benchmarks.

Supports both single-node multi-device (CUDA/TPU) and multi-node multi-host
Borg executions.
"""

import enum
import gc
import glob
import os
import shutil
from typing import Sequence

from absl import app
from absl import logging
from fairscale.nn.model_parallel import initialize as fairscale_init
import torch
import torch.multiprocessing as mp
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from perflab import common
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import distributed_models
from perflab import export as export_lib
from perflab import flags as flags_lib
from perflab import measure as measure_lib
from perflab import metrics as metrics_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_device_ops
from tests.distributed import distributed_utils

discovery_lib.import_submodules(distributed_models)
discovery_lib.import_submodules(steps)

_FRAMEWORK = mode_lib.Framework.TORCH


class BarrierPolicy(enum.Enum):
  """Policy for collective synchronization barrier during state cleanup."""

  SYNCHRONIZE = "synchronize"
  SKIP = "skip"


def resolve_benchmark_targets(
    test_filter: str | None = None,
) -> list[tuple[registry_lib.BenchmarkSpec, common.RunMode]]:
  """Resolves benchmark specs and modes deterministically from a filter string.

  Args:
    test_filter: Space-delimited target filter string. Each token must be an
      explicit target name with run mode suffix (e.g.
      'llama_8b_inference_eager_default'). If None or empty, returns all
      registered targets across all modes.

  Returns:
    List of (BenchmarkSpec, RunMode) tuples to execute.
  """
  target_map = {}
  for name, spec in sorted(registry_lib.REGISTRY.items()):
    for mode in common.RunMode:
      target_map[f"{name}_{mode.value}"] = (spec, mode)

  if not test_filter:
    return list(target_map.values())

  tokens = test_filter.split()
  if invalid := set(tokens) - target_map.keys():
    raise ValueError(
        f"Could not resolve benchmark target(s): {sorted(invalid)}. "
        f"Valid targets are: {sorted(target_map)}"
    )

  return [target_map[token] for token in tokens]


def _cleanup_distributed_worker_state(
    rank: int,
    target: target_lib.Target | None = None,
    barrier_policy: BarrierPolicy = BarrierPolicy.SYNCHRONIZE,
) -> None:
  """Flushes TPU/CUDA caches and tears down model parallel & process groups.

  Args:
    rank: Distributed rank index of current worker.
    target: Target runtime specification.
    barrier_policy: Barrier policy before process group destruction. Setting to
      BarrierPolicy.SKIP prevents barrier timeouts/deadlocks when unwinding from
      an unhandled exception on one or more ranks.
  """
  logging.info("Rank %s starting inter-test state cleanup...", rank)
  try:
    if fairscale_init.model_parallel_is_initialized():
      fairscale_init.destroy_model_parallel()
  except (RuntimeError, AttributeError) as e:
    logging.debug("Fairscale cleanup skipped on rank %s: %s", rank, e)

  if torch.distributed.is_initialized():
    # Avoid barrier deadlocks when unwinding from an exception, as failed ranks
    # will not reach this point to join the barrier.
    if barrier_policy == BarrierPolicy.SYNCHRONIZE:
      try:
        torch.distributed.barrier()
      except (RuntimeError, torch.distributed.DistError) as e:
        logging.warning("Barrier failed during cleanup on rank %s: %s", rank, e)
    torch.distributed.destroy_process_group()
    logging.info("Rank %s process group destroyed.", rank)

  tt_testing.reset_eager_state()
  torch.compiler.reset()

  # Run cyclic garbage collection BEFORE flushing the device allocator cache so
  # any unreachable Python `nn.Module` / tensor objects from the completed
  # benchmark release their storage back to the caching allocator first.
  gc.collect()
  if target is not None:
    if target.device_kind == target_lib.DeviceKind.TPU and hasattr(
        torch, "tpu"
    ):
      torch.tpu._clear_cache()
    elif (
        target.device_kind == target_lib.DeviceKind.CUDA
        and torch.cuda.is_available()
    ):
      torch.cuda.empty_cache()

  logging.info("Rank %s inter-test state cleanup completed.", rank)


def _export_benchmark_result(
    spec: registry_lib.BenchmarkSpec,
    platform: target_lib.Platform,
    mode: common.RunMode,
    succeeded: bool,
    metrics: metrics_lib.PerformanceMetrics,
) -> None:
  """Exports benchmark outcome and metrics to output storage."""
  export_lib.export(
      export_lib.BenchmarkData(
          spec_name=spec.name,
          platform=platform,
          framework=_FRAMEWORK,
          run_mode=mode,
          succeeded=succeeded,
          metrics=metrics,
      )
  )


def _run_single_target_task(
    spec_name: str,
    mode_value: str,
    platform_name: str,
) -> None:
  """Worker task executed on each rank/host for a single benchmark target."""
  # Required for OSS/GCP: Subprocesses spawned via torch.multiprocessing.spawn
  # bypass app.run() and start with unparsed absl flags.
  multiprocessing.parse_absl_flags()
  rank = int(os.environ.get("RANK", "0"))  # PY_ENVIRON_OK=benchmark
  exception_raised = False

  # Resolve platform, spec, and mode
  platform = target_lib.Platform(platform_name)
  spec = registry_lib.REGISTRY[spec_name]
  mode = common.RunMode(mode_value)

  # Setup Target for benchmark execution and cleanup
  target = target_lib.make_target(platform, dtype=spec.dtype)
  # resolve_benchmark_targets() expands all common.RunMode values when
  # unfiltered. Filter against mode_lib.modes_for() (matching base_test.py) so
  # TPU-only modes like eager_optimized or debug modes like
  # eager_defer_never_and_launch_blocking do not run and export duplicate rows
  # on CUDA or TPU.
  if mode.value in spec.skipped_run_modes or mode not in mode_lib.modes_for(
      _FRAMEWORK, target.device_kind
  ):
    logging.info(
        "Skipping benchmark %s on mode %s",
        spec.name,
        mode.value,
    )
    return
  backend = (
      "nccl" if target.device_kind == target_lib.DeviceKind.CUDA else "tpu_dist"
  )
  if target.device_kind == target_lib.DeviceKind.CUDA:
    torch.cuda.set_device(
        int(os.environ.get("LOCAL_RANK", "0"))  # PY_ENVIRON_OK=benchmark
    )

  try:
    logging.info(
        "Rank %s initializing process group for %s (%s) with backend %s...",
        rank,
        spec.name,
        mode.value,
        backend,
    )
    torch.distributed.init_process_group(backend=backend)
    logging.info("Rank %s process group initialized.", rank)

    logging.info(
        "Rank %s executing pre-benchmark barrier synchronization...", rank
    )
    torch.distributed.barrier()
    logging.info("Rank %s passed pre-benchmark barrier synchronization.", rank)

    logging.info(
        "Rank %s starting benchmark execution for %s (%s)...",
        rank,
        spec.name,
        mode.value,
    )

    # Harness setup
    device_ops = torch_device_ops.TorchDeviceOps(target)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    with mode_lib.run_mode_context(mode, target):
      # Build run step
      common.seed_rngs()
      logging.info("Rank %s calling spec.factory...", rank)
      factory_res = spec.factory(ctx)
      logging.info("Rank %s spec.factory returned.", rank)
      stepper = step_lib.resolve_stepper(spec.stepper, **spec.stepper_kwargs)
      stepper.init_with_benchmark_args(*factory_res)

      # Apply compile if needed
      if common.is_torch_compile(mode):
        compile_config = spec.compile_config or compile_lib.CompileConfig()
        stepper.compile(compile_config, ctx.target)

      # Measure
      logging.info("Rank %s calling measure...", rank)
      metrics = measure_lib.measure(
          stepper,
          device_ops,
          name=f"{spec.name}_{mode.value}",
          enable_xprof=flags_lib.ENABLE_XPROF.value,
      )
      logging.info(
          "Metrics for %s_%s on rank %s: %s",
          spec.name,
          mode.value,
          rank,
          metrics,
      )

    # Only rank 0 validates and exports results to persistent storage
    # (MLCompass / Sponge) to avoid duplicate entries and write collisions
    # across distributed workers.
    if rank == 0:
      if (
          metrics.e2e_wall_time_seconds <= 0.0
          or metrics.first_step_time_seconds <= 0.0
      ):
        raise AssertionError(
            f"Invalid metrics measured on rank {rank}: {metrics}"
        )
      _export_benchmark_result(
          spec, target.platform, mode, succeeded=True, metrics=metrics
      )

  except target_lib.UnsupportedBenchmark as e:
    logging.info("Skipping unsupported benchmark on rank %s: %s", rank, e)
  except Exception as e:
    exception_raised = True
    logging.error("FATAL DISTRIBUTED WORKER ERROR: %s: %s", type(e).__name__, e)
    # Only rank 0 reports failure status to persistent storage to prevent
    # duplicate failure records across worker processes.
    if rank == 0:
      _export_benchmark_result(
          spec,
          target.platform,
          mode,
          succeeded=False,
          metrics=metrics_lib.PerformanceMetrics(),
      )
    raise
  finally:
    # Explicitly drop local references to the model, example inputs, and stepper
    # before calling `_cleanup_distributed_worker_state` so `gc.collect()` can
    # reclaim their device memory prior to `empty_cache()`.
    factory_res = None
    stepper = None
    barrier_policy = (
        BarrierPolicy.SKIP if exception_raised else BarrierPolicy.SYNCHRONIZE
    )
    _cleanup_distributed_worker_state(
        rank, target=target, barrier_policy=barrier_policy
    )


def _clear_persistent_compilation_caches() -> None:
  """Clears shared-memory and on-disk compilation caches between benchmark runs.

  Ensures each distributed benchmark target measures true cold compilation
  latency without being polluted by cached graph artifacts or compiled binaries
  from preceding targets.
  """
  # 1. Clear TorchTPU Tier-2 host-local shared memory compilation cache.
  shutil.rmtree("/dev/shm/torch_tpu_cache", ignore_errors=True)

  # 2. Clear PyTorch Inductor and compile on-disk caches.
  cache_patterns = [
      "/tmp/torchinductor*",
      "/tmp/torch_compile*",
  ]
  if custom_inductor_cache := os.environ.get(  # PY_ENVIRON_OK=benchmark
      "TORCHINDUCTOR_CACHE_DIR"
  ):
    shutil.rmtree(custom_inductor_cache, ignore_errors=True)

  for pattern in cache_patterns:
    for p in glob.glob(pattern):
      shutil.rmtree(p, ignore_errors=True)


def main(argv: Sequence[str]) -> None:
  """Main entry point for discovering and executing distributed benchmarks."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  try:
    platform = target_lib.platform_from_env()
  except ValueError as e:
    raise ValueError(f"Unsupported platform: {e}") from e

  # Resolve topology
  target = target_lib.make_target(platform)
  topology = target.platform_spec.topology
  nnodes = topology.nnodes
  local_nproc = topology.nprocs_per_node

  if nnodes <= 1 and local_nproc <= 1 and platform != target_lib.Platform.CPU:
    logging.warning(
        "Running on a single-node single-process platform: %s", platform
    )

  if target.device_kind == target_lib.DeviceKind.TPU and nnodes == 1:
    # Populate TORCH_TPU_SLICEBUILDER_ADDRESSES and TORCH_TPU_TOPOLOGY when
    # running standalone single-host TPU benchmarks (e.g., on GCP or GitHub
    # Actions runners). This is a no-op if already set in os.environ.
    # Imported lazily so GPU runs (TORCH_DEVICE_BACKEND_AUTOLOAD=0) never depend
    # on `torch.tpu`, which only exists once the TorchTPU backend is loaded.
    from torch.tpu import distributed  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    distributed.set_tpu_launch_env(nproc_per_node=local_nproc)

  test_filter = os.environ.get(  # PY_ENVIRON_OK=benchmark
      "TESTBRIDGE_TEST_ONLY"
  )
  targets_to_run = resolve_benchmark_targets(test_filter)

  if not targets_to_run:
    raise ValueError(
        f"No matching benchmark targets found for platform {platform.value}"
        f" with filter '{test_filter}'."
    )

  logging.info(
      "Launching %d benchmark target(s) on platform %s with nnodes=%d,"
      " local_nproc=%d",
      len(targets_to_run),
      platform.value,
      nnodes,
      local_nproc,
  )

  for idx, (spec, mode) in enumerate(targets_to_run):
    target_name = f"{spec.name}_{mode.value}"
    logging.info(
        "Executing batched target %d/%d: %s",
        idx + 1,
        len(targets_to_run),
        target_name,
    )
    # Evict shared-memory and on-disk compilation caches between targets so each
    # target measures true cold compilation latency.
    _clear_persistent_compilation_caches()
    distributed_utils.dist_run(
        local_nproc,
        _run_single_target_task,
        spec.name,
        mode.value,
        platform.value,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  multiprocessing.handle_main(main)
