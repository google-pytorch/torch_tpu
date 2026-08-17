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

"""Production worker runner for multi-host performance benchmarks on Borg."""

import enum
import gc
import os
from typing import Sequence

from absl import app
from absl import logging
from fairscale.nn.model_parallel import initialize as fairscale_init
import torch
import torch.multiprocessing as mp
from torch_tpu.csrc.internal import testing as tt_testing
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import discovery as discovery_lib

# Unused import required to register MLCompass and runner flags with absl.
from examples.benchmarks.e2e.harness import flags as _  # pylint: disable=unused-import  # noqa: F401
from examples.benchmarks.e2e.harness import measure as measure_lib
from examples.benchmarks.e2e.harness import mode as mode_lib
from examples.benchmarks.e2e.harness import models
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import steps
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import distributed_utils

discovery_lib.import_submodules(models)
discovery_lib.import_submodules(steps)


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
      explicit spec name with run mode suffix (e.g.,
      'llama_8b_inference_eager_default'). If None or empty, returns all specs
      across all modes.

  Returns:
    List of (BenchmarkSpec, RunMode) tuples to execute.
  """
  target_map = {}
  for name, spec in sorted(registry_lib.REGISTRY.items()):
    for mode in common.RunMode:
      target_map[f"{name}_{mode.value}"] = (spec, mode)

  if not test_filter:
    return list(target_map.values())

  tokens = dict.fromkeys(test_filter.split())
  if invalid := tokens.keys() - target_map.keys():
    raise ValueError(
        f"Could not resolve benchmark target(s): {sorted(invalid)}. "
        f"Valid targets are: {sorted(target_map)}"
    )

  return [target_map[token] for token in tokens]


def _cleanup_multihost_worker_state(
    rank: int,
    barrier_policy: BarrierPolicy = BarrierPolicy.SYNCHRONIZE,
) -> None:
  """Flushes TPU caches and tears down model parallel & process groups.

  Args:
    rank: Distributed rank index of current worker.
    barrier_policy: Barrier policy before process group destruction. Setting to
      BarrierPolicy.SKIP prevents barrier timeouts/deadlocks when unwinding from
      an unhandled exception on one or more ranks.
  """
  logging.info("Rank %s starting inter-test state cleanup...", rank)
  tt_testing.reset_eager_state()
  torch.compiler.reset()
  gc.collect()

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
  logging.info("Rank %s inter-test state cleanup completed.", rank)


def _run_single_target_task(
    spec_name: str,
    mode_value: str,
    platform_name: str,
) -> None:
  """Worker task executed on each rank/host for a single benchmark target."""
  rank = int(os.environ.get("RANK", "0"))
  exception_raised = False

  # Resolve platform, spec, and mode
  platform = target_lib.Platform(platform_name)
  spec = registry_lib.REGISTRY[spec_name]
  mode = common.RunMode(mode_value)

  # Setup Target for benchmark execution and cleanup
  target = target_lib.make_target(platform, dtype=spec.dtype)

  try:
    logging.info(
        "Rank %s initializing process group for %s (%s)...",
        rank,
        spec.name,
        mode.value,
    )
    torch.distributed.init_process_group(backend="tpu_dist")
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
    run_scope_str = os.environ.get("RUN_SCOPE", context_lib.RunScope.FULL.value)
    run_scope = context_lib.RunScope(run_scope_str.lower())
    ctx = context_lib.Context(target=target, run_scope=run_scope)

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
      # TODO: b/543140638 - Add a synchronization barrier inside the
      # measurement path (between warmup and post-warmup timed runs) to ensure
      # all ranks enter the timed loop simultaneously without desynchronized
      # waiting overhead.
      metrics = measure_lib.measure(stepper, device_ops, name=spec.name)
      logging.info("Rank %s measure returned.", rank)
      logging.info("Metrics for %s: %s", spec.name, metrics)

  except target_lib.UnsupportedBenchmark as e:
    logging.info("Skipping unsupported benchmark on rank %s: %s", rank, e)
  except Exception as e:
    exception_raised = True
    logging.error("FATAL WORKER ERROR: %s: %s", type(e).__name__, e)
    raise
  finally:
    barrier_policy = (
        BarrierPolicy.SKIP if exception_raised else BarrierPolicy.SYNCHRONIZE
    )
    _cleanup_multihost_worker_state(rank, barrier_policy=barrier_policy)


def main(argv: Sequence[str]) -> None:
  """Main entry point for discovering and executing multi-host benchmarks."""
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

  if nnodes <= 1 and platform != target_lib.Platform.CPU:
    logging.warning("Running on a single-node platform: %s", platform)

  test_filter = os.environ.get("TESTBRIDGE_TEST_ONLY")
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
