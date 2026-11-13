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

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import enum
import fnmatch
import gc
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from typing import NamedTuple
import uuid
from xml.etree import ElementTree

from absl import app
from absl import logging
from fairscale.nn.model_parallel import initialize as fairscale_init
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from perflab import cases
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
from perflab import workload as workload_lib
from tests.distributed import distributed_utils

# Import all models and steps and trigger registration of benchmarks. main()
# fails the run if any of them fails to import. The torchtitan-dependent models
# are not exported to OSS (see copy.bara.sky), so they never fail here.
_IMPORT_FAILURES: tuple[discovery_lib.ImportFailure, ...] = (
    *discovery_lib.import_submodules(distributed_models),
    *discovery_lib.import_submodules(steps),
)

_FRAMEWORK = mode_lib.Framework.TORCH
# Characters XML 1.0 forbids (all C0 controls except tab, newline and CR).
_XML_INVALID_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INDUCTOR_CACHE_ENV = "TORCHINDUCTOR_CACHE_DIR"
# The Tier-2 compilation cache named by this env var lives in a subdirectory of
# the host-wide root. Keep these constants and the enablement logic in
# `_owned_compilation_caches` in sync with `kTier2CacheRootDir` and
# `GetTier2CacheName()` in csrc/common/tier2_compilation_cache.cc.
_TIER2_CACHE_ENV = "TORCH_TPU_TIER2_COMPILATION_CACHE"
_TIER2_CACHE_DISABLED = "disabled"
_TIER2_CACHE_ROOT = "/dev/shm/torch_tpu_cache"
_TIER3_CACHE_ROOT_ENV = "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"

_BACKEND_BY_DEVICE_KIND = {
    target_lib.DeviceKind.CPU: "gloo",
    target_lib.DeviceKind.CUDA: "nccl",
    target_lib.DeviceKind.TPU: "tpu_dist",
}


class BarrierPolicy(enum.Enum):
  """Policy for collective synchronization barrier during state cleanup."""

  SYNCHRONIZE = "synchronize"
  SKIP = "skip"


class TargetStatus(enum.Enum):
  """Outcome of a single batched benchmark target."""

  PASSED = "PASSED"
  FAILED = "FAILED"
  SKIPPED = "SKIPPED"


def _get_skip_reason(
    spec: registry_lib.BenchmarkSpec,
    mode: common.RunMode,
    device_kind: target_lib.DeviceKind,
) -> str | None:
  """Returns why `spec` should not run in `mode` on `device_kind`, or None.

  resolve_benchmark_targets() expands all common.RunMode values when unfiltered.
  Filter against mode_lib.modes_for() (matching cases.get_cases()) so TPU-only
  modes like eager_optimized or debug modes like
  eager_defer_never_and_launch_blocking do not run and export duplicate rows on
  CUDA or TPU.

  Args:
    spec: Benchmark spec to check.
    mode: Run mode to check.
    device_kind: Device kind of the current platform.
  """
  if mode not in mode_lib.modes_for(_FRAMEWORK, device_kind):
    return f"Run mode {mode.value} is not supported on {device_kind.value}"
  return cases.get_skip_reason(
      spec, mode, framework=_FRAMEWORK, device_kind=device_kind
  )


def _failure_reason(e: Exception) -> str:
  """Returns a one-line failure reason for the summary table."""
  lines = str(e).strip().splitlines()
  return lines[-1] if lines else type(e).__name__


class _TargetResult(NamedTuple):
  """Outcome of a single batched benchmark target."""

  name: str
  status: TargetStatus
  reason: str = ""
  duration_seconds: float = 0.0


def _format_summary(
    platform: target_lib.Platform,
    results: Sequence[_TargetResult],
) -> str:
  """Formats a per-target pass/fail summary table."""
  counts = {
      status: sum(1 for r in results if r.status == status)
      for status in TargetStatus
  }
  name_width = max((len(r.name) for r in results), default=0)
  lines = [
      f"Distributed benchmark summary ({platform.value}):"
      f" {counts[TargetStatus.PASSED]} passed,"
      f" {counts[TargetStatus.FAILED]} failed,"
      f" {counts[TargetStatus.SKIPPED]} skipped"
  ]
  for r in results:
    lines.append(
        f"  {r.name:<{name_width}}  {r.status.value:<7}  {r.reason}".rstrip()
    )
  return "\n".join(lines)


def _write_test_xml(
    path: str,
    platform: target_lib.Platform,
    results: Sequence[_TargetResult],
) -> None:
  """Writes `results` as a JUnit XML report, one test case per target.

  Blaze passes the report path in `XML_OUTPUT_FILE`; Sponge reads it to list
  test cases. Without it, Sponge only shows the pass/fail status of each shard.
  Test case names are the target names `TESTBRIDGE_TEST_ONLY` accepts, so a
  failed case can be rerun with `--test_filter=<name>`.

  Args:
    path: Output path.
    platform: Platform the targets ran on, used as the test case class name.
    results: Targets to report.
  """
  suite = ElementTree.Element(
      "testsuite",
      name="distributed_runner",
      tests=str(len(results)),
      failures=str(sum(r.status == TargetStatus.FAILED for r in results)),
      skipped=str(sum(r.status == TargetStatus.SKIPPED for r in results)),
      time=f"{sum(r.duration_seconds for r in results):.3f}",
  )
  for r in results:
    case = ElementTree.SubElement(
        suite,
        "testcase",
        name=r.name,
        classname=f"distributed_benchmarks.{platform.value}",
        time=f"{r.duration_seconds:.3f}",
    )
    # ElementTree escapes markup characters but not control characters, which
    # XML 1.0 forbids; one (e.g. an ANSI color code in an exception message)
    # would make Sponge drop every test case of the shard.
    reason = _XML_INVALID_CHARS.sub("", r.reason)
    if r.status == TargetStatus.FAILED:
      ElementTree.SubElement(case, "failure", message=reason).text = reason
    elif r.status == TargetStatus.SKIPPED:
      ElementTree.SubElement(case, "skipped", message=reason)
  root = ElementTree.Element("testsuites")
  root.append(suite)
  ElementTree.ElementTree(root).write(
      path, encoding="utf-8", xml_declaration=True
  )


def resolve_benchmark_targets(
    test_filter: str | None = None,
) -> Sequence[tuple[registry_lib.BenchmarkSpec, common.RunMode]]:
  """Resolves benchmark specs and modes deterministically from a filter string.

  Args:
    test_filter: Space-delimited target filter string. Each token is either an
      explicit target name with run mode suffix (e.g.
      'llama_8b_inference_eager_default') or, as absltest accepts in
      `TESTBRIDGE_TEST_ONLY`, an fnmatch wildcard pattern over target names
      (e.g. '*_compiled'). If None or empty, returns all registered targets
      across all modes.

  Returns:
    List of (BenchmarkSpec, RunMode) tuples to execute, in token order. The
    targets a wildcard pattern matches are ordered by benchmark name, then by
    `common.RunMode` order.

  Raises:
    ValueError: If a token matches no target.
  """
  target_map = {}
  for name, spec in sorted(registry_lib.REGISTRY.items()):
    for mode in common.RunMode:
      target_map[f"{name}_{mode.value}"] = (spec, mode)

  if not test_filter:
    return list(target_map.values())

  targets = []
  invalid = set()
  for token in test_filter.split():
    matches = [name for name in target_map if fnmatch.fnmatchcase(name, token)]
    if not matches:
      invalid.add(token)
    targets.extend(target_map[name] for name in matches)
  if invalid:
    raise ValueError(
        f"Could not resolve benchmark target(s): {sorted(invalid)}. "
        f"Valid targets are: {sorted(target_map)}"
    )

  return targets


def _shard_env() -> tuple[int, int]:
  """Returns `(shard_index, total_shards)` from the blaze sharding env.

  Returns `(0, 1)` without the sharding environment (e.g. multi-host Borg runs).

  Raises:
    ValueError: If the shard index is out of range.
  """
  total_shards = int(
      os.environ.get("TEST_TOTAL_SHARDS", "1")  # PY_ENVIRON_OK=benchmark
  )
  shard_index = int(
      os.environ.get("TEST_SHARD_INDEX", "0")  # PY_ENVIRON_OK=benchmark
  )
  if not 0 <= shard_index < total_shards:
    raise ValueError(
        f"TEST_SHARD_INDEX={shard_index} is out of range for"
        f" TEST_TOTAL_SHARDS={total_shards}."
    )
  return shard_index, total_shards


def _shard_targets(
    targets: Sequence[tuple[registry_lib.BenchmarkSpec, common.RunMode]],
) -> Sequence[tuple[registry_lib.BenchmarkSpec, common.RunMode]]:
  """Returns the subset of `targets` that this blaze test shard runs.

  Implements the blaze test sharding protocol (go/test-sharding): reads
  `TEST_TOTAL_SHARDS` / `TEST_SHARD_INDEX`, assigns targets round-robin, and
  touches `TEST_SHARD_STATUS_FILE` to tell blaze sharding is supported. Without
  these variables (e.g. multi-host Borg runs) every target is returned.

  Args:
    targets: Runnable targets in a deterministic order. Callers should drop
      targets that will be skipped first, so shards get similar numbers of
      targets that actually run.
  """
  if status_file := os.environ.get(  # PY_ENVIRON_OK=benchmark
      "TEST_SHARD_STATUS_FILE"
  ):
    with open(status_file, "a"):
      pass
  shard_index, total_shards = _shard_env()
  return list(targets[shard_index::total_shards])


def _cleanup_distributed_worker_state(
    rank: int,
    target: target_lib.Target | None = None,
    barrier_policy: BarrierPolicy = BarrierPolicy.SYNCHRONIZE,
) -> None:
  """Flushes TPU/CUDA caches and tears down model parallel & process groups.

  Args:
    rank: Distributed rank index of current worker.
    target: Target runtime specification. If None, device-specific cleanup
      (TorchTPU eager state reset and device cache flush) is skipped.
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

  if target is not None and target.device_kind == target_lib.DeviceKind.TPU:
    # Imported lazily so that CUDA and CPU runs never load TorchTPU testing
    # hooks; there is no TorchTPU eager state to reset on those devices.
    from torch_tpu._internal import testing as tt_testing  # pylint: disable=g-import-not-at-top

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


def _write_metrics(path: str, metrics: metrics_lib.PerformanceMetrics) -> None:
  """Writes `metrics` to `path` for `_read_metrics`."""
  # Write then rename, so that the launcher never reads a partial file if the
  # rank is killed while writing.
  with open(f"{path}.tmp", "wb") as f:
    pickle.dump(metrics, f)
  os.replace(f"{path}.tmp", path)


def _read_metrics(path: str) -> metrics_lib.PerformanceMetrics | None:
  """Returns the metrics `_write_metrics` wrote to `path`, or None if none."""
  if not os.path.exists(path):
    return None
  with open(path, "rb") as f:
    return pickle.load(f)


def _run_single_target_task(
    spec_name: str,
    mode_value: str,
    platform_name: str,
    skip_reason_file: str | None = None,
    metrics_file: str | None = None,
) -> None:
  """Worker task executed on each rank/host for a single benchmark target.

  Args:
    spec_name: Registered benchmark name.
    mode_value: `common.RunMode` value.
    platform_name: `target_lib.Platform` value.
    skip_reason_file: If set, rank 0 writes the reason here when the benchmark
      factory raises `UnsupportedBenchmark`, so the launcher reports a skip.
    metrics_file: If set, rank 0 writes the validated metrics here (see
      `_write_metrics`) once every rank has measured. The launcher exports them,
      so that each target has exactly one record.
  """
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
  skip_reason = _get_skip_reason(spec, mode, target.device_kind)
  if skip_reason is not None:
    logging.info(
        "Skipping benchmark %s on mode %s: %s",
        spec.name,
        mode.value,
        skip_reason,
    )
    return
  backend = _BACKEND_BY_DEVICE_KIND[target.device_kind]
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
    ctx = registry_lib.resolve_context(
        spec,
        context_lib.Context(
            target=target, run_scope=context_lib.RUN_SCOPE.value
        ),
    )

    with mode_lib.run_mode_context(mode, target):
      # Build run step
      common.seed_rngs()
      logging.info("Rank %s calling spec.factory...", rank)
      factory_res = spec.factory(ctx)
      logging.info("Rank %s spec.factory returned.", rank)
      # The ranks of a replica (e.g. TP) share its inputs, so report each
      # device's share.
      step_work = None
      if ctx.workload is not None:
        world_size = torch.distributed.get_world_size()
        devices_per_replica = spec.devices_per_replica(world_size)
        if devices_per_replica < 1 or world_size % devices_per_replica:
          raise ValueError(
              f"{spec.name} declares {devices_per_replica} devices per replica,"
              f" which must be >= 1 and divide the world size {world_size}."
          )
        step_work = workload_lib.step_work(
            ctx.workload,
            spec.stepper,
            spec.stepper_kwargs,
            devices_per_replica=devices_per_replica,
        )
      stepper = step_lib.resolve_stepper(spec.stepper, **spec.stepper_kwargs)
      stepper.init_with_benchmark_args(*factory_res)

      # Apply compile if needed
      if common.is_torch_compile(mode):
        compile_config = spec.compile_config or compile_lib.CompileConfig()
        stepper.compile(compile_config, ctx.target)

      # Measure. Only rank 0 collects xprof and reports metrics: TPU xprof
      # profiles every chip on the host from a single process, and starting
      # concurrent sessions across processes sharing a multi-core chip races on
      # the chip profiler lock. Every rank still runs the profiled steps, so all
      # ranks run the same steps and their collectives match.
      logging.info("Rank %s calling measure...", rank)
      metrics = measure_lib.measure(
          stepper,
          device_ops,
          name=f"{spec.name}_{mode.value}",
          enable_xprof=flags_lib.ENABLE_XPROF.value,
          capture_xprof=(rank == 0),
      )
      metrics.step_work = step_work
      logging.info(
          "Metrics for %s_%s on rank %s: %s",
          spec.name,
          mode.value,
          rank,
          metrics,
      )

    # Report metrics only after every rank has measured; if a peer fails,
    # mp.spawn tears down rank 0 here and the launcher reports the failure.
    torch.distributed.barrier()

    # Only rank 0 validates and reports the metrics, which the launcher then
    # exports to persistent storage (MLCompass / Sponge).
    if rank == 0:
      if (
          metrics.e2e_wall_time_seconds <= 0.0
          or metrics.first_step_time_seconds <= 0.0
      ):
        raise AssertionError(
            f"Invalid metrics measured on rank {rank}: {metrics}"
        )
      if metrics_file:
        _write_metrics(metrics_file, metrics)

  except target_lib.UnsupportedBenchmark as e:
    logging.info("Skipping unsupported benchmark on rank %s: %s", rank, e)
    # The ranks exit normally, so tell the launcher this was a skip rather than
    # a pass. Every rank runs the same factory checks; rank 0 reports for all.
    if rank == 0 and skip_reason_file:
      with open(skip_reason_file, "w") as f:
        f.write(str(e))
  except Exception as e:
    exception_raised = True
    logging.error("FATAL DISTRIBUTED WORKER ERROR: %s: %s", type(e).__name__, e)
    # The launcher in main() reports the FAILED status once per target, which
    # also covers ranks that die without reaching this handler.
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


@contextlib.contextmanager
def _scoped_env(updates: Mapping[str, str]) -> Iterator[None]:
  """Sets the given env vars, restoring their previous values on exit."""
  saved_env = {
      name: os.environ.get(name) for name in updates  # PY_ENVIRON_OK=benchmark
  }
  os.environ.update(updates)  # PY_ENVIRON_OK=benchmark
  try:
    yield
  finally:
    for name, value in saved_env.items():
      if value is None:
        os.environ.pop(name, None)  # PY_ENVIRON_OK=benchmark
      else:
        os.environ[name] = value  # PY_ENVIRON_OK=benchmark


@contextlib.contextmanager
def _owned_compilation_caches(
    nproc_per_node: int, *, tier2_cache_root: str | None = None
) -> Iterator[None]:
  """Points the workers of a target at fresh compilation caches it owns.

  The workers inherit the environment, so they compile into empty Inductor and
  Tier-2 caches instead of reusing the entries of preceding targets. Only the
  owned caches are removed afterwards: the default locations are shared with
  other jobs on the host, which may be using them. A Tier-3 cache root set by
  the user is left as is, so targets can still hit its entries.

  Args:
    nproc_per_node: Number of workers of the target on this node.
    tier2_cache_root: Directory holding the host-wide Tier-2 caches. Defaults to
      the one TorchTPU uses.

  Yields:
    None. The owned caches are active for the duration.
  """
  inductor_cache_dir = tempfile.mkdtemp(prefix="perflab_inductor_")
  owned_env = {_INDUCTOR_CACHE_ENV: inductor_cache_dir}
  # The world size the workers see, as set by
  # `distributed_utils.set_worker_env`.
  world_size = int(
      os.environ.get("WORLD_SIZE", nproc_per_node)  # PY_ENVIRON_OK=benchmark
  )
  # Mirrors GetTier2CacheName(), so that the Tier-2 cache stays enabled exactly
  # when it would be otherwise.
  if user_tier2_cache := os.environ.get(  # PY_ENVIRON_OK=benchmark
      _TIER2_CACHE_ENV
  ):
    uses_tier2_cache = user_tier2_cache != _TIER2_CACHE_DISABLED
  else:
    uses_tier2_cache = world_size > 1 or bool(
        os.environ.get(_TIER3_CACHE_ROOT_ENV)  # PY_ENVIRON_OK=benchmark
    )
  tier2_cache_dir = None
  if uses_tier2_cache:
    # A cache name set by the user is replaced too, so that the target compiles
    # cold and never deletes that cache.
    tier2_cache_name = f"perflab_{uuid.uuid4().hex}"
    owned_env[_TIER2_CACHE_ENV] = tier2_cache_name
    tier2_cache_dir = os.path.join(
        tier2_cache_root or _TIER2_CACHE_ROOT, tier2_cache_name
    )
  try:
    with _scoped_env(owned_env):
      yield
  finally:
    shutil.rmtree(inductor_cache_dir, ignore_errors=True)
    if tier2_cache_dir is not None:
      shutil.rmtree(tier2_cache_dir, ignore_errors=True)


def main(
    argv: Sequence[str],
    *,
    import_failures: Sequence[discovery_lib.ImportFailure] = _IMPORT_FAILURES,
) -> None:
  """Main entry point for discovering and executing distributed benchmarks.

  Args:
    argv: Command-line arguments.
    import_failures: The benchmark modules that failed to import, reported as a
      failed `benchmark_imports` result.
  """
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

  results: list[_TargetResult] = []
  if import_failures:
    reason = "benchmark module(s) failed to import:\n  " + "\n  ".join(
        str(f) for f in import_failures
    )
    logging.error("%s", reason)
    # Like the platform skips below, every shard reports the failure (and exits
    # non-zero), but only the first shard writes it to the test XML.
    results.append(
        _TargetResult("benchmark_imports", TargetStatus.FAILED, reason)
    )

  test_filter = os.environ.get(  # PY_ENVIRON_OK=benchmark
      "TESTBRIDGE_TEST_ONLY"
  )
  try:
    targets_to_run = resolve_benchmark_targets(test_filter)
  except ValueError:
    if not import_failures:
      raise
    # The filter may select targets of a module that failed to import, so
    # report the import failure rather than fail on the filter.
    logging.exception("Could not resolve the benchmark targets.")
    targets_to_run = []
  if not targets_to_run and not import_failures:
    raise ValueError(
        f"No matching benchmark targets found for platform {platform.value}"
        f" with filter '{test_filter}'."
    )

  runnable_targets = []
  node_rank = int(os.environ.get("NODE_RANK", "0"))  # PY_ENVIRON_OK=benchmark
  # Every shard skips the same known failures; only node 0 of the first shard
  # exports their skip records, so that each is recorded once per run.
  exports_skips = node_rank == 0 and _shard_env()[0] == 0
  for spec, mode in targets_to_run:
    skip_reason = _get_skip_reason(spec, mode, target.device_kind)
    if skip_reason is None:
      runnable_targets.append((spec, mode))
      continue
    target_name = f"{spec.name}_{mode.value}"
    logging.info(
        "Skipping benchmark %s on %s: %s",
        target_name,
        platform.value,
        skip_reason,
    )
    # Every shard reports the skipped targets, so each shard's summary is
    # complete on its own.
    results.append(
        _TargetResult(target_name, TargetStatus.SKIPPED, skip_reason)
    )
    # Like base_test, which never generates cases for the modes a platform does
    # not run, record known failures only in the modes the platform runs.
    if exports_skips and mode in mode_lib.modes_for(
        _FRAMEWORK, target.device_kind
    ):
      export_lib.export_known_failure_skip(
          spec_name=spec.name,
          platform=platform,
          framework=_FRAMEWORK,
          run_mode=mode,
      )

  # Shard after dropping skipped targets: most distributed benchmarks only run
  # compiled, so sharding the full (spec x mode) list round-robin could put all
  # runnable targets on the same shard.
  shard_targets = _shard_targets(runnable_targets)

  logging.info(
      "Launching %d of %d runnable benchmark target(s) on platform %s with"
      " nnodes=%d, local_nproc=%d",
      len(shard_targets),
      len(runnable_targets),
      platform.value,
      nnodes,
      local_nproc,
  )

  # `results` so far only holds the results every shard repeats: the import
  # failure and the targets every shard skips for the platform.
  num_shared_results = len(results)
  # Rank 0 writes a target's skip reason here if its factory raises
  # `UnsupportedBenchmark` at run time, and its metrics once every rank has
  # measured. This is a directory on the launcher's host, so the launcher of
  # node 0 reads what global rank 0 writes only because torchrun and `dist_run`
  # assign global ranks node by node: global rank 0 always runs on node 0.
  skip_dir = tempfile.mkdtemp(prefix="distributed_runner_skips_")
  for idx, (spec, mode) in enumerate(shard_targets):
    target_name = f"{spec.name}_{mode.value}"
    logging.info(
        "Executing batched target %d/%d: %s",
        idx + 1,
        len(shard_targets),
        target_name,
    )
    skip_reason_file = os.path.join(skip_dir, f"{idx}.txt")
    metrics_file = os.path.join(skip_dir, f"{idx}.metrics.pkl")
    start = time.monotonic()
    # Isolate failures per target: `mp.spawn` raises `ProcessRaisedException`
    # or `ProcessExitedException` after terminating the surviving ranks, so a
    # broken target must not abort the remaining ones.
    try:
      with _owned_compilation_caches(local_nproc):
        distributed_utils.dist_run(
            local_nproc,
            _run_single_target_task,
            spec.name,
            mode.value,
            platform.value,
            skip_reason_file,
            metrics_file,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Benchmark target %s FAILED.", target_name)
      results.append(
          _TargetResult(
              target_name,
              TargetStatus.FAILED,
              _failure_reason(e),
              time.monotonic() - start,
          )
      )
      # The launcher, not the worker, exports every result, so each target has
      # exactly one record, and failures that never reach the worker's
      # exception handler (e.g. a rank killed by a signal) are still visible on
      # dashboards. Only node 0 exports to avoid duplicate records in
      # multi-host runs.
      #
      # If rank 0 wrote metrics, every rank measured and the job failed only
      # afterwards, e.g. in teardown. The metrics are then valid, so the
      # dashboard records them as a success, while the run summary, the test
      # XML and the exit code still report the target as FAILED: that is where
      # harness breakage such as a failed teardown is triaged, and a failed
      # record would drop a valid measurement from the dashboards.
      if node_rank == 0:
        metrics = _read_metrics(metrics_file)
        export_lib.export(
            export_lib.make_benchmark_data(
                spec,
                platform=platform,
                framework=_FRAMEWORK,
                run_mode=mode,
                succeeded=metrics is not None,
                metrics=metrics,
            )
        )
    else:
      duration = time.monotonic() - start
      if os.path.exists(skip_reason_file):
        with open(skip_reason_file) as f:
          reason = f.read()
        logging.info("Benchmark target %s skipped: %s", target_name, reason)
        results.append(
            _TargetResult(target_name, TargetStatus.SKIPPED, reason, duration)
        )
        if node_rank == 0:
          export_lib.export_skip(
              spec_name=spec.name,
              platform=platform,
              framework=_FRAMEWORK,
              run_mode=mode,
          )
      else:
        results.append(
            _TargetResult(target_name, TargetStatus.PASSED, "", duration)
        )
        if node_rank == 0:
          if (metrics := _read_metrics(metrics_file)) is not None:
            export_lib.export(
                export_lib.make_benchmark_data(
                    spec,
                    platform=platform,
                    framework=_FRAMEWORK,
                    run_mode=mode,
                    succeeded=True,
                    metrics=metrics,
                )
            )
          else:
            logging.warning(
                "Benchmark target %s passed but global rank 0 wrote no"
                " metrics on node 0, so no result is exported.",
                target_name,
            )
  shutil.rmtree(skip_dir, ignore_errors=True)

  logging.info("%s", _format_summary(platform, results))
  if xml_output_file := os.environ.get(  # PY_ENVIRON_OK=benchmark
      "XML_OUTPUT_FILE"
  ):
    # Sponge merges the reports of all shards, so only the first shard reports
    # the results every shard repeats; otherwise each would be listed once per
    # shard.
    first_shard = _shard_env()[0] == 0
    _write_test_xml(
        xml_output_file,
        platform,
        (results[:num_shared_results] if first_shard else [])
        + results[num_shared_results:],
    )
  if any(r.status == TargetStatus.FAILED for r in results):
    sys.exit(1)


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  multiprocessing.handle_main(main)
