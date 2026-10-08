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

"""Provides logic to resolve benchmark cases."""

import hashlib
import os
from typing import Iterator, Tuple

from perflab import common
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import target as target_lib


def _case_order_key(name: str, seed: str) -> str:
  """Returns a deterministic hash key for ordering benchmark cases."""
  payload = f"{seed}:{name}" if seed else name
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cases(
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    *,
    order_seed: str | None = None,
) -> Iterator[Tuple[str, registry_lib.BenchmarkSpec, common.RunMode]]:
  """One case per (benchmark, applicable mode).

  The mode matrix is resolved here, at collection, so generated cases match the
  cell exactly: a CPU run never emits an eager_optimized case just to skip it.
  Ordering uses a deterministic hash (seeded by TORCH_TPU_BENCHMARK_ORDER_SEED
  when set) so all shards agree on the partition without alphabetical tail bias.

  Args:
    platform: Target hardware platform on which the benchmarks will run.
    framework: Execution framework (e.g. TORCH or TORCHAX) filtering modes.
    order_seed: Seed string used for deterministic benchmark ordering; falls
      back to the TORCH_TPU_BENCHMARK_ORDER_SEED environment variable if None.

  Yields:
    Tuples of (case_name, spec, run_mode) in deterministic hashed order.
  """
  if order_seed is None:
    order_seed = os.environ.get(  # PY_ENVIRON_OK=benchmark
        "TORCH_TPU_BENCHMARK_ORDER_SEED", ""
    )
  target_kind = target_lib.make_target(platform).device_kind
  ordered_names = sorted(
      registry_lib.REGISTRY,
      key=lambda name: _case_order_key(name, order_seed),
  )
  for name in ordered_names:
    spec = registry_lib.REGISTRY[name]
    for mode in mode_lib.modes_for(framework, target_kind):
      yield f"{spec.name}_{mode.value}", spec, mode


def get_skip_reason(
    spec: registry_lib.BenchmarkSpec,
    mode: common.RunMode,
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
) -> str | None:
  """Returns why a case should be skipped, or None if it should run.

  A case is skipped if it matches an entry in `known_failures.KNOWN_FAILURES` or
  if `spec.skipped_run_modes` lists `mode`.
  """
  failure_reason = known_failures.get_known_failure_reason(
      spec.name, framework=framework, device_kind=device_kind, run_mode=mode
  )
  if failure_reason is not None:
    return f"Benchmark {spec.name} is known to fail: {failure_reason}"
  if mode.value in spec.skipped_run_modes:
    return f"Benchmark {spec.name} explicitly skips run mode {mode.value}"
  return None
