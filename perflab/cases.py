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

from typing import Iterable, Iterator, Tuple

from perflab import common
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import target as target_lib


def get_cases(
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
) -> Iterator[Tuple[str, registry_lib.BenchmarkSpec, common.RunMode]]:
  """One case per (benchmark, applicable mode).

  The mode matrix is resolved here, at collection, so generated cases match the
  cell exactly: a CPU run never emits an eager_optimized case just to skip it.
  """
  target_kind = target_lib.make_target(platform).device_kind
  for name in sorted(registry_lib.REGISTRY):
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


def get_selected_cases(
    case_params: Iterable[tuple[registry_lib.BenchmarkSpec, common.RunMode]],
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
    skip_behavior: str,
) -> list[tuple[registry_lib.BenchmarkSpec, common.RunMode]]:
  """Returns the (spec, mode) cases that a run with `skip_behavior` executes.

  Applies `--skip_behavior` to the reasons from `get_skip_reason` the same way
  the benchmark binaries do: `skip` drops the skipped cases, `run_skipped` keeps
  only them, and `ignore` and `assert_raise` keep every case. Skips that are
  only known at run time (e.g. `UnsupportedBenchmark`) are not accounted for.

  Args:
    case_params: The (spec, mode) pairs generated for the binary.
    framework: The framework the binary measures.
    device_kind: The device kind the binary targets.
    skip_behavior: The value of `--skip_behavior`.

  Returns:
    The selected (spec, mode) pairs, in input order.
  """
  selected = []
  for spec, mode in case_params:
    skipped = (
        get_skip_reason(
            spec, mode, framework=framework, device_kind=device_kind
        )
        is not None
    )
    if skip_behavior == "skip" and skipped:
      continue
    if skip_behavior == "run_skipped" and not skipped:
      continue
    selected.append((spec, mode))
  return selected
