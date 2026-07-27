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

"""Tool to list all cases registered in the benchmark suite.

This functionality isn't supported by targets natively.
"""

from absl import app
from examples.benchmarks.e2e.harness import cases
from examples.benchmarks.e2e.harness import discovery
from examples.benchmarks.e2e.harness import mode as mode_lib
from examples.benchmarks.e2e.harness import models
from examples.benchmarks.e2e.harness import target as target_lib


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  failures = discovery.import_submodules(models)
  if failures:
    for f in failures:
      print(f)
    return

  framework = mode_lib.Framework.TORCH
  platform = target_lib.platform_from_env()

  all_cases = list(cases.get_cases(platform, framework))

  run_modes = ["compiled", "eager_default", "eager_optimized"]
  skip_counts = {mode: 0 for mode in run_modes}
  unskip_counts = {mode: 0 for mode in run_modes}

  for _, spec, _ in all_cases:
    for mode in run_modes:
      if spec.skipped_run_modes and mode in spec.skipped_run_modes:
        skip_counts[mode] += 1
      else:
        unskip_counts[mode] += 1

  print("Summary")
  print("=" * 80)
  print(f"Total Benchmarks: {len(all_cases)}")
  for mode in run_modes:
    print(
        f"Mode {mode}: {unskip_counts[mode]} unskipped, {skip_counts[mode]}"
        " skipped"
    )
  print("=" * 80)
  print()

  print("Registered Benchmark Cases")
  print("=" * 80)
  for case_name, spec, _ in all_cases:
    print(f"Case Name: {case_name}")
    print(f"  Benchmark Name : {spec.name}")
    print(f"  Step Function  : {spec.step.value if spec.step else 'None'}")
    print(f"  DType          : {spec.dtype.value if spec.dtype else 'None'}")
    if spec.step_kwargs:
      print(f"  Step Kwargs    : {spec.step_kwargs}")
    if spec.compile_config:
      print(f"  Compile Config : {spec.compile_config}")
    if spec.skipped_run_modes:
      modes = ", ".join(sorted(list(spec.skipped_run_modes)))
      print(f"  Skipped Modes  : {modes}")
    print("-" * 80)


if __name__ == "__main__":
  app.run(main)
