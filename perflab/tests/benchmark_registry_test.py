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

"""Tests that the BAP benchmark registry matches the benchmark harness.

BAP runs the benchmarks of perflab/benchmark_registry.pbtxt on GCP and ingests
only the TensorBoard scalars that match a metric pattern of the benchmark, so a
metric that no pattern matches never reaches the benchmark dashboards.
"""

from collections.abc import Set
import dataclasses
import os
import re

from absl.testing import absltest
from absl.testing import flagsaver
from perflab import export as export_lib
from perflab import measure as measure_lib
from perflab import target as target_lib
from perflab import workload as workload_lib
from tests import repo_paths
from tests import seed_test_utils

_REGISTRY_PATH = "perflab/benchmark_registry.pbtxt"
_TORCH_BENCHMARK_SCRIPT = "perflab/torch_benchmark_test.py"
_LAYER_BENCHMARK_SCRIPT = "perflab/torch_layer_benchmark_test.py"
_DISTRIBUTED_RUNNER_SCRIPT = "perflab/distributed_runner.py"

# The scripts that run every case on a single device, and the name pattern of
# their shards, whose groups are the shard index and the number of shards.
# Layer shards are not named `torch_*`: the MLCompass regression alerts select
# benchmarks by that prefix.
_SINGLE_DEVICE_SHARD_NAME_RES = {
    _TORCH_BENCHMARK_SCRIPT: re.compile(
        r"^torch_benchmarks_shard_(\d+)_of_(\d+)$"
    ),
    _LAYER_BENCHMARK_SCRIPT: re.compile(
        r"^layer_benchmarks_shard_(\d+)_of_(\d+)$"
    ),
}

# The OSS export adapter writes these next to the metrics of a benchmark case,
# and a batch sweep case that runs out of memory exports `out_of_memory`.
_STATUS_METRICS = (
    "wall_time",
    "succeeded",
    "skipped",
    export_lib.OUT_OF_MEMORY_METRIC,
)

# An example test name, under which the OSS export adapter writes the metrics
# of a case: torch_tpu/<platform>/<benchmark>/<run mode>/<framework>.
_TEST_NAME = "torch_tpu/v7_1x1x1/model_forward/compiled/torch"

# The torch_benchmark_test flag that makes it count the model FLOPs of each case
# and export the FLOPs metrics.
_COUNT_FLOPS_FLAG = "--count_flops"

_NAME_RE = re.compile(r'^  name: "([^"]+)"$')
_PATTERN_RE = re.compile(r'^  metrics \{ pattern: "([^"]+)"')
_ACTION_INPUT_RE = re.compile(
    r'^\s+(?:workload_)?action_inputs \{ key: "(\w+)" value: "([^"]*)" \}$'
)
_ENVIRONMENT_RE = re.compile(r'^    id: "([^"]+)"$')


@dataclasses.dataclass
class _Benchmark:
  """The parts of a registry `benchmarks` entry that the tests check."""

  name: str = ""
  script: str = ""
  runtime_flags: list[str] = dataclasses.field(default_factory=list)
  env_vars: dict[str, str] = dataclasses.field(default_factory=dict)
  # Environment id -> hardware-specific environment variables.
  env_vars_hw: dict[str, dict[str, str]] = dataclasses.field(
      default_factory=dict
  )
  metric_patterns: list[str] = dataclasses.field(default_factory=list)


def _parse_env_vars(value: str) -> dict[str, str]:
  return dict(pair.split("=", 1) for pair in value.split(",") if pair)


def _parse_registry(text: str) -> list[_Benchmark]:
  """Parses the registry, whose entries keep one field per line."""
  benchmarks = []
  environment = None
  for line in text.splitlines():
    if line == "benchmarks {":
      benchmarks.append(_Benchmark())
      continue
    if not benchmarks:
      continue
    benchmark = benchmarks[-1]
    if match := _NAME_RE.match(line):
      benchmark.name = match.group(1)
    elif match := _PATTERN_RE.match(line):
      benchmark.metric_patterns.append(match.group(1))
    elif match := _ENVIRONMENT_RE.match(line):
      environment = match.group(1)
    elif match := _ACTION_INPUT_RE.match(line):
      key, value = match.groups()
      if key == "script":
        benchmark.script = value
      elif key == "runtime_flags":
        benchmark.runtime_flags = value.split()
      elif key == "env_vars":
        benchmark.env_vars = _parse_env_vars(value)
      elif key == "env_vars_hw":
        benchmark.env_vars_hw[environment] = _parse_env_vars(value)
  return benchmarks


class _Stepper:
  """Stepper whose step does nothing."""

  def get_step_fn(self):
    return lambda: None

  def pre_warmup_init(self):
    pass

  def post_warmup_hook(self):
    pass


class _DeviceOps:
  """DeviceOps of a device that reports its peak memory but is not profiled."""

  def await_result(self, out):
    del out

  def check_result(self, out):
    del out

  def reset_peak_memory(self):
    pass

  def peak_memory_mb(self):
    return 1.0

  def supports_event_device_time(self) -> bool:
    return True

  def measure_step_device_time(self, step_fn):
    out = step_fn()
    self.await_result(out)
    self.check_result(out)
    return 0.5

  def compile_count(self):
    return 0


def _exported_metrics(count_flops: bool = False) -> Set[str]:
  """Returns the metrics that a successful GCP benchmark case exports.

  Args:
    count_flops: Whether the case is run with `--count_flops`, which also
      exports the FLOPs metrics of the case.
  """
  with flagsaver.flagsaver(
      min_warmup_steps=2, max_warmup_steps=2, post_warmup_steps=1
  ):
    metrics = measure_lib.measure(
        _Stepper(), _DeviceOps(), name="registry_test", enable_xprof=False
    )
  # As the benchmark binaries set it for a case whose workload is known.
  metrics.step_work = workload_lib.StepWork(
      workload=workload_lib.Workload(batch_size=2, seq_len=8),
      samples_per_step=2,
      tokens_per_step=16,
  )
  if count_flops:
    # As torch_benchmark_test sets them for a counted case on a device whose
    # peak FLOP/s are known.
    metrics.model_flops_per_step = 1e12
    metrics.peak_flops_per_second = 1e15
  return set(metrics.metric_map()) | set(_STATUS_METRICS)


class BenchmarkRegistryTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    path = os.path.join(repo_paths.get_repo_root_in_runfiles(), _REGISTRY_PATH)
    with open(path, encoding="utf-8") as f:
      cls.benchmarks = _parse_registry(f.read())

  def test_registry_is_parsed(self):
    self.assertNotEmpty(self.benchmarks)
    for benchmark in self.benchmarks:
      with self.subTest(benchmark=benchmark.name):
        self.assertTrue(benchmark.name)
        self.assertIn(
            benchmark.script,
            (
                _TORCH_BENCHMARK_SCRIPT,
                _LAYER_BENCHMARK_SCRIPT,
                _DISTRIBUTED_RUNNER_SCRIPT,
            ),
        )
        self.assertNotEmpty(benchmark.env_vars_hw)
        self.assertNotEmpty(benchmark.metric_patterns)

  def test_every_exported_metric_is_ingested(self):
    exported_metrics = _exported_metrics()
    for benchmark in self.benchmarks:
      for metric in sorted(exported_metrics):
        with self.subTest(benchmark=benchmark.name, metric=metric):
          tag = f"{_TEST_NAME}/{metric}"
          self.assertTrue(
              any(re.match(p, tag) for p in benchmark.metric_patterns),
              msg=f"no metric pattern of {benchmark.name} matches {tag}",
          )

  def test_every_flops_metric_is_ingested(self):
    flops_metrics = _exported_metrics(count_flops=True) - _exported_metrics()
    self.assertNotEmpty(flops_metrics)
    counted = [
        b for b in self.benchmarks if _COUNT_FLOPS_FLAG in b.runtime_flags
    ]
    self.assertNotEmpty(counted)
    for benchmark in counted:
      with self.subTest(benchmark=benchmark.name):
        # Only torch_benchmark_test counts FLOPs.
        self.assertEqual(benchmark.script, _TORCH_BENCHMARK_SCRIPT)
      for metric in sorted(flops_metrics):
        with self.subTest(benchmark=benchmark.name, metric=metric):
          tag = f"{_TEST_NAME}/{metric}"
          self.assertTrue(
              any(re.match(p, tag) for p in benchmark.metric_patterns),
              msg=f"no metric pattern of {benchmark.name} matches {tag}",
          )

  def test_flops_metrics_are_ingested_only_when_counted(self):
    flops_metrics = _exported_metrics(count_flops=True) - _exported_metrics()
    for benchmark in self.benchmarks:
      if _COUNT_FLOPS_FLAG in benchmark.runtime_flags:
        continue
      for metric in sorted(flops_metrics):
        with self.subTest(benchmark=benchmark.name, metric=metric):
          tag = f"{_TEST_NAME}/{metric}"
          self.assertFalse(
              any(re.match(p, tag) for p in benchmark.metric_patterns),
              msg=f"{benchmark.name} does not count FLOPs but ingests {tag}",
          )

  def _assert_shards_cover_every_index_once(self, script: str):
    """Checks that the shards of `script` run each of its cases exactly once."""
    shard_name_re = _SINGLE_DEVICE_SHARD_NAME_RES[script]
    shards = [b for b in self.benchmarks if b.script == script]
    self.assertNotEmpty(shards)
    total_shards = {int(b.env_vars["TEST_TOTAL_SHARDS"]) for b in shards}
    self.assertLen(total_shards, 1)
    (total,) = total_shards
    self.assertCountEqual(
        [int(b.env_vars["TEST_SHARD_INDEX"]) for b in shards], range(total)
    )
    for benchmark in shards:
      with self.subTest(benchmark=benchmark.name):
        match = shard_name_re.match(benchmark.name)
        self.assertIsNotNone(match)
        self.assertEqual(
            (int(match.group(1)), int(match.group(2))),
            (int(benchmark.env_vars["TEST_SHARD_INDEX"]), total),
        )

  def test_torch_benchmark_shards_cover_every_index_once(self):
    self._assert_shards_cover_every_index_once(_TORCH_BENCHMARK_SCRIPT)

  def test_layer_benchmark_shards_cover_every_index_once(self):
    self._assert_shards_cover_every_index_once(_LAYER_BENCHMARK_SCRIPT)

  def test_platforms_match_the_script(self):
    for benchmark in self.benchmarks:
      for environment, env_vars in benchmark.env_vars_hw.items():
        with self.subTest(benchmark=benchmark.name, environment=environment):
          platform = target_lib.Platform(env_vars["BENCHMARK_PLATFORM"])
          topology = target_lib.make_target(platform).platform_spec.topology
          num_devices = topology.nnodes * topology.nprocs_per_node
          if benchmark.script in _SINGLE_DEVICE_SHARD_NAME_RES:
            # torch_benchmark_test and torch_layer_benchmark_test run every
            # case on a single device.
            self.assertEqual(num_devices, 1)
          else:
            # distributed_runner, the only other script that
            # test_registry_is_parsed allows, runs on several devices.
            self.assertGreater(num_devices, 1)


if __name__ == "__main__":
  absltest.main()
