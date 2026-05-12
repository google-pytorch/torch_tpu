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

"""Global registry for end-to-end benchmarks separated by silos."""

import dataclasses
from typing import Any, Callable, Iterator

from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils


@dataclasses.dataclass
class RegisteredBenchmarkItem:
  """An item yielded by a registered benchmark generator.

  Attributes:
    testcase_name: The name of the test method to be added to the TestCase. Must
      be unique within the TestCase. Should ideally start with 'test_'.
    config: The PerformanceBenchmarkConfig to run.
    benchmark_name: The name of the benchmark (used for MLCompass).
  """

  testcase_name: str
  config: performance_utils.PerformanceBenchmarkConfig
  benchmark_name: str


_REGISTRY: list[Callable[[], Iterator[RegisteredBenchmarkItem]]] = []


def register_benchmark(
    benchmark_name: str,
    run_modes: list[benchmark_utils.RunMode],
):
  """Decorator to register a benchmark function across multiple run modes.

  The decorated function should accept a single 'run_mode' argument and return
  a PerformanceBenchmarkConfig.

  Args:
    benchmark_name: The name of the benchmark (used for MLCompass).
    run_modes: List of run modes to generate tests for.
  """

  def decorator(
      func: Callable[
          [benchmark_utils.RunMode],
          performance_utils.PerformanceBenchmarkConfig,
      ],
  ):
    def generator() -> Iterator[RegisteredBenchmarkItem]:
      for params in test_utils.generate_run_mode_configs(run_modes):
        run_mode = params["run_mode"]
        run_mode_name = params["testcase_name"]

        config = func(run_mode)

        base_name = getattr(func, "__name__")
        assert base_name.startswith(
            "test_"
        ), f"Benchmark function name must start with 'test_', got {base_name}"
        testcase_name = f"{base_name}_{run_mode_name}"

        yield RegisteredBenchmarkItem(
            testcase_name=testcase_name,
            config=config,
            benchmark_name=benchmark_name,
        )

    _REGISTRY.append(generator)
    return func

  return decorator


def populate_standalone_test_class(test_class: Any):
  """Populates a test class with all registered benchmarks."""
  for gen_func in _REGISTRY:
    for item in gen_func():
      # Capture item in a closure
      def make_test_method(benchmark_item):
        def test_method(self):
          self.run_performance_benchmark_test(
              benchmark_item.config,
              benchmark_item.benchmark_name,
          )

        return test_method

      method = make_test_method(item)
      method.__name__ = item.testcase_name
      setattr(test_class, item.testcase_name, method)
