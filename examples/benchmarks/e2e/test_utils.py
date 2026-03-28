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

"""Utility functions for benchmarks."""

import dataclasses
from typing import Any, Sequence

from absl.testing import parameterized
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import performance_utils

# CUDA only has EAGER and COMPILED run modes. Other run modes are applicable to
# TPU only.
CUDA_RUN_MODES = (
    benchmark_utils.RunMode.EAGER_DEFAULT,
    benchmark_utils.RunMode.COMPILED,
)


def get_base_test_name(
    test_method_name: str, microbenchmark_name: str | None
) -> str:
  """Returns the base test name by removing the microbenchmark name suffix."""
  if not microbenchmark_name:
    return test_method_name
  return test_method_name.removesuffix(f"_{microbenchmark_name}")


class BenchmarkTest(parameterized.TestCase):
  """Tests for end-to-end performance benchmarks."""

  def run_performance_benchmark_test(
      self,
      config: performance_utils.PerformanceBenchmarkConfig,
      benchmark_name: str,
      microbenchmark_name: str | None = None,
  ) -> None:
    """Runs a benchmark test.

    Args:
      config: The benchmark config.
      benchmark_name: The name of the benchmark. This should match the last part
        of target in the MLCompass config file. See
        go/torchtpu-mlcompass#configuration-structure for more details.
      microbenchmark_name: This is used to export microbenchmark results to
        MLCompass. If a benchmark test is composed of multiple microbenchmarks,
        this should be set to the name of the microbenchmark. For example, when
        testing a linear layer with different tensor shapes, each one should be
        exported to MLCompass as a microbenchmark. See
        go/mlcompass-microbenchmark-guide for more details.
    """

    platform = benchmark_utils.PLATFORM.value
    if platform not in config.supported_platforms:
      self.skipTest(
          f"Platform {benchmark_utils.PLATFORM.value} not in"
          f" {config.supported_platforms}"
      )
    if (
        platform
        in (
            benchmark_utils.Platform.B200_8,
            benchmark_utils.Platform.B200_4,
            benchmark_utils.Platform.B200_1,
        )
        and config.run_mode not in CUDA_RUN_MODES
    ):
      self.skipTest(
          f"Run mode {config.run_mode} not applicable to platform {platform}"
      )

    # MlCompass only looks for base test name after the benchmark run.
    base_test_name = get_base_test_name(
        self._testMethodName, microbenchmark_name
    )

    performance_utils.run_benchmark(
        config=config,
        test_method_name=base_test_name,
        benchmark_name=benchmark_name,
        microbenchmark_name=microbenchmark_name,
    )


def get_microbenchmark_name(config_dataclass) -> str:
  """Returns a string representation of the dataclass suitable for a test name."""
  config_dict = dataclasses.asdict(config_dataclass)
  name_parts = []
  for k, v in config_dict.items():
    if k == "dtype":
      v = str(v).replace("torch.", "")
    elif isinstance(v, (tuple, list)):
      v = "x".join(map(str, v))
    name_parts.append(f"{k}_{v}")
  return "_".join(name_parts)


def generate_layer_test_configs(
    run_modes: Sequence[Any],
    is_trainings: Sequence[Any],
    layer_configs_list: Sequence[Any],
):
  """Generates layer test configs for parameterized tests.

  Args:
    run_modes: The run modes to generate test parameters for.
    is_trainings: The training modes to generate test parameters for.
    layer_configs_list: The layer configs to generate test parameters for.

  Yields:
    A dictionary containing the test parameters.
  """
  for is_training in is_trainings:
    for run_mode in run_modes:
      for layer_config in layer_configs_list:
        train_str = "train" if is_training else "eval"
        microbenchmark_name = get_microbenchmark_name(layer_config)
        testcase_name = f"{train_str}_{run_mode.value}_{microbenchmark_name}"

        yield dict(
            testcase_name=testcase_name,
            run_mode=run_mode,
            is_training=is_training,
            layer_config=layer_config,
        )


def generate_run_mode_and_train_configs(
    run_modes: Sequence[Any],
    is_training: Sequence[Any],
):
  """Generates test parameters from a list of run modes and training modes.

  Args:
    run_modes: The run modes to generate test parameters for.
    is_training: The training modes to generate test parameters for.

  Yields:
    A dictionary containing the test parameters.
  """
  for training_mode in is_training:
    for run_mode in run_modes:
      name_parts = []
      name_parts.append("train" if training_mode else "eval")
      name_parts.append(f"{run_mode.value}")
      testcase_name = "_".join(name_parts)
      yield dict(
          testcase_name=testcase_name,
          run_mode=run_mode,
          is_training=training_mode,
      )


def generate_run_mode_configs(
    run_modes: Sequence[Any],
):
  """Generates test parameters from a list of run modes.

  Args:
    run_modes: The run modes to generate test parameters for.

  Yields:
    A dictionary containing the test parameters.
  """
  for run_mode in run_modes:
    yield dict(
        testcase_name=run_mode.value,
        run_mode=run_mode,
    )
