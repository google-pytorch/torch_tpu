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
import os
import queue
import sys
from typing import Any, Sequence

from absl import flags
from absl import logging
from absl.testing import parameterized
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

USE_SUBPROCESS = flags.DEFINE_bool(
    "torch_tpu_internal_use_subprocess_for_benchmarks",
    False,
    "Whether to run each benchmark in a separate subprocess for isolation.",
)

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


def _run_benchmark_redirected(q: queue.Queue, **kwargs):
  class QueueWriter:

    def __init__(self, q):
      self.q = q

    def write(self, s):
      if s:
        self.q.put(s)

    def flush(self):
      pass

  qw = QueueWriter(q)
  sys.stdout = qw
  sys.stderr = qw
  performance_utils.run_benchmark(**kwargs)


class BenchmarkTest(parameterized.TestCase):
  """Tests for end-to-end performance benchmarks."""

  @classmethod
  def _is_torchax_backend(cls) -> bool:
    return benchmark_utils.BACKEND.value == benchmark_utils.Backend.TORCHAX

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls._is_torchax_backend():
      # Import TorchAX specific libraries only if the backend is TorchAX
      # otherwise we will attempt to overwrite privatedeviceuse1 backend.
      import torchax

      torchax.enable_globally()

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

    if performance_utils.BOUNDED_DYNAMIC.value:
      if not benchmark_utils.is_torch_compile(config.run_mode):
        self.skipTest(
            "Only compiled mode benchmarks with bounded dynamic shapes is"
            " supported."
        )
      if config.is_training:
        self.skipTest("Training is not supported under bounded dynamic mode.")

      benchmark_name = f"{benchmark_name}_bounded_dynamic"
      config.model_and_input_args.is_bounded_dynamic = True
      if config.model_and_input_args.sequence_length is not None:
        seq_len = config.model_and_input_args.sequence_length
        config.model_and_input_args.sequence_length = (
            model_utils.DynamicDimension(
                min_value=max(2, seq_len // 2), max_value=seq_len * 2
            )
        )

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

    if (
        self._is_torchax_backend()
        and config.run_mode != benchmark_utils.RunMode.COMPILED
    ):
      self.skipTest("TorchAX only supports compiled run mode.")

    if self._is_torchax_backend() and platform in (
        benchmark_utils.Platform.B200_8,
        benchmark_utils.Platform.B200_4,
        benchmark_utils.Platform.B200_1,
    ):
      self.skipTest("TorchAX is not supported for GPU benchmarks.")

    # MlCompass only looks for base test name after the benchmark run.
    base_test_name = get_base_test_name(
        self._testMethodName, microbenchmark_name
    )

    if self._is_torchax_backend():
      # Import TorchAX specific libraries only if the backend is TorchAX otherwise we
      # will attempt to overwrite privatedeviceuse1 backend.
      from examples.benchmarks.e2e.torchax import performance_utils as torchax_perf_utils

      torchax_perf_utils.run_benchmark(
          config=config,
          test_method_name=base_test_name,
          benchmark_name=f"{benchmark_name}_torchax",
          microbenchmark_name=microbenchmark_name,
      )
      return

    if USE_SUBPROCESS.value:
      logging.info("Running benchmark in a subprocess for isolation.")
      ctx = g3_multiprocessing.get_context(g3_multiprocessing.ABSL_SPAWN)
      q = ctx.Queue()
      p = ctx.Process(
          target=_run_benchmark_redirected,
          args=(q,),
          kwargs=dict(
              config=config,
              test_method_name=base_test_name,
              benchmark_name=benchmark_name,
              microbenchmark_name=microbenchmark_name,
          ),
      )
      p.start()

      while p.is_alive() or not q.empty():
        try:
          output = q.get(timeout=0.1)
          sys.stderr.write(output)
          sys.stderr.flush()
        except queue.Empty:
          continue

      p.join()
      if p.exitcode != 0:
        self.fail(f"Benchmark subprocess failed with exit code {p.exitcode}")
    else:
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
