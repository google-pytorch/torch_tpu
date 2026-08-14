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
import errno
import os
import queue
import sys
from typing import Any, Sequence

from absl import flags
from absl import logging
from absl.testing import parameterized
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import common
from examples.benchmarks.e2e import mlcompass_utils
from examples.benchmarks.e2e import model_configs
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils
from torch_tpu._internal.distributed import multiprocessing

USE_SUBPROCESS = flags.DEFINE_bool(
    "torch_tpu_internal_use_subprocess_for_benchmarks",
    False,
    "Whether to run each benchmark in a separate subprocess for isolation.",
)

_DRY_RUN = flags.DEFINE_bool(
    "dry_run",
    False,
    "Whether to run in dry-run mode. If True, actual benchmark execution is"
    " skipped, and the list of targets that would have run are logged and"
    " written to the test outputs directory specified by environment variable"
    " TEST_UNDECLARED_OUTPUTS_DIR.",
)
_DRY_RUN_OUTPUT_FILE = "mlcompass_test_targets.txt"

# CUDA only has EAGER and COMPILED run modes. Other run modes are applicable to
# TPU only.
CUDA_RUN_MODES = (
    common.RunMode.EAGER_DEFAULT,
    common.RunMode.COMPILED,
)


def get_base_test_name(
    test_method_name: str, microbenchmark_name: str | None
) -> str:
  """Returns the base test name by removing the microbenchmark name suffix."""
  if not microbenchmark_name:
    return test_method_name
  return test_method_name.removesuffix(f"_{microbenchmark_name}")


def _get_output_dir(key: str) -> str:
  try:
    return os.environ[key]
  except KeyError:
    raise RuntimeError(
        f"Output directory not set using environment variable {key}"
    )


def _dry_run_test(
    platform: common.Platform,
    base_test_name: str,
    benchmark_name: str,
) -> None:
  """Logs and writes dry run test details to a text file."""
  logging.info(
      "[DRY RUN]: Would have run benchmark with name %s and test method"
      " name %s",
      benchmark_name,
      base_test_name,
  )
  output_dir = _get_output_dir("TEST_UNDECLARED_OUTPUTS_DIR")
  txt_path = os.path.join(output_dir, _DRY_RUN_OUTPUT_FILE)
  try:
    os.makedirs(os.path.dirname(txt_path))
  except OSError as ex:
    if ex.errno != errno.EEXIST:
      raise
  with open(txt_path, mode="a", encoding="utf-8") as f:
    mlcompass_test_name = mlcompass_utils.get_mlcompass_test_name(
        platform, base_test_name, benchmark_name
    )
    f.write(mlcompass_test_name + "\n")
    print(f"[DRY RUN] mlcompass test name: {mlcompass_test_name}", flush=True)


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
    return common.BACKEND.value == common.Backend.TORCHAX

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls._is_torchax_backend():
      # Import TorchAX specific libraries only if the backend is TorchAX
      # otherwise we will attempt to overwrite privatedeviceuse1 backend.
      import torchax

      torchax.enable_globally()
      torchax.enable_performance_mode()

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
      if not common.is_torch_compile(config.run_mode):
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
                min_value=max(2, seq_len // 2), max_value=seq_len * 2  # pyrefly: ignore[unsupported-operation]
            )
        )

    platform = common.PLATFORM.value
    if platform not in config.supported_platforms:
      self.skipTest(
          f"Platform {common.PLATFORM.value} not in"
          f" {config.supported_platforms}"
      )
    if (
        platform
        in (
            common.Platform.B200_8,
            common.Platform.B200_4,
            common.Platform.B200_1,
        )
        and config.run_mode not in CUDA_RUN_MODES
    ):
      self.skipTest(
          f"Run mode {config.run_mode} not applicable to platform {platform}"
      )

    if (
        self._is_torchax_backend()
        and config.run_mode != common.RunMode.COMPILED
    ):
      self.skipTest("TorchAX only supports compiled run mode.")

    if self._is_torchax_backend() and platform in (
        common.Platform.B200_8,
        common.Platform.B200_4,
        common.Platform.B200_1,
    ):
      self.skipTest("TorchAX is not supported for GPU benchmarks.")

    # MlCompass only looks for base test name after the benchmark run.
    base_test_name = get_base_test_name(
        self._testMethodName, microbenchmark_name
    )
    benchmark_name = (
        f"{benchmark_name}_torchax"
        if self._is_torchax_backend()
        else benchmark_name
    )

    if _DRY_RUN.value:
      _dry_run_test(platform, base_test_name, benchmark_name)
      return

    if self._is_torchax_backend():
      # Import TorchAX specific libraries only if the backend is TorchAX otherwise we
      # will attempt to overwrite privatedeviceuse1 backend.
      from examples.benchmarks.e2e.torchax import performance_utils as torchax_perf_utils

      torchax_perf_utils.run_benchmark(
          config=config,
          test_method_name=base_test_name,
          benchmark_name=benchmark_name,
          microbenchmark_name=microbenchmark_name,
      )
      return

    if USE_SUBPROCESS.value:
      logging.info("Running benchmark in a subprocess for isolation.")
      ctx = multiprocessing.get_context("spawn")
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
    run_modes: Sequence[common.RunMode] = (
        common.RunMode.EAGER_DEFAULT,
        common.RunMode.EAGER_OPTIMIZED,
        common.RunMode.COMPILED,
    ),
    exclude_run_modes: Sequence[common.RunMode] | None = None,
):
  """Generates test parameters from a list of run modes.

  Args:
    run_modes: The run modes to generate test parameters for.
    exclude_run_modes: Run modes to exclude from generation.

  Yields:
    A dictionary containing the test parameters.
  """
  if exclude_run_modes is not None:
    run_modes = [rm for rm in run_modes if rm not in exclude_run_modes]
  for run_mode in run_modes:
    yield dict(
        testcase_name=run_mode.value,
        run_mode=run_mode,
    )


def generate_model_test_configs(
    model_configs_list: Sequence[model_configs.ModelConfig],
    run_modes: Sequence[common.RunMode] = (
        common.RunMode.EAGER_DEFAULT,
        common.RunMode.EAGER_OPTIMIZED,
        common.RunMode.COMPILED,
    ),
    exclude_run_modes: Sequence[common.RunMode] | None = None,
):
  """Generates model test configs for parameterized tests.

  Args:
    model_configs_list: The model configs to generate test parameters for.
    run_modes: The run modes to generate test parameters for.
    exclude_run_modes: Run modes to exclude from generation.

  Yields:
    A dictionary containing the test parameters.
  """
  if exclude_run_modes is not None:
    run_modes = [rm for rm in run_modes if rm not in exclude_run_modes]
  for run_mode in run_modes:
    for model_config in model_configs_list:
      microbenchmark_name = get_microbenchmark_name(model_config)
      testcase_name = f"{run_mode.value}_{microbenchmark_name}"

      yield dict(
          testcase_name=testcase_name,
          run_mode=run_mode,
          model_config=model_config,
      )
