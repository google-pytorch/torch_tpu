# Copyright 2025 Google LLC
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

"""Utilities for testing StableHLO compilation speed."""

import time
from typing import Callable

from absl import flags
from absl import logging
from torch_tpu.csrc.internal import testing as tt_testing
from torch_tpu.csrc.internal.compile import tpu_torch_compile
from tests import seed_test_utils

_NUM_RUNS = flags.DEFINE_integer(
    "num_runs", 10, "Number of runs for each benchmark."
)


class StableHloCompileTimeTestBase(seed_test_utils.RepeatableTest):
  """StableHLO compilation benchmarks."""

  # Tell the type checker that these attributes exist.
  _preheat_xla_done: bool

  @classmethod
  def setUpClass(cls):
    """Initializes the TPU runtime for all tests."""
    super().setUpClass()
    cls._preheat_xla_done = False  # Whether XLA has been preheated.

  def setUp(self):
    """Initializes the TPU runtime for each test."""
    super().setUp()
    tt_testing.reset_eager_state()
    if not self._preheat_xla_done:
      self._preheat_xla()
      self._preheat_xla_done = True

  def get_compilation_time(
      self, stablehlo_str: str, eager: bool = True
  ) -> float:
    """Serializes a StableHLO module and benchmarks the module's compilation.

    This helper function orchestrates the end-to-end process of compiling an
    MLIR module and measuring its performance. It performs the following steps:

    1.  Parses the input "stablehlo_str" using the `parse_mlir_text` C++
        binding. This step performs full syntax parsing and semantic
        verification of the MLIR.
    2.  Submits the resulting MLIR module to the `compile_mlir` C++
        binding for compilation.
    3.  Measures the wall-clock time of the `compile_mlir` call.
    4.  Verifies success by asserting that compilation returns a non-None
        PjRtLoadedExecutable.

    Args:
      stablehlo_str: A string containing a complete StableHLO MLIR module.
      eager: If true, use the compiler profile optimized for eager execution;
        otherwise, use the profile optimized for torch.compile.

    Returns:
      The elapsed time in seconds for the compilation.

    Raises:
      AssertionError: If serialization or compilation fails, or if the
      compilation returns None.
    """
    logging.info("SHLO String (text format): %d bytes", len(stablehlo_str))

    try:
      mlir_module = tpu_torch_compile.parse_mlir_text(stablehlo_str)
    except RuntimeError as e:
      self.fail(
          "MLIR parsing failed with an error: "
          f"{e}\n--- Input MLIR String ---\n{stablehlo_str}"
      )

    logging.info("MLIR parsing successful.")

    try:
      start_time = time.monotonic()
      # compile() is synchronous, so the elapsed time includes the
      # compilation time.
      executable = tpu_torch_compile.compile_mlir(
          mlir_module, fast_compile=eager
      )
      elapsed_time = time.monotonic() - start_time
    except RuntimeError as e:
      self.fail(
          "MLIR compilation failed with an error: "
          f"{e}\n--- Input MLIR String ---\n{stablehlo_str}"
      )

    logging.info("Compile time: %.4f seconds", elapsed_time)
    self.assertIsNotNone(executable)
    return elapsed_time

  def do_test(
      self, bm_name: str, make_shlo: Callable[[int], str], eager: bool = True
  ) -> None:
    """Runs the benchmark and logs the results.

    Args:
      bm_name: The name of the benchmark.
      make_shlo: A function that takes an iteration index as input and returns a
        StableHLO module string. This allows us to vary the StableHLO module
        across different iterations in the same test case.
      eager: If true, use the compiler profile optimized for eager execution;
        otherwise, use the profile optimized for torch.compile. Defaults to True
        to match default user experience. In general, we care about the
        compilation speed more in the eager mode than in the torch.compile mode.
        Therefore it makes more sense for the benchmarks to default to eager.
    """

    num_runs = _NUM_RUNS.value
    compile_times = []
    logging.info(
        "--- Running Benchmark %s for %d trials ---", bm_name, num_runs
    )
    for i in range(_NUM_RUNS.value):
      elapsed_time = self.get_compilation_time(make_shlo(i), eager=eager)
      compile_times.append(elapsed_time)

    logging.info("Benchmark %s raw times (seconds): %s", bm_name, compile_times)

  def _preheat_xla(self) -> None:
    """Pre-compiles a trivial StableHLO module to warm up XLA.

    This ensures that the measurements in the tests are fair.
    """

    logging.info("Pre-compiling a trivial StableHLO module to warm up XLA...")
    stablehlo_str = """
      module {
        func.func @main() -> tensor<i64> {
          %c = stablehlo.constant dense<0> : tensor<i64>
          return %c : tensor<i64>
        }
      }
    """
    self.get_compilation_time(stablehlo_str)
