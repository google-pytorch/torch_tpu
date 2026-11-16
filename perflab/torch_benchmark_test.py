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

"""Torch benchmark binary.

Runs every registered single-host benchmark, once per applicable execution
mode, on the platform named by BENCHMARK_PLATFORM. This assembles a step
function according to the benchmark spec, resolves the run-scope, and
calls measure on the step function to get metrics.

Environment & Flags:
  BENCHMARK_PLATFORM   cpu | b200_1 | b200_8 | v6e_1x1 | ...   (default: cpu)
  --run_scope          full | presubmit                        (default: full)
  --run_in_subprocess  isolate each case in a child process    (default: true)
  --fake_tensor_mode   build and run each case under FakeTensorMode, and check
                       that building it fits in the HBM of a v7 TensorCore;
                       only valid with BENCHMARK_PLATFORM=cpu; also skips the
                       weights cache dir                       (default: false)
  --count_flops        after measuring each case, count the model FLOPs of a
                       step on a fake CPU copy and export TFLOP/s and MFU
                                                               (default: false)
  --batch_sweep        in addition to the default cases, measure each benchmark
                       of batch_sweep_benchmarks.py in each run mode of
                       --batch_sweep_run_modes at the other batch sizes of
                       --batch_sweep_sizes                  (default: false)
  --batch_sweep_only   with --batch_sweep, run only the batch sweep, including
                       the default batch sizes              (default: false)
"""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu._internal.distributed import multiprocessing
from perflab import base_test
from perflab import cases
from perflab import common
from perflab import discovery as discovery_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_base_test

# This binary IS the torch run. Not a flag -- see module docstring.
_FRAMEWORK = mode_lib.Framework.TORCH

# Resolved once at import/collection time without acquiring device handles.
_PLATFORM = target_lib.platform_from_env()

# Import all models and trigger registration of benchmarks.
failures = discovery_lib.import_submodules(models)
failures += discovery_lib.import_submodules(steps)


_BATCH_SWEEP = flags_lib.batch_sweep_flag_values()
# A batch sweep runs the default cases (unless --batch_sweep_only) and adds
# cases at more batch sizes.
_CASES: list[cases.Case] = list(cases.get_cases(_PLATFORM, _FRAMEWORK))
if _BATCH_SWEEP.enabled:
  _CASES += cases.get_batch_sweep_cases(_PLATFORM, _FRAMEWORK)


class BenchmarkTest(
    torch_base_test.BaseTorchBenchmarkTest,
    parameterized.TestCase,  # ABSLTEST_OK=benchmark
):
  """One test method, parameterised over the registry x mode matrix."""

  framework = _FRAMEWORK
  platform = _PLATFORM
  case_params = base_test.case_params(_CASES)
  import_failures = failures

  @parameterized.named_parameters(*_CASES)
  def test_benchmark(
      self,
      spec: registry_lib.BenchmarkSpec,
      mode: common.RunMode,
      batch_size: int | None = None,
  ):
    """Measures a benchmark in a run mode.

    Args:
      spec: The benchmark.
      mode: The run mode.
      batch_size: The batch size of a batch sweep case, or None to measure the
        benchmark at its own batch size.
    """
    self.run_case(spec, mode, batch_size)


def _test_main(*args: Any, **kwargs: Any) -> None:
  absltest.main(*args, testLoader=base_test.BenchmarkTestLoader(), **kwargs)


if __name__ == "__main__":
  multiprocessing.handle_test_main(_test_main)
