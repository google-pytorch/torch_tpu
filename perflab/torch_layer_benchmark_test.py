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

"""Torch layer benchmark binary.

Runs every registered layer benchmark (`perflab.layers`), once per applicable
execution mode, on the platform named by BENCHMARK_PLATFORM. Each layer
benchmark steps one layer, e.g. `nn.Linear`, at one shape, for training or
evaluation. This binary does not run the model benchmarks of
`torch_benchmark_test`; both share the test base class in `torch_base_test`.

Environment & Flags:
  BENCHMARK_PLATFORM   cpu | b200_1 | v7_1x1x1 | ...           (default: cpu)
  --run_scope          full | presubmit                        (default: full)
  --run_in_subprocess  isolate each case in a child process    (default: true)
  --fake_tensor_mode   build and run each case under FakeTensorMode, and check
                       that building it fits in the HBM of a v7 TensorCore;
                       only valid with BENCHMARK_PLATFORM=cpu  (default: false)
"""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu._internal.distributed import multiprocessing
from perflab import base_test
from perflab import cases
from perflab import common
from perflab import discovery as discovery_lib
from perflab import layers
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import steps
from perflab import target as target_lib
from perflab import torch_base_test

# Layer benchmarks run only under the torch framework.
_FRAMEWORK = mode_lib.Framework.TORCH

# Resolved once at import/collection time without acquiring device handles.
_PLATFORM = target_lib.platform_from_env()

# Import all layers and trigger registration of benchmarks.
failures = discovery_lib.import_submodules(layers)
failures += discovery_lib.import_submodules(steps)


_CASES = list(cases.get_cases(_PLATFORM, _FRAMEWORK))


class LayerBenchmarkTest(
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
      self, spec: registry_lib.BenchmarkSpec, mode: common.RunMode
  ):
    self.run_case(spec, mode)


def _test_main(*args: Any, **kwargs: Any) -> None:
  absltest.main(*args, testLoader=base_test.BenchmarkTestLoader(), **kwargs)


if __name__ == "__main__":
  multiprocessing.handle_test_main(_test_main)
