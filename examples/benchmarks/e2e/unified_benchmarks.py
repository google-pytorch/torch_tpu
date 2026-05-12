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

"""Unified buildable unit to run benchmarks by silo."""

import sys

from absl import flags
from absl.testing import absltest
import torch.multiprocessing as mp
from examples.benchmarks.e2e import benchmark_registry

# Import registered benchmark modules here so their decorators execute.
# pylint: disable=unused-import
from examples.benchmarks.e2e import registered_model_benchmarks

# pylint: enable=unused-import
from examples.benchmarks.e2e import test_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

BENCHMARK_SILO = flags.DEFINE_string(
    "benchmark_silo",
    None,
    "The benchmark silo to run (e.g., 'model', 'layer', 'compile', 'quality').",
    required=True,
)


class UnifiedBenchmarkTest(test_utils.BenchmarkTest):
  """Dynamically populated tests for end-to-end benchmarks."""

  pass


def _populate_test_case():
  """Populates UnifiedBenchmarkTest with test methods from the registry."""
  # Parse flags known so far to extract benchmark_silo before absltest.main().
  flags.FLAGS(sys.argv, known_only=True)
  silo = BENCHMARK_SILO.value
  if not silo:
    return

  benchmark_registry.populate_standalone_test_class(UnifiedBenchmarkTest, silo)


if __name__ == "__main__":
  _populate_test_case()
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
