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

"""Muse Glimmer 30B forward benchmark.

Runs as a test target and as a plain Python script, e.g.:

  python3 examples/muse_glimmer/benchmark_test.py --smoke_test=false
"""

from absl.testing import absltest
from examples.muse_glimmer import benchmark


class MuseGlimmerBenchmarkTest(absltest.TestCase):  # ABSLTEST_OK=benchmark

  def test_forward_benchmark(self):
    metrics = benchmark.run_benchmark()

    self.assertGreater(metrics.first_step_time_seconds, 0.0)
    self.assertGreater(metrics.post_warmup_step_time_seconds, 0.0)
    self.assertGreater(metrics.e2e_wall_time_seconds, 0.0)


if __name__ == "__main__":
  absltest.main()
