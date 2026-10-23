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

"""Tests for _profiler_backend C++ APIs."""

import concurrent.futures
import os
import shutil
import tempfile

from absl.testing import absltest
import portpicker
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.profiler import _profiler_backend as profiler_backend
from tests import seed_test_utils


class ProfilerBackendTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.tmpdir = tempfile.mkdtemp()
    # Ensure state is clean before each test.
    try:
      profiler_backend.stop_profiler_server()
    except RuntimeError:
      pass
    try:
      profiler_backend.stop_trace(os.path.join(self.tmpdir, 'unused.pb'))
    except RuntimeError:
      pass

  def tearDown(self):
    shutil.rmtree(self.tmpdir)
    super().tearDown()

  def test_thread_safety_server_start(self):
    """Calling start_profiler_server() from multiple threads should be OK."""

    port = portpicker.pick_unused_port()
    num_threads = 10

    def start_server():
      try:
        profiler_backend.start_profiler_server(port)
        return True
      except RuntimeError:
        return False

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      results = list(executor.map(lambda _: start_server(), range(num_threads)))

    try:
      self.assertEqual(results.count(True), 1)
      self.assertEqual(results.count(False), num_threads - 1)
    finally:
      profiler_backend.stop_profiler_server()

  def test_thread_safety_trace_start(self):
    """Calling start_trace() from multiple threads should be OK."""

    num_threads = 10

    def start_trace():
      try:
        profiler_backend.start_trace(self.tmpdir)
        return True
      except RuntimeError:
        return False

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      results = list(executor.map(lambda _: start_trace(), range(num_threads)))

    try:
      self.assertEqual(results.count(True), 1)
      self.assertEqual(results.count(False), num_threads - 1)
    finally:
      profiler_backend.stop_trace(os.path.join(self.tmpdir, 'trace.pb'))


if __name__ == '__main__':
  absltest.main()
