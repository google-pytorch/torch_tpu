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

  def test_double_start_server_fails(self):
    """Calling start_profiler_server() twice should fail."""

    port = portpicker.pick_unused_port()
    profiler_backend.start_profiler_server(port)
    try:
      with self.assertRaisesRegex(
          RuntimeError, 'server has already been started'
      ):
        profiler_backend.start_profiler_server(port)
    finally:
      profiler_backend.stop_profiler_server()

  def test_double_start_trace_fails(self):
    """Calling start_trace() twice should fail."""

    profiler_backend.start_trace(self.tmpdir)
    try:
      with self.assertRaisesRegex(
          RuntimeError, 'session has already been started'
      ):
        profiler_backend.start_trace(self.tmpdir)
    finally:
      profiler_backend.stop_trace(os.path.join(self.tmpdir, 'trace.pb'))

  def test_stop_server_without_start_fails(self):
    """Calling stop_profiler_server() without start_profiler_server() should fail."""

    with self.assertRaisesRegex(RuntimeError, 'server has not been started'):
      profiler_backend.stop_profiler_server()

  def test_stop_trace_without_start_fails(self):
    """Calling stop_trace() without start_trace() should fail."""

    with self.assertRaisesRegex(RuntimeError, 'session has not been started'):
      profiler_backend.stop_trace(os.path.join(self.tmpdir, 'trace.pb'))

  def test_stop_server_twice_fails(self):
    """Calling stop_profiler_server() twice should fail."""

    port = portpicker.pick_unused_port()
    profiler_backend.start_profiler_server(port)
    profiler_backend.stop_profiler_server()
    with self.assertRaisesRegex(RuntimeError, 'server has not been started'):
      profiler_backend.stop_profiler_server()

  def test_stop_trace_twice_fails(self):
    """Calling stop_trace() twice should fail."""

    profiler_backend.start_trace(self.tmpdir)
    profiler_backend.stop_trace(os.path.join(self.tmpdir, 'trace.pb'))
    with self.assertRaisesRegex(RuntimeError, 'session has not been started'):
      profiler_backend.stop_trace(os.path.join(self.tmpdir, 'trace.pb'))

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

  def test_retry_start_after_stop_failure(self):
    """Calling stop_trace() failure should still allow a subsequent start."""

    profiler_backend.start_trace(self.tmpdir)
    # Use a directory path as a filename, which should fail on stop.
    # In Stop(), it tries to NewWritableFile(filename, &outfile).
    # If self.tmpdir is a directory, NewWritableFile likely fails.
    invalid_filename = self.tmpdir
    with self.assertRaisesRegex(RuntimeError, 'failed to create file'):
      profiler_backend.stop_trace(invalid_filename)

    # Now verify that we can start again.
    profiler_backend.start_trace(self.tmpdir)
    profiler_backend.stop_trace(os.path.join(self.tmpdir, 'trace.pb'))


if __name__ == '__main__':
  absltest.main()
