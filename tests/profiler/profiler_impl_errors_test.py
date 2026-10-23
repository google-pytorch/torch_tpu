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

"""Tests for _internal.profiler._impl internal APIs."""

from absl.testing import absltest
import portpicker
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.profiler import _impl as profiler_impl
from tests import seed_test_utils


class ProfilerInternalTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    profiler_impl._profile_state.reset()
    # Clean up any previous server potentially left running from a crashed test
    if profiler_impl._profiler_server is not None:
      profiler_impl.stop_server()

  def test_double_start_server_fails(self):
    """Tests that starting the server twice fails."""
    server = None
    port1 = portpicker.pick_unused_port()
    try:
      server = profiler_impl.start_server(port1)
      self.assertIsNotNone(server)
      with self.assertRaises(ValueError):
        # Try to start on the same port - should fail inside start_server
        profiler_impl.start_server(port1)
    finally:
      if server is not None:
        profiler_impl.stop_server()

  def test_double_start_server_diff_ports_fails(self):
    """Tests that starting the server twice fails."""
    server = None
    port1 = portpicker.pick_unused_port()
    port2 = portpicker.pick_unused_port()
    try:
      server = profiler_impl.start_server(port1)
      self.assertIsNotNone(server)
      with self.assertRaises(ValueError):
        # Try to start on the different ports
        profiler_impl.start_server(port2)
    finally:
      if server is not None:
        profiler_impl.stop_server()

  def test_double_start_trace_fails(self):
    tmpdir = self.create_tempdir('test_double_start_trace_fails').full_path
    try:
      profiler_impl.start_trace(tmpdir)
      with self.assertRaises(RuntimeError):
        profiler_impl.start_trace(tmpdir)
    finally:
      # Ensure stop_trace is called if start_trace succeeded the first time
      if profiler_impl._profile_state.profile_session is not None:
        profiler_impl.stop_trace()

  def test_stop_trace_without_start_fails(self):
    with self.assertRaises(RuntimeError):
      profiler_impl.stop_trace()


if __name__ == '__main__':
  absltest.main()
