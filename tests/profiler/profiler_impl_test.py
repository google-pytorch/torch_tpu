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

import os

from absl.testing import absltest
import portpicker
import torch
from torch_tpu._internal import sync as tpu_sync
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

  def test_start_server(self):
    port = portpicker.pick_unused_port()
    server = None
    try:
      server = profiler_impl.start_server(port)
      self.assertIsNotNone(server)
    finally:
      if server is not None:
        profiler_impl.stop_server()

  def test_start_stop_trace(self):
    """Tests the start_trace and stop_trace functions."""
    output_dir = self.create_tempdir('start_stop_trace').full_path
    try:
      profiler_impl.start_trace(output_dir)
      a = torch.randn(2, 2)
      b = torch.randn(2, 2)
      _ = a - b
    finally:
      if profiler_impl._profile_state.profile_session is not None:
        profiler_impl.stop_trace()
    self.assertTrue(
        os.path.exists(os.path.join(output_dir, 'plugins', 'profile'))
    )

  def test_start_stop_trace_tpu(self):
    """Tests the start_trace and stop_trace functions with TPU operations."""
    try:
      device = torch.device('tpu')
    except RuntimeError as e:
      self.fail(f'Failed to get TPU device: {e}')

    self.assertEqual(device.type, 'tpu', 'Device type should be TPU')

    output_dir = self.create_tempdir('start_stop_trace_tpu').full_path
    options = profiler_impl.ProfileOptions()
    options.device_tracer_level = 1  # Enable device trace collection
    options.host_tracer_level = 2  # Increase host trace verbosity
    options.python_tracer_level = 0  # Disable Python tracer
    try:
      profiler_impl.start_trace(output_dir, profiler_options=options)

      a = torch.randn((32, 32)).to(device)
      b = torch.randn((32, 32)).to(device)
      c = a + b
      tpu_sync.synchronize(c)
    finally:
      if profiler_impl._profile_state.profile_session is not None:
        profiler_impl.stop_trace()

    self.assertTrue(
        os.path.exists(os.path.join(output_dir, 'plugins', 'profile'))
    )


if __name__ == '__main__':
  absltest.main()
