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

"""Tests for _internal.profiler.profile APIs."""

import os

from absl.testing import absltest
import torch
import torch.accelerator
from torch_tpu._internal import profiler
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal.profiler import _impl as profiler_impl
from torch_tpu._internal.profiler import profiler_api


class ProfilerApiTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    profiler_impl._profile_state.reset()
    # Clean up any previous server potentially left running from a crashed test
    if profiler_impl._profiler_server is not None:
      profiler_impl.stop_server()

  def test_profile_api(self):
    output_dir = self.create_tempdir('profile_api').full_path
    handler = profiler.xprof_trace_handler(dir_name=output_dir)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        on_trace_ready=handler,
    ):
      # Simulate some work
      a = torch.randn(10, 10)
      b = torch.randn(10, 10)
      _ = a + b

    # Check if trace files are created
    plugins_dir = os.path.join(output_dir, 'plugins', 'profile')
    self.assertTrue(
        os.path.isdir(plugins_dir), msg=f'Plugins dir not found: {plugins_dir}'
    )
    trace_dirs = os.listdir(plugins_dir)
    self.assertLen(
        trace_dirs, 1, msg=f'Expected 1 trace dir, found: {trace_dirs}'
    )
    trace_dir = os.path.join(plugins_dir, trace_dirs[0])
    found_files = os.listdir(trace_dir)
    self.assertTrue(
        any(f.endswith('.xplane.pb') for f in found_files),
        msg=(
            f'No .xplane.pb file found in {trace_dir}. Files found:'
            f' {found_files}'
        ),
    )

  def test_profile_api_tpu(self):
    """Tests the profile context manager with TPU operations."""
    try:
      device = torch.accelerator.current_accelerator()
    except RuntimeError as e:
      self.fail(f'Failed to get TPU device: {e}')
    self.assertEqual(device.type, 'tpu', 'Device type should be TPU')

    output_dir = self.create_tempdir('profile_api_tpu').full_path
    handler = profiler.xprof_trace_handler(dir_name=output_dir)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.TPU],
        on_trace_ready=handler,
    ):
      a = torch.randn((16, 16)).to(device)
      b = torch.randn((16, 16)).to(device)
      c = a @ b
      tpu_sync.synchronize(c)

    # Check if trace files are created
    plugins_dir = os.path.join(output_dir, 'plugins', 'profile')
    self.assertTrue(
        os.path.isdir(plugins_dir), msg=f'Plugins dir not found: {plugins_dir}'
    )
    trace_dirs = os.listdir(plugins_dir)
    self.assertLen(
        trace_dirs, 1, msg=f'Expected 1 trace dir, found: {trace_dirs}'
    )
    trace_dir = os.path.join(plugins_dir, trace_dirs[0])
    found_files = os.listdir(trace_dir)
    self.assertTrue(
        any(f.endswith('.xplane.pb') for f in found_files),
        msg=(
            f'No .xplane.pb file found in {trace_dir}. Files found:'
            f' {found_files}'
        ),
    )

  def test_profile_api_gpu_unsupported(self):
    output_dir = self.create_tempdir('profile_api_gpu_unsupported').full_path
    handler = profiler.xprof_trace_handler(dir_name=output_dir)
    with self.assertRaisesRegex(ValueError, 'GPU profiling is not supported.'):
      with profiler.profile(
          activities=[profiler.ProfilerActivity.GPU],
          on_trace_ready=handler,
      ):
        pass

  def test_profiler_options_tpu_only(self):
    options = profiler_api._get_profile_options([profiler.ProfilerActivity.TPU])
    self.assertEqual(options.device_tracer_level, 1)
    self.assertEqual(options.host_tracer_level, 0)
    self.assertEqual(options.python_tracer_level, 0)

  def test_profiler_options_cpu_only(self):
    options = profiler_api._get_profile_options([profiler.ProfilerActivity.CPU])
    self.assertEqual(options.device_tracer_level, 0)
    self.assertEqual(options.host_tracer_level, 2)
    self.assertEqual(options.python_tracer_level, 1)

  def test_profiler_options_tpu_and_cpu(self):
    options = profiler_api._get_profile_options(
        activities=[
            profiler.ProfilerActivity.TPU,
            profiler.ProfilerActivity.CPU,
        ]
    )
    self.assertEqual(options.device_tracer_level, 1)
    self.assertEqual(options.host_tracer_level, 2)
    self.assertEqual(options.python_tracer_level, 1)

  def test_profiler_options_none(self):
    options = profiler_api._get_profile_options([])
    self.assertEqual(options.device_tracer_level, 0)
    self.assertEqual(options.host_tracer_level, 0)
    self.assertEqual(options.python_tracer_level, 0)


if __name__ == '__main__':
  absltest.main()
