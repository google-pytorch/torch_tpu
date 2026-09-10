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
import pathlib
from unittest import mock

from absl.testing import absltest
import portpicker
import torch
from torch_tpu._internal import profiler
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.profiler import _api
from torch_tpu._internal.profiler import _impl as profiler_impl
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig
from tests import seed_test_utils

# pylint: disable=g-direct-tensorflow-import
from tsl.profiler.protobuf import xplane_pb2

# pylint: enable=g-direct-tensorflow-import


class ProfilerApiTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
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

  def test_xprof_trace_handler_with_rank(self):
    handler = profiler.xprof_trace_handler(dir_name='/tmp', worker_rank='42')
    self.assertEqual(getattr(handler, 'worker_rank', None), '42')

  @mock.patch('socket.gethostname', return_value='test_host')
  def test_profile_api_with_rank(self, _):
    output_dir = self.create_tempdir('profile_api_with_rank').full_path
    handler = profiler.xprof_trace_handler(
        dir_name=output_dir, worker_rank='42'
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        on_trace_ready=handler,
    ):
      # Simulate some work
      a = torch.randn(10, 10)
      b = torch.randn(10, 10)
      _ = a + b

    # Check if trace files are created with correct rank
    plugins_dir = os.path.join(output_dir, 'plugins', 'profile')
    self.assertTrue(os.path.isdir(plugins_dir))
    trace_dirs = os.listdir(plugins_dir)
    self.assertLen(trace_dirs, 1)
    trace_dir = os.path.join(plugins_dir, trace_dirs[0])
    found_files = os.listdir(trace_dir)
    self.assertIn('test_host_42.xplane.pb', found_files)

  def test_profile_api_with_invalid_rank(self):
    output_dir = self.create_tempdir('profile_api_with_invalid_rank').full_path
    handler = profiler.xprof_trace_handler(
        dir_name=output_dir, worker_rank='../../invalid'
    )
    with self.assertRaisesRegex(
        ValueError, 'worker_rank cannot contain path separators'
    ):
      with profiler.profile(
          activities=[profiler.ProfilerActivity.CPU],
          on_trace_ready=handler,
      ):
        pass

  @mock.patch('socket.gethostname', return_value='test_host')
  def test_profile_api_without_rank(self, _):
    output_dir = self.create_tempdir('profile_api_without_rank').full_path
    handler = profiler.xprof_trace_handler(dir_name=output_dir)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        on_trace_ready=handler,
    ):
      # Simulate some work
      a = torch.randn(10, 10)
      b = torch.randn(10, 10)
      _ = a + b

    # Check if trace files are created without rank
    plugins_dir = os.path.join(output_dir, 'plugins', 'profile')
    self.assertTrue(os.path.isdir(plugins_dir))
    trace_dirs = os.listdir(plugins_dir)
    self.assertLen(trace_dirs, 1)
    trace_dir = os.path.join(plugins_dir, trace_dirs[0])
    found_files = os.listdir(trace_dir)
    self.assertIn('test_host.xplane.pb', found_files)

  def test_profile_api_tpu(self):
    """Tests the profile context manager with TPU operations."""
    try:
      device = torch.device('tpu')
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
    options = _api._get_profile_options([profiler.ProfilerActivity.TPU])
    self.assertEqual(options.device_tracer_level, 1)
    self.assertEqual(options.host_tracer_level, 0)
    self.assertEqual(options.python_tracer_level, 0)

  def test_profiler_options_cpu_only(self):
    options = _api._get_profile_options([profiler.ProfilerActivity.CPU])
    self.assertEqual(options.device_tracer_level, 0)
    self.assertEqual(options.host_tracer_level, 2)
    self.assertEqual(options.python_tracer_level, 1)

  def test_profiler_options_tpu_and_cpu(self):
    options = _api._get_profile_options(
        activities=[
            profiler.ProfilerActivity.TPU,
            profiler.ProfilerActivity.CPU,
        ]
    )
    self.assertEqual(options.device_tracer_level, 1)
    self.assertEqual(options.host_tracer_level, 2)
    self.assertEqual(options.python_tracer_level, 1)

  def test_profiler_options_none(self):
    options = _api._get_profile_options([])
    self.assertEqual(options.device_tracer_level, 0)
    self.assertEqual(options.host_tracer_level, 0)
    self.assertEqual(options.python_tracer_level, 0)

  def test_tpu_profiler_config_run_dir(self):
    gcs_uri = 'gs://my-bucket/my-dir'
    local_path = '/tmp/my-dir'

    # Mock __init__ to capture kwargs for assertion.
    # We must call original_init because PyTorch C++ bindings enforce that the
    # base class constructor is always invoked.
    original_init = torch.profiler._ExperimentalConfig.__init__
    called_kwargs = {}

    def fake_init(self, *args, **kwargs):
      called_kwargs.clear()
      called_kwargs.update(kwargs)
      original_init(self, *args, **kwargs)

    with mock.patch.object(
        torch.profiler._ExperimentalConfig, '__init__', new=fake_init
    ):
      # Test that passing a GCS string preserves the protocol completely
      TpuProfilerConfig(run_dir=gcs_uri)
      custom_config = called_kwargs.get('custom_profiler_config', '')
      self.assertIn('host_tracer_level:', custom_config)
      self.assertIn(f'run_dir:{gcs_uri}', custom_config)

      # Test that passing a pathlib.Path works for local paths
      TpuProfilerConfig(run_dir=pathlib.Path(local_path))
      custom_config = called_kwargs.get('custom_profiler_config', '')
      self.assertIn(f'run_dir:{local_path}', custom_config)

  def test_exported_start_and_stop_server(self):
    """Tests start_server and stop_server at exported module paths."""
    # Test from torch_tpu._internal.profiler
    port = portpicker.pick_unused_port()
    server_port = profiler.start_server(port)
    self.assertEqual(server_port, port)
    profiler.stop_server()

  def test_start_server_registers_xprof_callback_handler(self):
    """Tests that starting server registers XProfCallbackHandler."""
    # 1. Start profiler server, which should trigger
    # XProfCallbackHandler::Register().
    port = portpicker.pick_unused_port()
    profiler.start_server(port)
    try:
      output_dir = self.create_tempdir('server_xprof_callback').full_path
      options = profiler_impl.ProfileOptions()
      options.host_tracer_level = 2

      # 2. Start a trace session and execute PyTorch ops inside RecordFunction.
      profiler_impl.start_trace(output_dir, profiler_options=options)
      try:
        with torch.autograd.profiler.record_function(
            'test_server_registered_pytorch_op'
        ):
          a = torch.randn(4, 4)
          b = torch.randn(4, 4)
          _ = a + b
      finally:
        profiler_impl.stop_trace()

      # 3. Verify that output profile directory and xplane trace files exist.
      plugins_dir = os.path.join(output_dir, 'plugins', 'profile')
      self.assertTrue(os.path.isdir(plugins_dir))
      trace_dirs = os.listdir(plugins_dir)
      self.assertLen(trace_dirs, 1)
      trace_dir = os.path.join(plugins_dir, trace_dirs[0])
      xplane_files = [
          f for f in os.listdir(trace_dir) if f.endswith('.xplane.pb')
      ]
      self.assertNotEmpty(xplane_files)
      xplane_file_path = os.path.join(trace_dir, xplane_files[0])

      # 4. Parse XSpace protobuf and verify the PyTorch op event was captured.
      with open(xplane_file_path, 'rb') as f:
        xspace = xplane_pb2.XSpace.FromString(f.read())

      found = False
      for plane in xspace.planes:
        for metadata in plane.event_metadata.values():
          if 'test_server_registered_pytorch_op' in metadata.name:
            found = True
            break
        if found:
          break
      self.assertTrue(
          found,
          msg=(
              'Did not find test_server_registered_pytorch_op in XSpace'
              ' metadata'
          ),
      )
    finally:
      # 5. Clean up by stopping the profiler server.
      profiler.stop_server()


if __name__ == '__main__':
  absltest.main()
