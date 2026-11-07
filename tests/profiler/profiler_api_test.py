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
import traceback
from unittest import mock

from absl.testing import absltest
import portpicker
import torch
from torch_tpu._internal import profiler
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from torch_tpu._internal.profiler import _api
from torch_tpu._internal.profiler import _impl as profiler_impl
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig
from tests import seed_test_utils

# pylint: disable=g-direct-tensorflow-import
from tsl.profiler.protobuf import xplane_pb2

# pylint: enable=g-direct-tensorflow-import


def _count_host_events(xspace: xplane_pb2.XSpace, event_name: str) -> int:
  """Counts events matching `event_name` on `/host:CPU` XPlanes."""
  total = 0
  for plane in xspace.planes:
    if plane.name == '/host:CPU':
      ids = {k for k, m in plane.event_metadata.items() if m.name == event_name}
      total += sum(
          e.metadata_id in ids for line in plane.lines for e in line.events
      )
  return total


def _fresh_process_door2_worker(output_dir: str, result_queue) -> None:
  """Runs `torch_tpu._internal.profiler.profile` in a fresh process."""
  try:
    multiprocessing.parse_absl_flags()
    marker = 'door2_fresh_process_unique_op'
    # Use CPU-only activities so the spawned worker does not contend for TPU
    # device nodes held by the parent test process.
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        on_trace_ready=profiler.xprof_trace_handler(dir_name=output_dir),
    ):
      with torch.autograd.profiler.record_function(marker):
        _ = torch.randn(4, 4) + torch.randn(4, 4)

    plugins_dir = pathlib.Path(output_dir) / 'plugins' / 'profile'
    xplane_files = list(plugins_dir.glob('**/*.xplane.pb'))
    if len(xplane_files) != 1:
      result_queue.put(f'Expected 1 xplane file, got {xplane_files}')
      return
    xspace = xplane_pb2.XSpace.FromString(xplane_files[0].read_bytes())
    result_queue.put(('OK', _count_host_events(xspace, marker)))
  except Exception:  # pylint: disable=broad-exception-caught
    result_queue.put(traceback.format_exc())


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
    """Tests that active trace sessions capture exact single host events."""
    # 1. Start profiler server; callback registration is scoped to active
    # ProfilerSessions rather than server startup.
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

      # 4. Parse XSpace protobuf and verify exact count == 1 (no duplicates).
      with open(xplane_file_path, 'rb') as f:
        xspace = xplane_pb2.XSpace.FromString(f.read())

      self.assertEqual(
          _count_host_events(xspace, 'test_server_registered_pytorch_op'),
          1,
      )
    finally:
      # 5. Clean up by stopping the profiler server.
      profiler.stop_server()

  def test_profile_api_fresh_process_captures_exact_single_event(self):
    """Verifies Door 2 captures RecordFunction in a fresh process."""
    output_dir = self.create_tempdir('door2_fresh_process').full_path
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_fresh_process_door2_worker,
        args=(output_dir, result_queue),
    )
    proc.start()
    proc.join()
    res = result_queue.get() if not result_queue.empty() else 'NO RESULT'
    self.assertEqual(
        proc.exitcode,
        0,
        f'Worker process exited with {proc.exitcode}:\n{res}',
    )
    self.assertEqual(res, ('OK', 1), f'Unexpected worker result:\n{res}')


if __name__ == '__main__':
  multiprocessing.handle_test_main(absltest.main)
