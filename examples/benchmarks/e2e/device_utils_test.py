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

"""Tests for device_utils."""

from unittest import mock

from absl.testing import absltest
import torch
from examples.benchmarks.e2e import device_utils
from torch_tpu._internal.profiler import xprof_adapter
from tests import seed_test_utils


class DeviceUtilsTest(seed_test_utils.RepeatableTest):

  def test_get_peak_host_compilation_memory_mb(self):
    class SimpleModel(torch.nn.Module):

      def forward(self, x):
        return x + 1

    device = torch.device('tpu')
    model = SimpleModel().to(device)
    x = torch.randn(10, device=device)

    # We need to use torch.compile with the tpu backend
    compiled_model = torch.compile(model, backend='tpu')

    # Run it to trigger compilation
    _ = compiled_model(x)
    torch.tpu.synchronize()

    peak_mem = device_utils.get_peak_host_compilation_memory_mb('tpu')
    self.assertIsNotNone(peak_mem)
    self.assertGreater(peak_mem, 0.0)

  def test_get_max_total_device_time_success(self):
    mock_client = mock.create_autospec(xprof_adapter.XprofAnalysisClient)
    mock_client.get_hosts.return_value = ['host1', 'host2']

    # Mock XSpace structure
    mock_event1 = mock.MagicMock()
    mock_event1.duration_ps = 1000000000000  # 1 second

    mock_event2 = mock.MagicMock()
    mock_event2.duration_ps = 2000000000000  # 2 seconds

    mock_line1 = mock.MagicMock()
    mock_line1.name = 'XLA Modules:...'
    mock_line1.events = [mock_event1]

    mock_line2 = mock.MagicMock()
    mock_line2.name = 'XLA Modules:...'
    mock_line2.events = [mock_event2]

    mock_plane1 = mock.MagicMock()
    mock_plane1.name = '/device:0'
    mock_plane1.lines = [mock_line1]

    mock_plane2 = mock.MagicMock()
    mock_plane2.name = '/device:1'
    mock_plane2.lines = [mock_line2]

    mock_xspace1 = mock.MagicMock()
    mock_xspace1.planes = [mock_plane1]

    mock_xspace2 = mock.MagicMock()
    mock_xspace2.planes = [mock_plane2]

    def get_xspace_side_effect(session_id, host):
      del session_id  # Unused
      if host == 'host1':
        return mock_xspace1
      elif host == 'host2':
        return mock_xspace2
      return None

    mock_client.get_xspace.side_effect = get_xspace_side_effect

    # Call the function
    max_time = device_utils.get_max_total_device_time(
        session_id='session_123', client=mock_client
    )

    # We expect max of 1s and 2s, which is 2s.
    self.assertEqual(max_time, 2.0)

  def test_get_max_total_device_time_missing_args(self):
    self.assertEqual(
        device_utils.get_max_total_device_time(session_id=None, client=None),
        -1.0,
    )
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id='session_123', client=None
        ),
        -1.0,
    )
    mock_client = mock.create_autospec(xprof_adapter.XprofAnalysisClient)
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id=None, client=mock_client
        ),
        -1.0,
    )

  def test_get_max_total_device_time_no_hosts(self):
    mock_client = mock.create_autospec(xprof_adapter.XprofAnalysisClient)
    mock_client.get_hosts.return_value = []
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id='session_123', client=mock_client
        ),
        -1.0,
    )

  def test_get_max_total_device_time_no_devices(self):
    mock_client = mock.create_autospec(xprof_adapter.XprofAnalysisClient)
    mock_client.get_hosts.return_value = ['host1']

    mock_plane = mock.MagicMock()
    mock_plane.name = '/cpu:0'  # Not a device
    mock_xspace = mock.MagicMock()
    mock_xspace.planes = [mock_plane]
    mock_client.get_xspace.return_value = mock_xspace

    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id='session_123', client=mock_client
        ),
        -1.0,
    )


if __name__ == '__main__':
  absltest.main()
