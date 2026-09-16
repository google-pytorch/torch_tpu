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

"""Tests for TorchAX performance_utils peak memory measurement."""

import sys

if sys.version_info >= (3, 14):
  # Torchax is not supported on Python 3.14+.
  sys.exit(0)

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import torch
from examples.benchmarks.e2e import benchmark_utils as pt_benchmark_utils
from examples.benchmarks.e2e import common as pt_common
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.e2e import performance_utils as pt_performance_utils
from examples.benchmarks.e2e.torchax import performance_utils as torchax_perf_utils
from tests import seed_test_utils


class TorchaxPerformanceUtilsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.synchronize_tensors_in_tear_down = False
    # Keep step counts minimal for fast unit testing
    self.enter_context(
        flagsaver.flagsaver(max_warmup_steps=1, post_warmup_steps=1)
    )

  def test_forward_pass_computes_peak_memory_when_xprof_disabled(self):
    """Regression test: layer benchmarks run with enable_xprof=False and must still compute peak memory."""
    mock_jittable = mock.MagicMock()
    mock_jittable.params = {'w': torch.randn(2, 2)}
    mock_jittable.buffers = {}

    mock_out = mock.MagicMock()
    mock_out.data = torch.randn(2, 2)

    with (
        mock.patch.object(
            torchax_perf_utils, '_call_functional_model', return_value=mock_out
        ),
        mock.patch.object(torchax_perf_utils, '_sync_jax_device'),
        mock.patch.object(
            device_utils, 'reset_peak_memory_stats'
        ) as mock_reset,
        mock.patch.object(
            device_utils, 'get_peak_memory_hbm', return_value=256.5
        ) as mock_get_peak,
    ):
      result = torchax_perf_utils._run_torchax_forward_pass(
          model_jittable=mock_jittable,
          inputs=torch.randn(2, 2),
          run_mode=pt_common.RunMode.EAGER_DEFAULT,
          enable_xprof=False,
      )

      mock_reset.assert_called_once_with('jax')
      mock_get_peak.assert_called_once_with('jax', None, None)
      self.assertEqual(result.peak_device_memory_mb, 256.5)

  def test_backward_pass_computes_peak_memory_when_xprof_disabled(self):
    """Regression test: training layer benchmarks run with enable_xprof=False and must still compute peak memory."""
    mock_jittable = mock.MagicMock()
    mock_jittable.params = {'w': torch.randn(2, 2)}
    mock_jittable.buffers = {}

    mock_config = mock.MagicMock(
        spec=pt_performance_utils.PerformanceBenchmarkConfig
    )
    mock_config.run_mode = pt_common.RunMode.EAGER_DEFAULT
    mock_config.benchmark_category = (
        pt_benchmark_utils.BenchmarkCategory.ML_LAYER
    )

    with (
        mock.patch.object(
            torchax_perf_utils.torchax.interop,
            'jax_value_and_grad',
            return_value=lambda diff_target: (torch.tensor(1.0), diff_target),
        ),
        mock.patch.object(torchax_perf_utils, '_sync_jax_device'),
        mock.patch.object(
            device_utils, 'reset_peak_memory_stats'
        ) as mock_reset,
        mock.patch.object(
            device_utils, 'get_peak_memory_hbm', return_value=512.75
        ) as mock_get_peak,
    ):
      result = torchax_perf_utils._run_torchax_backward_pass(
          model_jittable=mock_jittable,
          inputs=torch.randn(2, 2),
          config=mock_config,
          enable_xprof=False,
      )

      mock_reset.assert_called_once_with('jax')
      mock_get_peak.assert_called_once_with('jax', None, None)
      self.assertEqual(result.peak_device_memory_mb, 512.75)

  def test_forward_pass_computes_peak_memory_when_xprof_enabled(self):
    """Regression test: model benchmarks with enable_xprof=True pass session_id and xprof_client to get_peak_memory_hbm."""
    mock_jittable = mock.MagicMock()
    mock_jittable.params = {'w': torch.randn(2, 2)}
    mock_jittable.buffers = {}

    mock_out = mock.MagicMock()
    mock_out.data = torch.randn(2, 2)

    mock_xprof_ctx = mock.MagicMock()
    mock_xprof_ctx.session_id = 'test_session_123'
    mock_xprof_ctx.__enter__.return_value = mock_xprof_ctx

    mock_client = mock.MagicMock()

    with (
        mock.patch.object(
            torchax_perf_utils, '_call_functional_model', return_value=mock_out
        ),
        mock.patch.object(torchax_perf_utils, '_sync_jax_device'),
        mock.patch.object(
            torchax_perf_utils,
            '_get_device_timings',
            return_value=(0.01, 0.01),
        ),
        mock.patch.object(
            pt_performance_utils, 'get_xprof_client', return_value=mock_client
        ),
        mock.patch.object(
            pt_benchmark_utils, 'XprofContext', return_value=mock_xprof_ctx
        ),
        mock.patch.object(
            device_utils, 'reset_peak_memory_stats'
        ) as mock_reset,
        mock.patch.object(
            device_utils, 'get_peak_memory_hbm', return_value=1024.0
        ) as mock_get_peak,
    ):
      result = torchax_perf_utils._run_torchax_forward_pass(
          model_jittable=mock_jittable,
          inputs=torch.randn(2, 2),
          run_mode=pt_common.RunMode.EAGER_DEFAULT,
          enable_xprof=True,
      )

      mock_reset.assert_called_once_with('jax')
      mock_get_peak.assert_called_once_with(
          'jax', 'test_session_123', mock_client
      )
      self.assertEqual(result.peak_device_memory_mb, 1024.0)


if __name__ == '__main__':
  absltest.main()
