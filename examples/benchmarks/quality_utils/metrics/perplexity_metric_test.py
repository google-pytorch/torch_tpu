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

import os
import sys

# Ensure the repo root is in sys.path first to avoid import shadowing of 'examples' package in CI.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)


import math
from unittest import mock

from absl.testing import absltest
import torch
from examples.benchmarks.quality_utils import quality_benchmark_model
from examples.benchmarks.quality_utils.metrics import perplexity_metric


class PerplexityMetricTest(absltest.TestCase):

  def test_assess_short_sequence(self):
    metric = perplexity_metric.PerplexityMetric()

    mock_model = mock.MagicMock(
        spec=quality_benchmark_model.QualityBenchmarkModel
    )
    mock_model.encode.return_value = [1]
    mock_model.max_seq_len = 10

    result = metric.assess("a", mock_model)
    self.assertTrue(torch.isinf(result))

  def test_assess_empty_sequence(self):
    metric = perplexity_metric.PerplexityMetric()

    mock_model = mock.MagicMock(
        spec=quality_benchmark_model.QualityBenchmarkModel
    )
    mock_model.encode.return_value = []
    mock_model.max_seq_len = 10

    result = metric.assess("", mock_model)
    self.assertTrue(torch.isinf(result))

  def test_assess_averages_chunks_correctly(self):
    metric = perplexity_metric.PerplexityMetric()

    mock_model = mock.MagicMock(
        spec=quality_benchmark_model.QualityBenchmarkModel
    )
    mock_model.encode.return_value = [10, 11, 12, 13, 14]
    mock_model.max_seq_len = 2

    def mock_format(chunk_tokens):
      mock_input = mock.MagicMock()
      mock_input.unpadded_length = len(chunk_tokens)
      return mock_input

    mock_model.format.side_effect = mock_format

    pred_logits = torch.zeros((3, 4))
    target = torch.tensor([0, 1, 2])
    mock_model.get_logits_and_targets.return_value = (pred_logits, target)

    result = metric.assess("abcdef", mock_model)

    self.assertAlmostEqual(result.item(), 4.0, places=4)
    self.assertEqual(mock_model.format.call_count, 2)
    self.assertEqual(mock_model.get_logits_and_targets.call_count, 2)


if __name__ == "__main__":
  absltest.main()
