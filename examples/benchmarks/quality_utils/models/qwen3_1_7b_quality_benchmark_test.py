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

import os
import sys

# Ensure the repo root is in sys.path first to avoid import shadowing of 'examples' package in CI.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from unittest import mock

from absl.testing import absltest
import torch
from examples.benchmarks.quality_utils import quality_benchmark_model
from examples.benchmarks.quality_utils.models import qwen3_1_7b_quality_benchmark
import transformers


class Qwen317BQualityBenchmarkModelTest(absltest.TestCase):

  def test_format_padding(self):
    model = qwen3_1_7b_quality_benchmark.Qwen317BQualityBenchmarkModel(
        device='cpu', max_seq_len=10
    )

    mock_tokenizer = mock.MagicMock(spec=transformers.PreTrainedTokenizer)
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.pad_token_id = 99

    model._tokenizer = mock_tokenizer

    formatted_input = model.format('hello')

    # total_len = 10 + 1 = 11
    expected_tokens = torch.tensor(
        [[1, 2, 3, 99, 99, 99, 99, 99, 99, 99, 99]], dtype=torch.long
    )
    self.assertTrue(torch.equal(formatted_input.input, expected_tokens))
    self.assertEqual(formatted_input.unpadded_length, 3)

  def test_format_padding_fallback(self):
    model = qwen3_1_7b_quality_benchmark.Qwen317BQualityBenchmarkModel(
        device='cpu', max_seq_len=10
    )

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.pad_token_id = None
    mock_tokenizer.eos_token_id = 100

    model._tokenizer = mock_tokenizer

    formatted_input = model.format('hello')

    # total_len = 10 + 1 = 11
    expected_tokens = torch.tensor(
        [[1, 2, 3, 100, 100, 100, 100, 100, 100, 100, 100]], dtype=torch.long
    )
    self.assertTrue(torch.equal(formatted_input.input, expected_tokens))
    self.assertEqual(formatted_input.unpadded_length, 3)

  def test_encode(self):
    model = qwen3_1_7b_quality_benchmark.Qwen317BQualityBenchmarkModel(
        device='cpu', max_seq_len=10
    )
    mock_tokenizer = mock.MagicMock(spec=transformers.PreTrainedTokenizer)
    mock_tokenizer.encode.return_value = [1, 2, 3]
    model._tokenizer = mock_tokenizer

    encoded = model.encode('hello')

    self.assertEqual(encoded, [1, 2, 3])
    mock_tokenizer.encode.assert_called_once_with(
        'hello', add_special_tokens=True
    )

  def test_get_logits_and_targets_alignment(self):
    model = qwen3_1_7b_quality_benchmark.Qwen317BQualityBenchmarkModel(
        device='cpu', max_seq_len=10
    )

    mock_tokenizer = mock.MagicMock(spec=transformers.PreTrainedTokenizer)
    mock_tokenizer.pad_token_id = 99
    model._tokenizer = mock_tokenizer

    mock_torch_model = mock.MagicMock(spec=torch.nn.Module)
    mock_outputs = mock.MagicMock(
        spec=transformers.modeling_outputs.ModelOutput
    )
    mock_outputs.logits = torch.ones((1, 10, 50))
    mock_torch_model.return_value = mock_outputs
    model._model = mock_torch_model

    # 11 tokens
    formatted_input_tensor = torch.arange(11).unsqueeze(0)
    formatted_input = quality_benchmark_model.FormattedInput(
        formatted_input_tensor, 11
    )

    pred_logits, target = model.get_logits_and_targets(formatted_input)

    expected_target = torch.arange(1, 11)

    self.assertEqual(pred_logits.shape, (10, 50))
    self.assertTrue(torch.equal(pred_logits, torch.ones((10, 50))))
    self.assertTrue(torch.equal(target, expected_target))


if __name__ == '__main__':
  absltest.main()
