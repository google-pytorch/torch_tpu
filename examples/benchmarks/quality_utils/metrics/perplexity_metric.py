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


"""Perplexity score computation for quality benchmarks."""

from typing import Sequence
from absl import logging
import torch
import torch.nn.functional as F
from examples.benchmarks.quality_utils import quality_benchmark_model


class PerplexityMetric(quality_benchmark_model.MetricProducer):
  """Perplexity metric for quality benchmarks."""

  def get_name(self) -> str:
    return "perplexity"

  @torch.inference_mode()
  def assess(
      self,
      model_input: str | Sequence[int],
      benchmark_model: quality_benchmark_model.QualityBenchmarkModel,
  ) -> torch.Tensor:
    """Runs perplexity score on a single sample.

    This function is used to run perplexity score on a single sample. It will
    split the input text into chunks of the given length, compute the cross
    entropy loss for each chunk, and return the exponentiated average loss over
    all processed tokens.

    Args:
      model_input: The text to run perplexity score on.
      benchmark_model: The benchmark model to use for encoding the text.

    Returns:
      The perplexity score.
    """
    if isinstance(model_input, str):
      tokens = benchmark_model.encode(model_input)
    else:
      tokens = model_input
    max_seq_len = benchmark_model.max_seq_len

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(tokens) - 1, max_seq_len):
      chunk_tokens = tokens[i : i + max_seq_len + 1]

      if len(chunk_tokens) < max_seq_len + 1:
        break

      formatted_input = benchmark_model.format(chunk_tokens)
      pred_logits, target = benchmark_model.get_logits_and_targets(
          formatted_input
      )

      # Manual cross entropy
      log_probs = torch.nn.functional.log_softmax(pred_logits.float(), dim=-1)
      target_expanded = target.long().unsqueeze(-1)
      target_log_probs = log_probs.gather(
          dim=-1, index=target_expanded
      ).squeeze(-1)
      ce = -target_log_probs

      target_len = formatted_input.unpadded_length - 1
      res = ce[:target_len].sum()

      total_loss += res
      total_tokens += target_len

    if total_tokens == 0:
      # Raise error if no tokens were processed to avoid hiding incorrect behavior.
      # TODO(b/499309732): Revert to raising an error after finding a solution
      logging.warning("No tokens processed for perplexity calculation.")
      return torch.tensor(float("inf"))

    return torch.exp(total_loss / total_tokens)
