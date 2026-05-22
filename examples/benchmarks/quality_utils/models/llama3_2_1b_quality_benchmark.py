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

"""Llama 3.2 1B quality benchmark model."""

from typing import Any

import torch
from examples import paths
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
import transformers


class Llama321BQualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """Llama 3.2 1B quality benchmark model."""

  def __init__(self, device: str, max_seq_len: int):
    super().__init__()
    self._device = device
    self._max_seq_len = max_seq_len
    self._model = None
    self._tokenizer = None

  def initialize(self) -> None:
    """Initializes the model and tokenizer."""
    self._model = transformers.AutoModelForCausalLM.from_pretrained(
        paths.XM_HOME + "weights/huggingface/meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
    )

    self._model.requires_grad_(False)
    # Using 8B tokenizer as done in reference test llama3_2_1b_training_test.py
    # since Llama 3 and 3.2 share the same tokenizer.
    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
        paths.XM_HOME
        + "weights/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/",
    )
    self._model.eval()
    self._model.to(self._device)

  @property
  def max_seq_len(self) -> int:
    return self._max_seq_len

  def encode(self, text: str) -> list[int]:
    """Encodes the text into tokens without padding or truncation.

    Args:
      text: The input string to encode.

    Returns:
      A list of token IDs.
    """
    assert self._tokenizer is not None, "Tokenizer not initialized"
    return self._tokenizer.encode(text, add_special_tokens=True)

  def format(self, raw_input: Any) -> quality_benchmark_model.FormattedInput:
    """Formats input for the model."""
    assert self._tokenizer is not None
    if isinstance(raw_input, str):
      tokens = self.encode(raw_input)
    elif isinstance(raw_input, list):
      tokens = raw_input
    else:
      raise ValueError(f"Unsupported input type: {type(raw_input)}")
    raw_token_length = len(tokens)

    total_len = self._max_seq_len + 1
    pad_token_id = self._tokenizer.pad_token_id
    if pad_token_id is None:
      pad_token_id = self._tokenizer.eos_token_id

    padded_tokens = torch.full(
        (1, total_len), pad_token_id, dtype=torch.long, device=self._device
    )

    tokens_len = min(raw_token_length, total_len)
    padded_tokens[0, :tokens_len] = torch.tensor(
        tokens[:tokens_len], dtype=torch.long, device=self._device
    )

    return quality_benchmark_model.FormattedInput(padded_tokens, tokens_len)

  def get_logits_and_targets(
      self, formatted_input: quality_benchmark_model.FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation."""
    assert self._model is not None
    input_ids = formatted_input.input[:, :-1].to(self._device)

    outputs = self._model(input_ids)

    pred_logits = outputs.logits[0].to(torch.float32)

    # The target consists of the sample_input shifted by one to the left when
    # compared to the input to the model. We also slice out the batch dimension.
    target = formatted_input.input[0, 1:].to(self._device)

    return pred_logits, target

  def _compile_model_once(self) -> None:
    """Compiles the model for the target device."""
    assert self._model is not None
    self._model = device_utils.torch_compile(self._model, self._device)

  def get_model(self) -> torch.nn.Module:
    """Gets the model."""
    return self._model
