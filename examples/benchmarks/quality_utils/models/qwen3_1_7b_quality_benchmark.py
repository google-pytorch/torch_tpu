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

"""Qwen 3 1.7B quality benchmark model."""

from typing import Any

import torch
from examples import paths
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
from tests import module_registry
import transformers


class Qwen317BQualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """Qwen 3 1.7B quality benchmark model."""

  def __init__(self, device: str, max_seq_len: int):
    super().__init__()
    self._device = device
    self._max_seq_len = max_seq_len
    self._model = None
    self._tokenizer = None

  def initialize(self) -> None:
    """Initializes the model and tokenizer."""
    registry = module_registry.ModuleRegistry()
    spec = registry.get_module_spec(
        "transformers",
        "Qwen/Qwen3-1.7B",
        load_weights=True,
        torch_dtype=torch.bfloat16,
    )
    self._model = spec.module_factory()
    self._tokenizer = spec.preprocessor_factory()

    self._model.eval()
    self._model.to(self._device)
    self._model.requires_grad_(False)

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
    assert self._tokenizer is not None
    assert self._model is not None
    input_ids = formatted_input.input[:, :-1].to(self._device)

    pad_token_id = self._tokenizer.pad_token_id
    if pad_token_id is None:
      pad_token_id = self._tokenizer.eos_token_id

    attention_mask = (input_ids != pad_token_id).long()

    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

    pred_logits = outputs.logits[0].to(torch.float32)
    target = formatted_input.input[0, 1:].to(self._device)

    return pred_logits, target

  def get_model(self) -> torch.nn.Module:
    """Gets the model."""
    return self._model

  def _compile_model_once(self) -> None:
    """Compiles the model for the target device."""
    assert self._model is not None
    self._model = device_utils.torch_compile(self._model, self._device)
