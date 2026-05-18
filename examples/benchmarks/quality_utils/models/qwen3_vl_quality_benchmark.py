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

"""Qwen3-VL quality benchmark model."""

import torch
from torch_tpu._internal.utils import device_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
from tests import module_registry


class Qwen3VLQualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """Qwen3-VL quality benchmark model.

  Attributes:
    _model: The Qwen3-VL model.
  """

  def __init__(self, model_name: str, device: str):
    super().__init__()
    self._model_name = model_name
    self._device_str = device
    self._device = None
    self._model = None
    self._registry = module_registry.ModuleRegistry()
    self._module_spec = None

  def initialize(self) -> None:
    """Initializes the model."""
    if self._device_str == "tpu":
      from torch_tpu import _loader

      _loader.load()
    self._device = torch.device(self._device_str)

    self._module_spec = self._registry.get_module_spec(
        "transformers", self._model_name, load_weights=False
    )
    self._model = self._module_spec.module_factory().to(self._device)
    self._model.eval()

  def get_model(self) -> torch.nn.Module:
    """Gets the model."""
    return self._model

  def _compile_model_once(self) -> None:
    """Compiles the model for the target device."""
    assert self._model is not None
    self._model = device_utils.torch_compile(self._model, self._device.type)

  def format(self, text: str) -> quality_benchmark_model.FormattedInput:
    """Encodes text for Qwen3-VL model.

    Uses a mock tokenization mapping characters to token IDs to enable offline
    pipeline verification without downloading the real tokenizer.

    Args:
      text: The text to encode.

    Returns:
      The encoded text as a tokens tensor and the length of the tokens.
    """
    assert self._module_spec is not None
    config = self._module_spec.config

    vocab_size = (
        config.text_config.vocab_size
        if hasattr(config, "text_config")
        else config.vocab_size
    )

    # Simple mock tokenization: ord(char) % vocab_size
    tokens = [ord(c) % vocab_size for c in text]
    max_len = 512
    tokens = tokens[:max_len]

    # We must make sure the sequence length is at least 2 for next-token prediction
    if len(tokens) < 2:
      tokens = [0, 0]

    tokens_tensor = torch.tensor(
        [tokens], dtype=torch.long, device=self._device
    )
    return quality_benchmark_model.FormattedInput(tokens_tensor, len(tokens))

  def get_logits_and_targets(
      self, formatted_input: quality_benchmark_model.FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation."""
    assert self._model is not None

    # Forward through the model (next-token prediction)
    input_ids = formatted_input.input[:, :-1]
    outputs = self._model(input_ids=input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    # Remove batch dimension (batch_size=1)
    pred_logits = logits[0]

    # Target is shifted by 1
    target = formatted_input.input[0, 1:]

    return pred_logits, target
