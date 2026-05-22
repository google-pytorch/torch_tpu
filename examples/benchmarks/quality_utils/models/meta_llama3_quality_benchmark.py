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

"""Llama3 quality benchmark model."""

from typing import Any

from fairscale.nn.model_parallel import initialize as fairscale_init
from llama_models.llama3 import generation
import torch
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
from examples.benchmarks.quality_utils.models import configs


RANDOM_SEED = 4242


class DistributedMetaLlama3QualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """Llama3 quality benchmark model.

  This model requires a minimum of 8 devices to run.

  Attributes:
    _generator: The Llama generator.
    _model: The Llama model.
  """

  _generator: generation.Llama
  _model: torch.nn.Module

  def __init__(
      self, device: str, world_size: int, model_config: str, max_seq_len: int
  ):
    """Creates a QualityBenchmarkModel to enable future model initialization.

    Args:
      device: The device to initialize the model on.
      world_size: The world size of the model. Must be 8.
      model_config: The config of the model to initialize.
      max_seq_len: The maximum sequence length of the model.
    """
    if world_size != 8:
      raise ValueError(
          "World size must be exactly 8 given checkpoints used. Got"
          f" {world_size}."
      )

    super().__init__()
    self._device = device
    self._world_size = world_size
    self._model_config = model_config
    self._max_seq_len = max_seq_len

  def initialize(self) -> None:
    """Initializes the model.

    The model is loaded from a pre-trained checkpoint based on the model
    config.
    """
    with torch.device(self._device):
      torch.manual_seed(RANDOM_SEED)
      if not fairscale_init.model_parallel_is_initialized():
        fairscale_init.initialize_model_parallel(self._world_size)

      torch.set_default_dtype(torch.bfloat16)

    with torch.inference_mode():
      self._generator = generation.Llama.build(
          ckpt_dir=configs.checkpoint_dir[self._model_config],
          max_seq_len=self._max_seq_len,
          max_batch_size=1,
          world_size=self._world_size,
          device=self._device,
      )
    self._model = self._generator.model

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
    return self._generator.formatter.encode_content(text).tokens

  def get_model(self) -> torch.nn.Module:
    """Gets the model from a pre-trained checkpoint."""
    return self._model

  def _compile_model_once(self) -> None:
    """Compiles the model for the target device."""
    self._model = device_utils.torch_compile(self._model, self._device)

  def format(self, raw_input: Any) -> quality_benchmark_model.FormattedInput:
    """Encodes text for meta llama3 model.

    Based on the implementation in llama3/generation.py to generate text.

    Args:
      raw_input: The text or tokens to encode.

    Returns:
      The encoded text as a tokens tensor with padding and the length of the
      unpadded tokens.
    """
    # Encode text to prompt_tokens.
    if isinstance(raw_input, str):
      raw_tokens = self.encode(raw_input)
    elif isinstance(raw_input, list):
      raw_tokens = raw_input
    else:
      raise ValueError(f"Unsupported input type: {type(raw_input)}")
    raw_token_length = len(raw_tokens)

    # Create buffer for pad for model interaction.
    total_len = self._generator.model.params.max_seq_len + 1

    # Model expects a batch dimension, but we only have one sample.
    tokens = torch.full(
        (1, total_len), self._generator.tokenizer.pad_id, dtype=torch.long
    )

    tokens_len = min(raw_token_length, total_len)
    tokens[0, :tokens_len] = torch.tensor(
        raw_tokens[:tokens_len], dtype=torch.long
    )

    return quality_benchmark_model.FormattedInput(tokens, tokens_len)

  def get_logits_and_targets(
      self, formatted_input: quality_benchmark_model.FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation."""
    # We forward formatted_input.input[:, :-1] to the model. For every
    # position in the input sequence, the model predicts a distribution for
    # the token right after that position, and we are measuring Perplexity for
    # the next token.
    pred_logits = self._model(formatted_input.input[:, :-1], 0)

    # Remove the batch dimension (batch_size=1) using direct indexing.
    pred_logits = pred_logits[0]

    # The target consists of the sample_input shifted by one to the left when
    # compared to the input to the model. We also slice out the batch dimension.
    target = formatted_input.input[0, 1:]

    return pred_logits, target

  def get_logits_and_targets(
      self, formatted_input: quality_benchmark_model.FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation."""
    # We forward formatted_input.input[:, :-1] to the model. For every
    # position in the input sequence, the model predicts a distribution for
    # the token right after that position, and we are measuring Perplexity for
    # the next token.
    pred_logits = self._model(formatted_input.input[:, :-1], 0)

    # Remove the batch dimension (batch_size=1) using direct indexing.
    pred_logits = pred_logits[0]

    # The target consists of the sample_input shifted by one to the left when
    # compared to the input to the model. We also slice out the batch dimension.
    target = formatted_input.input[0, 1:]

    return pred_logits, target
