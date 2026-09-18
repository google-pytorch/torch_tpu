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

"""Common model loading and initialization helpers for benchmark harness."""

from collections.abc import Callable
import functools
import typing
from typing import Any

import torch
from tests import module_registry


def get_vocab_size(config: Any, default: int = 128256) -> int:
  """Gets the vocab size from a config, with a fallback default."""
  direct_vocab_size = getattr(config, "vocab_size", None)
  if direct_vocab_size is not None:
    return direct_vocab_size

  text_config = getattr(config, "text_config", None)
  if text_config is not None:
    text_vocab_size = getattr(text_config, "vocab_size", None)
    if text_vocab_size is not None:
      return text_vocab_size

  return default


@functools.cache
def get_module_registry() -> module_registry.ModuleRegistry:
  """Returns a cached ModuleRegistry instance."""
  return module_registry.ModuleRegistry()


def load_hf_llm(
    model_name: str,
    dtype: torch.dtype,
    device_str: str,
    is_training: bool = False,
    seq_len: int = 2048,
    batch_size: int = 1,
    modify_config_hook: Callable[[Any], Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace LLM and prepares inputs for benchmarking.

  Args:
    model_name: HuggingFace model identifier.
    dtype: PyTorch data type for the model and inputs.
    device_str: Device string representation (e.g. 'cpu', 'cuda', 'tpu').
    is_training: Whether the model is prepared for training or inference.
    seq_len: Sequence length for benchmark runs.
    batch_size: Batch size for benchmark runs.
    modify_config_hook: Optional hook to modify model configuration.

  Returns:
    A tuple of (model, inputs dictionary).
  """
  registry = get_module_registry()

  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
      modify_config_hook=modify_config_hook,
  )

  with torch.device(device_str):
    model = module_spec.module_factory().to(dtype=dtype)

  if is_training:
    model.train()
  else:
    model.eval()

  _, inputs = module_spec.sample_inputs_factory(
      (batch_size, seq_len), device_str
  )

  inputs.pop("attention_mask", None)

  for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
      inputs[k] = v.to(dtype)

  if is_training:
    vocab_size = get_vocab_size(model.config)
    inputs["labels"] = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device_str,
        dtype=torch.long,
        requires_grad=False,
    )

  return model, inputs
