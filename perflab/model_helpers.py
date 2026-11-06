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
from perflab import context as context_lib
from perflab import mode as mode_lib

if typing.TYPE_CHECKING:
  from tests import module_registry

# Batch size and sequence length of the models that run in presubmit scope only
# in the FakeTensorMode CPU smoke.
SMOKE_BATCH_SIZE = 1
SMOKE_SEQ_LEN = 128


def is_presubmit_smoke(ctx: context_lib.Context) -> bool:
  """Returns whether this is the presubmit-scope FakeTensorMode CPU smoke.

  A model too large for presubmit is registered with
  `PresubmitPolicy.CPU_SMOKE_ONLY`, so in presubmit scope it runs only in the
  FakeTensorMode CPU smoke, which allocates no storage, and there it builds at
  reduced sizes (e.g. `SMOKE_BATCH_SIZE` and `SMOKE_SEQ_LEN`).

  Args:
    ctx: The benchmark context.
  """
  return (
      ctx.run_scope is context_lib.RunScope.PRESUBMIT
      and mode_lib.fake_tensor_smoke_enabled()
  )


def init_small_random_weights(module: torch.nn.Module) -> None:
  """Re-initializes a module to N(0, 0.01) weights and zero biases.

  Randomly initialized Diffusers and vision transformers overflow in bfloat16
  with their default initialization; use with `model.apply(...)`.
  """
  with torch.no_grad():
    if (
        isinstance(getattr(module, "weight", None), torch.Tensor)
        and module.weight.is_floating_point()
    ):
      module.weight.normal_(std=0.01)
    if isinstance(getattr(module, "bias", None), torch.Tensor):
      module.bias.zero_()


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
def get_module_registry() -> "module_registry.ModuleRegistry":
  """Returns a cached ModuleRegistry instance.

  Under `--fake_tensor_mode`, the registry has no weights cache directory: it
  reads model configs from the in-tree resources and otherwise falls back to
  GCS as usual.
  """
  # Imported lazily rather than at module scope. Importing module_registry
  # is likely to probe the CUDA driver at import time and thereby
  # poisons fork() for CUDA.
  from tests import module_registry  # pylint: disable=g-import-not-at-top

  if mode_lib.fake_tensor_smoke_enabled():
    # FakeTensorMode never loads weights, and reading the weights cache
    # directory made the CPU smoke shards time out: the registry falls back only
    # on a clean miss, not on a timeout, and raises if the directory is missing.
    return module_registry.ModuleRegistry(base_path="")
  return module_registry.ModuleRegistry()


def load_hf_llm(
    model_name: str,
    dtype: torch.dtype,
    device_str: str,
    is_training: bool = False,
    seq_len: int = 2048,
    batch_size: int = 1,
    modify_config_hook: Callable[[Any], Any] | None = None,
    keep_vision_inputs: bool = False,
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
    keep_vision_inputs: Whether to keep the `pixel_values` and
      `image_position_ids` sample inputs. Vision and vision-language models need
      them; they are dropped by default.

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
    model = module_spec.module_factory(dtype=dtype)

  if is_training:
    model.train()
  else:
    model.eval()

  _, inputs = module_spec.sample_inputs_factory(
      (batch_size, seq_len), device_str
  )

  # Providing an attention mask with a full-length sequence is redundant and
  # causes extra overhead.
  inputs.pop("attention_mask", None)

  # These inputs are only used for multimodal models. Enable these after
  # verifying that they work on TPU.
  if not keep_vision_inputs:
    inputs.pop("pixel_values", None)
    inputs.pop("image_position_ids", None)

  for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
      inputs[k] = v.to(dtype)

  if is_training:
    vocab_size = get_vocab_size(model.config)
    labels_shape = (
        inputs["input_ids"].shape
        if "input_ids" in inputs
        else (batch_size, seq_len)
    )
    inputs["labels"] = torch.randint(
        0,
        vocab_size,
        labels_shape,
        device=device_str,
        dtype=torch.long,
        requires_grad=False,
    )

  return model, inputs
