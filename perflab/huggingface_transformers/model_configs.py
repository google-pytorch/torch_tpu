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

"""Model configurations for HuggingFace Transformers examples."""

from typing import Callable

from etils import epath
import transformers

_MINI_MODEL_IDS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507-MINI",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-MINI",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct-MINI",
]

_ALL_MODEL_IDS = _MINI_MODEL_IDS + [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Llama-3.2-1B",
]

# Get the parent package name. The absolute name differs internally vs
# externally, but the relative name is the same.
_RESOURCE_PATH = __name__.rpartition(".")[0]

ConfigLoader = Callable[[], transformers.configuration_utils.PretrainedConfig]


def create_path_for_model_id(model_id: str) -> epath.Path:
  """Creates and validates a model id.

  Args:
    model_id: The HuggingFace model ID, such as `Qwen/Qwen3-0.6B`.

  Returns:
    The path to the model config.

  Raises:
    ValueError: If the model_id is empty.
  """
  if not model_id:
    raise ValueError("Model ID is empty.")

  path = (
      epath.resource_path(_RESOURCE_PATH)
      / f"model_configs/{model_id}/config.json"
  )
  if not path.exists():
    raise ValueError(f"Model config not found: {model_id}")

  return path


def create_config_loader(
    model_id: str,
) -> ConfigLoader:
  """Creates a callable that loads a model config.

  No file access is done until the callable is called. This allows
  parameterizization of tests without triggering
  file access before the test framework has initialized.

  Args:
    model_id: The HuggingFace model ID, such as `Qwen/Qwen3-0.6B`.

  Returns:
    A callable that loads the model config.
  """

  def _loader():
    # To delay file access, do not move the following outside the closure.
    path = create_path_for_model_id(model_id)
    return transformers.AutoConfig.from_pretrained(path)

  return _loader


def get_mini_model_configs() -> list[tuple[str, transformers.AutoConfig]]:
  """Returns a list of (model_id, config) for mini models."""
  configs = []
  for model_id in _MINI_MODEL_IDS:
    configs.append((model_id, create_config_loader(model_id)()))
  return configs


def get_all_model_configs() -> list[tuple[str, transformers.AutoConfig]]:
  """Returns a list of (model_id, config) for all models."""
  configs = []
  for model_id in _ALL_MODEL_IDS:
    configs.append((model_id, create_config_loader(model_id)()))
  return configs


def get_lazy_mini_model_configs() -> list[tuple[str, ConfigLoader]]:
  """Returns a list of (model_id, config_loader_func) for mini models."""
  configs = []
  for model_id in _MINI_MODEL_IDS:
    configs.append((model_id, create_config_loader(model_id)))
  return configs


def get_lazy_all_model_configs() -> list[tuple[str, ConfigLoader]]:
  """Returns a list of (model_id, config_loader_func) for all models."""
  configs = []
  for model_id in _ALL_MODEL_IDS:
    configs.append((model_id, create_config_loader(model_id)))
  return configs
