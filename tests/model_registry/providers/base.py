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

"""Abstract base class for model source providers."""

import abc
from collections.abc import Callable
from typing import Any


class BaseProvider(abc.ABC):
  """Abstract base class for model source providers.

  All concrete providers (TransformersProvider, DiffusersProvider,
  TimmProvider, TorchvisionProvider) inherit from this class.
  """

  def __init__(self, base_path: str = ""):
    self.base_path = base_path

  @abc.abstractmethod
  def get(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
  ) -> Any:
    """Instantiates and returns a ModuleSpec for one model.

    Args:
      name: Model identifier within this provider, e.g. a Hugging Face repo id
        such as "google/gemma-2-2b" or a timm / torchvision model name.
      load_weights: If True, loads pretrained weights; otherwise the model is
        initialized with random weights.
      modify_config_hook: Optional callable that receives the model config and
        returns a modified config before the model is instantiated.

    Returns:
      A ModuleSpec whose factories build the model, its sample inputs and
      its preprocessor.
    """
