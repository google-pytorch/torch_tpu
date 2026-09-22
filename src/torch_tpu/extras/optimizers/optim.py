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

"""Stateless functional optimizer base class and shared types for TPU tracing."""

import abc
from typing import Any, ClassVar

import torch


class Optimizer(abc.ABC):
  """Abstract base class for stateless functional optimizers operating on nested ParamGroup."""

  ParamGroup: ClassVar[Any]

  def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0):
    self.lr = lr
    self.weight_decay = weight_decay

  @abc.abstractmethod
  def init_param_group(
      self, params: dict[str, torch.Tensor] | torch.nn.Module
  ) -> Any:
    """Initializes a fresh ParamGroup instance from model parameters or nn.Module."""
    pass

  @abc.abstractmethod
  def step(self, param_group: Any, grads: dict[str, torch.Tensor]) -> Any:
    """Performs optimizer step on param_group given grads and returns updated ParamGroup."""
    pass

  def __call__(self, param_group: Any, grads: dict[str, torch.Tensor]) -> Any:
    return self.step(param_group, grads)
