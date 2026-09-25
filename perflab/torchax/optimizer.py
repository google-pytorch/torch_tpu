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

"""Optimizer mapping and utilities for TorchAX."""

from typing import Callable

import optax
import torch


def _convert_adam(
    torch_optimizer: torch.optim.Optimizer,
) -> optax.GradientTransformation:
  """Converts torch.optim.Adam to optax.adam."""
  param_group = torch_optimizer.param_groups[0]
  lr = param_group.get("lr", 0.001)
  betas = param_group.get("betas", (0.9, 0.999))
  eps = param_group.get("eps", 1e-8)
  return optax.adam(
      learning_rate=lr,
      b1=betas[0],
      b2=betas[1],
      eps=eps,
  )


def _convert_adamw(
    torch_optimizer: torch.optim.Optimizer,
) -> optax.GradientTransformation:
  """Converts torch.optim.AdamW to optax.adamw."""
  param_group = torch_optimizer.param_groups[0]
  lr = param_group.get("lr", 0.001)
  betas = param_group.get("betas", (0.9, 0.999))
  eps = param_group.get("eps", 1e-8)
  weight_decay = param_group.get("weight_decay", 0.01)
  return optax.adamw(
      learning_rate=lr,
      b1=betas[0],
      b2=betas[1],
      eps=eps,
      weight_decay=weight_decay,
  )


_OPTIMIZER_CONVERTERS: dict[
    type[torch.optim.Optimizer],
    Callable[[torch.optim.Optimizer], optax.GradientTransformation],
] = {
    torch.optim.Adam: _convert_adam,
    torch.optim.AdamW: _convert_adamw,
}


def get_optax_optimizer(
    torch_optimizer: torch.optim.Optimizer,
) -> optax.GradientTransformation:
  """Initializes an Optax optimizer based on the PyTorch optimizer class.

  Args:
    torch_optimizer: The PyTorch optimizer instance from the benchmark factory.

  Returns:
    An equivalent Optax GradientTransformation.

  Raises:
    ValueError: If torch_optimizer is an unsupported optimizer class.
  """
  converter = _OPTIMIZER_CONVERTERS.get(type(torch_optimizer))
  if converter is None:
    supported_names = [c.__name__ for c in _OPTIMIZER_CONVERTERS]
    raise ValueError(
        f"Unsupported torch optimizer class: {type(torch_optimizer).__name__}."
        f" Supported optimizers are: {supported_names}."
    )

  return converter(torch_optimizer)
