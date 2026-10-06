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

"""SGD optimizer implementations for TPU tracing."""

import abc
import dataclasses
from typing import ClassVar

import torch
import torch.optim.sgd as _torch_sgd
from torch.utils import _pytree
from torch_tpu._internal.optimizers import optim as _optim
from torch_tpu._internal.utils import annotations as _annotations

__all__ = [
    "ReferenceSgd",
    "TorchSgd",
]


class _SgdBase(_optim.Optimizer):
  """Internal base class for SGD optimizer variants with nested ParamGroup."""

  @_annotations.experimental(
      "ParamGroup is experimental and may change or be removed without notice."
  )
  @dataclasses.dataclass
  class ParamGroup:
    """Holds model parameters and SGD optimizer states (m, steps)."""

    params: dict[str, torch.Tensor]
    opt_state_m: dict[str, torch.Tensor] = dataclasses.field(
        default_factory=dict
    )
    opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

  _HYPERPARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
      "lr",
      "momentum",
      "dampening",
      "weight_decay",
      "nesterov",
  )

  def __init__(
      self,
      lr: float = 1e-3,
      momentum: float = 0.0,
      dampening: float = 0.0,
      weight_decay: float = 0.0,
      nesterov: bool = False,
  ):
    super().__init__(lr=lr, weight_decay=weight_decay)
    self._momentum = momentum
    self._dampening = dampening
    self._nesterov = nesterov

  @_annotations.experimental(
      "init_param_group() is experimental and may change or be removed without"
      " notice."
  )
  def init_param_group(
      self, params: dict[str, torch.Tensor] | torch.nn.Module
  ) -> ParamGroup:
    """Initializes SGD optimizer state from model parameters or nn.Module."""
    if isinstance(params, torch.nn.Module):
      params = {
          name.replace(".", "_"): param.detach()
          for name, param in params.named_parameters()
      }
    if not params:
      return self.ParamGroup(params={})

    device = next(iter(params.values())).device

    opt_state_m = (
        {
            name: torch.zeros_like(
                param,
                memory_format=torch.preserve_format,
                device=device,
            )
            for name, param in params.items()
        }
        if self._momentum != 0.0
        else {}
    )
    opt_steps = {
        name: torch.tensor(1.0, dtype=torch.float32, device=device)
        for name in params.keys()
    }

    return self.ParamGroup(
        params=params,
        opt_state_m=opt_state_m,
        opt_steps=opt_steps,
    )

  @abc.abstractmethod
  @_annotations.experimental(
      "step() is experimental and may change or be removed without notice."
  )
  def step(
      self, param_group: ParamGroup, grads: dict[str, torch.Tensor]
  ) -> ParamGroup:
    """Performs optimizer step on param_group given grads and returns updated ParamGroup."""
    pass


class ReferenceSgd(_SgdBase):
  """Reference pure PyTorch implementation of SGD.

  Internal: not part of the public API. Use
  `torch.tpu.optimizers.TorchSgd` instead.
  """

  def _reference_sgd_update(
      self,
      param: torch.Tensor,
      grad: torch.Tensor,
      m: torch.Tensor | None = None,
      step: float | torch.Tensor = 1.0,
  ) -> None:
    """Performs an in-place reference SGD update on param and momentum buffer m."""
    if self._weight_decay != 0.0:
      d_p = grad + self._weight_decay * param
    else:
      d_p = grad

    if self._momentum != 0.0 and m is not None:
      new_m = torch.where(
          step == 1.0, d_p, self._momentum * m + (1.0 - self._dampening) * d_p
      )
      if self._nesterov:
        d_p = d_p + self._momentum * new_m
      else:
        d_p = new_m
      m.copy_(new_m)

    new_p = param - self._lr * d_p
    param.copy_(new_p)

  def step(
      self, param_group: _SgdBase.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> _SgdBase.ParamGroup:
    """Reference pure PyTorch implementation of SGD step."""
    for name, param in param_group.params.items():
      step_tensor = param_group.opt_steps[name]
      m = param_group.opt_state_m.get(name, None)
      self._reference_sgd_update(
          param=param,
          grad=grads[name],
          m=m,
          step=step_tensor,
      )
      step_tensor.add_(1.0)
    return param_group


@_annotations.experimental(
    "TorchSgd is experimental and may change or be removed without notice."
)
class TorchSgd(_SgdBase):
  """Torch SGD implementation calling torch.optim.sgd.sgd directly."""

  _HYPERPARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
      "lr",
      "momentum",
      "dampening",
      "weight_decay",
      "nesterov",
      "maximize",
      "foreach",
      "fused",
      "grad_scale",
      "found_inf",
      "has_sparse_grad",
  )

  def __init__(
      self,
      lr: float = 1e-3,
      momentum: float = 0.0,
      dampening: float = 0.0,
      weight_decay: float = 0.0,
      nesterov: bool = False,
      maximize: bool = False,
      foreach: bool | None = False,
      fused: bool | None = True,
      grad_scale: torch.Tensor | None = None,
      found_inf: torch.Tensor | None = None,
      has_sparse_grad: bool = False,
  ):
    super().__init__(
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    self._maximize = maximize
    self._foreach = foreach
    self._fused = fused
    self._grad_scale = grad_scale
    self._found_inf = found_inf
    self._has_sparse_grad = has_sparse_grad

  @_annotations.experimental(
      "step() is experimental and may change or be removed without notice."
  )
  def step(
      self, param_group: _SgdBase.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> _SgdBase.ParamGroup:
    """Torch SGD step calling torch.optim.sgd.sgd directly."""
    param_keys = list(param_group.params.keys())
    param_list = [param_group.params[k] for k in param_keys]
    grad_list = [grads[k] for k in param_keys]
    m_list = (
        [param_group.opt_state_m[k] for k in param_keys]
        if self._momentum != 0.0
        else []
    )
    step_list = [param_group.opt_steps[k] for k in param_keys]

    _torch_sgd.sgd(
        params=param_list,
        d_p_list=grad_list,
        momentum_buffer_list=m_list,
        has_sparse_grad=self._has_sparse_grad,
        foreach=self._foreach,
        fused=self._fused,
        grad_scale=self._grad_scale,
        found_inf=self._found_inf,
        weight_decay=self._weight_decay,
        momentum=self._momentum,
        lr=self._lr,
        dampening=self._dampening,
        nesterov=self._nesterov,
        maximize=self._maximize,
    )
    torch.ops.aten._foreach_add_(step_list, 1.0)

    return param_group


def _sgd_param_group_flatten(group: _SgdBase.ParamGroup):
  children = (
      group.params,
      group.opt_state_m,
      group.opt_steps,
  )
  return children, None


def _sgd_param_group_unflatten(children, _context):
  return _SgdBase.ParamGroup(
      params=children[0],
      opt_state_m=children[1],
      opt_steps=children[2],
  )


_pytree.register_pytree_node(
    _SgdBase.ParamGroup,
    _sgd_param_group_flatten,
    _sgd_param_group_unflatten,
)
