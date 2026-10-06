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

"""AdamW optimizer implementations for TPU tracing."""

import abc
import dataclasses
from typing import ClassVar

import torch
import torch.optim.adamw as _torch_adamw
from torch.utils import _pytree
from torch_tpu._internal.optimizers import optim as _optim
from torch_tpu._internal.utils import annotations as _annotations

__all__ = [
    "ReferenceAdamW",
    "TorchAdamW",
]


class _AdamWBase(_optim.Optimizer):
  """Internal base class for AdamW optimizer variants with nested ParamGroup."""

  @_annotations.experimental(
      "ParamGroup is experimental and may change or be removed without notice."
  )
  @dataclasses.dataclass
  class ParamGroup:
    """Holds model parameters and Adam optimizer states (m, v, steps)."""

    params: dict[str, torch.Tensor]
    opt_state_m: dict[str, torch.Tensor] = dataclasses.field(
        default_factory=dict
    )
    opt_state_v: dict[str, torch.Tensor] = dataclasses.field(
        default_factory=dict
    )
    opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

  _HYPERPARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
      "lr",
      "weight_decay",
      "beta1",
      "beta2",
      "eps",
      "use_bfloat16_moments",
  )

  def __init__(
      self,
      lr: float = 1e-3,
      weight_decay: float = 1e-2,
      beta1: float = 0.9,
      beta2: float = 0.999,
      eps: float = 1e-8,
      use_bfloat16_moments: bool = True,
  ):
    super().__init__(lr=lr, weight_decay=weight_decay)
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps
    self._use_bfloat16_moments = use_bfloat16_moments

  @_annotations.experimental(
      "init_param_group() is experimental and may change or be removed without"
      " notice."
  )
  def init_param_group(
      self, params: dict[str, torch.Tensor] | torch.nn.Module
  ) -> ParamGroup:
    """Initializes Adam optimizer state from model parameters or nn.Module."""
    if isinstance(params, torch.nn.Module):
      params = {
          name.replace(".", "_"): param.detach()
          for name, param in params.named_parameters()
      }
    if not params:
      return self.ParamGroup(params={})

    device = next(iter(params.values())).device
    moment_dtype = torch.bfloat16 if self._use_bfloat16_moments else None

    opt_state_m = {
        name: torch.zeros_like(
            param,
            dtype=moment_dtype,
            memory_format=torch.preserve_format,
            device=device,
        )
        for name, param in params.items()
    }
    opt_state_v = {
        name: torch.zeros_like(
            param,
            dtype=moment_dtype,
            memory_format=torch.preserve_format,
            device=device,
        )
        for name, param in params.items()
    }
    opt_steps = {
        name: torch.tensor(0.0, dtype=torch.float32, device=device)
        for name in params.keys()
    }

    return self.ParamGroup(
        params=params,
        opt_state_m=opt_state_m,
        opt_state_v=opt_state_v,
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


class ReferenceAdamW(_AdamWBase):
  """Reference pure PyTorch implementation of AdamW.

  Internal: not part of the public API. Use
  `torch.tpu.optimizers.TorchAdamW` instead.
  """

  def _reference_adamw_update(
      self,
      param: torch.Tensor,
      grad: torch.Tensor,
      m: torch.Tensor,
      v: torch.Tensor,
      step: float | torch.Tensor = 1.0,
  ) -> None:
    """Performs an in-place reference AdamW update on param, m, and v."""
    if self._weight_decay != 0.0:
      p_decayed = param * (1.0 - self._lr * self._weight_decay)
    else:
      p_decayed = param

    bias_correction1 = 1.0 - (self._beta1**step)
    bias_correction2 = 1.0 - (self._beta2**step)
    sqrt_bc2 = bias_correction2**0.5
    step_size = (self._lr / bias_correction1) * sqrt_bc2
    scaled_eps = self._eps * sqrt_bc2

    new_m = self._beta1 * m + (1.0 - self._beta1) * grad
    new_v = self._beta2 * v + (1.0 - self._beta2) * (grad * grad)
    denom = torch.sqrt(new_v) + scaled_eps
    new_p = p_decayed - (step_size * new_m) / denom

    param.copy_(new_p)
    m.copy_(new_m)
    v.copy_(new_v)

  def step(
      self, param_group: _AdamWBase.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> _AdamWBase.ParamGroup:
    """Reference pure PyTorch implementation of AdamW step."""
    for name, param in param_group.params.items():
      step_tensor = param_group.opt_steps[name]
      step_tensor.add_(1.0)
      self._reference_adamw_update(
          param=param,
          grad=grads[name],
          m=param_group.opt_state_m[name],
          v=param_group.opt_state_v[name],
          step=step_tensor,
      )
    return param_group


@_annotations.experimental(
    "TorchAdamW is experimental and may change or be removed without notice."
)
class TorchAdamW(_AdamWBase):
  """Torch AdamW implementation calling torch.optim.adamw.adamw directly."""

  _HYPERPARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
      "lr",
      "weight_decay",
      "beta1",
      "beta2",
      "eps",
      "use_bfloat16_moments",
      "amsgrad",
      "maximize",
      "foreach",
      "capturable",
      "differentiable",
      "fused",
      "grad_scale",
      "found_inf",
  )

  def __init__(
      self,
      lr: float = 1e-3,
      weight_decay: float = 1e-2,
      beta1: float = 0.9,
      beta2: float = 0.999,
      eps: float = 1e-8,
      use_bfloat16_moments: bool = True,
      amsgrad: bool = False,
      maximize: bool = False,
      foreach: bool | None = False,
      capturable: bool = True,
      differentiable: bool = False,
      fused: bool | None = False,
      grad_scale: torch.Tensor | None = None,
      found_inf: torch.Tensor | None = None,
  ):
    super().__init__(
        lr=lr,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        use_bfloat16_moments=use_bfloat16_moments,
    )
    self._amsgrad = amsgrad
    self._maximize = maximize
    self._foreach = foreach
    self._capturable = capturable
    self._differentiable = differentiable
    self._fused = fused
    self._grad_scale = grad_scale
    self._found_inf = found_inf

  @_annotations.experimental(
      "step() is experimental and may change or be removed without notice."
  )
  def step(
      self, param_group: _AdamWBase.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> _AdamWBase.ParamGroup:
    """Torch AdamW step calling torch.optim.adamw.adamw directly."""
    param_keys = list(param_group.params.keys())
    param_list = [param_group.params[k] for k in param_keys]
    grad_list = [grads[k] for k in param_keys]
    m_list = [param_group.opt_state_m[k] for k in param_keys]
    v_list = [param_group.opt_state_v[k] for k in param_keys]
    step_list = [param_group.opt_steps[k] for k in param_keys]

    _torch_adamw.adamw(
        params=param_list,
        grads=grad_list,
        exp_avgs=m_list,
        exp_avg_sqs=v_list,
        max_exp_avg_sqs=[],
        state_steps=step_list,
        foreach=self._foreach,
        capturable=self._capturable,
        differentiable=self._differentiable,
        fused=self._fused,
        grad_scale=self._grad_scale,
        found_inf=self._found_inf,
        amsgrad=self._amsgrad,
        beta1=self._beta1,
        beta2=self._beta2,
        lr=self._lr,
        weight_decay=self._weight_decay,
        eps=self._eps,
        maximize=self._maximize,
    )
    return param_group


def _adam_param_group_flatten(group: _AdamWBase.ParamGroup):
  children = (
      group.params,
      group.opt_state_m,
      group.opt_state_v,
      group.opt_steps,
  )
  return children, None


def _adam_param_group_unflatten(children, _context):
  return _AdamWBase.ParamGroup(
      params=children[0],
      opt_state_m=children[1],
      opt_state_v=children[2],
      opt_steps=children[3],
  )


_pytree.register_pytree_node(
    _AdamWBase.ParamGroup,
    _adam_param_group_flatten,
    _adam_param_group_unflatten,
)
