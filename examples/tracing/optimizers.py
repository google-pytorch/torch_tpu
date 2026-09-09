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

"""Stateless functional optimizer library for compiling and running PyTorch training steps on TPU."""

import abc
import dataclasses
from typing import Any, ClassVar

import torch
import torch.optim.adamw as torch_adamw
import torch.optim.sgd as torch_sgd
from torch.utils import _pytree


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


class AdamW(Optimizer):
  """Base class for AdamW optimizer variants with nested ParamGroup."""

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
    extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

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
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.use_bfloat16_moments = use_bfloat16_moments

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
    moment_dtype = torch.bfloat16 if self.use_bfloat16_moments else None

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
  def step(
      self, param_group: ParamGroup, grads: dict[str, torch.Tensor]
  ) -> ParamGroup:
    """Performs optimizer step on param_group given grads and returns updated ParamGroup."""
    pass


class SGD(Optimizer):
  """Base class for SGD optimizer variants with nested ParamGroup."""

  @dataclasses.dataclass
  class ParamGroup:
    """Holds model parameters and SGD optimizer states (m, steps)."""

    params: dict[str, torch.Tensor]
    opt_state_m: dict[str, torch.Tensor] = dataclasses.field(
        default_factory=dict
    )
    opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __init__(
      self,
      lr: float = 1e-3,
      momentum: float = 0.0,
      dampening: float = 0.0,
      weight_decay: float = 0.0,
      nesterov: bool = False,
  ):
    super().__init__(lr=lr, weight_decay=weight_decay)
    self.momentum = momentum
    self.dampening = dampening
    self.nesterov = nesterov

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
        if self.momentum != 0.0
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
  def step(
      self, param_group: ParamGroup, grads: dict[str, torch.Tensor]
  ) -> ParamGroup:
    """Performs optimizer step on param_group given grads and returns updated ParamGroup."""
    pass


class ReferenceAdamw(AdamW):
  """Reference pure PyTorch implementation of AdamW."""

  def _reference_adamw_update(
      self,
      param: torch.Tensor,
      grad: torch.Tensor,
      m: torch.Tensor,
      v: torch.Tensor,
      step: float | torch.Tensor = 1.0,
  ) -> None:
    """Performs an in-place reference AdamW update on param, m, and v."""
    if self.weight_decay != 0.0:
      p_decayed = param * (1.0 - self.lr * self.weight_decay)
    else:
      p_decayed = param

    bias_correction1 = 1.0 - (self.beta1**step)
    bias_correction2 = 1.0 - (self.beta2**step)
    sqrt_bc2 = bias_correction2**0.5
    step_size = (self.lr / bias_correction1) * sqrt_bc2
    scaled_eps = self.eps * sqrt_bc2

    new_m = self.beta1 * m + (1.0 - self.beta1) * grad
    new_v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
    denom = torch.sqrt(new_v) + scaled_eps
    new_p = p_decayed - (step_size * new_m) / denom

    param.copy_(new_p)
    m.copy_(new_m)
    v.copy_(new_v)

  def step(
      self, param_group: AdamW.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> AdamW.ParamGroup:
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


class TorchAdamw(AdamW):
  """Torch AdamW implementation calling torch.optim.adamw.adamw directly."""

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
      fused: bool | None = True,
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
    self.amsgrad = amsgrad
    self.maximize = maximize
    self.foreach = foreach
    self.capturable = capturable
    self.differentiable = differentiable
    self.fused = fused
    self.grad_scale = grad_scale
    self.found_inf = found_inf

  def step(
      self, param_group: AdamW.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> AdamW.ParamGroup:
    """Torch AdamW step calling torch.optim.adamw.adamw directly."""
    param_keys = list(param_group.params.keys())
    param_list = [param_group.params[k] for k in param_keys]
    grad_list = [grads[k] for k in param_keys]
    m_list = [param_group.opt_state_m[k] for k in param_keys]
    v_list = [param_group.opt_state_v[k] for k in param_keys]
    step_list = [param_group.opt_steps[k] for k in param_keys]

    torch_adamw.adamw(
        params=param_list,
        grads=grad_list,
        exp_avgs=m_list,
        exp_avg_sqs=v_list,
        max_exp_avg_sqs=[],
        state_steps=step_list,
        foreach=self.foreach,
        capturable=self.capturable,
        differentiable=self.differentiable,
        fused=self.fused,
        grad_scale=self.grad_scale,
        found_inf=self.found_inf,
        amsgrad=self.amsgrad,
        beta1=self.beta1,
        beta2=self.beta2,
        lr=self.lr,
        weight_decay=self.weight_decay,
        eps=self.eps,
        maximize=self.maximize,
    )
    return param_group


class ReferenceSgd(SGD):
  """Reference pure PyTorch implementation of SGD."""

  def _reference_sgd_update(
      self,
      param: torch.Tensor,
      grad: torch.Tensor,
      m: torch.Tensor | None = None,
      step: float | torch.Tensor = 1.0,
  ) -> None:
    """Performs an in-place reference SGD update on param and momentum buffer m."""
    if self.weight_decay != 0.0:
      d_p = grad + self.weight_decay * param
    else:
      d_p = grad

    if self.momentum != 0.0 and m is not None:
      new_m = torch.where(
          step == 1.0, d_p, self.momentum * m + (1.0 - self.dampening) * d_p
      )
      if self.nesterov:
        d_p = d_p + self.momentum * new_m
      else:
        d_p = new_m
      m.copy_(new_m)

    new_p = param - self.lr * d_p
    param.copy_(new_p)

  def step(
      self, param_group: SGD.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> SGD.ParamGroup:
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


class TorchSgd(SGD):
  """Torch SGD implementation calling torch.optim.sgd.sgd directly."""

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
    self.maximize = maximize
    self.foreach = foreach
    self.fused = fused
    self.grad_scale = grad_scale
    self.found_inf = found_inf
    self.has_sparse_grad = has_sparse_grad

  def step(
      self, param_group: SGD.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> SGD.ParamGroup:
    """Torch SGD step calling torch.optim.sgd.sgd directly."""
    param_keys = list(param_group.params.keys())
    param_list = [param_group.params[k] for k in param_keys]
    grad_list = [grads[k] for k in param_keys]
    m_list = (
        [param_group.opt_state_m[k] for k in param_keys]
        if self.momentum != 0.0
        else []
    )
    step_list = [param_group.opt_steps[k] for k in param_keys]

    torch_sgd.sgd(
        params=param_list,
        d_p_list=grad_list,
        momentum_buffer_list=m_list,
        has_sparse_grad=self.has_sparse_grad,
        foreach=self.foreach,
        fused=self.fused,
        grad_scale=self.grad_scale,
        found_inf=self.found_inf,
        weight_decay=self.weight_decay,
        momentum=self.momentum,
        lr=self.lr,
        dampening=self.dampening,
        nesterov=self.nesterov,
        maximize=self.maximize,
    )
    torch.ops.aten._foreach_add_(step_list, 1.0)

    return param_group


# Register PyTree handlers for nested ParamGroup classes
def _adam_param_group_flatten(group: AdamW.ParamGroup):
  children = (
      group.params,
      group.opt_state_m,
      group.opt_state_v,
      group.opt_steps,
      group.extra_state,
  )
  return children, None


def _adam_param_group_unflatten(children, _context):
  return AdamW.ParamGroup(
      params=children[0],
      opt_state_m=children[1],
      opt_state_v=children[2],
      opt_steps=children[3],
      extra_state=children[4],
  )


_pytree.register_pytree_node(
    AdamW.ParamGroup,
    _adam_param_group_flatten,
    _adam_param_group_unflatten,
)


def _sgd_param_group_flatten(group: SGD.ParamGroup):
  children = (
      group.params,
      group.opt_state_m,
      group.opt_steps,
      group.extra_state,
  )
  return children, None


def _sgd_param_group_unflatten(children, _context):
  return SGD.ParamGroup(
      params=children[0],
      opt_state_m=children[1],
      opt_steps=children[2],
      extra_state=children[3],
  )


_pytree.register_pytree_node(
    SGD.ParamGroup,
    _sgd_param_group_flatten,
    _sgd_param_group_unflatten,
)


class NoOpOptimizer(Optimizer):
  """No-op optimizer that does not update parameters.

  Provides a no-op state container compatible with the Optimizer interface
  without mutating parameters. Note: For pure forward+backward benchmarking
  where gradients must be preserved as graph outputs against dead-code
  elimination (DCE) and set on model parameters for device synchronization,
  SingleTraceTrainer(optimizer=None) is the designated mechanism.
  """

  donate_inputs: bool = False

  @dataclasses.dataclass
  class ParamGroup:
    """Holds model parameters without optimizer state."""

    params: dict[str, torch.Tensor]
    extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

  def init_param_group(
      self, params: dict[str, torch.Tensor] | torch.nn.Module
  ) -> ParamGroup:
    """Initializes ParamGroup from model parameters or nn.Module."""
    if isinstance(params, torch.nn.Module):
      params = {
          name.replace(".", "_"): param.detach()
          for name, param in params.named_parameters()
      }
    return self.ParamGroup(params=params)

  def step(
      self, param_group: ParamGroup, grads: dict[str, torch.Tensor]
  ) -> ParamGroup:
    """Performs no-op step, returning param_group unchanged."""
    del grads
    return param_group


def _noop_param_group_flatten(group: NoOpOptimizer.ParamGroup):
  children = (group.params, group.extra_state)
  return children, None


def _noop_param_group_unflatten(children, _context):
  return NoOpOptimizer.ParamGroup(
      params=children[0],
      extra_state=children[1],
  )


_pytree.register_pytree_node(
    NoOpOptimizer.ParamGroup,
    _noop_param_group_flatten,
    _noop_param_group_unflatten,
)
