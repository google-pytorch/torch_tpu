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

    if self.momentum != 0.0:
      opt_state_m = {
          name: torch.zeros_like(
              param,
              memory_format=torch.preserve_format,
              device=device,
          )
          for name, param in params.items()
      }
    else:
      opt_state_m = {}
    opt_steps = {
        name: torch.tensor(0.0, dtype=torch.float32, device=device)
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

  def step(
      self, param_group: AdamW.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> AdamW.ParamGroup:
    """Reference pure PyTorch implementation of AdamW step."""
    if not param_group.params:
      return param_group

    # Mathematical formulation of AdamW optimizer step:
    # -------------------------------------------------------------
    # For each parameter p_t with gradient g_t at step t:
    #   1. Weight decay:
    #        p_decayed = p_{t-1} * (1 - lr * weight_decay)
    #   2. First and second momentum moment estimates:
    #        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    #        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    #   3. Bias corrections:
    #        \hat{m}_t = m_t / (1 - beta1^t)
    #        \hat{v}_t = v_t / (1 - beta2^t)
    #   4. Parameter update:
    #        p_t = p_decayed - lr * \hat{m}_t / (\sqrt{\hat{v}_t} + eps)
    #
    # Scaled equivalent formulation (hoisting scalar operations outside loop):
    #   effective_step_size = (lr / (1 - beta1^t)) * \sqrt{1 - beta2^t}
    #   scaled_epsilon = eps * \sqrt{1 - beta2^t}
    #   p_t = p_decayed - (m_t * effective_step_size) / (\sqrt{v_t} + scaled_epsilon)
    # -------------------------------------------------------------
    first_step = next(iter(param_group.opt_steps.values()))
    new_step = first_step + 1.0

    bias_correction1 = 1.0 - (self.beta1**new_step)
    bias_correction2 = 1.0 - (self.beta2**new_step)
    sqrt_bias_correction2 = (
        bias_correction2**0.5
        if isinstance(new_step, (int, float))
        else torch.sqrt(bias_correction2)
    )
    effective_step_size = (self.lr / bias_correction1) * sqrt_bias_correction2
    scaled_epsilon = self.eps * sqrt_bias_correction2

    weight_decay_factor = (
        1.0 - self.lr * self.weight_decay if self.weight_decay != 0.0 else 1.0
    )
    beta1 = self.beta1
    one_minus_beta1 = 1.0 - self.beta1
    beta2 = self.beta2
    one_minus_beta2 = 1.0 - self.beta2

    new_params = {}
    new_opt_state_m = {}
    new_opt_state_v = {}
    new_opt_steps = {}

    for name, param in param_group.params.items():
      grad = grads[name]
      momentum = param_group.opt_state_m[name]
      variance = param_group.opt_state_v[name]

      param_decayed = (
          param * weight_decay_factor if weight_decay_factor != 1.0 else param
      )
      new_momentum = momentum * beta1 + grad * one_minus_beta1
      new_variance = variance * beta2 + (grad * grad) * one_minus_beta2
      denominator = torch.sqrt(new_variance) + scaled_epsilon
      new_param = (
          param_decayed - (new_momentum * effective_step_size) / denominator
      )

      new_params[name] = new_param.to(dtype=param.dtype)
      new_opt_state_m[name] = new_momentum.to(dtype=momentum.dtype)
      new_opt_state_v[name] = new_variance.to(dtype=variance.dtype)
      new_opt_steps[name] = param_group.opt_steps[name] + 1.0

    return self.ParamGroup(
        params=new_params,
        opt_state_m=new_opt_state_m,
        opt_state_v=new_opt_state_v,
        opt_steps=new_opt_steps,
        extra_state=param_group.extra_state,
    )


class FusedAdamw(AdamW):
  """Fused AdamW implementation calling torch.optim.adamw.adamw directly with fused=True."""

  def step(
      self, param_group: AdamW.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> AdamW.ParamGroup:
    """Fused AdamW step calling torch.optim.adamw.adamw directly."""
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
        foreach=False,
        capturable=True,
        fused=True,
        amsgrad=False,
        beta1=self.beta1,
        beta2=self.beta2,
        lr=self.lr,
        weight_decay=self.weight_decay,
        eps=self.eps,
        maximize=False,
    )
    return param_group


class ReferenceSgd(SGD):
  """Reference pure PyTorch implementation of SGD."""

  def _reference_sgd_update(
      self,
      param: torch.Tensor,
      grad: torch.Tensor,
      momentum_buffer: torch.Tensor | None = None,
      step: float | torch.Tensor = 1.0,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Performs functional reference SGD update on param and momentum buffer."""
    # Mathematical formulation of SGD with momentum and weight decay:
    # -------------------------------------------------------------
    #   1. Weight decay:
    #        d_p = grad + weight_decay * param
    #   2. Momentum buffer:
    #        if step == 1:
    #          b_1 = d_p
    #        else:
    #          b_t = momentum * b_{t-1} + (1 - dampening) * d_p
    #   3. Nesterov momentum:
    #        d_p = d_p + momentum * b_t  (if nesterov)
    #        d_p = b_t                   (otherwise)
    #   4. Parameter update:
    #        param_t = param_{t-1} - lr * d_p
    # -------------------------------------------------------------
    if self.weight_decay != 0.0:
      grad_with_weight_decay = grad + self.weight_decay * param
    else:
      grad_with_weight_decay = grad

    new_momentum_buffer = None
    if self.momentum != 0.0 and momentum_buffer is not None:
      new_momentum_buffer = torch.where(
          step == 1.0,
          grad_with_weight_decay,
          self.momentum * momentum_buffer
          + (1.0 - self.dampening) * grad_with_weight_decay,
      )
      if self.nesterov:
        grad_with_weight_decay = (
            grad_with_weight_decay + self.momentum * new_momentum_buffer
        )
      else:
        grad_with_weight_decay = new_momentum_buffer

    new_param = param - self.lr * grad_with_weight_decay
    return new_param, new_momentum_buffer

  def step(
      self, param_group: SGD.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> SGD.ParamGroup:
    """Reference pure PyTorch implementation of SGD step."""
    if not param_group.params:
      return param_group

    new_params = {}
    new_opt_state_m = {}
    new_opt_steps = {}

    for name, param in param_group.params.items():
      step_tensor = param_group.opt_steps[name] + 1.0
      momentum_buffer = param_group.opt_state_m.get(name, None)
      new_param, new_momentum_buffer = self._reference_sgd_update(
          param=param,
          grad=grads[name],
          momentum_buffer=momentum_buffer,
          step=step_tensor,
      )
      new_params[name] = new_param.to(dtype=param.dtype)
      if new_momentum_buffer is not None and momentum_buffer is not None:
        new_opt_state_m[name] = new_momentum_buffer.to(
            dtype=momentum_buffer.dtype
        )
      elif new_momentum_buffer is not None:
        new_opt_state_m[name] = new_momentum_buffer
      new_opt_steps[name] = step_tensor

    return self.ParamGroup(
        params=new_params,
        opt_state_m=new_opt_state_m,
        opt_steps=new_opt_steps,
        extra_state=param_group.extra_state,
    )


class FusedSgd(SGD):
  """Fused SGD implementation calling torch.optim.sgd.sgd directly with fused=True."""

  def step(
      self, param_group: SGD.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> SGD.ParamGroup:
    """Fused SGD step calling torch.optim.sgd.sgd directly."""
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
        weight_decay=self.weight_decay,
        momentum=self.momentum,
        lr=self.lr,
        dampening=self.dampening,
        nesterov=self.nesterov,
        maximize=False,
        foreach=False,
        fused=True,
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


def _adam_param_group_unflatten(children, context):
  del context  # Unused.
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


def _sgd_param_group_unflatten(children, context):
  del context  # Unused.
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
