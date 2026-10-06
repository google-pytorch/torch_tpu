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

"""Muon optimizer implementations for TPU tracing."""

import abc
import dataclasses
import math
from typing import Any, ClassVar

import torch
import torch.optim._muon as _torch_muon
from torch.utils import _pytree
from torch_tpu._internal.optimizers import optim as _optim
from torch_tpu._internal.utils import annotations as _annotations

__all__ = [
    "ReferenceMuon",
    "TorchMuon",
]


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
  """Newton-Schulz iteration to compute zeroth power / orthogonalization of 2D grad."""
  if ns_steps >= 100:
    raise ValueError(
        "Number of steps must be less than 100 for computational efficiency"
    )
  if len(grad.shape) != 2:
    raise ValueError("Input tensor gradient must be a 2D matrix")
  if len(ns_coefficients) != 3:
    raise ValueError("Coefficients must be a tuple of exactly 3 values")
  a, b, c = ns_coefficients
  ortho_grad = grad.to(dtype=torch.bfloat16, copy=True)
  if grad.size(0) > grad.size(1):
    ortho_grad = ortho_grad.T
  ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
  for _ in range(ns_steps):
    gram_matrix = ortho_grad @ ortho_grad.T
    gram_update = torch.addmm(
        gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
    )
    ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
  if grad.size(0) > grad.size(1):
    ortho_grad = ortho_grad.T
  return ortho_grad


def _adjust_lr(
    lr: float, adjust_lr_fn: str | None, param_shape: torch.Size
) -> float:
  """Default learning rate adjustment used by Muon."""
  a, b = param_shape[:2]
  if adjust_lr_fn is None or adjust_lr_fn == "original":
    adjusted_ratio = math.sqrt(max(1, a / b))
  elif adjust_lr_fn == "match_rms_adamw":
    adjusted_ratio = 0.2 * math.sqrt(max(a, b))
  elif adjust_lr_fn == "spectral_unclamped":
    adjusted_ratio = math.sqrt(a / b)
  else:
    adjusted_ratio = 1.0
  return lr * adjusted_ratio


class _MuonBase(_optim.Optimizer):
  """Internal base class for Muon optimizer variants with nested ParamGroup."""

  DEFAULT_A: float = 3.4445
  DEFAULT_B: float = -4.7750
  DEFAULT_C: float = 2.0315
  DEFAULT_NS_STEPS: int = 5
  EPS: float = 1e-7

  __tt_api_stages__: ClassVar[dict[str, _annotations.ApiStageInfo]] = {
      'DEFAULT_A': _annotations.ApiStageInfo(
          stage=_annotations.Stage.EXPERIMENTAL,
          reason=(
              'DEFAULT_A is experimental and may change or be removed without'
              ' notice.'
          ),
      ),
      'DEFAULT_B': _annotations.ApiStageInfo(
          stage=_annotations.Stage.EXPERIMENTAL,
          reason=(
              'DEFAULT_B is experimental and may change or be removed without'
              ' notice.'
          ),
      ),
      'DEFAULT_C': _annotations.ApiStageInfo(
          stage=_annotations.Stage.EXPERIMENTAL,
          reason=(
              'DEFAULT_C is experimental and may change or be removed without'
              ' notice.'
          ),
      ),
      'DEFAULT_NS_STEPS': _annotations.ApiStageInfo(
          stage=_annotations.Stage.EXPERIMENTAL,
          reason=(
              'DEFAULT_NS_STEPS is experimental and may change or be removed'
              ' without notice.'
          ),
      ),
      'EPS': _annotations.ApiStageInfo(
          stage=_annotations.Stage.EXPERIMENTAL,
          reason=(
              'EPS is experimental and may change or be removed without notice.'
          ),
      ),
  }

  @_annotations.experimental(
      'ParamGroup is experimental and may change or be removed without notice.'
  )
  @dataclasses.dataclass
  class ParamGroup:
    """Holds model parameters and Muon optimizer states (m, steps)."""

    params: dict[str, torch.Tensor]
    opt_state_m: dict[str, torch.Tensor] = dataclasses.field(
        default_factory=dict
    )
    opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __init__(
      self,
      lr: float = 1e-3,
      weight_decay: float = 0.1,
      momentum: float = 0.95,
      nesterov: bool = True,
      ns_coefficients: tuple[float, float, float] = (
          DEFAULT_A,
          DEFAULT_B,
          DEFAULT_C,
      ),
      eps: float = EPS,
      ns_steps: int = DEFAULT_NS_STEPS,
      adjust_lr_fn: str | None = None,
      use_bfloat16_moments: bool = False,
  ):
    super().__init__(lr=lr, weight_decay=weight_decay)
    self.momentum = momentum
    self.nesterov = nesterov
    self.ns_coefficients = ns_coefficients
    self.eps = eps
    self.ns_steps = ns_steps
    self.adjust_lr_fn = adjust_lr_fn
    self.use_bfloat16_moments = use_bfloat16_moments

  @_annotations.experimental(
      'init_param_group() is experimental and may change or be removed without'
      ' notice.'
  )
  def init_param_group(
      self, params: dict[str, torch.Tensor] | torch.nn.Module
  ) -> ParamGroup:
    """Initializes Muon optimizer state from model parameters or nn.Module."""
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
      'step() is experimental and may change or be removed without notice.'
  )
  def step(
      self, param_group: ParamGroup, grads: dict[str, torch.Tensor]
  ) -> ParamGroup:
    """Performs optimizer step on param_group given grads and returns updated ParamGroup."""
    pass


class ReferenceMuon(_MuonBase):
  """Reference pure PyTorch implementation of Muon.

  Internal: not part of the public API. Use
  `torch.tpu.optimizers.TorchMuon` instead.
  """

  def _reference_muon_update(
      self,
      param: torch.Tensor,
      grad: torch.Tensor,
      m: torch.Tensor,
  ) -> None:
    """Performs an in-place reference Muon update on param and momentum buffer m."""
    if grad.ndim != 2:
      raise ValueError("Param gradient must be a 2D matrix")

    m.lerp_(grad, 1.0 - self.momentum)
    update = grad.lerp(m, self.momentum) if self.nesterov else m

    update = _zeropower_via_newtonschulz(
        update, self.ns_coefficients, self.ns_steps, self.eps
    )
    adjusted_lr = _adjust_lr(self.lr, self.adjust_lr_fn, param.shape)

    param.mul_(1.0 - self.lr * self.weight_decay)
    param.add_(update, alpha=-adjusted_lr)

  def step(
      self, param_group: _MuonBase.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> _MuonBase.ParamGroup:
    """Reference pure PyTorch implementation of Muon step."""
    for name, param in param_group.params.items():
      self._reference_muon_update(
          param=param,
          grad=grads[name],
          m=param_group.opt_state_m[name],
      )
      param_group.opt_steps[name].add_(1.0)
    return param_group


@_annotations.experimental(
    'TorchMuon is experimental and may change or be removed without notice.'
)
class TorchMuon(_MuonBase):
  """Torch Muon implementation calling torch.optim._muon.muon directly."""

  def __init__(
      self,
      lr: float = 1e-3,
      weight_decay: float = 0.1,
      momentum: float = 0.95,
      nesterov: bool = True,
      ns_coefficients: tuple[float, float, float] = (
          _MuonBase.DEFAULT_A,
          _MuonBase.DEFAULT_B,
          _MuonBase.DEFAULT_C,
      ),
      eps: float = _MuonBase.EPS,
      ns_steps: int = _MuonBase.DEFAULT_NS_STEPS,
      adjust_lr_fn: str | None = None,
      use_bfloat16_moments: bool = False,
      foreach: bool | None = False,
      has_complex: bool = False,
  ):
    super().__init__(
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        eps=eps,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
        use_bfloat16_moments=use_bfloat16_moments,
    )
    self.foreach = foreach
    self.has_complex = has_complex

  @_annotations.experimental(
      'step() is experimental and may change or be removed without notice.'
  )
  def step(
      self, param_group: _MuonBase.ParamGroup, grads: dict[str, torch.Tensor]
  ) -> _MuonBase.ParamGroup:
    """Torch Muon step calling torch.optim._muon.muon directly."""
    param_keys = list(param_group.params.keys())
    param_list = [param_group.params[k] for k in param_keys]
    grad_list = [grads[k] for k in param_keys]
    m_list = [param_group.opt_state_m[k] for k in param_keys]
    step_list = [param_group.opt_steps[k] for k in param_keys]

    _torch_muon.muon(
        params=param_list,
        grads=grad_list,
        muon_momentum_bufs=m_list,
        foreach=self.foreach,
        lr=self.lr,
        weight_decay=self.weight_decay,
        momentum=self.momentum,
        nesterov=self.nesterov,
        ns_coefficients=self.ns_coefficients,
        ns_steps=self.ns_steps,
        eps=self.eps,
        adjust_lr_fn=self.adjust_lr_fn,
        has_complex=self.has_complex,
    )
    torch.ops.aten._foreach_add_(step_list, 1.0)

    return param_group


def _muon_param_group_flatten(group: _MuonBase.ParamGroup):
  children = (
      group.params,
      group.opt_state_m,
      group.opt_steps,
      group.extra_state,
  )
  return children, None


def _muon_param_group_unflatten(children, _context):
  return _MuonBase.ParamGroup(
      params=children[0],
      opt_state_m=children[1],
      opt_steps=children[2],
      extra_state=children[3],
  )


_pytree.register_pytree_node(
    _MuonBase.ParamGroup,
    _muon_param_group_flatten,
    _muon_param_group_unflatten,
)
