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

"""Optimizer implementations and optimization utilities."""

# pylint: disable=protected-access

from typing import Any
import torch
import torch.optim.adam as adam_module
import torch.optim.optimizer as optimizer_module
from torch_tpu._internal.optim import patch


class AdamW(torch.optim.AdamW):
  """A TPU-friendly AdamW optimizer.

  NOTE: This class is a fork of `torch.optim.AdamW` to bypass graph compilation
  limitations. Because it is a fork, it has maintenance overhead and will not
  automatically pick up upstream improvements to `torch.optim.AdamW`.

  Original implementation reference:
  - https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py

  Why we fork:
  - Dynamically allocated optimizer states during `torch.compile` (e.g. `step`
    counter hosted on CPU) cause compilation errors or graph breaks on TPU
    backends.
  - AdamW bypasses this by pre-allocating state tensors eagerly on the target
    TPU device during optimizer initialization.
  - Standard AdamW uses `_init_group` which is intercepted by PyTorch Dynamo.
    If there are pending mutations on parameters (like gradient updates from
    backward), Dynamo triggers an unconditional graph break. AdamW avoids
    this by using a custom `_my_init_group` method to prevent interception.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._eager_init_state()

  def _eager_init_state(self):
    for group in self.param_groups:
      for p in group["params"]:
        if p.requires_grad:
          state = self.state[p]
          if len(state) == 0:
            if group["fused"]:
              optimizer_module._device_dtype_check_for_fused(p)
            if group["capturable"] or group["fused"]:
              state["step"] = torch.zeros(
                  (),
                  dtype=optimizer_module._get_scalar_dtype(
                      is_fused=group["fused"]
                  ),
                  device=p.device,
              )
            else:
              state["step"] = torch.tensor(
                  0.0, dtype=optimizer_module._get_scalar_dtype(), device="cpu"
              )
            state["exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if group["amsgrad"]:
              state["max_exp_avg_sq"] = torch.zeros_like(
                  p, memory_format=torch.preserve_format
              )

  # patch starts here:
  # PyTorch Dynamo intercepts `_init_group` and triggers graph breaks if there
  # are pending mutations on parameters (like gradient updates from backward).
  # We use a custom `_my_init_group` name to prevent Dynamo from intercepting
  # this call, avoiding graph breaks.
  # Original signature:
  # def _init_group(self, group, params_with_grad, grads, exp_avgs, ...):
  def _my_init_group(
      self,
      group,
      params_with_grad,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
  ):
    # patch ends here
    has_complex = False
    for p in group["params"]:
      if p.grad is not None:
        has_complex |= torch.is_complex(p)
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError(
              "Adam does not support sparse gradients, please consider"
              " SparseAdam instead"
          )
        grads.append(p.grad)

        state = self.state[p]
        # Lazy state initialization
        if len(state) == 0:
          if group["fused"]:
            optimizer_module._device_dtype_check_for_fused(p)
          # Deliberately host `step` on CPU if both capturable and fused
          # are off.
          if group["capturable"] or group["fused"]:
            state["step"] = torch.zeros(
                (),
                dtype=optimizer_module._get_scalar_dtype(
                    is_fused=group["fused"]
                ),
                device=p.device,
            )
          else:
            state["step"] = torch.tensor(
                0.0, dtype=optimizer_module._get_scalar_dtype(), device="cpu"
            )
          # Exponential moving average of gradient values
          state["exp_avg"] = torch.zeros_like(
              p, memory_format=torch.preserve_format
          )
          # Exponential moving average of squared gradient values
          state["exp_avg_sq"] = torch.zeros_like(
              p, memory_format=torch.preserve_format
          )
          if group["amsgrad"]:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state["max_exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if group["amsgrad"]:
          max_exp_avg_sqs.append(state["max_exp_avg_sq"])
        if group["differentiable"] and state["step"].requires_grad:
          raise RuntimeError(
              "`requires_grad` is not supported for `step` in differentiable"
              " mode"
          )

        # Foreach without capturable does not support a tensor lr
        if (
            group["foreach"]
            and torch.is_tensor(group["lr"])
            and not group["capturable"]
        ):
          raise RuntimeError(
              "lr as a Tensor is not supported for capturable=False and"
              " foreach=True"
          )

        state_steps.append(state["step"])
    return has_complex

  @patch.use_grad_for_differentiable
  def step(self, closure=None):
    self._accelerator_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group["betas"]

      # patch starts here: call _my_init_group instead of _init_group to
      # avoid Dynamo graph breaks
      # has_complex = self._init_group(group, ...)
      has_complex = self._my_init_group(
          # patch ends here
          group,
          params_with_grad,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
      )

      adam_fn: Any = adam_module.adam
      adam_fn(
          params_with_grad,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          amsgrad=group["amsgrad"],
          has_complex=has_complex,
          beta1=beta1,
          beta2=beta2,
          lr=group["lr"],
          weight_decay=group["weight_decay"],
          eps=group["eps"],
          maximize=group["maximize"],
          foreach=group["foreach"],
          capturable=group["capturable"],
          differentiable=group["differentiable"],
          fused=group["fused"],
          grad_scale=getattr(self, "grad_scale", None),
          found_inf=getattr(self, "found_inf", None),
          decoupled_weight_decay=group["decoupled_weight_decay"],
      )

    return loss
