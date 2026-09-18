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

"""Public API for torch_tpu._internal.compile."""

from typing import Any

import torch
from torch._dynamo.backends import registry
from torch._functorch._aot_autograd import utils as aot_utils
from torch_tpu._internal.compile._backend import async_compile
from torch_tpu._internal.compile._backend import AsyncCompilationSubmitted
from torch_tpu._internal.compile._backend import resolve_compilations
from torch_tpu._internal.compile._backend import TpuBackend
from torch_tpu._internal.compile.debug import TpuCompileDebug

# Register "tpu" backend
registry.register_backend(compiler_fn=TpuBackend(), name="tpu")  # pyrefly: ignore[bad-argument-type]


def _prepare_module_for_tpu(module: torch.nn.Module) -> None:
  """Applies the TPU model rewrites that have to happen before tracing."""
  try:
    # Not a declared dep: `cache` needs `transformers`, which most TPU programs
    # do not depend on, so this stays an optional, best-effort import.
    # pylint: disable-next=g-import-not-at-top
    from torch_tpu._internal import cache as cache_lib  # pyrefly: ignore[missing-module-attribute]
  except ImportError:
    return
  cache_lib.register_tpu_cache_hook(module)


def tpu_compile(model: Any, *, backend: Any = "tpu", **kwargs: Any) -> Any:
  """Compiles `model` for TPU.

  Equivalent to `torch.compile(model, backend="tpu")`, except that it also
  applies the TPU-specific rewrites that `TpuBackend` cannot apply itself: the
  backend is only handed a `GraphModule`, so anything that has to change what
  gets traced (currently, upgrading a Hugging Face `DynamicCache` to its
  preallocated in-place equivalent so decode does not retrace every step) has
  to happen here. Prefer this over calling `torch.compile` directly.

  Args:
    model: The object to compile. Rewrites are applied only if it is an
      `nn.Module`; plain callables are passed straight through.
    backend: The Dynamo backend, overridable to pass a configured `TpuBackend`
      instance instead of the registered default.
    **kwargs: Forwarded verbatim to `torch.compile`.

  Returns:
    The compiled object.
  """
  if isinstance(model, torch.nn.Module):
    _prepare_module_for_tpu(model)
  return torch.compile(model, backend=backend, **kwargs)


def _initialize_graphsafe_rng():
  """Initializes graphsafe RNG operations for TPU.

  Registers 'privateuseone' (TPU) for graphsafe RNG and enables
  necessary config flags.
  """
  aot_utils.register_graphsafe_rng_device_type("privateuseone")
  # Add "tpu" to the supported device types for graphsafe RNG, as torch_tpu
  # renames "privateuseone" to "tpu".
  aot_utils._GRAPHSAFE_RNG_DEVICE_TYPES.add("tpu")  # pylint: disable=protected-access

  # pylint: disable=protected-access
  torch._functorch.config.graphsafe_rng_functionalization = True
  torch._functorch.config.functionalize_rng_ops = False
  # pylint: enable=protected-access


# TODO: b/496168350 - Remove if-condition once generic graphsafe RNG is
# available in OSS PyTorch release.
if hasattr(aot_utils, "register_graphsafe_rng_device_type"):
  _initialize_graphsafe_rng()


# New Dynamo flag that enables tracing through the `backward` function call.
# This enables the possibility of generating of a single fx graph containing
# both forward and backward subgraphs, rather than a discrete fx graph for each.
# See b/480979694 for details.
# pylint: disable=protected-access
if hasattr(torch._dynamo.config, "trace_autograd_ops"):
  torch._dynamo.config.trace_autograd_ops = True
# Disables PyTorch Dynamo's default graph break on RNN, LSTM, and GRU modules,
# allowing Dynamo to trace recurrent layers into aten::lstm and aten::gru nodes
# for lowering to TorchTPU's StableHLO kernels.
if hasattr(torch._dynamo.config, "allow_rnn"):
  torch._dynamo.config.allow_rnn = True
# pylint: enable=protected-access


def _register_scan_operator() -> None:
  """Registers the custom scan operator implementation for TPU."""
  from torch_tpu._internal.compile import scan as _  # pylint: disable=g-import-not-at-top  # noqa: F401


_register_scan_operator()


def _register_fused_optimizer_meta_kernels() -> None:
  """Registers missing PyTorch Meta kernels for ATen fused optimizers."""
  # pylint: disable=g-import-not-at-top,protected-access
  from torch._decomp import _add_op_to_registry
  from torch._decomp import meta_table
  from torch._meta_registrations import meta__fused_adam
  from torch._meta_registrations import meta__fused_adam_

  def _register_meta_op(op_overload, fn):
    _add_op_to_registry(meta_table, op_overload, fn)
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        op_overload.name(), "Meta"
    ):
      op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

  _register_meta_op(torch.ops.aten._fused_adam_.tensor_lr, meta__fused_adam_)
  _register_meta_op(torch.ops.aten._fused_adamw_.tensor_lr, meta__fused_adam_)
  _register_meta_op(torch.ops.aten._fused_adam.tensor_lr, meta__fused_adam)
  _register_meta_op(torch.ops.aten._fused_adamw.default, meta__fused_adam)
  _register_meta_op(torch.ops.aten._fused_adamw.tensor_lr, meta__fused_adam)

  def _meta_fused_sgd_(
      self,
      grads,
      momentum_buffer_list,
      *,
      weight_decay,
      momentum,
      lr,
      dampening,
      nesterov,
      maximize,
      is_first_step,
      grad_scale=None,
      found_inf=None,
  ):
    del weight_decay, momentum, lr, dampening, nesterov
    del maximize, is_first_step, grad_scale, found_inf
    for l in (self, grads, momentum_buffer_list):
      torch._check(
          isinstance(l, list),
          lambda: f"expected tensor list but got {type(l)}",
      )

  def _meta_fused_sgd(
      self,
      grads,
      momentum_buffer_list,
      *,
      weight_decay,
      momentum,
      lr,
      dampening,
      nesterov,
      maximize,
      is_first_step,
      grad_scale=None,
      found_inf=None,
  ):
    _meta_fused_sgd_(
        self,
        grads,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        is_first_step=is_first_step,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return (
        [torch.empty_like(t) for t in self],
        [torch.empty_like(t) for t in grads],
        [torch.empty_like(t) for t in momentum_buffer_list],
    )

  _register_meta_op(torch.ops.aten._fused_sgd_.default, _meta_fused_sgd_)
  _register_meta_op(torch.ops.aten._fused_sgd_.tensor_lr, _meta_fused_sgd_)
  _register_meta_op(torch.ops.aten._fused_sgd.default, _meta_fused_sgd)
  _register_meta_op(torch.ops.aten._fused_sgd.tensor_lr, _meta_fused_sgd)

  def _meta_fused_adagrad_(
      self,
      grads,
      state_sums,
      state_steps,
      *,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale=None,
      found_inf=None,
  ):
    del lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf
    for l in (self, grads, state_sums, state_steps):
      torch._check(
          isinstance(l, list),
          lambda: f"expected tensor list but got {type(l)}",
      )

  def _meta_fused_adagrad_default(
      self,
      grads,
      state_sums,
      state_steps,
      *,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale=None,
      found_inf=None,
  ):
    _meta_fused_adagrad_(
        self,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return (
        [torch.empty_like(t) for t in self],
        [torch.empty_like(t) for t in grads],
        [torch.empty_like(t) for t in state_sums],
        [torch.empty_like(t) for t in state_steps],
    )

  def _meta_fused_adagrad_tensor_lr(
      self,
      grads,
      state_sums,
      state_steps,
      *,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale=None,
      found_inf=None,
  ):
    _meta_fused_adagrad_(
        self,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return (
        [torch.empty_like(t) for t in self],
        [torch.empty_like(t) for t in grads],
        [torch.empty_like(t) for t in state_sums],
    )

  _register_meta_op(
      torch.ops.aten._fused_adagrad_.default, _meta_fused_adagrad_
  )
  _register_meta_op(
      torch.ops.aten._fused_adagrad_.tensor_lr, _meta_fused_adagrad_
  )
  _register_meta_op(
      torch.ops.aten._fused_adagrad.default, _meta_fused_adagrad_default
  )
  _register_meta_op(
      torch.ops.aten._fused_adagrad.tensor_lr, _meta_fused_adagrad_tensor_lr
  )
  # pylint: enable=g-import-not-at-top,protected-access


_register_fused_optimizer_meta_kernels()

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "AsyncCompilationSubmitted",
    "TpuBackend",
    "TpuCompileDebug",
    "async_compile",
    "resolve_compilations",
    "tpu_compile",
    # go/keep-sorted end
]
