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

"""TPU backend registrations for PyTorch functional RNG higher-order ops."""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch._higher_order_ops import utils as hop_utils
from torch._prims import rng_prims
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.utils import _pytree as pytree
from torch_tpu._internal.device import _device_ops_backend

_TPU_DEVICE_TYPES = frozenset(("tpu", "privateuseone"))
_registered = False


def _is_tpu_device(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
  return rng_prims.get_device(args, kwargs) in _TPU_DEVICE_TYPES


def _resolve_tpu_device_index(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> int:
  """Returns the TPU device ordinal from op args/kwargs, or -1 for current."""
  device = kwargs.get("device")
  if device is not None:
    if isinstance(device, str):
      device = torch.device(device)
    if (
        isinstance(device, torch.device)
        and device.type in _TPU_DEVICE_TYPES
        and device.index is not None
    ):
      return device.index
    return -1
  for arg in args:
    if isinstance(arg, torch.Tensor) and arg.device.type in _TPU_DEVICE_TYPES:
      return arg.device.index if arg.device.index is not None else -1
    if isinstance(arg, torch.device) and arg.type in _TPU_DEVICE_TYPES:
      return arg.index if arg.index is not None else -1
  return -1


def _run_and_save_rng_state_tpu(
    op: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[torch.Tensor, Any]:
  # pylint: disable=protected-access
  if torch._guards.detect_fake_mode() is not None:
    rng_state = torch.empty((16,), dtype=torch.uint8, device="cpu")
  else:
    device_index = _resolve_tpu_device_index(args, kwargs)
    rng_state = _device_ops_backend.get_rng_state(device_index)
  # pylint: enable=protected-access
  return rng_state, op(*args, **kwargs)


def _run_with_rng_state_tpu(
    rng_state: torch.Tensor,
    op: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
  device_index = _resolve_tpu_device_index(args, kwargs)
  current_state = _device_ops_backend.get_rng_state(device_index)
  _device_ops_backend.set_rng_state(rng_state.cpu(), device_index)
  try:
    return op(*args, **kwargs)
  finally:
    _device_ops_backend.set_rng_state(current_state, device_index)


def _patch_run_and_save_rng_state() -> None:
  """Registers TPU support for `torch.ops.higher_order.run_and_save_rng_state`."""
  # pylint: disable=protected-access
  hop = rng_prims.run_and_save_rng_state
  privateuse1_key = torch._C.DispatchKey.PrivateUse1
  backend_select_key = torch._C.DispatchKey.BackendSelect

  if privateuse1_key not in hop.py_kernels:
    hop.py_impl(privateuse1_key)(_run_and_save_rng_state_tpu)

  orig_backend_select = hop.py_kernels.pop(backend_select_key, None)
  if orig_backend_select is not None:

    def _backend_select(op: Callable[..., Any], *args: Any, **kwargs: Any):
      if _is_tpu_device(args, kwargs):
        return _run_and_save_rng_state_tpu(op, *args, **kwargs)
      return orig_backend_select(op, *args, **kwargs)

    hop.py_impl(backend_select_key)(_backend_select)
    hop._dispatch_cache.clear()

  # Upstream's FakeTensorMode and ProxyTorchDispatchMode handlers capture
  # upstream's local `impl_backend_select` closure directly, so wrap them too.
  registered_fake_fns = getattr(hop_utils, "registered_hop_fake_fns", None)
  if registered_fake_fns is not None and hop in registered_fake_fns:
    orig_fake = registered_fake_fns[hop]

    def _fake_impl(op: Callable[..., Any], *args: Any, **kwargs: Any):
      if _is_tpu_device(args, kwargs):
        return _run_and_save_rng_state_tpu(op, *args, **kwargs)
      return orig_fake(op, *args, **kwargs)

    registered_fake_fns[hop] = _fake_impl
  elif fake_tensor.FakeTensorMode in hop.python_key_table:
    orig_fake_mode = hop.python_key_table.pop(fake_tensor.FakeTensorMode)

    def _fake_mode_impl(
        mode: fake_tensor.FakeTensorMode,
        op: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
      if _is_tpu_device(args, kwargs):
        with mode:
          return _run_and_save_rng_state_tpu(op, *args, **kwargs)
      return orig_fake_mode(mode, op, *args, **kwargs)

    hop.py_impl(fake_tensor.FakeTensorMode)(_fake_mode_impl)

  orig_proxy = hop.python_key_table.pop(
      proxy_tensor.ProxyTorchDispatchMode, None
  )
  if orig_proxy is not None:

    def _proxy_impl(
        mode: proxy_tensor.ProxyTorchDispatchMode,
        op: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
      if _is_tpu_device(args, kwargs):
        out = _run_and_save_rng_state_tpu(op, *args, **kwargs)
        tracer: Any = mode.tracer
        proxy_args = pytree.tree_map(tracer.unwrap_proxy, (op, *args))
        proxy_kwargs = pytree.tree_map(tracer.unwrap_proxy, kwargs)
        out_proxy = tracer.create_proxy(
            "call_function", hop, proxy_args, proxy_kwargs
        )
        return proxy_tensor.track_tensor_tree(
            out, out_proxy, constant=None, tracer=tracer
        )
      return orig_proxy(mode, op, *args, **kwargs)

    hop.py_impl(proxy_tensor.ProxyTorchDispatchMode)(_proxy_impl)
  # pylint: enable=protected-access


def _patch_run_with_rng_state() -> None:
  """Registers TPU support for `torch.ops.higher_order.run_with_rng_state`."""
  # pylint: disable=protected-access
  hop = rng_prims.run_with_rng_state
  privateuse1_key = torch._C.DispatchKey.PrivateUse1
  backend_select_key = torch._C.DispatchKey.BackendSelect

  if privateuse1_key not in hop.py_kernels:
    hop.py_impl(privateuse1_key)(_run_with_rng_state_tpu)

  orig_backend_select = hop.py_kernels.pop(backend_select_key, None)
  if orig_backend_select is not None:

    def _backend_select(
        rng_state: torch.Tensor,
        op: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
      if _is_tpu_device(args, kwargs):
        return _run_with_rng_state_tpu(rng_state, op, *args, **kwargs)
      return orig_backend_select(rng_state, op, *args, **kwargs)

    hop.py_impl(backend_select_key)(_backend_select)
    hop._dispatch_cache.clear()
  # pylint: enable=protected-access


def register() -> None:
  """Registers TPU support for `run_and_save_rng_state` and `run_with_rng_state`."""
  global _registered
  if _registered:
    return
  _patch_run_and_save_rng_state()
  _patch_run_with_rng_state()
  _registered = True
